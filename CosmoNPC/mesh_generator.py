import numpy as np
from pmesh import ParticleMesh
from numpy.lib import format
from mpi4py import MPI
import logging
# from nbodykit.lab import *
from .catalog_processor import npy_reader,fits_reader, add_completeness_weight,catalog_reader
import warnings
import gc


warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_mesh_box(catalogs, correlation_mode, nmesh, geometry, boxsize, 
                 sampler, interlaced, column_names, comm, statistic, tracer_type = "ab"):
    """
    Generate mesh fields and compute number densities for multi-galaxy catalogs in periodic boxes.
    """

    rank = comm.Get_rank()

    Nmesh = np.array(nmesh)
    boxsize = np.array(boxsize)

    # Load data catalogs
    # tracer a
    data_a = catalog_reader(catalogs["data_a"], geometry, column_names, None, None, None, comm)
    sub_N_gal_a = data_a['WEIGHT'].shape[0]
    sub_weight_sum_a = np.sum(data_a['WEIGHT'])
    N_gal_a = comm.reduce(sub_N_gal_a, op=MPI.SUM, root=0)
    weight_sum_a = comm.reduce(sub_weight_sum_a, op=MPI.SUM, root=0)
    NZ_a = weight_sum_a / np.prod(boxsize) if rank == 0 else None

    if correlation_mode == "cross":
        data_b = catalog_reader(catalogs["data_b"], geometry, column_names, None, None, None, comm)
        sub_N_gal_b = data_b['WEIGHT'].shape[0]
        sub_weight_sum_b = np.sum(data_b['WEIGHT'])
        N_gal_b = comm.reduce(sub_N_gal_b, op=MPI.SUM, root=0)
        weight_sum_b = comm.reduce(sub_weight_sum_b, op=MPI.SUM, root=0)
        NZ_b = weight_sum_b / np.prod(boxsize) if rank == 0 else None

    if tracer_type == "abc":
        data_c = catalog_reader(catalogs["data_c"], geometry, column_names, None, None, None, comm)
        sub_N_gal_c = data_c['WEIGHT'].shape[0]
        sub_weight_sum_c = np.sum(data_c['WEIGHT'])
        N_gal_c = comm.reduce(sub_N_gal_c, op=MPI.SUM, root=0)
        weight_sum_c = comm.reduce(sub_weight_sum_c, op=MPI.SUM, root=0)
        NZ_c = weight_sum_c / np.prod(boxsize) if rank == 0 else None
        if rank == 0:
            logging.info(f"Total number of galaxies for tracer_c: {N_gal_c}, mean number density NZ_c: {NZ_c}")


    # Mesh attributes
    mesh_attrs = {}
    if rank == 0:
        if correlation_mode == "auto":
            mesh_attrs.update({'NZ_a': NZ_a, "N_gal_a": N_gal_a})
        else:
            mesh_attrs.update({'NZ_a': NZ_a, 'NZ_b': NZ_b, "N_gal_a": N_gal_a, "N_gal_b": N_gal_b})
        if tracer_type == "abc":
            mesh_attrs.update({'NZ_c': NZ_c, "N_gal_c": N_gal_c})
        mesh_attrs.update({'interlaced': interlaced, 'sampler': sampler})

    # boardcast mesh_attrs to all ranks
    mesh_attrs = comm.bcast(mesh_attrs, root=0)

    # Mesh generation
    rfield_a = pm_painter(data_a['Position'], data_a['WEIGHT'], Nmesh, boxsize, sampler, interlaced, comm)
    if rank == 0:
        logging.info(f"Mesh for tracer_a generated.")
    if correlation_mode == "cross":
        rfield_b = pm_painter(data_b['Position'], data_b['WEIGHT'], Nmesh, boxsize, sampler, interlaced, comm)
        if rank == 0:
            logging.info(f"Mesh for tracer_b generated.")
    else:
        rfield_b = None


    if tracer_type == "abc":
        rfield_c = pm_painter(data_c['Position'], data_c['WEIGHT'], Nmesh, boxsize, sampler, interlaced, comm)
        if rank == 0:
            logging.info(f"Mesh for tracer_c generated.")
    else:
        rfield_c = None
    
    # Memory cleanup
    del data_a
    if correlation_mode == "cross":
        del data_b
    if tracer_type == "abc":
        del data_c
    gc.collect()

    if statistic == "pk":
        return mesh_attrs, rfield_a, rfield_b
    elif statistic == "bk_sugi":
        return mesh_attrs, rfield_a, rfield_b, rfield_c



def pm_painter(Position, WEIGHT, Nmesh, boxsize, sampler,interlaced,comm, boxcenter = None):
    """
    Paint particles onto a mesh using ParticleMesh.
    Returns:
        rfield (ndarray): Painted mesh field.
    """
    rank = comm.Get_rank()

    if boxcenter is not None:
        Position = Position - boxcenter  # to be checked later

    # pmesh does not support the name "ngp", map it to "NEAREST"
    if sampler == "ngp":
        sampler = "NEAREST"

    pm = ParticleMesh(BoxSize=boxsize, Nmesh=Nmesh, dtype='complex128', resampler=sampler, comm=comm)

    if not interlaced:
        lay = pm.decompose(Position, smoothing=0.5 * pm.resampler.support)
    else:
        lay = pm.decompose(Position,smoothing= 1.0 * pm.resampler.support)

    if not interlaced:
        rfield = pm.paint(Position, mass=WEIGHT, layout=lay)
    else:
        shifted = pm.affine.shift(0.5)
        p = lay.exchange(Position)
        w = lay.exchange(WEIGHT)
        c1 = pm.paint(p, mass=w).r2c()
        c2 = pm.paint(p, mass=w, transform=shifted).r2c()
        logging.info(f"Interlaced painting and R2C done in Rank {rank}, start combining fields")

        # 2. Combine the two fields in the Fourier domain
        H = pm.BoxSize / pm.Nmesh
        for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
            kH = sum(k[i] * H[i] for i in range(3))  # compute k·H (wavevector dot cell size)
            # Core merging formula: weighted average with phase correction
            s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * np.exp(0.5 * 1j * kH)
        rfield = c1.c2r()

        del c1, c2, p, w
        gc.collect()

    return rfield

 
def get_mesh_pk_survey(catalogs, correlation_mode, nmesh, geometry, column_names, \
                       boxsize, sampler, interlaced,z_range, comp_weight_plan, para_cosmo, \
                        comm, normalization_scheme,alpha_scheme = "pypower"):
    
    """
        Generate meshes for power spectrum measurement from survey-like catalog in both single and multiple tracer cases.
        Calculate Normaliation I factors and Shot noise N0.
        2 approaches for normalization are provided: particle-normalization and mixed-mesh normalization.
    """
    # Initialize MPI
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Store attributes for the mesh
    mesh_attrs = {}

    # Convert nmesh and boxsize to numpy arrays
    Nmesh = np.array(nmesh)
    boxsize = np.array(boxsize)
    boxcenter = None

    # Load data and random catalogs
    # firstly we use randoms_a to determine the boxcenter
    randoms_a, boxcenter = catalog_reader(catalogs["randoms_a"], geometry, column_names, z_range, \
                               comp_weight_plan, para_cosmo, comm,catalog_type="randoms",\
                                normalization_scheme=normalization_scheme)
    
    mesh_attrs.update({'boxcenter': boxcenter})

    data_a = catalog_reader(catalogs["data_a"], geometry, column_names, z_range, \
                           comp_weight_plan, para_cosmo, comm,boxcenter=boxcenter, catalog_type="data",\
                            normalization_scheme=normalization_scheme)

    # obtain some necessary sums for alpha, particle-normalization and shot noise calculation
    particle_sums_a = compute_particle_sums_pk(data_a, randoms_a, correlation_mode, comm)

    if rank == 0:
        (weight_sum_data_a, weight_sum_randoms_a,
         weight_sum_data_wofkp_a, weight_sum_randoms_wofkp_a,
         I_sum_data_a, I_sum_randoms_a,
         N_sum_data_a, N_sum_randoms_a) = particle_sums_a
        
        # calculate alpha
        if alpha_scheme == "pypower":
            alpha_a = weight_sum_data_a / weight_sum_randoms_a # pypower scheme
            if rank == 0:
                logging.info(f"Using same alpha scheme as pypower.")
        elif alpha_scheme == "nbodykit":
            alpha_a = weight_sum_data_wofkp_a / weight_sum_randoms_wofkp_a  # nbodykit scheme
            if rank == 0:
                logging.info(f"Using same alpha scheme as nbodykit.")

        logging.info(f"alpha_a = {alpha_a}")


        # calculate normalization I, when using particle-normalization scheme
        # if normalization_scheme == "particle":
        I_rand = alpha_a * I_sum_randoms_a
        I_data = I_sum_data_a
        logging.info(f"Normalization a: I_rand = {I_rand}, I_data = {I_data}")

        # store attributes in mesh_attrs
        mesh_attrs.update({
            'alpha_a': alpha_a,
            'I_rand': I_rand,
            'I_data': I_data,
        })
    
    # broadcast mesh_attrs to all ranks
    mesh_attrs = comm.bcast(mesh_attrs, root=0)
    weight_sum_data_a = comm.bcast(weight_sum_data_a if rank == 0 else None, root=0)
    weight_sum_randoms_a = comm.bcast(weight_sum_randoms_a if rank == 0 else None, root=0)
    alpha_a = comm.bcast(alpha_a if rank == 0 else None, root=0)


    # Generate meshes for tracer_a
    if rank == 0:
        logging.info(f"Start generating meshes for data catalog of tracer_a.")
    rfield_data_a = pm_painter(data_a['Position'], data_a['WEIGHT']*data_a["WEIGHT_FKP"],\
                           Nmesh, boxsize, sampler, interlaced, comm, boxcenter=boxcenter)
    
    if rank == 0:
        logging.info(f"Start generating meshes for randoms catalog of tracer_a.")
    rfield_randoms_a = pm_painter(randoms_a['Position'], randoms_a['WEIGHT']*randoms_a["WEIGHT_FKP"],\
                           Nmesh, boxsize, sampler, interlaced, comm, boxcenter=boxcenter)
    
    # revover to physical number density
    rfield_data_a /= np.prod(boxsize) / np.prod(nmesh)
    rfield_randoms_a /= np.prod(boxsize) / np.prod(nmesh)

    # FKP overdensity field
    rfield_a = rfield_data_a - alpha_a * rfield_randoms_a



    if correlation_mode == "auto":
        rfield_b = rfield_a
        # mixed mesh normalization
    else:
        # Load data and random catalogs for tracer_b
        data_b = catalog_reader(catalogs["data_b"], geometry, column_names, z_range, \
                               comp_weight_plan, para_cosmo, comm, boxcenter=boxcenter, catalog_type="data",\
                                normalization_scheme=normalization_scheme)
        randoms_b = catalog_reader(catalogs["randoms_b"], geometry, column_names, z_range, \
                                   comp_weight_plan, para_cosmo, comm, boxcenter=boxcenter, catalog_type="randoms",\
                                    normalization_scheme=normalization_scheme)
        
        # obtain some necessary sums for alpha, particle-normalization and shot noise calculation
        particle_sums_b = compute_particle_sums_pk(data_b, randoms_b, correlation_mode, comm)
        if rank == 0:
            (weight_sum_data_b, weight_sum_randoms_b,
             weight_sum_data_wofkp_b, weight_sum_randoms_wofkp_b,
             I_sum_data_b, I_sum_randoms_b,
             N_sum_data_b, N_sum_randoms_b) = particle_sums_b
            
            # calculate alpha
            if alpha_scheme == "pypower":
                alpha_b = weight_sum_data_b / weight_sum_randoms_b # pypower scheme
            elif alpha_scheme == "nbodykit":
                alpha_b = weight_sum_data_wofkp_b / weight_sum_randoms_wofkp_b  # nbodykit scheme
            logging.info(f"Alpha_b = {alpha_b}")  
            # store attributes in mesh_attrs
            mesh_attrs.update({
                'alpha_b': alpha_b,
            })

        # broadcast updated mesh_attrs to all ranks
        mesh_attrs = comm.bcast(mesh_attrs, root=0) 
        alpha_b = comm.bcast(alpha_b if rank == 0 else None, root=0)

        # Generate meshes for tracer_b
        if rank == 0:
            logging.info(f"Start generating meshes for data catalog of tracer_b.")
        rfield_data_b = pm_painter(data_b['Position'], data_b['WEIGHT']*data_b["WEIGHT_FKP"],\
                               Nmesh, boxsize, sampler, interlaced, comm, boxcenter=boxcenter)
        if rank == 0:
            logging.info(f"Start generating meshes for randoms catalog of tracer_b.")
        rfield_randoms_b = pm_painter(randoms_b['Position'], randoms_b['WEIGHT']*randoms_b["WEIGHT_FKP"],\
                               Nmesh, boxsize, sampler, interlaced, comm, boxcenter=boxcenter)
        
        # revover to physical number density
        rfield_data_b /= np.prod(boxsize) / np.prod(nmesh)
        rfield_randoms_b /= np.prod(boxsize) / np.prod(nmesh)

        # FKP overdensity field
        rfield_b = rfield_data_b - alpha_b * rfield_randoms_b

    # calculate the normalization and shot noise
    if normalization_scheme == "particle":
        if correlation_mode == "auto":
            if rank == 0:
                N0 = (N_sum_data_a + alpha_a**2 * N_sum_randoms_a) / I_rand
                mesh_attrs.update({
                    'N0': N0,
                })
        else:
            raise ValueError("Particle-normalization scheme will not work for cross-correlation!")
    else: # mixed-mesh normalization
        """
        Following the mixed-mesh normalization scheme in pypower:
        I = 1/dV \frac{\alpha_{2} \sum_{i} n_{d,1}^{i} n_{r,2}^{i} + \alpha_{1} \sum_{i} n_{d,2}^{i} n_{r,1}^{i}}{2}
        """
        if correlation_mode == "auto":
            I_mesh_sub = alpha_a * np.sum(rfield_data_a * rfield_randoms_a) \
                * np.prod(boxsize)/np.prod(nmesh)
        else: # cross-correlation
            I_mesh_sub = 0.5* np.prod(boxsize)/np.prod(nmesh) * \
                            (alpha_b * np.sum(rfield_data_a * rfield_randoms_b) \
                            + alpha_a * np.sum(rfield_data_b * rfield_randoms_a))
        I_mesh = comm.reduce(I_mesh_sub, op=MPI.SUM, root=0)
        I_mesh = np.float64(I_mesh)
        if rank == 0:
            if correlation_mode == "auto":
                N0 = (N_sum_data_a + alpha_a**2 * N_sum_randoms_a) / I_mesh
            else:
                N0 = 0.0  # shot noise for cross-correlation is set to 0.0
            mesh_attrs.update({
                'I_mesh': I_mesh,
                'N0': N0,
            })

    # broadcast updated mesh_attrs to all ranks
    mesh_attrs = comm.bcast(mesh_attrs, root=0)
    
    # Memory cleanup
    del data_a, randoms_a, rfield_data_a, rfield_randoms_a
    if correlation_mode == "cross":
        del data_b, randoms_b, rfield_data_b, rfield_randoms_b
    gc.collect()

    return mesh_attrs, rfield_a, rfield_b



def compute_particle_sums_pk(data, randoms, correlation_mode, comm):
    """
    Compute necessary sums for alpha, normalization, and shot noise calculations for power spectrum measurement.
    Returns results only on rank 0.
    """
    rank = comm.Get_rank()

    # for alpha
    sub_weight_sum_data = np.sum(data['WEIGHT'] * data['WEIGHT_FKP'])
    sub_weight_sum_randoms = np.sum(randoms['WEIGHT'] * randoms['WEIGHT_FKP'])

    # for comparison between different alpha calculation methods
    sub_weight_sum_data_wofkp = np.sum(data['WEIGHT'])
    sub_weight_sum_randoms_wofkp = np.sum(randoms['WEIGHT'])

    # for particle-normalization
    sub_I_sum_data = np.sum(data['WEIGHT'] * data['NZ'] * data['WEIGHT_FKP'] * data['WEIGHT_FKP'])
    sub_I_sum_randoms = np.sum(randoms['WEIGHT'] * randoms['NZ'] * randoms['WEIGHT_FKP'] * randoms['WEIGHT_FKP'])

    # for shot noise
    if correlation_mode == "auto":
        sub_N_sum_data = np.sum((data['WEIGHT'] * data['WEIGHT_FKP'])**2)
        sub_N_sum_randoms = np.sum((randoms['WEIGHT'] * randoms['WEIGHT_FKP'])**2)
    else:
        sub_N_sum_data = 0.0 # although these values are not used in cross-correlation, we still set them to 0.0 to avoid errors
        sub_N_sum_randoms = 0.0

    # collect all necessary sub_sums to rank 0 and sum them up
    gathered_results = comm.gather(
        [sub_weight_sum_data, sub_weight_sum_randoms,
            sub_weight_sum_data_wofkp, sub_weight_sum_randoms_wofkp,
            sub_I_sum_data, sub_I_sum_randoms,
            sub_N_sum_data, sub_N_sum_randoms],
        root=0
    )
    if rank == 0:
        # Aggregate results from all ranks
        results = map(np.sum, zip(*gathered_results))
        return tuple(results)
    else:
        return None


    
    

    