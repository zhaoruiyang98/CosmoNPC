import numpy as np
from pmesh import ParticleMesh
from numpy.lib import format
from mpi4py import MPI
import logging
from .math_funcs import *
from .catalog_processor import *
import warnings
import gc


warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_mesh_box(catalogs, correlation_mode, nmesh, geometry, boxsize, 
                 sampler, interlaced, column_names, comm, statistic, \
                tracer_type = "ab", apply_rsd = False,\
                      para_cosmo = None, redshift_box = None, los = None):
    """
    Generate mesh fields and compute number densities for multi-galaxy catalogs in periodic boxes.
    """

    rank = comm.Get_rank()

    Nmesh = np.array(nmesh)
    boxsize = np.array(boxsize)

    # Load data catalogs
    # tracer a
    data_a = catalog_reader(catalogs["data_a"], geometry, column_names, None, None, \
                            comm, para_cosmo=para_cosmo, apply_rsd=apply_rsd, \
                                redshift_box=redshift_box, boxsize=boxsize, los=los)
    sub_N_gal_a = data_a['WEIGHT'].shape[0]
    sub_weight_sum_a = np.sum(data_a['WEIGHT'])
    N_gal_a = comm.reduce(sub_N_gal_a, op=MPI.SUM, root=0)
    weight_sum_a = comm.reduce(sub_weight_sum_a, op=MPI.SUM, root=0)
    NZ_a = weight_sum_a / np.prod(boxsize) if rank == 0 else None

    if correlation_mode == "cross":
        data_b = catalog_reader(catalogs["data_b"], geometry, column_names, None, None, \
                                comm, para_cosmo=para_cosmo, apply_rsd=apply_rsd, \
                                    redshift_box=redshift_box, boxsize=boxsize, los=los)
        sub_N_gal_b = data_b['WEIGHT'].shape[0]
        sub_weight_sum_b = np.sum(data_b['WEIGHT'])
        N_gal_b = comm.reduce(sub_N_gal_b, op=MPI.SUM, root=0)
        weight_sum_b = comm.reduce(sub_weight_sum_b, op=MPI.SUM, root=0)
        NZ_b = weight_sum_b / np.prod(boxsize) if rank == 0 else None

    if tracer_type == "abc":
        data_c = catalog_reader(catalogs["data_c"], geometry, column_names, None, None, \
                                comm, para_cosmo=para_cosmo, apply_rsd=apply_rsd, \
                                    redshift_box=redshift_box, boxsize=boxsize, los=los)
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
                               comp_weight_plan, comm, para_cosmo=para_cosmo, boxsize=boxsize, catalog_type="randoms",\
                                normalization_scheme=normalization_scheme)
    
    mesh_attrs.update({'boxcenter': boxcenter})

    data_a = catalog_reader(catalogs["data_a"], geometry, column_names, z_range, \
                           comp_weight_plan, comm, para_cosmo=para_cosmo, boxsize=boxsize, boxcenter=boxcenter, catalog_type="data",\
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
                               comp_weight_plan, comm, para_cosmo=para_cosmo, boxsize=boxsize, boxcenter=boxcenter, catalog_type="data",\
                                normalization_scheme=normalization_scheme)
        randoms_b = catalog_reader(catalogs["randoms_b"], geometry, column_names, z_range, \
                                   comp_weight_plan, comm, para_cosmo=para_cosmo, boxsize=boxsize, boxcenter=boxcenter, catalog_type="randoms",\
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
        r"""
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



def get_mesh_bk_survey(catalogs, correlation_mode, tracer_type, angu_config, nmesh, geometry, column_names, \
                       boxsize, sampler, interlaced, z_range, comp_weight_plan, para_cosmo, \
                       comm, normalization_scheme="particle", alpha_scheme="pypower"):
    """
    Framework function for survey-like bispectrum meshes.
    Returns mesh_attrs, rfield_a, rfield_b, rfield_c.
    """
    rank = comm.Get_rank()

    assert geometry == "survey-like", "get_mesh_bk_survey only supports survey-like geometry."
    assert correlation_mode in ["auto", "cross"], "correlation_mode must be 'auto' or 'cross'."
    assert tracer_type in ["aaa", "aab", "abb", "abc"], "tracer_type must be one of 'aaa', 'aab', 'abb', 'abc'."
    assert normalization_scheme in ["particle", "mesh", "mixed-mesh"], \
        "normalization_scheme must be one of 'particle', 'mesh', 'mixed-mesh'."
    if correlation_mode == "auto":
        assert tracer_type == "aaa", "For auto-correlation, tracer_type must be 'aaa'."

    Nmesh = np.array(nmesh)
    boxsize = np.array(boxsize)
    vol_per_cell = np.prod(boxsize) / np.prod(nmesh)
    mesh_attrs = {"normalization_scheme": normalization_scheme}
    boxcenter = None
    tracer_cache = {}

    def _prepare_tracer(label, data_key, random_key, keep_data_mesh=False, keep_randoms_mesh=False):
        """
        Prepare one tracer and return three fields with a fixed interface:
        - rfield: signal field F = D - alpha * R (always needed by bispectrum signal)
        - rfield_data: mesh-painted data field (kept only when normalization needs it)
        - rfield_randoms: mesh-painted random field (kept only when normalization needs it)
        """
        randoms = catalog_reader(
            catalogs[random_key], geometry, column_names, z_range, comp_weight_plan, comm,
            para_cosmo=para_cosmo, boxsize=boxsize, boxcenter=boxcenter,
            catalog_type="randoms", normalization_scheme=normalization_scheme
        )
        data = catalog_reader(
            catalogs[data_key], geometry, column_names, z_range, comp_weight_plan, comm,
            para_cosmo=para_cosmo, boxsize=boxsize, boxcenter=boxcenter,
            catalog_type="data", normalization_scheme=normalization_scheme
        )

        sums_bk = compute_particle_sums_bk(data, randoms, correlation_mode, angu_config, comm)
        if rank == 0:
            (weight_sum_data, weight_sum_randoms,
             weight_sum_data_wofkp, weight_sum_randoms_wofkp,
             I_33, S_LM_data, S_LM_randoms) = sums_bk
            if alpha_scheme == "pypower":
                alpha = weight_sum_data / weight_sum_randoms
                logging.info("Using same alpha scheme as pypower.")
            elif alpha_scheme == "nbodykit":
                alpha = weight_sum_data_wofkp / weight_sum_randoms_wofkp
                logging.info("Using same alpha scheme as nbodykit.")
            else:
                raise ValueError("alpha_scheme must be either 'pypower' or 'nbodykit'.")

            I_33 *= alpha
            S_LM = S_LM_data - alpha**3 * S_LM_randoms
            mesh_attrs[f"alpha_{label}"] = alpha
            mesh_attrs[f"I_33_{label}"] = I_33
            mesh_attrs[f"S_LM_{label}"] = S_LM
        else:
            alpha = None
        alpha = comm.bcast(alpha, root=0)

        if rank == 0:
            logging.info(f"Start generating mesh for data catalog of tracer_{label}.")
        rfield_data = pm_painter(
            data["Position"], data["WEIGHT"] * data["WEIGHT_FKP"],
            Nmesh, boxsize, sampler, interlaced, comm, boxcenter=boxcenter
        )
        if rank == 0:
            logging.info(f"Start generating mesh for randoms catalog of tracer_{label}.")
        rfield_randoms = pm_painter(
            randoms["Position"], randoms["WEIGHT"] * randoms["WEIGHT_FKP"],
            Nmesh, boxsize, sampler, interlaced, comm, boxcenter=boxcenter
        )

        rfield_data *= 1/vol_per_cell
        rfield_randoms *= alpha/vol_per_cell
        rfield = rfield_data - rfield_randoms

        del data, randoms
        gc.collect()

        if not keep_data_mesh:
            del rfield_data
            rfield_data = None
        if not keep_randoms_mesh:
            del rfield_randoms
            rfield_randoms = None
        gc.collect()

        return rfield, rfield_data, rfield_randoms

    # Build tracer_a first (defines boxcenter from randoms_a if needed).
    randoms_a_preview, boxcenter = catalog_reader(
        catalogs["randoms_a"], geometry, column_names, z_range, comp_weight_plan, comm,
        para_cosmo=para_cosmo, boxsize=boxsize, catalog_type="randoms",
        normalization_scheme=normalization_scheme
    )
    del randoms_a_preview
    gc.collect()

    mesh_attrs["boxcenter"] = boxcenter

    # Decide which auxiliary meshes should be kept for normalization.
    keep_flags = {
        "a": {"keep_data_mesh": False, "keep_randoms_mesh": False},
        "b": {"keep_data_mesh": False, "keep_randoms_mesh": False},
        "c": {"keep_data_mesh": False, "keep_randoms_mesh": False},
    }
    if normalization_scheme == "mesh":
        keep_flags["a"]["keep_randoms_mesh"] = True
    elif normalization_scheme == "mixed-mesh":
        if correlation_mode == "auto":
            # Auto mixed-mesh needs D_a from cache, but R_a must be split odd/even so it is rebuilt later.
            keep_flags["a"]["keep_data_mesh"] = True
        else:
            if tracer_type == "abc":
                keep_flags["a"]["keep_randoms_mesh"] = True
                keep_flags["b"]["keep_randoms_mesh"] = True
                keep_flags["c"]["keep_randoms_mesh"] = True
            elif tracer_type == "aab":
                # Correct formula: D_a * R_a * R_b
                keep_flags["a"]["keep_data_mesh"] = True
                keep_flags["a"]["keep_randoms_mesh"] = True
                keep_flags["b"]["keep_randoms_mesh"] = True
            elif tracer_type == "abb":
                # Correct formula: R_a * R_b * D_b
                keep_flags["a"]["keep_randoms_mesh"] = True
                keep_flags["b"]["keep_randoms_mesh"] = True
                keep_flags["b"]["keep_data_mesh"] = True

    rfield_a, rfield_data_a, rfield_randoms_a = _prepare_tracer(
        "a", "data_a", "randoms_a",
        keep_data_mesh=keep_flags["a"]["keep_data_mesh"],
        keep_randoms_mesh=keep_flags["a"]["keep_randoms_mesh"]
    )
    tracer_cache["a"] = {
        "rfield": rfield_a,
        "rfield_data": rfield_data_a,
        "rfield_randoms": rfield_randoms_a,
    }

    if correlation_mode == "auto":
        rfield_b = None
        rfield_c = None
        tracer_cache["b"] = {"rfield": None, "rfield_data": None, "rfield_randoms": None}
        tracer_cache["c"] = {"rfield": None, "rfield_data": None, "rfield_randoms": None}
    else:
        rfield_b, rfield_data_b, rfield_randoms_b = _prepare_tracer(
            "b", "data_b", "randoms_b",
            keep_data_mesh=keep_flags["b"]["keep_data_mesh"],
            keep_randoms_mesh=keep_flags["b"]["keep_randoms_mesh"]
        )
        tracer_cache["b"] = {
            "rfield": rfield_b,
            "rfield_data": rfield_data_b,
            "rfield_randoms": rfield_randoms_b,
        }
        if tracer_type == "abc":
            rfield_c, rfield_data_c, rfield_randoms_c = _prepare_tracer(
                "c", "data_c", "randoms_c",
                keep_data_mesh=keep_flags["c"]["keep_data_mesh"],
                keep_randoms_mesh=keep_flags["c"]["keep_randoms_mesh"]
            )
            tracer_cache["c"] = {
                "rfield": rfield_c,
                "rfield_data": rfield_data_c,
                "rfield_randoms": rfield_randoms_c,
            }
        else:
            rfield_c = None
            tracer_cache["c"] = {"rfield": None, "rfield_data": None, "rfield_randoms": None}


    # Normalization architecture placeholders.
    # if rank == 0:
    if normalization_scheme == "particle":
        assert correlation_mode == "auto", "Particle-normalization scheme is only applicable for auto-correlation."
        if rank == 0:
            mesh_attrs["I_norm"] = mesh_attrs[f"I_33_a"]
    elif normalization_scheme == "mesh":
        assert correlation_mode == "auto", "Mesh-normalization scheme is only applicable for auto-correlation."
        rfield_randoms_a = tracer_cache["a"]["rfield_randoms"]
        if rfield_randoms_a is None:
            raise RuntimeError("rfield_randoms_a is required for mesh normalization but is None.")
        sub_I_mesh = vol_per_cell * np.sum(rfield_randoms_a**3)
        I_mesh = comm.reduce(sub_I_mesh, op=MPI.SUM, root=0)
        if rank == 0:
            mesh_attrs["I_norm"] = I_mesh        
    else:  # mixed-mesh
        if correlation_mode == "auto":
            rfield_data_a = tracer_cache["a"]["rfield_data"]
            if rfield_data_a is None:
                raise RuntimeError("rfield_data_a is required for mixed-mesh auto normalization but is None.")
            
            data_a = catalog_reader(catalogs["data_a"], geometry, column_names, z_range, comp_weight_plan, comm,\
                                    para_cosmo=para_cosmo, boxsize=boxsize, boxcenter=boxcenter, catalog_type="data",\
                                    normalization_scheme=normalization_scheme)

            randoms_a = catalog_reader(catalogs["randoms_a"], geometry, column_names, z_range, comp_weight_plan, comm,
                para_cosmo=para_cosmo, boxsize=boxsize, boxcenter=boxcenter,
                catalog_type="randoms", normalization_scheme=normalization_scheme)
                
            # split the randoms catalog into 2 sub-catalogs by odd-even indexing
            if rank == 0:
                logging.info(f"Start generating meshes for data and randoms catalogs of tracer_a for mixed-mesh normalization.")
            mask_odd = np.arange(randoms_a['Position'].shape[0]) % 2 == 1
            randoms_a_odd = randoms_a[mask_odd]
            randoms_a_even = randoms_a[~mask_odd]

            # get alphas for randoms_a_odd and randoms_a_even
            if alpha_scheme == "pypower":
                sub_weight_sum_data_a= np.sum(data_a['WEIGHT'] * data_a['WEIGHT_FKP'])
                sub_weight_sum_randoms_a_odd = np.sum(randoms_a_odd['WEIGHT'] * randoms_a_odd['WEIGHT_FKP'])
                sub_weight_sum_randoms_a_even = np.sum(randoms_a_even['WEIGHT'] * randoms_a_even['WEIGHT_FKP'])
            elif alpha_scheme == "nbodykit":
                sub_weight_sum_data_a = np.sum(data_a['WEIGHT'])
                sub_weight_sum_randoms_a_odd = np.sum(randoms_a_odd['WEIGHT'])
                sub_weight_sum_randoms_a_even = np.sum(randoms_a_even['WEIGHT'])
            else:
                raise ValueError("alpha_scheme must be either 'pypower' or 'nbodykit'.")
            
            weight_sum_data_a = comm.reduce(sub_weight_sum_data_a, op=MPI.SUM, root=0)
            weight_sum_randoms_a_odd = comm.reduce(sub_weight_sum_randoms_a_odd, op=MPI.SUM, root=0)
            weight_sum_randoms_a_even = comm.reduce(sub_weight_sum_randoms_a_even, op=MPI.SUM, root=0)
            alpha_a_odd = weight_sum_data_a / weight_sum_randoms_a_odd if rank == 0 else None
            alpha_a_even = weight_sum_data_a / weight_sum_randoms_a_even if rank == 0 else None
            alpha_a_odd = comm.bcast(alpha_a_odd, root=0)
            alpha_a_even = comm.bcast(alpha_a_even, root=0)

            rfield_randoms_a_odd = pm_painter(randoms_a_odd['Position'], randoms_a_odd['WEIGHT']*randoms_a_odd["WEIGHT_FKP"],\
                                Nmesh, boxsize, sampler, interlaced, comm, boxcenter=boxcenter)
            rfield_randoms_a_even = pm_painter(randoms_a_even['Position'], randoms_a_even['WEIGHT']*randoms_a_even["WEIGHT_FKP"],\
                                Nmesh, boxsize, sampler, interlaced, comm, boxcenter=boxcenter)
            rfield_randoms_a_odd *= alpha_a_odd/vol_per_cell
            rfield_randoms_a_even *= alpha_a_even/vol_per_cell

            sub_I_mesh = vol_per_cell * \
                (np.sum(rfield_data_a * rfield_randoms_a_odd * rfield_randoms_a_even))
            
            I_mesh = comm.reduce(sub_I_mesh, op=MPI.SUM, root=0)
            if rank == 0:
                mesh_attrs["I_norm"] = I_mesh
            del data_a, randoms_a, randoms_a_odd, randoms_a_even, rfield_randoms_a_odd, rfield_randoms_a_even
            gc.collect()
        else:
            if tracer_type == "abc":
                rfield_randoms_a = tracer_cache["a"]["rfield_randoms"]
                rfield_randoms_b = tracer_cache["b"]["rfield_randoms"]
                rfield_randoms_c = tracer_cache["c"]["rfield_randoms"]
                if (rfield_randoms_a is None) or (rfield_randoms_b is None) or (rfield_randoms_c is None):
                    raise RuntimeError("random meshes are required for tracer_type='abc' mixed-mesh normalization.")
                sub_I_mesh = vol_per_cell * (np.sum(rfield_randoms_a * rfield_randoms_b * rfield_randoms_c))
            elif tracer_type == "aab":
                # Correct scheme: D_a * R_a * R_b
                rfield_data_a = tracer_cache["a"]["rfield_data"]
                rfield_randoms_a = tracer_cache["a"]["rfield_randoms"]
                rfield_randoms_b = tracer_cache["b"]["rfield_randoms"]
                if (rfield_data_a is None) or (rfield_randoms_a is None) or (rfield_randoms_b is None):
                    raise RuntimeError("D_a, R_a and R_b meshes are required for tracer_type='aab'.")
                sub_I_mesh = vol_per_cell * \
                    (np.sum(rfield_data_a * rfield_randoms_a * rfield_randoms_b))
            elif tracer_type == "abb":
                # Correct scheme: R_a * R_b * D_b
                rfield_randoms_a = tracer_cache["a"]["rfield_randoms"]
                rfield_randoms_b = tracer_cache["b"]["rfield_randoms"]
                rfield_data_b = tracer_cache["b"]["rfield_data"]
                if (rfield_randoms_a is None) or (rfield_randoms_b is None) or (rfield_data_b is None):
                    raise RuntimeError("R_a, R_b and D_b meshes are required for tracer_type='abb'.")
                sub_I_mesh = vol_per_cell * \
                    (np.sum(rfield_randoms_a * rfield_randoms_b * rfield_data_b))
                
            I_mesh = comm.reduce(sub_I_mesh, op=MPI.SUM, root=0)
            if rank == 0:
                mesh_attrs["I_norm"] = I_mesh

    mesh_attrs = comm.bcast(mesh_attrs, root=0)
    # if rank == 0:
    #     print("Mesh attributes:", mesh_attrs)

    return mesh_attrs, rfield_a, rfield_b, rfield_c


def get_N_field(catalogs, tracer_flag, alpha, nmesh, geometry, column_names, boxsize,
                sampler, interlaced, z_range, comp_weight_plan, para_cosmo,
                boxcenter, comm, normalization_scheme="particle"):
    """
    Build N-field for shot-noise pipeline in real space (no Y_lm and no FFT here).

    The new painting weights are squared total weights:
        (WEIGHT * WEIGHT_FKP)^2

    Final field:
        N_field = N_data + alpha^2 * N_randoms
    """
    rank = comm.Get_rank()

    assert geometry == "survey-like", "get_N_field currently supports survey-like geometry only."
    assert tracer_flag in ["a", "b", "c"], "tracer_flag must be one of 'a', 'b', or 'c'."

    data_key = f"data_{tracer_flag}"
    randoms_key = f"randoms_{tracer_flag}"

    Nmesh = np.array(nmesh)
    boxsize = np.array(boxsize)
    vol_per_cell = np.prod(boxsize) / np.prod(Nmesh)

    if rank == 0:
        logging.info(f"Start generating N_field for tracer_{tracer_flag} with alpha={alpha:.8e}.")

    randoms = catalog_reader(
        catalogs[randoms_key], geometry, column_names, z_range, comp_weight_plan, comm,
        para_cosmo=para_cosmo, boxsize=boxsize, boxcenter=boxcenter,
        catalog_type="randoms", normalization_scheme=normalization_scheme
    )

    data = catalog_reader(
        catalogs[data_key], geometry, column_names, z_range, comp_weight_plan, comm,
        para_cosmo=para_cosmo, boxsize=boxsize, boxcenter=boxcenter,
        catalog_type="data", normalization_scheme=normalization_scheme
    )

    weight_data_sq = (data["WEIGHT"] * data["WEIGHT_FKP"]) ** 2
    weight_randoms_sq = (randoms["WEIGHT"] * randoms["WEIGHT_FKP"]) ** 2

    if rank == 0:
        logging.info(f"Start generating squared-weight mesh for data catalog of tracer_{tracer_flag}.")
    rfield_data = pm_painter(
        data["Position"], weight_data_sq,
        Nmesh, boxsize, sampler, interlaced, comm, boxcenter=boxcenter
    )

    if rank == 0:
        logging.info(f"Start generating squared-weight mesh for randoms catalog of tracer_{tracer_flag}.")
    rfield_randoms = pm_painter(
        randoms["Position"], weight_randoms_sq,
        Nmesh, boxsize, sampler, interlaced, comm, boxcenter=boxcenter
    )

    rfield_data /= vol_per_cell
    rfield_randoms /= vol_per_cell

    rfield_N = rfield_data + alpha**2 * rfield_randoms

    del data, randoms, weight_data_sq, weight_randoms_sq, rfield_data, rfield_randoms
    gc.collect()

    if rank == 0:
        logging.info(f"N_field for tracer_{tracer_flag} is ready.")

    return rfield_N


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

def compute_particle_sums_bk(data, randoms, correlation_mode, angu_config, comm):
    """
    Compute necessary sums for alpha, normalization, and shot noise calculations for bispectrum measurement.
    Returns results only on rank 0.
    """
    rank = comm.Get_rank()

    # for alpha, pypower scheme includes FKP weights
    sub_weight_sum_data = np.sum(data['WEIGHT'] * data['WEIGHT_FKP'])
    sub_weight_sum_randoms = np.sum(randoms['WEIGHT'] * randoms['WEIGHT_FKP'])

    # for comparison between different alpha calculation methods, nbodykit's alpha does not include FKP weights
    sub_weight_sum_data_wofkp = np.sum(data['WEIGHT'])
    sub_weight_sum_randoms_wofkp = np.sum(randoms['WEIGHT'])

    # for particle-normalization of auto correlation, I_33
    sub_I_33 = np.sum(np.array(randoms['NZ']) ** 2 * np.array(randoms['WEIGHT']) ** 2 \
                      * np.array(randoms['WEIGHT_FKP']) ** 3)

    # for shot noise, S_LM
    if correlation_mode == "auto":
        data_pos = data['Position']
        randoms_pos = randoms['Position']
        data_unit = data_pos / np.linalg.norm(data_pos, axis=1, keepdims=True)
        randoms_unit = randoms_pos / np.linalg.norm(randoms_pos, axis=1, keepdims=True)

        sub_S_LM_list_data = []
        sub_S_LM_list_randoms = []
        for mm in range(0, angu_config[2] + 1):
            ylm = get_Ylm(angu_config[2], mm, Racah_normalized=True)
            if rank == 0:
                logging.info(f"Start calculating S_LM with spherical harmonics Y_{angu_config[2]}^{mm} = {ylm.expr}")
            S_LM_data_mm = np.sum(((data['WEIGHT'] * data['WEIGHT_FKP']))**3 \
                                  * np.conj(ylm(data_unit[:,0], data_unit[:,1], data_unit[:,2])))
            S_LM_randoms_mm = np.sum(((randoms['WEIGHT'] * randoms['WEIGHT_FKP']))**3 \
                                  * np.conj(ylm(randoms_unit[:,0], randoms_unit[:,1], randoms_unit[:,2])))
            sub_S_LM_list_data.append(S_LM_data_mm)
            sub_S_LM_list_randoms.append(S_LM_randoms_mm)
        
        sub_S_LM_list_data = np.array(sub_S_LM_list_data)
        sub_S_LM_list_randoms = np.array(sub_S_LM_list_randoms)
    else:
        sub_S_LM_list_data = 0.0 # although this value is not used in cross-correlation, we still set it to 0.0 to avoid errors
        sub_S_LM_list_randoms = 0.0

    # collect all necessary sub_sums to rank 0 and sum them up
    gathered_results = comm.gather(
        [sub_weight_sum_data, sub_weight_sum_randoms,
            sub_weight_sum_data_wofkp, sub_weight_sum_randoms_wofkp,
            sub_I_33, sub_S_LM_list_data, sub_S_LM_list_randoms],
        root=0
    )
    if rank == 0:
        # Aggregate results from all ranks
        # print("Gathered results from all ranks:", gathered_results)
        results = []
        for col in zip(*gathered_results):
            first_item = col[0]
            if isinstance(first_item, np.ndarray):
                results.append(np.sum(col, axis=0))
            else:
                results.append(np.sum(col))
        return results
    else:        
        return None
