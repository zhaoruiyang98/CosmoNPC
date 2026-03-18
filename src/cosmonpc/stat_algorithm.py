import numpy as np
import time
from mpi4py import MPI
import logging
from sympy.physics.wigner import wigner_3j
from .math_funcs import *
import gc


def calculate_bk_sco_box(rfield, stat_attrs, comm, **kwargs):
    rank = comm.Get_rank()

    # Extract mesh attributes
    poles = stat_attrs['poles']
    boxsize, nmesh = np.array(stat_attrs['boxsize']), np.array(stat_attrs['nmesh'])
    # boxcenter = np.array(stat_attrs['boxcenter'])
    k_min, k_max, k_bins = stat_attrs['k_min'], stat_attrs['k_max'], stat_attrs['k_bins']
    sampler, interlaced = stat_attrs['sampler'], stat_attrs['interlaced']
    P_shot = stat_attrs['P_shot']
    NZ = stat_attrs['NZ']
    rsd = np.array(stat_attrs['rsd'])

    if rank == 0:
        logging.info(f"Rank {rank}: Using rsd = {rsd}.")

    # Define some useful variables
    k_edge = np.linspace(k_min, k_max, k_bins + 1)
    k_center = 0.5 * (k_edge[:-1] + k_edge[1:])
    dk = (k_max - k_min) / k_bins

    if rank == 0:
        logging.info(f"Rank {rank}: k_edge = {k_edge}")
    comm.Barrier()

    # Calculate the Fourier transform of the density field
    cfield = rfield.r2c()
    # Compensate the cfield depending on the type of mesh
    compensation = get_compensation(interlaced, sampler)
    cfield.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    if rank == 0:
        logging.info(f"{compensation[0][1].__name__} applied to the density field")

    # Validate the poles
    validate_poles(poles)

    # Get the kgrid, knorm for binning and Legendre polynomials
    if poles == [0]:  # For ell = 0 only, we do not need to calculate the kgrid
        knorm = np.sqrt(sum(np.real(kk)**2 for kk in cfield.x))
        kgrid = None
    else:
        kgrid, knorm = get_kgrid(cfield)

    # clear zero-mode in the Fourier space
    cfield[knorm == 0.] = 0.
    if rank == 0:
        logging.info(f"Rank {rank}: Zero-mode in Fourier space cleared")


    """ 
        Before getting the bispectrum, we firstly get the power spectrum monopole
        as well as k_eff, k_num for the bk shot noise
    """
    P_field = np.real(cfield[:])**2 + np.imag(cfield[:])**2

    results = {} if rank == 0 else None

    """
    Note that poles at least contain one even number...
    """
    binned_F0x_list = get_binned_ifft_field(cfield, k_bins, k_edge, knorm, 0, comm)


    # Loop over the poles
    for pole in poles:
        if rank == 0:
            logging.info(f"Rank {rank}: Processing pole {pole}")

        if pole & 1:
            # for ell = odd, the multipoles are 0 strictly
            res = np.zeros(k_bins).astype("float128")
            if rank == 0:
                logging.info(f"Rank {rank}: ell = {pole}, Odd multipoles are automatically set to zero")
        else:
            if pole == 0:
                binned_list_k1 = binned_F0x_list
                F_ell = None
            else:
                L_ells = get_legendre(pole, rsd[0], rsd[1], rsd[2])
                if rank == 0:
                    logging.info(f"Rank {rank}: L_ells = {L_ells.expr}")
                F_ell = cfield * L_ells(kgrid[0], kgrid[1], kgrid[2])
                binned_list_k1 = get_binned_ifft_field(F_ell, k_bins, k_edge, knorm, pole, comm)
            binned_list_k2 = binned_F0x_list
            binned_list_k3 = binned_F0x_list

    # loop over all possible combinations of binned_list_k1, k2, k3
    # with k1 >= k2 >= k3

        sub_res,tri_config, N_tri= [], [] ,[]

        for i in range(k_bins):
            for j in range(i, k_bins):
                for k in range(j, k_bins):
                    # if rank == 0:
                    #     logging.info(f"Rank {rank}: Processing bins with index: {i}, {j}, {k}")
                    # get the bispectrum
                    if i + j >= k-1:
                        sub_res.append(np.sum(binned_list_k1[i] * binned_list_k2[j] * binned_list_k3[k]))
                        tri_config.append((i, j, k))
                        # $ V_T^{ANA} = 8\pi^2k_1k_2k_3\Delta k^3 $
                        N_tri.append(8 * np.pi**2 *k_center[i] * k_center[j] * k_center[k] * dk **3)

        sub_res = np.array(sub_res).astype("float128")
        
        # Gather the results from all ranks
        bk_res = comm.reduce(sub_res, op=MPI.SUM, root=0)

        # get NT, the number of triangles
        if rank == 0:
            N_tri = np.array(N_tri)
            bk_res *= (2* pole + 1) / N_tri
            vol_per_cell = boxsize.prod() / nmesh.prod()
            bk_res *= vol_per_cell**3 / boxsize.prod() # we need further check this
            
            # store the results
            results.update({
                f'B{pole}': bk_res,
            })
            logging.info(f"Rank {rank}: B{pole} calculated")
            if "N_tri" not in results:
                results["N_tri"] = N_tri
                logging.info(f"Rank {rank}: N_tri stored")
            if "tri_config" not in results:
                results["tri_config"] = tri_config
                logging.info(f"Rank {rank}: tri_config stored")

    # Free memory
    del cfield, P_field, binned_F0x_list, binned_list_k1, binned_list_k2, binned_list_k3
    gc.collect()
    if rank == 0:
        logging.info(f"Rank {rank}: Finished processing all poles")
        return results
    else:
        return None
        

            
def get_binned_ifft_field(kfield, k_bins, k_edge, knorm, pole, comm):
    """
    Get the binned inverse Fourier transform of the density field
    """
    rank = comm.Get_rank()
    verbose_level = stat_attrs.get("verbose_level", 2)

    def _log(level, stage, msg):
        if rank == 0 and verbose_level >= level:
            logging.info(f"[BK-SURVEY][{stage}] {msg}")

    list_to_return = []
    # Loop over the bins and sum the values in each bin
    for i in range(k_bins):
        mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i+1])
        if pole == 0:
            list_to_return.append(np.real((kfield * mask).c2r())) # it is strictly to be real
        else:
            list_to_return.append(2 * np.real((kfield * mask).c2r())) # for G_ell, we take the real part and double it
    del mask
    gc.collect()

    if rank == 0:
        logging.info(f"Rank {rank}: Mesh shape of binned ifft field = {list_to_return[0].shape}")

    return list_to_return


def calculate_power_spectrum_survey(stat_attrs, rfield_a, rfield_b, correlation_mode, comm, **kwargs):
    rank = comm.Get_rank()

    # Extract mesh attributes
    poles = stat_attrs['poles']
    boxsize, nmesh = np.array(stat_attrs['boxsize']), np.array(stat_attrs['nmesh'])
    boxcenter = np.array(stat_attrs['boxcenter'])
    k_min, k_max, k_bins = stat_attrs['k_min'], stat_attrs['k_max'], stat_attrs['k_bins']
    sampler, interlaced = stat_attrs['sampler'], stat_attrs['interlaced']
    N0 = stat_attrs['N0']

    if stat_attrs["normalization_scheme"] == "particle":
        I_norm = stat_attrs['I_rand']
        if rank == 0:
            logging.info(f"Rank {rank}: Using particle normalization with I_rand = {I_norm}.")
    else:
        I_norm = stat_attrs['I_mesh']
        if rank == 0:
            logging.info(f"Rank {rank}: Using mixed-mesh normalization with I_mesh = {I_norm}.")

    # Define some useful variables
    k_edge = np.linspace(k_min, k_max, k_bins + 1)
    if rank == 0:
        logging.info(f"Rank {rank}: k_edge = {k_edge}")

    comm.Barrier()

    # Calculate the Fourier transform of the density fields
    cfield_a = rfield_a.r2c()

    # Compensate the cfield depending on the type of mesh
    compensation = get_compensation(interlaced, sampler)
    cfield_a.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    cfield_a[:] *= boxsize.prod()  # normalize the cfield, very interesting, if the normalization is not done here, the result will be wrong,
    if rank == 0:
        logging.info(f"Rank {rank}: {compensation[0][1].__name__} applied to the density field a")

    if correlation_mode == "auto":
        cfield_b = cfield_a
        if rank == 0:
            logging.info(f"Rank {rank}: Auto-correlation mode, using the same density field for b")
    else:
        cfield_b = rfield_b.r2c()
        cfield_b.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        cfield_b[:] *= boxsize.prod()
        if rank == 0:
            logging.info(f"Rank {rank}: {compensation[0][1].__name__} applied to the density field b")


    # Get the kgrid, knorm, and x_grid for binning and spherical harmonics
    if poles == [0]:  # For ell = 0 only, we do not need to calculate the kgrid
        knorm = np.sqrt(sum(np.real(kk)**2 for kk in cfield_a.x))
        kgrid = None
    else:
        kgrid, knorm = get_kgrid(cfield_a)
        xgrid = get_xgrid(rfield_a, boxcenter, boxsize, nmesh)

    # No need to clear zero-mode in the Fourier space, see math_funcs.get_Ylm for explanation
    results = {} if rank == 0 else None

    # get k_eff and k_num in one particular k_bin
    sub_count, sub_knorm_sum = get_kbin_count(k_bins, k_edge, knorm)
    # gather the results from all ranks
    k_num = comm.reduce(sub_count, op=MPI.SUM, root=0)
    total_knorm_sum = comm.reduce(sub_knorm_sum, op=MPI.SUM, root=0)
    if rank == 0:
        k_eff = total_knorm_sum / k_num
    results = {'k_eff': k_eff, "k_num":k_num} if rank == 0 else None

    # Loop over the poles
    for pole in poles:
        if rank == 0:
            logging.info(f"Rank {rank}: Processing pole {pole}")

        if pole == 0:
            if correlation_mode == "auto":
                P_ell_field = np.real(cfield_a[:])**2 + np.imag(cfield_a[:])**2
            else:
                P_ell_field = cfield_a[:] * np.conj(cfield_b[:])
        else:
            """
            IMPORTANT NOTE:
            Here we temporarily put the rsd effect into rfield_b
            """
            G_ell_b = get_G_ell(rfield_b, pole, kgrid, xgrid, compensation, boxsize, comm)
            P_ell_field = cfield_a[:] * np.conj(G_ell_b[:])

        # Radial binning
        sub_sum = radial_binning(P_ell_field, k_bins, k_edge, knorm)

        # Gather the results from all ranks
        total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)

        if rank == 0:
            res = total_sum / k_num
            # Add some factors, substract the shot noise and take the complex conjugate for ell > 0
            res *= (2*pole+1)/ I_norm 
            
            if pole == 0:
                res -= N0
                logging.info(f"Rank {rank}: Shot noise subtracted from P0")
            else:
                if pole & 1: # ell = odd
                    res = 2j * np.imag(res)
                else:
                    res = 2 * np.real(res)

            # Store the results
            results.update({
                f'P{pole}': res,
            })
            logging.info(f"Rank {rank}: P{pole} calculated")

    # Free memory
    del cfield_a, cfield_b, P_ell_field
    gc.collect()

    if rank == 0:
        logging.info(f"Rank {rank}: Finished processing all poles")
        return results
    else:
        return None



def calculate_power_spectrum_box(rfield_a, rfield_b, correlation_mode, \
                                 stat_attrs, comm, **kwargs):
    rank = comm.Get_rank()

    # Extract mesh attributes
    poles = stat_attrs['poles']
    boxsize, nmesh = np.array(stat_attrs['boxsize']), np.array(stat_attrs['nmesh'])
    k_min, k_max, k_bins = stat_attrs['k_min'], stat_attrs['k_max'], stat_attrs['k_bins']
    sampler, interlaced = stat_attrs['sampler'], stat_attrs['interlaced']
    rsd = np.array(stat_attrs['rsd'])
    P_shot = 1.0/stat_attrs['NZ_a'] if correlation_mode == "auto" else 0.0

    if correlation_mode == "cross":
        NZ_a = stat_attrs['NZ_a']
        NZ_b = stat_attrs['NZ_b']
    else:
        NZ_a = stat_attrs['NZ_a']

    if rank == 0:
        logging.info(f"Rank {rank}: Using rsd = {rsd}.")

    # Define some useful variables
    k_edge = np.linspace(k_min, k_max, k_bins + 1)
    if rank == 0:
        logging.info(f"Rank {rank}: k_edge = {k_edge}")
    comm.Barrier()


    # get the compensation mode
    compensation = get_compensation(interlaced, sampler)

    # Calculate the Fourier transform of the density field and compensate it
    cfield_a =  rfield_a.r2c()
    cfield_a.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])

    logging.info(f"Rank {rank}: The shape of rfield_a is {rfield_a.shape}, the shape of cfield_a is {cfield_a.shape}.")
    if rank == 0:
        logging.info(f"Rank {rank}: {compensation[0][1].__name__} applied to the density field")


    if correlation_mode == "cross":
        cfield_b =  rfield_b.r2c()
        cfield_b.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        if rank == 0:
            logging.info(f"Rank {rank}: {compensation[0][1].__name__} applied to the density field")
    else:
        cfield_b = cfield_a
    
    # Validate the poles
    validate_poles(poles)

    # Get the kgrid, knorm for binning and Legendre polynomials
    if poles == [0]:  # For ell = 0 only, we do not need to calculate the kgrid
        knorm = np.sqrt(sum(np.real(kk)**2 for kk in cfield_a.x))
        kgrid = None
    else:
        kgrid, knorm = get_kgrid(cfield_a)

    # clear zero-mode in the Fourier space
    # note that even for cross power spectrum, we only need to clear the zero-mode of cfield_a
    # cfield_a[knorm == 0.] = 0. 
    # we'd better use a smarter way to handle the zero-mode issue
    if knorm[0, 0, 0] == 0.0:
        cfield_a[0, 0, 0] = 0.0 + 0j
        logging.info(f"Rank {rank}: Zero-mode in Fourier space of cfield_a cleared!")

    P_field = np.real(cfield_a[:]) * np.real(cfield_b[:]) + np.imag(cfield_a[:]) * np.imag(cfield_b[:])
    """
    This operation leverages the hermitian symmetry of the Fourier transform of real fields.
    When summing over all k-modes in a spherical shell (with thickness), it can be shown that:
    
        F_a(k) * F_b*(k) = [F_a(-k) * F_b*(-k)]*

    For two opposite points in k-space, the imaginary parts cancel out during the summation over all k-modes in the shell.
    """

    # save some memory, cfield, rfield are not needed anymore
    del rfield_a, rfield_b, cfield_a, cfield_b
    gc.collect()

    # create the results container
    results = {} if rank == 0 else None

    # get k_eff and k_num in one particular k_bin
    sub_count, sub_knorm_sum = get_kbin_count(k_bins, k_edge, knorm)
    # gather the results from all ranks
    k_num = comm.reduce(sub_count, op=MPI.SUM, root=0)
    total_knorm_sum = comm.reduce(sub_knorm_sum, op=MPI.SUM, root=0)
    if rank == 0:
        k_eff = total_knorm_sum / k_num

    results = {'k_eff': k_eff, "k_num":k_num} if rank == 0 else None

    # Loop over the poles
    for pole in poles:
        if rank == 0:
            logging.info(f"Rank {rank}: Processing pole {pole}")

        if pole & 1: # for ell = odd, the multipoles are 0 strictly
            res = np.zeros(k_bins).astype("float128")
            if rank == 0:
                logging.info(f"Rank {rank}: ell = {pole}, Odd multipoles are automatically set to zero")
        else:
            if pole == 0:
                sub_sum = radial_binning(P_field, k_bins, k_edge, knorm)
            else:
                L_ells = get_legendre(pole, rsd[0], rsd[1], rsd[2])
                if rank == 0:
                    logging.info(f"Rank {rank}: L_ells = {L_ells.expr}")
                sub_sum = radial_binning(P_field * L_ells(kgrid[0],kgrid[1],kgrid[2]), \
                                        k_bins, k_edge, knorm)
            
            # Gather the results from all ranks
            total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)

            if rank == 0:
                res = total_sum / k_num
                if correlation_mode == "auto":
                    res *= boxsize.prod() **(-1)  * (nmesh.prod()**2) / NZ_a**2
                else:
                    res *= boxsize.prod() **(-1)  * nmesh.prod()**2 / (NZ_a * NZ_b)
                """
                Correction details:
                Reference: 
                https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/algorithms/fftpower.html#FFTPower

                1. For every overdensity field, multiply a factor:
                   nmesh.prod() / N_gal = nmesh.prod() / (N_Z * boxsize.prod())
                   This recovers 1 + δ at every grid point.

                2. For the FFT operation:
                   Similar to the survey-like case, correct by boxsize.prod()**2.
                   However, divide by Volume (similar to dividing by I_norm in the survey case).

                3. By definition:
                   P(k) = |δ(k)|^2 / V

                Combining these corrections:
                - For auto-power spectrum:
                  boxsize.prod() ** (-1) * (nmesh.prod() ** 2) / NZ ** 2
                - For cross-power spectrum:
                  boxsize.prod() ** (-1) * (nmesh.prod() ** 2) / (NZ_a * NZ_b)
                """
                if pole == 0:
                    res -= P_shot
                    logging.info(f"Rank {rank}: Shot noise subtracted from P0")
                else:
                    res *= 2*pole+1

        # Store the results
        if rank == 0:
            results.update({
                f'P{pole}': res,
                "P_shot": P_shot,
            })
            logging.info(f"Rank {rank}: P{pole} calculated")

    # Free memory
    del P_field
    gc.collect()
    if rank == 0:
        logging.info(f"Rank {rank}: Finished processing all poles")
        return results
    else:
        return None


def calculate_bk_sugi_box(rfield_a, rfield_b, rfield_c, correlation_mode, \
                          stat_attrs, comm, **kwargs):
    rank = comm.Get_rank()
    # Extract mesh attributes
    data_vector_mode = stat_attrs.get("data_vector_mode", "diagonal")
    block_size = stat_attrs.get("block_size", 1 if data_vector_mode == "diagonal" else "full")
    [ell_1, ell_2, L] = stat_attrs['angu_config']
    boxsize = np.array(stat_attrs['boxsize'])
    nmesh = np.array(stat_attrs['nmesh'])
    k_min, k_max, k_bins = stat_attrs['k_min'], stat_attrs['k_max'], stat_attrs['k_bins']
    sampler = stat_attrs['sampler']
    interlaced = stat_attrs['interlaced']
    tracer_type = stat_attrs['tracer_type']  # "aaa", "aab", "abb" or "abc"
    vol_per_cell = boxsize.prod() / nmesh.prod()

    # constants related to angular momenta



    M = 0 # magnetic quantum number for L, only M=0 is considered here
    N_ells = (2 * ell_1 + 1) * (2 * ell_2 + 1) * (2 * L + 1)
    H_ells = np.float64(wigner_3j(ell_1, ell_2, L, 0, 0, 0))
    # find all sub-configurations that satisfy the triangular condition
    magnetic_configs, three_j_values = get_magnetic_configs_box(ell_1, ell_2, L)
    if rank == 0:
        logging.info(f"Rank {rank}: Magnetic configurations found: {magnetic_configs}")

    # Extract number density and galaxy count based on correlation mode
    if correlation_mode == "cross":
        NZ_a, N_gal_a = stat_attrs['NZ_a'], stat_attrs['N_gal_a']
        NZ_b, N_gal_b = stat_attrs['NZ_b'], stat_attrs['N_gal_b']
        if tracer_type == "abc":
            NZ_c, N_gal_c = stat_attrs['NZ_c'], stat_attrs['N_gal_c']
    else:
        NZ_a, N_gal_a = stat_attrs['NZ_a'], stat_attrs['N_gal_a']

    # Define k-bin edges
    k_edge = np.linspace(k_min, k_max, k_bins + 1)
    if rank == 0:
        logging.info(f"Rank {rank}: k_edge = {k_edge}")

    # Get the compensation mode in k-space
    compensation = get_compensation_bk_sugi(sampler)
    if rank == 0:
        logging.info("In Sugiyama estimator, the interlacing technique is not supported currently.\
                      Falling back to non-interlaced mode.")

    # Recover the density field to physical overdensity
    rfield_a[:] = (rfield_a[:] / vol_per_cell - NZ_a)
    if correlation_mode == "cross":
        rfield_b[:] = (rfield_b[:] / vol_per_cell - NZ_b)
        if tracer_type == "abc":
            rfield_c[:] = (rfield_c[:] / vol_per_cell - NZ_c)
    

    # Calculate the Fourier transform of the density field and apply compensation
    cfield_a = rfield_a.r2c()
    cfield_a.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    # logging.info(f"Rank {rank}: The shape of rfield_a is {rfield_a.shape}, the shape of cfield_a is {cfield_a.shape}.")
    if rank == 0:
        logging.info(f"Rank {rank}: {compensation[0][1].__name__} applied to the density field a")

    if correlation_mode == "cross":
        cfield_b = rfield_b.r2c()
        cfield_b.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        if rank == 0:
            logging.info(f"Rank {rank}: {compensation[0][1].__name__} applied to the density field b")

    if correlation_mode == "cross" and tracer_type == "abc":
        cfield_c = rfield_c.r2c()
        cfield_c.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        if rank == 0:
            logging.info(f"Rank {rank}: {compensation[0][1].__name__} applied to the density field c")


    # Get the kgrid, knorm for binning and spherical harmonics
    kgrid, knorm = get_kgrid(cfield_a)
    if rank == 0:
        logging.info(f"Rank {rank}: kgrid and knorm obtained.")    

    if correlation_mode == "auto":
        G_00 = cfield_a.c2r()
    else:
        if tracer_type == "abc":
            G_00 = cfield_c.c2r()
        else:
            G_00 = cfield_b.c2r()
    G_00 = np.real(G_00) # G_00 is strictly to be real
    if rank == 0:
        logging.info(f"Rank {rank}: G_00 generated.")

    # create the results container
    results = {} if rank == 0 else None
    
    if data_vector_mode == "full":
        if block_size == "full":
            block_size = k_bins
        elif isinstance(block_size, int) and 1 <= block_size <= k_bins:
            pass
        else:
            raise ValueError("For full mode, block_size must be 1, 'full', or an integer in [1, k_bins].")
        if rank == 0:
            logging.info(f"Rank {rank}: Using block_size = {block_size} for full data vector mode.")
        # create a full k_bins x k_bins array to store the results,
        total_res = np.zeros((k_bins, k_bins)).astype("complex128")

        for ss in range(len(magnetic_configs)):
            (m1, m2, M) = magnetic_configs[ss]
            if rank == 0:
                logging.info(
                    f"Rank {rank}: {'='*20}Processing magnetic configuration m1={m1}, m2={m2}, M={M}{'='*20}"
                )

            sub_res = np.zeros((k_bins, k_bins)).astype("complex128")
            ylm_1 = get_Ylm(ell_1, m1, Racah_normalized=True)
            ylm_2 = get_Ylm(ell_2, m2, Racah_normalized=True)
            if rank == 0:
                logging.info(
                    f"Rank {rank}: Processing spherical harmonics Y_{ell_1}^{m1} as {ylm_1.expr} and Y_{ell_2}^{m2} as {ylm_2.expr}"
                )

            ylm_weighted_cfield_1 = cfield_a * ylm_1(kgrid[0], kgrid[1], kgrid[2])
            if tracer_type in ["aaa", "aab"]:
                ylm_weighted_cfield_2 = cfield_a * ylm_2(kgrid[0], kgrid[1], kgrid[2])
            else:
                ylm_weighted_cfield_2 = cfield_b * ylm_2(kgrid[0], kgrid[1], kgrid[2])

            # Symmetry can be used only when both legs are same tracer source and same ell.
            same_tracer = (correlation_mode == "auto" or tracer_type == "aab")
            same_ell = (ell_1 == ell_2)
            can_sym_equal = same_tracer and same_ell and (m1 == m2)
            can_sym_conj = same_tracer and same_ell and (m1 == -m2)
            use_symmetry = can_sym_equal or can_sym_conj

            for bi in range(0, k_bins, block_size):
                i_end = min(k_bins, bi + block_size)
                if rank == 0:
                    logging.info("-"*50)
                    logging.info(f"Rank {rank}: Processing block rows {bi+1} to {i_end}...")

                # Reuse cache_1 across all bj blocks within this bi block.
                cache_1 = {}
                cache_2 = {}

                bj_start = bi if use_symmetry else 0
                for bj in range(bj_start, k_bins, block_size):
                    j_end = min(k_bins, bj + block_size)
                    block_on_diag = (bi == bj)
                    if rank == 0:
                        logging.info(f"Rank {rank}: Processing block columns {bj+1} to {j_end}...")

                    for i in range(bi, i_end):
                        if i not in cache_1:
                            mask_i = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i + 1])
                            cache_1[i] = (ylm_weighted_cfield_1 * mask_i).c2r()
                        binned_field_1 = cache_1[i]

                        j_local_start = bj
                        if use_symmetry and block_on_diag:
                            j_local_start = max(j_local_start, i)

                        for j in range(j_local_start, j_end):
                            if can_sym_equal and block_on_diag:
                                if j not in cache_1:
                                    mask_j = np.logical_and(knorm >= k_edge[j], knorm < k_edge[j + 1])
                                    cache_1[j] = (ylm_weighted_cfield_1 * mask_j).c2r()
                                binned_field_2 = cache_1[j]
                            elif can_sym_conj and block_on_diag:
                                if j not in cache_1:
                                    mask_j = np.logical_and(knorm >= k_edge[j], knorm < k_edge[j + 1])
                                    cache_1[j] = (ylm_weighted_cfield_1 * mask_j).c2r()
                                # The phase factor only applies in conjugate symmetry on diagonal blocks.
                                binned_field_2 = (-1)**(m1 + ell_1) * np.conj(cache_1[j])
                            else:
                                # Off-diagonal blocks must build cache_2 independently.
                                if j not in cache_2:
                                    mask_j = np.logical_and(knorm >= k_edge[j], knorm < k_edge[j + 1])
                                    cache_2[j] = (ylm_weighted_cfield_2 * mask_j).c2r()
                                binned_field_2 = cache_2[j]

                            sub_sig_sum = np.sum(G_00 * binned_field_1 * binned_field_2)
                            total_sig_sum = comm.reduce(sub_sig_sum, op=MPI.SUM, root=0)
                            if rank == 0:
                                sub_res[i, j] = three_j_values[ss] * total_sig_sum

                del cache_1, cache_2
                gc.collect()

            if rank == 0 and use_symmetry:
                for i in range(k_bins):
                    for j in range(i + 1, k_bins):
                        if can_sym_equal:
                            sub_res[j, i] = sub_res[i, j]
                        elif can_sym_conj:
                            sub_res[j, i] = np.conj(sub_res[i, j])

            if rank == 0:
                if magnetic_configs[ss] == (0, 0, 0):
                    total_res += sub_res
                else:
                    total_res += 2 * np.real(sub_res)
    elif data_vector_mode == "diagonal":
        total_res = np.zeros(k_bins).astype("complex128")

        for ss in range(len(magnetic_configs)):
            (m1, m2, M) = magnetic_configs[ss]
            if rank == 0:
                logging.info(
                    f"Rank {rank}: {'='*20}Processing magnetic configuration m1 = {m1}, m2 = {m2}, M = {M}...{'='*20}"
                )
            sub_res = np.zeros(k_bins).astype("complex128")
            ylm_1 = get_Ylm(ell_1, m1, Racah_normalized=True)
            ylm_2 = get_Ylm(ell_2, m2, Racah_normalized=True)
            if rank == 0:
                logging.info(f"Rank {rank}: Processing spherical harmonics Y_{ell_1}^{m1} as {ylm_1.expr} and Y_{ell_2}^{m2} as {ylm_2.expr}")
            ylm_weighted_cfield_1 = cfield_a * ylm_1(kgrid[0], kgrid[1], kgrid[2])
            if tracer_type in ["aaa", "aab"]:
                if ell_1 != ell_2 or m1 != m2:
                    ylm_weighted_cfield_2 = cfield_a * ylm_2(kgrid[0], kgrid[1], kgrid[2])
            else:
                ylm_weighted_cfield_2 = cfield_b * ylm_2(kgrid[0], kgrid[1], kgrid[2])

            # binning in k-space then ifft it to real space
            if rank == 0:
                logging.info(f"Rank {rank}: Closing triangles in k-space by binning and iffting the weighted cfield...")
            for i in range(k_bins):
                if rank == 0:
                    logging.info(f"Rank {rank}: Processing k-bin {i+1}/{k_bins}...")
                mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i + 1])
                binned_field_1 = (ylm_weighted_cfield_1 * mask).c2r()
                if tracer_type in ["aaa", "aab"]:
                    if ell_1 == ell_2:
                        if m1 == 0:
                            binned_field_2 = binned_field_1
                            if rank == 0 and i == 0:
                                logging.info(f"Rank {rank}: Using the same weighted cfield for b as for a.")
                        else:
                            binned_field_2 =  (-1)**(m1 + ell_1) * np.conj(binned_field_1)  #the additional phase factor can be found in our methodology paper
                            if rank == 0 and i == 0:
                                logging.info(f"Rank {rank}: Using the conjugate of weighted cfield a for b.")
                    else:
                        binned_field_2 = (ylm_weighted_cfield_2 * mask).c2r()
                else:
                    binned_field_2 = (ylm_weighted_cfield_2 * mask).c2r()
                sub_sig_sum = np.sum(G_00 * binned_field_1 * binned_field_2)
                # gather the results from all ranks
                total_sig_sum = comm.reduce(sub_sig_sum, op=MPI.SUM, root=0)
                if rank == 0:
                    sub_res[i] = three_j_values[ss] * total_sig_sum

            if rank == 0:
                if magnetic_configs[ss] == (0, 0, 0):
                    total_res += sub_res
                else:
                    total_res += 2 * np.real(sub_res)

    # Get k_eff and k_num in one particular k_bin
    k_center = 0.5 * (k_edge[1:] + k_edge[:-1])
    k3_min, k3_max = 2 * k_min, 2 * k_max
    k3_edge = np.linspace(k3_min, k3_max, k_bins * 2 + 1)[:-1]
    k3_center = 0.5 * (k3_edge[1:] + k3_edge[:-1])
    k3_bins = len(k3_center)

    sub_count, sub_knorm_sum = get_kbin_count(k3_bins, k3_edge, knorm)
    k_num = comm.reduce(sub_count, op=MPI.SUM, root=0)
    total_knorm_sum = comm.reduce(sub_knorm_sum, op=MPI.SUM, root=0)
    if rank == 0:
        k_eff = total_knorm_sum / k_num

    
    """
    Now we obtain the shot-noise, we define 4 terms in EQ45 of Sugiyama 2019 as SN0, SN1, SN2, SN3
    """
    time_shotnoise_start = time.time()
    if tracer_type == "abc":
        SN0, SN1, SN2, SN3 = 0.0, 0.0, 0.0, 0.0
        if rank == 0:
            logging.info("Tracer type 'abc' detected, shot-noise terms are all set to zero.")
    else:
        # SN0
        if correlation_mode == "auto" and [ell_1, ell_2, L] == [0, 0, 0]:
            SN0 = N_gal_a  # normalization factor I is N_gal_a^3/boxsize^2, we will divide it later
        else:
            SN0 = 0.0

        """
        The logic of this part can be found in our methodology paper. The Q_0 term is necessary.
        """
        P_field = cfield_a[:] * np.conj(cfield_a[:] if correlation_mode == "auto" else cfield_b[:])
        P_field *= boxsize.prod() ** 2

        if correlation_mode == "auto":
            S_field = cfield_a.copy()
            S_field[:] = SN0
            S_field.apply(out=Ellipsis, func=get_compensation_shot_sugi(sampler)[0][1], kind="circular")
            if rank == 0:
                logging.info(f"Rank {rank}: Shot-noise compensation applied for field a.")
            P_field[:] -= S_field[:]
        
        if L != 0: # multiply by y_L0
            y_L0 = get_Ylm(L, 0, Racah_normalized=True)
            P_field *= y_L0(kgrid[0], kgrid[1], kgrid[2])
            if rank == 0:
                logging.info(f"Rank {rank}: Multiplying by spherical harmonic Y_{L}^{0} as {y_L0.expr} for SN1 and SN2 calculation.")

        # Perform radial binning
        sub_sum = radial_binning(P_field, k3_bins, k3_edge, knorm)
        total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)

        if rank == 0:
            SN1, SN2 = 0.0, 0.0
            if ell_1 == L and ell_2 == 0:
                if correlation_mode == "auto":
                    SN1 = (total_sum / k_num)[:len(k_center)]
                    SN1 *= 2 * L + 1
                    if ell_1 == 0:
                        SN2 = SN1.copy()
                elif correlation_mode == "cross" and tracer_type == "abb":
                    SN1 = (total_sum / k_num)[:len(k_center)]
                    SN1 *= 2 * L + 1
            # Convert the 1d vector to (k1,k2) plane, note that SN1 is only related to k1, SN2 is only related to k2,
            if data_vector_mode == "full":
                SN1_full = np.zeros((k_bins, k_bins)).astype('c16')
                SN2_full = np.zeros((k_bins, k_bins)).astype('c16')
                if np.isscalar(SN1) or np.ndim(SN1) == 0:
                    SN1_full[:] = SN1
                else:
                    for i in range(k_bins):
                        SN1_full[i, :] = SN1[i]
                if np.isscalar(SN2) or np.ndim(SN2) == 0:
                    SN2_full[:] = SN2
                else:
                    for i in range(k_bins):
                        SN2_full[:, i] = SN2[i]
                SN1, SN2 = SN1_full, SN2_full
                del SN1_full, SN2_full
                gc.collect()

            # SN3
            if tracer_type == "abb":
                SN3 = 0.0
            else:
                weighted_k_num = k_num / k3_center
                if data_vector_mode == "diagonal":
                    SN3 = np.zeros(k_bins).astype('c16')
                    if rank == 0:
                        for i in range(k_bins):
                            coeff_assoc_legendre = np.zeros(k3_bins).astype('f8')
                            for xx in range(len(magnetic_configs)):
                                (m1, m2, M) = magnetic_configs[xx]
                                # three_j = three_j_values[xx]
                                q_ells  = get_q_ells(i,i,k_center, k_min,k_max,k_bins, ell_1, ell_2, L, k3_bins)
                            SN3[i] = np.sum(q_ells * total_sum / k3_center) / np.sum(weighted_k_num[:2 * i+1])
                        SN3 *= H_ells * N_ells
                elif data_vector_mode == "full":
                    SN3 = np.zeros((k_bins, k_bins)).astype('c16')
                    if rank == 0:
                        for i in range(k_bins):
                            for j in range(k_bins):
                                q_ells  = get_q_ells(i,j,k_center, k_min,k_max,k_bins, ell_1, ell_2, L, k3_bins)
                                valid_k3_bins = get_valid_k3_bins(k_center[i], k_center[j], k_min, k_max, k_bins)
                                SN3[i, j] = np.sum(q_ells * total_sum / k3_center) / np.sum(weighted_k_num * valid_k3_bins)
                    SN3 *= H_ells * N_ells

    time_shotnoise_end = time.time()
    if rank == 0:
        logging.info(f"Rank{rank}: Time to compute shot noise terms: {time_shotnoise_end - time_shotnoise_start:.2f} seconds")

    # Normalize the bispectrum
    if rank == 0:
        if correlation_mode == "auto":
            I_norm = (N_gal_a ** 3) / (boxsize.prod() ** 2)
        else:
            if tracer_type == "aab":
                I_norm = (N_gal_a ** 2 * N_gal_b) / (boxsize.prod() ** 2)
            elif tracer_type == "abb":
                I_norm = (N_gal_a * N_gal_b ** 2) / (boxsize.prod() ** 2)
            elif tracer_type == "abc":
                I_norm = (N_gal_a * N_gal_b * N_gal_c) / (boxsize.prod() ** 2)


        # Combine shot-noise terms.
        # For full mode we keep shot-noise subtraction disabled for now to validate signal first.
        # We will re-enable a mathematically consistent full-mode shot-noise model later.
        total_shot_noise = SN0 + SN1 + SN2 + SN3

        # Normalize the bispectrum result.
        total_res *= N_ells * H_ells  / I_norm
        if data_vector_mode == "full":
            total_res *= (boxsize.prod()) ** 2 / np.outer(k_num[:k_bins],k_num[:k_bins])
        else:
            total_res *= (boxsize.prod()) ** 2 / (k_num[:len(total_res)])**2

        total_res *= vol_per_cell
        total_shot_noise *= 1 / I_norm
        # Final bispectrum result after subtracting shot-noise
        final_bk = total_res  - total_shot_noise

        logging.info(f"Rank {rank}: Final bispectrum calculated.")

            # Store the results
        results.update({
            'B_sugi': final_bk,
            'SN_terms': {
                'SN0': SN0/I_norm,
                'SN1': SN1/I_norm,
                'SN2': SN2/I_norm,
                'SN3': SN3/I_norm
            },
            'I_norm': I_norm,
            "Shot_noise": total_shot_noise,
            "Bk_raw": total_res,
            "k_eff": k_eff,
            'nmodes': k_num
        })


    # broadcast results to all ranks
    results = comm.bcast(results, root=0)
    if rank == 0:
        logging.info(f"Rank {rank}: Finished processing bispectrum calculation.")

    return results


def calculate_bk_sugi_survey(rfield_a, rfield_b, rfield_c, correlation_mode, \
                             stat_attrs, comm, **kwargs):
    """
    Survey-like Sugiyama bispectrum estimator (signal-only version).

    Implementation notes
    --------------------
    1) For each angu_config = (ell_1, ell_2, L), we first build G_{L,M} from
       the third tracer:
          G_{L,M}(k) = FFT[ rfield3(x) * y_{L,M}(x) ]
       using stream-style reuse when M changes.
    2) Then for each magnetic sub-configuration (m1, m2, M), build
       F_{ell_1,m1} and F_{ell_2,m2} and perform binned inverse FFTs.
    3) Use symmetry accelerations:
       - Generic conjugate symmetry between (m1,m2,M) and (-m1,-m2,-M):
         only one half is explicitly evaluated (as provided by
         get_magnetic_configs_survey), and non-self-conjugate terms use 2*Re(.).
       - If tracer1 == tracer2 and ell_1 == ell_2 and m1 == m2, evaluate only
         upper triangle in full mode and fill lower triangle by transpose.
       - If tracer1 == tracer2 and ell_1 == ell_2 and swapped configuration
         (m2,m1,M) exists, reuse by transpose instead of recomputing.
    4) Shot noise is intentionally disabled at this stage (all SN terms are zero).
    """
    rank = comm.Get_rank()

    # Keep a single log format for readability; always emit on rank 0.
    def _log(_level, stage, msg):
        if rank == 0:
            logging.info(f"[BK-SURVEY][{stage}] {msg}")
    data_vector_mode = stat_attrs.get("data_vector_mode", "diagonal")
    block_size = stat_attrs.get("block_size", 1 if data_vector_mode == "diagonal" else "full")
    [ell_1, ell_2, L] = stat_attrs["angu_config"]
    boxsize = np.array(stat_attrs["boxsize"])
    nmesh = np.array(stat_attrs["nmesh"])
    boxcenter = np.array(stat_attrs["boxcenter"])
    k_min, k_max, k_bins = stat_attrs["k_min"], stat_attrs["k_max"], stat_attrs["k_bins"]
    sampler = stat_attrs["sampler"]
    tracer_type = stat_attrs["tracer_type"]
    I_norm = stat_attrs["I_norm"]
    vol_per_cell = boxsize.prod() / nmesh.prod()

    N_ells = (2 * ell_1 + 1) * (2 * ell_2 + 1) * (2 * L + 1)
    H_ells = np.float64(wigner_3j(ell_1, ell_2, L, 0, 0, 0))
    magnetic_configs, three_j_values = get_magnetic_configs_survey(ell_1, ell_2, L)
    if rank == 0:
        logging.info(f"[BK-SURVEY][INIT] magnetic_configs={magnetic_configs}")
        logging.info(
            f"[BK-SURVEY][INIT] angu=({ell_1},{ell_2},{L}) mode={data_vector_mode} "
            f"k_bins={k_bins} tracer_type={tracer_type} total_cfg={len(magnetic_configs)}"
        )

    k_edge = np.linspace(k_min, k_max, k_bins + 1)
    if rank == 0:
        logging.info(f"[BK-SURVEY][INIT] k_edge={k_edge}")

    # Window-function compensation used by Sugiyama estimator in k-space.
    compensation = get_compensation_bk_sugi(sampler)
    compensation_shot_sugi = get_compensation_shot_sugi(sampler)

    cfield_a = rfield_a.r2c()
    cfield_a.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    cfield_a[:] *= boxsize.prod()

    if correlation_mode == "auto":
        cfield_b = cfield_a
    else:
        cfield_b = rfield_b.r2c()
        cfield_b.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        cfield_b[:] *= boxsize.prod()

    kgrid, knorm = get_kgrid(cfield_a)
    xgrid = get_xgrid(rfield_a, boxcenter, boxsize, nmesh)

    sub_count, sub_knorm_sum = get_kbin_count(k_bins, k_edge, knorm)
    k_num = comm.reduce(sub_count, op=MPI.SUM, root=0)
    total_knorm_sum = comm.reduce(sub_knorm_sum, op=MPI.SUM, root=0)
    if rank == 0:
        k_eff = total_knorm_sum / k_num

    # Select the third tracer field for G_{L,M}.
    if correlation_mode == "auto":
        rfield3 = rfield_a
    else:
        rfield3 = rfield_c if tracer_type == "abc" else rfield_b


    # ------------------------------------------------------------------
    # Step A: Prepare stream-style G_{L,M} evaluation.
    # We keep only one G_{L,M} in memory at any time:
    #   G_{L,M}(k) = FFT[ rfield3(x) * Y_{L,M}(x) ]
    # Since magnetic_configs are ordered by M, we recompute only when M changes.
    # ------------------------------------------------------------------
    current_M = None
    G_LM_current = None

    results = {} if rank == 0 else None
    # For quick lookup of swapped configuration (m2, m1, M).
    cfg_to_idx = {cfg: ii for ii, cfg in enumerate(magnetic_configs)}
    processed_cfg = set()
    # Whether the first and second factors use the same tracer source.
    same_tracer_12 = (correlation_mode == "auto" or tracer_type == "aab")
    same_ell_12 = (ell_1 == ell_2)
    total_cfg_count = len(magnetic_configs)
    skipped_cfg_count = 0
    evaluated_cfg_count = 0

    if data_vector_mode == "full":
        if block_size == "full":
            block_size = k_bins
        elif isinstance(block_size, int) and 1 <= block_size <= k_bins:
            pass
        else:
            raise ValueError("For full mode, block_size must be 1, 'full', or an integer in [1, k_bins].")
        _log(1, "INIT", f"full_mode block_size={block_size}")

        total_res = np.zeros((k_bins, k_bins)).astype("complex128")

        for ss in range(len(magnetic_configs)):
            (m1, m2, M) = magnetic_configs[ss]
            cfg_t0 = time.time()
            if (m1, m2, M) in processed_cfg:
                if rank == 0:
                    skipped_cfg_count += 1
                _log(
                    1,
                    "CFG",
                    f"sub_cfg={ss+1}/{total_cfg_count} (m1,m2,M)=({m1},{m2},{M}) action=skip reason=covered_by_symmetry",
                )
                continue

            sub_res = np.zeros((k_bins, k_bins)).astype("complex128")
            # Recompute G_{L,M} only when M changes; otherwise reuse.
            if M != current_M:
                if G_LM_current is not None:
                    del G_LM_current
                    gc.collect()
                ylm_LM = get_Ylm(L, M, Racah_normalized=True)
                rfield_weighted_third = rfield3 * ylm_LM(xgrid[0], xgrid[1], xgrid[2])
                cfield_GLM = rfield_weighted_third.r2c()
                cfield_GLM.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
                # cfield_GLM[:] *= boxsize.prod()
                G_LM_current = cfield_GLM.c2r()
                current_M = M
                _log(1, "GLM", f"sub_cfg={ss+1}/{total_cfg_count} M={M} action=recompute")
                del rfield_weighted_third, cfield_GLM
                gc.collect()

            ylm_1 = get_Ylm(ell_1, m1, Racah_normalized=True)
            ylm_2 = get_Ylm(ell_2, m2, Racah_normalized=True)
            _log(
                1,
                "CFG",
                f"{'='*16} sub_cfg={ss+1}/{total_cfg_count} (m1,m2,M)=({m1},{m2},{M}) {'='*16}",
            )
            _log(
                2,
                "CFG",
                f"ylm1=Y_{ell_1}^{m1}:{ylm_1.expr} | ylm2=Y_{ell_2}^{m2}:{ylm_2.expr}",
            )

            ylm_weighted_cfield_1 = cfield_a * ylm_1(kgrid[0], kgrid[1], kgrid[2])
            if tracer_type in ["aaa", "aab"]:
                if ell_1 == ell_2 and m1 == m2:
                    ylm_weighted_cfield_2 = ylm_weighted_cfield_1
                    _log(2, "CFG", "field2_source=reuse_field1 (same tracer, same ell, m1==m2)")
                else:
                    ylm_weighted_cfield_2 = cfield_a * ylm_2(kgrid[0], kgrid[1], kgrid[2])
                    _log(2, "CFG", "field2_source=tracer_a")
            else:
                ylm_weighted_cfield_2 = cfield_b * ylm_2(kgrid[0], kgrid[1], kgrid[2])
                _log(2, "CFG", "field2_source=tracer_b")

            # Within one configuration:
            # if m1 == m2 and same tracer + same ell, the (i,j) matrix is symmetric,
            # so only upper-triangle is needed.
            can_sym_equal = same_tracer_12 and same_ell_12 and (m1 == m2)
            # If m1 == -m2, binned_field_2 can be obtained from binned_field_1 by
            # conjugation with phase factor.
            can_sym_conj = same_tracer_12 and same_ell_12 and (m1 == -m2)
            use_symmetry = can_sym_equal or can_sym_conj
            if rank == 0:
                evaluated_cfg_count += 1
            _log(
                1,
                "CFG",
                f"sub_cfg={ss+1}/{total_cfg_count} (m1,m2,M)=({m1},{m2},{M}) action=evaluate "
                f"3j_value={three_j_values[ss]:+.6e} sym_equal={can_sym_equal} sym_conj={can_sym_conj} use_sym={use_symmetry}",
            )

            for bi in range(0, k_bins, block_size):
                i_end = min(k_bins, bi + block_size)
                cache_1 = {}
                cache_2 = {}
                bj_start = bi if use_symmetry else 0
                _log(2, "BLOCK", f"sub_cfg={ss+1}/{total_cfg_count} row=[{bi},{i_end-1}]")

                for bj in range(bj_start, k_bins, block_size):
                    j_end = min(k_bins, bj + block_size)
                    block_on_diag = (bi == bj)
                    _log(
                        2,
                        "BLOCK",
                        f"sub_cfg={ss+1}/{total_cfg_count} col=[{bj},{j_end-1}] diag={block_on_diag}",
                    )
                    for i in range(bi, i_end):
                        if i not in cache_1:
                            mask_i = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i + 1])
                            cache_1[i] = (ylm_weighted_cfield_1 * mask_i).c2r()
                        binned_field_1 = cache_1[i]

                        j_local_start = bj
                        if use_symmetry and block_on_diag:
                            j_local_start = max(j_local_start, i)

                        for j in range(j_local_start, j_end):
                            if can_sym_equal and block_on_diag:
                                if j not in cache_1:
                                    mask_j = np.logical_and(knorm >= k_edge[j], knorm < k_edge[j + 1])
                                    cache_1[j] = (ylm_weighted_cfield_1 * mask_j).c2r()
                                binned_field_2 = cache_1[j]
                            elif can_sym_conj and block_on_diag:
                                if j not in cache_1:
                                    mask_j = np.logical_and(knorm >= k_edge[j], knorm < k_edge[j + 1])
                                    cache_1[j] = (ylm_weighted_cfield_1 * mask_j).c2r()
                                binned_field_2 = (-1) ** (m1 + ell_1) * np.conj(cache_1[j])
                            else:
                                if j not in cache_2:
                                    mask_j = np.logical_and(knorm >= k_edge[j], knorm < k_edge[j + 1])
                                    cache_2[j] = (ylm_weighted_cfield_2 * mask_j).c2r()
                                binned_field_2 = cache_2[j]

                            sub_sig_sum = np.sum(G_LM_current * binned_field_1 * binned_field_2)
                            total_sig_sum = comm.reduce(sub_sig_sum, op=MPI.SUM, root=0)
                            if rank == 0:
                                sub_res[i, j] = three_j_values[ss] * total_sig_sum

                del cache_1, cache_2
                gc.collect()

            # Fill lower triangle from upper triangle when m1 == m2 or m1 == -m2 optimization is used.
            if rank == 0 and use_symmetry:
                n_filled = 0
                for i in range(k_bins):
                    for j in range(i + 1, k_bins):
                        if can_sym_equal:
                            sub_res[j, i] = sub_res[i, j]
                            n_filled += 1
                        elif can_sym_conj:
                            # if m1 == -m2, then M is strictly 0, G_LM is real
                            sub_res[j, i] = np.conj(sub_res[i, j])
                            n_filled += 1
                if n_filled > 0:
                    fill_mode = "transpose" if can_sym_equal else "conjugate-transpose"
                    _log(
                        2,
                        "SYM",
                        f"sub_cfg={ss+1}/{total_cfg_count} action=fill_lower count={n_filled} method={fill_mode}",
                    )

            # Swapped-config acceleration:
            # if (m2, m1, M) exists and ell_1 == ell_2 with same tracer source,
            # its result is the transpose of current one.
            swap_cfg = (m2, m1, M)
            has_swap = same_tracer_12 and same_ell_12 and (swap_cfg in cfg_to_idx) and (swap_cfg != (m1, m2, M))
            add_swap_now = has_swap and (swap_cfg not in processed_cfg)

            if rank == 0:
                sub_res_to_add = sub_res
                if add_swap_now:
                    sub_res_to_add = sub_res + sub_res.T
                    _log(
                        2,
                        "SYM",
                        f"sub_cfg={ss+1}/{total_cfg_count} action=swap_reuse swap_cfg={swap_cfg} method=transpose",
                    )

                # Generic conjugate-half acceleration:
                # configs are generated only on one half-space; for non-self-conjugate
                # configs, add 2*Re(.) to include conjugate partner.
                if magnetic_configs[ss] == (0, 0, 0):
                    total_res += sub_res_to_add
                    _log(2, "ACC", f"sub_cfg={ss+1}/{total_cfg_count} action=accumulate rule=self_conjugate_once")
                else:
                    total_res += 2 * np.real(sub_res_to_add)
                    _log(2, "ACC", f"sub_cfg={ss+1}/{total_cfg_count} action=accumulate rule=2*Re")
                _log(1, "CFG", f"sub_cfg={ss+1}/{total_cfg_count} action=done elapsed={time.time()-cfg_t0:.2f}s")

            # IMPORTANT (MPI consistency): update processed_cfg identically on all ranks.
            if add_swap_now:
                processed_cfg.add(swap_cfg)
            processed_cfg.add((m1, m2, M))

            del ylm_weighted_cfield_1, ylm_weighted_cfield_2
            gc.collect()

    elif data_vector_mode == "diagonal":
        total_res = np.zeros(k_bins).astype("complex128")
        diagonal_mask_cache = {}
        diagonal_binned_field2_cache = {}
        diagonal_reuse_fixed_m2_leg = (ell_2 == 0 and all(cfg[1] == 0 for cfg in magnetic_configs))
        diagonal_shared_weighted_cfield_2 = None

        if diagonal_reuse_fixed_m2_leg:
            ylm_2_shared = get_Ylm(ell_2, 0, Racah_normalized=True)
            if tracer_type in ["aaa", "aab"]:
                diagonal_shared_weighted_cfield_2 = cfield_a * ylm_2_shared(kgrid[0], kgrid[1], kgrid[2])
                _log(1, "DIAG-CACHE", "reuse_enabled: fixed second leg uses tracer_a with ell_2=0, m2=0 across sub-configs.")
            else:
                diagonal_shared_weighted_cfield_2 = cfield_b * ylm_2_shared(kgrid[0], kgrid[1], kgrid[2])
                _log(1, "DIAG-CACHE", "reuse_enabled: fixed second leg uses tracer_b with ell_2=0, m2=0 across sub-configs.")
        else:
            _log(1, "DIAG-CACHE", "reuse_disabled: second leg is not globally fixed to ell_2=0, m2=0.")

        for ss in range(len(magnetic_configs)):
            (m1, m2, M) = magnetic_configs[ss]
            if (m1, m2, M) in processed_cfg:
                if rank == 0:
                    skipped_cfg_count += 1
                _log(
                    1,
                    "CFG",
                    f"sub_cfg={ss+1}/{total_cfg_count} (m1,m2,M)=({m1},{m2},{M}) action=skip reason=covered_by_symmetry",
                )
                continue

            sub_res = np.zeros(k_bins).astype("complex128")
            # Recompute G_{L,M} only when M changes; otherwise reuse.
            if M != current_M:
                if G_LM_current is not None:
                    del G_LM_current
                    gc.collect()
                ylm_LM = get_Ylm(L, M, Racah_normalized=True)
                rfield_weighted_third = rfield3 * ylm_LM(xgrid[0], xgrid[1], xgrid[2])
                cfield_GLM = rfield_weighted_third.r2c()
                cfield_GLM.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
                # cfield_GLM[:] *= boxsize.prod()
                G_LM_current = cfield_GLM.c2r()
                current_M = M
                _log(1, "GLM", f"sub_cfg={ss+1}/{total_cfg_count} M={M} action=recompute")
                del rfield_weighted_third, cfield_GLM
                gc.collect()

            ylm_1 = get_Ylm(ell_1, m1, Racah_normalized=True)
            ylm_2 = get_Ylm(ell_2, m2, Racah_normalized=True)

            ylm_weighted_cfield_1 = cfield_a * ylm_1(kgrid[0], kgrid[1], kgrid[2])
            if tracer_type in ["aaa", "aab"]:
                ylm_weighted_cfield_2 = cfield_a * ylm_2(kgrid[0], kgrid[1], kgrid[2])
            else:
                ylm_weighted_cfield_2 = cfield_b * ylm_2(kgrid[0], kgrid[1], kgrid[2])

            can_sym_equal = same_tracer_12 and same_ell_12 and (m1 == m2)
            can_sym_conj = same_tracer_12 and same_ell_12 and (m1 == -m2)
            if rank == 0:
                evaluated_cfg_count += 1
            _log(
                1,
                "CFG",
                f"sub_cfg={ss+1}/{total_cfg_count} (m1,m2,M)=({m1},{m2},{M}) action=evaluate "
                f"3j_value={three_j_values[ss]:+.6e} sym_equal={can_sym_equal} sym_conj={can_sym_conj} mode=diagonal",
            )

            for i in range(k_bins):
                mask_i = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i + 1])
                binned_field_1 = (ylm_weighted_cfield_1 * mask_i).c2r()
                if can_sym_equal:
                    binned_field_2 = binned_field_1
                elif can_sym_conj:
                    binned_field_2 = (-1) ** (m1 + ell_1) * np.conj(binned_field_1)
                else:
                    binned_field_2 = (ylm_weighted_cfield_2 * mask_i).c2r()

                sub_sig_sum = np.sum(G_LM_current * binned_field_1 * binned_field_2)
                total_sig_sum = comm.reduce(sub_sig_sum, op=MPI.SUM, root=0)
                if rank == 0:
                    sub_res[i] = three_j_values[ss] * total_sig_sum

            swap_cfg = (m2, m1, M)
            has_swap = same_tracer_12 and same_ell_12 and (swap_cfg in cfg_to_idx) and (swap_cfg != (m1, m2, M))
            add_swap_now = has_swap and (swap_cfg not in processed_cfg)

            if rank == 0:
                sub_res_to_add = sub_res
                # Diagonal mode: transpose relation becomes equality on diagonal bins.
                if add_swap_now:
                    sub_res_to_add = 2.0 * sub_res
                    _log(
                        2,
                        "SYM",
                        f"sub_cfg={ss+1}/{total_cfg_count} action=swap_reuse swap_cfg={swap_cfg} method=diagonal_double",
                    )

                if magnetic_configs[ss] == (0, 0, 0):
                    total_res += sub_res_to_add
                    _log(2, "ACC", f"sub_cfg={ss+1}/{total_cfg_count} action=accumulate rule=self_conjugate_once")
                else:
                    total_res += 2 * np.real(sub_res_to_add)
                    _log(2, "ACC", f"sub_cfg={ss+1}/{total_cfg_count} action=accumulate rule=2*Re")

            # IMPORTANT (MPI consistency): update processed_cfg identically on all ranks.
            if add_swap_now:
                processed_cfg.add(swap_cfg)
            processed_cfg.add((m1, m2, M))

            del ylm_weighted_cfield_1, ylm_weighted_cfield_2
            gc.collect()
    else:
        raise ValueError("data_vector_mode must be either 'diagonal' or 'full'.")

    # Release the last streamed G_{L,M} field.
    if G_LM_current is not None:
        del G_LM_current
        gc.collect()
        _log(2, "GLM", "action=release_current")

    """
    Shot-noise preparation:
    first construct N_field(s) for survey-like geometry.
    The detailed shot-noise combination is still pending, but N_field is now
    generated here as the first step.
    """
    from .mesh_generator import get_N_field

    catalogs = kwargs.get("catalogs", None)
    if catalogs is None:
        raise ValueError("catalogs must be provided to calculate_bk_sugi_survey for N_field construction.")
    
    

    # Keep only the necessary rfield references for shot-noise terms.
    # These are references (no memory copy).
    time_shotnoise_start = time.time()
    rfield_shot_a = None
    rfield_shot_b = None
    rfield_shot_c = None

    N_field_a = None
    N_field_b = None
    N_field_c = None

    # Requested logic:
    # 1) tracer_type == "abc": no rfield / no N_field (shot noise is strictly zero)
    # 2) tracer_type == "aab": only rfield_b; only N_field_a
    # 3) tracer_type == "abb": only rfield_a; only N_field_b
    # 4) all other cases: only rfield_a; only N_field_a
    if tracer_type == "abc":
        _log(1, "SHOT", "tracer_type=abc -> skip rfield/N_field construction (shotnoise=0).")
    elif tracer_type == "aab":
        rfield_shot_b = rfield_b
        N_field_a = get_N_field(
            catalogs=catalogs,
            tracer_flag="a",
            alpha=stat_attrs["alpha_a"],
            nmesh=stat_attrs["nmesh"],
            geometry="survey-like",
            column_names=stat_attrs["column_names"],
            boxsize=stat_attrs["boxsize"],
            sampler=stat_attrs["sampler"],
            interlaced=stat_attrs["interlaced"],
            z_range=stat_attrs["z_range"],
            comp_weight_plan=stat_attrs["comp_weight_plan"],
            para_cosmo=stat_attrs["cosmology"],
            boxcenter=stat_attrs["boxcenter"],
            comm=comm,
            normalization_scheme=stat_attrs.get("normalization_scheme", "particle"),
        )
        _log(1, "SHOT", "selection: use rfield_b + N_field_a for shot-noise terms.")
    elif tracer_type == "abb":
        rfield_shot_a = rfield_a
        N_field_b = get_N_field(
            catalogs=catalogs,
            tracer_flag="b",
            alpha=stat_attrs["alpha_b"],
            nmesh=stat_attrs["nmesh"],
            geometry="survey-like",
            column_names=stat_attrs["column_names"],
            boxsize=stat_attrs["boxsize"],
            sampler=stat_attrs["sampler"],
            interlaced=stat_attrs["interlaced"],
            z_range=stat_attrs["z_range"],
            comp_weight_plan=stat_attrs["comp_weight_plan"],
            para_cosmo=stat_attrs["cosmology"],
            boxcenter=stat_attrs["boxcenter"],
            comm=comm,
            normalization_scheme=stat_attrs.get("normalization_scheme", "particle"),
        )
        _log(1, "SHOT", "selection: use rfield_a + N_field_b for shot-noise terms.")
    else:
        rfield_shot_a = rfield_a
        N_field_a = get_N_field(
            catalogs=catalogs,
            tracer_flag="a",
            alpha=stat_attrs["alpha_a"],
            nmesh=stat_attrs["nmesh"],
            geometry="survey-like",
            column_names=stat_attrs["column_names"],
            boxsize=stat_attrs["boxsize"],
            sampler=stat_attrs["sampler"],
            interlaced=stat_attrs["interlaced"],
            z_range=stat_attrs["z_range"],
            comp_weight_plan=stat_attrs["comp_weight_plan"],
            para_cosmo=stat_attrs["cosmology"],
            boxcenter=stat_attrs["boxcenter"],
            comm=comm,
            normalization_scheme=stat_attrs.get("normalization_scheme", "particle"),
        )
        _log(1, "SHOT", "selection: use rfield_a + N_field_a for shot-noise terms.")

    _log(1, "SHOT", "N_field construction finished")

    _log(1, "SUMMARY", f"evaluated={evaluated_cfg_count} skipped={skipped_cfg_count} total={total_cfg_count}")

    k_center = 0.5 * (k_edge[1:] + k_edge[:-1])
    k3_min, k3_max = 2 * k_min, 2 * k_max
    k3_edge = np.linspace(k3_min, k3_max, k_bins * 2 + 1)[:-1]
    k3_center = 0.5 * (k3_edge[1:] + k3_edge[:-1])
    k3_bins = len(k3_center)
    sub_count_k3, _ = get_kbin_count(k3_bins, k3_edge, knorm)
    k_num_k3 = comm.reduce(sub_count_k3, op=MPI.SUM, root=0)
    k_num_k3 = comm.bcast(k_num_k3 if rank == 0 else None, root=0)

    # calculate shot-noise terms
    def calculate_shot_noise_S1_like(rfield_shot, N_field_shot, S_LM_shot = None):
        """
        Compute S1-like one-dimensional shot-noise term S(k1) for survey mode.
        This helper is kept inside calculate_bk_sugi_survey to reuse local variables.

        Implemented conditions:
        - survives only when ell_1 == L and ell_2 == 0
        - returns zeros when required shot-noise fields are unavailable
        - the S_LM correction term is included only for same-tracer case (aaa/auto)
        """
        S1_local = np.zeros(k_bins, dtype="complex128")

        if not (ell_1 == L and ell_2 == 0):
            _log(1, "SHOT", "S1-like skipped: Kronecker condition ell_1==L and ell_2==0 not satisfied.")
            return S1_local
        if (rfield_shot is None) or (N_field_shot is None):
            _log(1, "SHOT", "S1-like skipped: rfield_shot or N_field_shot is None.")
            return S1_local

        cfield_delta = rfield_shot.r2c()
        cfield_delta.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        cfield_delta[:] *= boxsize.prod()

        # Sbar term is only needed for same-tracer case.
        use_sbar_term = (correlation_mode == "auto" and tracer_type == "aaa")
        if use_sbar_term:
            if S_LM_shot is None:
                S_LM_arr = np.zeros(L + 1, dtype="complex128")
                _log(1, "SHOT-S1", "use S_LM correction with default zeros (S_LM_shot is None)")
            elif np.isscalar(S_LM_shot):
                S_LM_arr = np.zeros(L + 1, dtype="complex128")
                S_LM_arr[0] = S_LM_shot
            else:
                S_LM_arr = np.asarray(S_LM_shot, dtype="complex128")
            _log(1, "SHOT-S1", f"use S_LM correction with values={S_LM_arr}")
        else:
            S_LM_arr = None
            _log(1, "SHOT-S1", "skip S_LM correction (not same-tracer auto case).")

        # Use conjugate symmetry: evaluate only M = 0..L.
        # eta=1 for M=0, eta=2 for M>0, with Re-part accumulation.
        _log(1, "SHOT-S1", f"start M-loop range=[0,{L}] use_sbar_term={use_sbar_term}")
        _log(1, "SHOT-S1", "strategy=accumulate_complex_field_then_bin_then_2Re")
        accum_field = np.zeros_like(cfield_delta[:], dtype="complex128")
        for M in range(0, L + 1):
            m_weight = 0.5 if M == 0 else 1.0
            _log(2, "SHOT-S1", f"processing M={M}, weight={m_weight}")
            y_LM = get_Ylm(L, M, Racah_normalized=True)

            rfield_NLM = N_field_shot * y_LM(xgrid[0], xgrid[1], xgrid[2])
            cfield_NLM = rfield_NLM.r2c()
            cfield_NLM.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
            cfield_NLM[:] *= boxsize.prod()

            term_field = cfield_delta[:] * np.conj(cfield_NLM[:])

            if use_sbar_term:
                Sbar_LM = S_LM_arr[M] 
                S_field = cfield_delta.copy()
                S_field[:] = Sbar_LM
                S_field.apply(out=Ellipsis, func=compensation_shot_sugi[0][1], kind=compensation_shot_sugi[0][2])
                term_field[:] -= S_field[:]
                del S_field
            
            term_field *= y_LM(kgrid[0], kgrid[1], kgrid[2])
            accum_field += m_weight * term_field
            del rfield_NLM, cfield_NLM, term_field
            gc.collect()

        sub_sum_total = radial_binning(accum_field, k_bins, k_edge, knorm)
        sub_sum_total = 2.0 * np.real(sub_sum_total)
        total_sum = comm.reduce(sub_sum_total, op=MPI.SUM, root=0)
        if rank == 0:
            valid_mask = k_num > 0
            S1_local[valid_mask] = (2 * L + 1) * total_sum[valid_mask] / k_num[valid_mask]
        _log(1, "SHOT-S1", "finished M accumulation, single radial binning, and 2*Re on binned vector.")
        S1_local = comm.bcast(S1_local if rank == 0 else None, root=0)

        del cfield_delta, accum_field, sub_sum_total
        gc.collect()
        return S1_local

    def calculate_Q_L_like(rfield_shot, N_field_shot, S_LM_shot=None):
        """
        Compute Q_L(k3) for S3-like shot-noise term.
        """
        Q_like = np.zeros(k3_bins, dtype="complex128")
        if (rfield_shot is None) or (N_field_shot is None):
            _log(1, "SHOT-S3", "Q_L skipped: rfield_shot or N_field_shot is None.")
            return Q_like

        use_sbar_term = (correlation_mode == "auto" and tracer_type == "aaa")
        if use_sbar_term:
            if S_LM_shot is None:
                S_LM_arr = np.zeros(L + 1, dtype="complex128")
            elif np.isscalar(S_LM_shot):
                S_LM_arr = np.zeros(L + 1, dtype="complex128")
                S_LM_arr[0] = S_LM_shot
            else:
                S_LM_arr = np.asarray(S_LM_shot, dtype="complex128")
            _log(1, "SHOT-S3", f"use S_LM correction with values={S_LM_arr}")
        else:
            S_LM_arr = None
            _log(1, "SHOT-S3", "skip S_LM correction (not same-tracer auto case).")

        rfield_N00 = N_field_shot
        cfield_N00 = rfield_N00.r2c()
        cfield_N00.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        cfield_N00[:] *= boxsize.prod()
        del rfield_N00
        gc.collect()

        _log(1, "SHOT-S3", f"start Q_L loop M=[0,{L}] use_sbar_term={use_sbar_term}")
        accum_field = np.zeros_like(cfield_N00[:], dtype="complex128")
        for M in range(0, L + 1):   
            M_weight = 0.5 if M == 0 else 1.0
            _log(2, "SHOT-S3", f"Q_L processing M={M}, weight={M_weight}")
            y_lm = get_Ylm(L, M, Racah_normalized=True)

            rfield_delta_lm = rfield_shot * np.conj(y_lm(xgrid[0], xgrid[1], xgrid[2]))
            cfield_delta_lm = rfield_delta_lm.r2c()
            cfield_delta_lm.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
            cfield_delta_lm[:] *= boxsize.prod()

            term_field = cfield_delta_lm[:] * np.conj(cfield_N00[:])
            if use_sbar_term:
                Sbar_lm = S_LM_arr[M]
                S_field = cfield_delta_lm.copy()
                S_field[:] = Sbar_lm
                S_field.apply(out=Ellipsis, func=compensation_shot_sugi[0][1], kind=compensation_shot_sugi[0][2])
                term_field[:] -= S_field[:]
                del S_field

            term_field *= y_lm(kgrid[0], kgrid[1], kgrid[2])
            accum_field += M_weight * term_field

            del rfield_delta_lm, cfield_delta_lm, term_field
            gc.collect()

        sub_sum = radial_binning(accum_field, k3_bins, k3_edge, knorm)
        sub_sum = 2.0 * np.real(sub_sum)
        total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)
        if rank == 0:
            Q_like = total_sum.astype("complex128")
        Q_like = comm.bcast(Q_like if rank == 0 else None, root=0)

        del cfield_N00, accum_field, sub_sum
        gc.collect()
        _log(1, "SHOT-S3", "Q_L finished.")
        return Q_like

    def calculate_shot_noise_S3_like(Q_like, mode):
        """
        Compute S3-like term with box-consistent branching:
        - diagonal: only i==j
        - full: all (i,j)
        """
        if tracer_type == "abb":
            _log(1, "SHOT-S3", f"S3 forced to zero in mode={mode} because tracer_type=abb.")
            if mode == "full":
                return np.zeros((k_bins, k_bins), dtype="complex128")
            return np.zeros(k_bins, dtype="complex128")

        weighted_k_num = np.zeros(k3_bins, dtype="float64")
        valid_k3 = k_num_k3 > 0
        weighted_k_num[valid_k3] = k_num_k3[valid_k3] / k3_center[valid_k3]

        if mode == "diagonal":
            S3_local = np.zeros(k_bins, dtype="complex128")
            if rank == 0:
                for i in range(k_bins):
                    q_ells = get_q_ells(
                        i, i, k_center, k_min, k_max, k_bins,
                        ell_1, ell_2, L, k3_bins
                    )
                    denom = np.sum(weighted_k_num[:2 * i + 1])
                    if denom > 0:
                        S3_local[i] = np.sum(q_ells * Q_like / k3_center) / denom
                S3_local *= H_ells * N_ells
            return comm.bcast(S3_local if rank == 0 else None, root=0)
        elif mode == "full":
            S3_local = np.zeros((k_bins, k_bins), dtype="complex128")
            if rank == 0:
                for i in range(k_bins):
                    for j in range(k_bins):
                        q_ells = get_q_ells(
                            i, j, k_center, k_min, k_max, k_bins,
                            ell_1, ell_2, L, k3_bins
                        )
                        valid_bins = get_valid_k3_bins(k_center[i], k_center[j], k_min, k_max, k_bins)
                        denom = np.sum(weighted_k_num * valid_bins)
                        if denom > 0:
                            S3_local[i, j] = np.sum(q_ells * Q_like / k3_center) / denom
                S3_local *= H_ells * N_ells
            return comm.bcast(S3_local if rank == 0 else None, root=0)
        else:
            raise ValueError("mode must be either 'diagonal' or 'full' for calculate_shot_noise_S3_like.")

    if data_vector_mode == "diagonal":
        SN0 = 0.0 + 0.0j
        SN1 = np.zeros(k_bins, dtype="complex128")
        SN2 = np.zeros(k_bins, dtype="complex128")
        SN3 = np.zeros(k_bins, dtype="complex128")

        if tracer_type == "abc":
            _log(1, "SHOT", "tracer_type=abc -> S0=S1=S2=S3=0.")
        else:
            rfield_shot = None
            N_field_shot = None
            S_LM_shot = None
            if tracer_type == "aab":
                rfield_shot = rfield_shot_b
                N_field_shot = N_field_a
                S_LM_shot = stat_attrs.get("S_LM_a", None)
                _log(1, "SHOT", "active tracer branch: aab -> rfield_shot=rfield_b, N_field_shot=N_field_a, S_LM_shot=S_LM_a")
            elif tracer_type == "abb":
                rfield_shot = rfield_shot_a
                N_field_shot = N_field_b
                S_LM_shot = stat_attrs.get("S_LM_b", None)
                _log(1, "SHOT", "active tracer branch: abb -> rfield_shot=rfield_a, N_field_shot=N_field_b, S_LM_shot=S_LM_b")
            else:
                rfield_shot = rfield_shot_a
                N_field_shot = N_field_a
                S_LM_shot = stat_attrs.get("S_LM_a", None)
                _log(1, "SHOT", "active tracer branch: aaa/other -> rfield_shot=rfield_a, N_field_shot=N_field_a, S_LM_shot=S_LM_a")

            # S0: only for tracer_type=="aaa" and (ell1,ell2,L)==(0,0,0).
            if tracer_type == "aaa" and (ell_1 == 0 and ell_2 == 0 and L == 0):
                S_LM_a = stat_attrs.get("S_LM_a", 0.0)
                SN0 = S_LM_a[0] if isinstance(S_LM_a, np.ndarray) else S_LM_a
                _log(1, "SHOT-S0", f"active: tracer_type=aaa and angu_config=(0,0,0), value={SN0}")
            else:
                SN0 = 0.0 + 0.0j
                _log(1, "SHOT-S0", "set_to_zero: requires tracer_type=aaa and angu_config=(0,0,0).")

            # S1: non-zero only for tracer_type in {"aaa","abb"} and ell_1 == L, ell_2 == 0.
            if (tracer_type in ["aaa", "abb"]) and (ell_1 == L and ell_2 == 0):
                SN1 = calculate_shot_noise_S1_like(rfield_shot, N_field_shot, S_LM_shot)
                _log(1, "SHOT", "SN1 active.")
            else:
                SN1 = np.zeros(k_bins, dtype="complex128")
                _log(1, "SHOT", "SN1 set to 0 (requires tracer_type in {aaa,abb} and ell_1==L, ell_2==0).")

            # S2: only for tracer_type=="aaa" and (0,0,0); then S2 = S1.
            if tracer_type == "aaa" and (ell_1 == 0 and ell_2 == 0 and L == 0):
                SN2 = SN1.copy()
                _log(1, "SHOT-S2", "active: reuse SN1 because tracer_type=aaa and angu_config=(0,0,0).")
            else:
                SN2 = np.zeros(k_bins, dtype="complex128")
                _log(1, "SHOT-S2", "set_to_zero: requires tracer_type=aaa and angu_config=(0,0,0).")

            # S3: zero for tracer_type=="abb"; otherwise keep dedicated branch.
            if tracer_type != "abb":
                Q_like = calculate_Q_L_like(rfield_shot, N_field_shot, S_LM_shot)
                SN3 = calculate_shot_noise_S3_like(Q_like, mode="diagonal")
                _log(1, "SHOT", "SN3 active.")
            else:
                SN3 = np.zeros(k_bins, dtype="complex128")
                _log(1, "SHOT", "SN3 set to 0 because tracer_type=abb.")
    else:
        SN0 = 0.0 + 0.0j
        SN1 = np.zeros((k_bins, k_bins), dtype="complex128")
        SN2 = np.zeros((k_bins, k_bins), dtype="complex128")
        SN3 = np.zeros((k_bins, k_bins), dtype="complex128")

        if tracer_type == "abc":
            _log(1, "SHOT", "full mode: tracer_type=abc -> S0=S1=S2=S3=0.")
        else:
            rfield_shot = None
            N_field_shot = None
            S_LM_shot = None
            if tracer_type == "aab":
                rfield_shot = rfield_shot_b
                N_field_shot = N_field_a
                S_LM_shot = stat_attrs.get("S_LM_a", None)
                _log(1, "SHOT", "full mode branch: aab -> rfield_shot=rfield_b, N_field_shot=N_field_a, S_LM_shot=S_LM_a")
            elif tracer_type == "abb":
                rfield_shot = rfield_shot_a
                N_field_shot = N_field_b
                S_LM_shot = stat_attrs.get("S_LM_b", None)
                _log(1, "SHOT", "full mode branch: abb -> rfield_shot=rfield_a, N_field_shot=N_field_b, S_LM_shot=S_LM_b")
            else:
                rfield_shot = rfield_shot_a
                N_field_shot = N_field_a
                S_LM_shot = stat_attrs.get("S_LM_a", None)
                _log(1, "SHOT", "full mode branch: aaa/other -> rfield_shot=rfield_a, N_field_shot=N_field_a, S_LM_shot=S_LM_a")

            # S0: only for tracer_type=="aaa" and (ell1,ell2,L)==(0,0,0).
            if tracer_type == "aaa" and (ell_1 == 0 and ell_2 == 0 and L == 0):
                S_LM_a = stat_attrs.get("S_LM_a", 0.0)
                SN0 = S_LM_a[0] if isinstance(S_LM_a, np.ndarray) else S_LM_a
                _log(1, "SHOT-S0", f"full_mode active: tracer_type=aaa and angu_config=(0,0,0), value={SN0}")
            else:
                SN0 = 0.0 + 0.0j
                _log(1, "SHOT-S0", "full_mode set_to_zero: requires tracer_type=aaa and angu_config=(0,0,0).")

            # S1 vector then expand to full (same style as box implementation).
            if (tracer_type in ["aaa", "abb"]) and (ell_1 == L and ell_2 == 0):
                SN1_vec = calculate_shot_noise_S1_like(rfield_shot, N_field_shot, S_LM_shot)
                _log(1, "SHOT", "full mode SN1 active (from 1D S1-like).")
            else:
                SN1_vec = np.zeros(k_bins, dtype="complex128")
                _log(1, "SHOT", "full mode SN1 set to 0 (requires tracer_type in {aaa,abb} and ell_1==L, ell_2==0).")

            for i in range(k_bins):
                SN1[i, :] = SN1_vec[i]

            # S2 vector then expand to full (same style as box implementation).
            if tracer_type == "aaa" and (ell_1 == 0 and ell_2 == 0 and L == 0):
                SN2_vec = SN1_vec.copy()
                _log(1, "SHOT-S2", "full_mode active: reuse SN1_vec because tracer_type=aaa and angu_config=(0,0,0).")
            else:
                SN2_vec = np.zeros(k_bins, dtype="complex128")
                _log(1, "SHOT-S2", "full_mode set_to_zero: requires tracer_type=aaa and angu_config=(0,0,0).")

            for i in range(k_bins):
                SN2[:, i] = SN2_vec[i]

            # S3 in full mode.
            if tracer_type != "abb":
                Q_like = calculate_Q_L_like(rfield_shot, N_field_shot, S_LM_shot)
                SN3 = calculate_shot_noise_S3_like(Q_like, mode="full")
                _log(1, "SHOT", "full mode SN3 active.")
            else:
                SN3 = np.zeros((k_bins, k_bins), dtype="complex128")
                _log(1, "SHOT", "full mode SN3 set to 0 because tracer_type=abb.")
    time_shotnoise_end = time.time()
    if rank == 0:
        logging.info(f"Rank {rank}: Time to compute shot noise terms: {time_shotnoise_end - time_shotnoise_start:.2f} seconds")
    if rank == 0:
        # Signal normalization (no shot-noise subtraction at this stage).
        total_res *= N_ells * H_ells / I_norm
        if data_vector_mode == "full":
            total_res *= 1.0 / np.outer(k_num, k_num)
            sn_shape = (k_bins, k_bins)
        else:
            total_res *= 1.0 / (k_num ** 2)
            sn_shape = (k_bins,)
        total_res *= vol_per_cell

        total_shot_noise = (SN0 + SN1 + SN2 + SN3) / I_norm
        final_bk = total_res - total_shot_noise

        results.update({
            "B_sugi": final_bk,
            "Bk_raw": total_res,
            "SN_terms": {"SN0": SN0/I_norm, "SN1": SN1/I_norm, "SN2": SN2/I_norm, "SN3": SN3/I_norm},
            "Shot_noise": total_shot_noise,
            "I_norm": I_norm,
            "k_eff": k_eff,
            "nmodes": k_num,
        })

    # Keep N_field/rfield construction in this stage, but do not serialize mesh objects.
    del rfield_shot_a, rfield_shot_b, rfield_shot_c
    del N_field_a, N_field_b, N_field_c
    gc.collect()

    results = comm.bcast(results, root=0)
    if rank == 0:
        logging.info(f"Rank {rank}: Finished survey bispectrum (signal-only) calculation.")
    return results


def get_G_ell(rfield, ell, kgrid, xgrid,compensation,boxsize,comm):
    r"""
    Calculate the G_ell function for the given ell.
    This function is a placeholder and should be replaced with the actual implementation.
    \mathcal{G}_\ell(\mathbf{k})= \frac{4\pi}{2\ell+1} \left[\frac{1}{2}F_\ell^0(\mathbf{k})
    +\sum_{m=1}^\ell F_\ell^m(\mathbf{k})\right]
    """
    rank = comm.Get_rank()

    Ylms = [get_Ylm(ell, m) for m in range(ell + 1)]
    rf = rfield * Ylms[0](xgrid[0], xgrid[1], xgrid[2])
    G_ell = rf.r2c() 
    G_ell[:] *= Ylms[0](kgrid[0], kgrid[1], kgrid[2])/2

    for m in range(1, ell + 1):
        rf = rfield * np.conj(Ylms[m](xgrid[0], xgrid[1], xgrid[2]))
        cf = rf.r2c()
        cf[:] *= Ylms[m](kgrid[0], kgrid[1], kgrid[2])
        G_ell[:] += cf[:]

    # recollect the memory
    del rf, cf
    gc.collect()

    G_ell.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    if rank == 0:
        logging.info(f"{compensation[0][1].__name__} applied to G_ell")
        # print(f"rank {rank}: type of G_ell = {type(G_ell)}")
    G_ell[:] *= 4*np.pi*boxsize.prod() / (2*ell + 1)

    return G_ell


def validate_tracer(tracer_type, correlation_mode):
    assert correlation_mode in ["auto", "cross"], "correlation_mode must be either 'auto' or 'cross'"
    assert tracer_type in ["aaa", "aab", "abb", "abc"], "tracer_type must be one of 'aaa', 'aab', 'abb', or 'abc'"
    if correlation_mode == "auto":
        assert tracer_type == "aaa", "For auto-correlation, tracer_type must be 'aaa'"


def validate_poles(poles):
    """
        Validate the poles input.
        Raise ValueError if the input is not valid.
        1. All elements must be non-negative integers.
        2. No duplicate values.
        3. If more than one value, they must be sorted in ascending order.
    """
    if not all(isinstance(p, int) and p >= 0 for p in poles):
        raise ValueError("All elements in 'poles' must be non-negative integers.")
    if len(poles) != len(set(poles)):
        raise ValueError("'poles' contains duplicate values.")
    if len(poles) > 1 and poles != sorted(poles):
        raise ValueError("'poles' must be sorted in ascending order if it contains more than one value.")
    

def validate_sugi_poles(angu_config, geometry):    
    """
        Validate the angu_config input for Sugiyama bispectrum estimator.
        Raise ValueError if the input is not valid.
        1. All ell elements must satisfy quantum angular number conditions.
        2. ell1 >= ell2 
        3. The sum of the three ell elements must be even.
    """
    ell1, ell2, L = angu_config
    if ell1 < ell2:
        raise ValueError("In 'angu_config', ell1 must be greater than or equal to ell2.")
    if (ell1 + ell2 + L) % 2 != 0:
        raise ValueError("The sum of ell1, ell2, and L in 'angu_config' must be even.")
    if  L % 2 != 0:
        raise ValueError("L must be even.")
    # Triangle condition
    if not (ell1 - ell2 <= L <= ell1 + ell2):
        raise ValueError("The values in 'angu_config' do not satisfy the triangle condition.")
    if geometry not in ["box-like", "survey-like"]:
        raise ValueError("geometry must be either 'box-like' or 'survey-like'.")





"""
Taken from nbodykit.algorithms.convpower.catalogmesh
see Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240> for details
"""
def get_compensation(interlaced, sampler):
    """
    Return the compensation function, which corrects for the
    windowing kernel.
    """
    if interlaced:
        d = {'cic' : CompensateCIC,
             'tsc' : CompensateTSC,
             'pcs' : CompensatePCS,
             'ngp' : CompensateNGP,
            }
    else:
        d = {'cic' : CompensateCICShotnoise,
             'tsc' : CompensateTSCShotnoise,
             'pcs' : CompensatePCSShotnoise,
             'ngp' : CompensateNGPShotnoise,
            }

    if not sampler in d:
        raise ValueError("compensation for window %s is not defined" % sampler)
    filter = d[sampler]
    return [('complex', filter, "circular")]


def get_compensation_bk_sugi(sampler):
    d = {'cic' : CompensateCIC,
        'tsc' : CompensateTSC,
        'pcs' : CompensatePCS,
        'ngp' : CompensateNGP,
        }        
    if not sampler in d:
        raise ValueError("compensation for window %s is not defined" % sampler)
    filter = d[sampler]
    return [('complex', filter, "circular")]


def get_compensation_shot_sugi(sampler):
    d = {'cic' : Compensate_bk_noise_cic,
        'tsc' : Compensate_bk_noise_tsc,
        'pcs' : Compensate_bk_noise_pcs,
        'ngp' : Compensate_bk_noise_ngp,
        }
    
    if not sampler in d:
        raise ValueError("compensation for window %s is not defined" % sampler)
    
    filter = d[sampler]
    return [('complex', filter, "circular")]
