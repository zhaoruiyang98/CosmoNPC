import numpy as np
from mpi4py import MPI
import logging
from sympy import legendre_poly
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
    cfield_a[knorm == 0.] = 0. 
    if rank == 0:
        logging.info(f"Rank {rank}: Zero-mode in Fourier space cleared")

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
                   nmesh.prod() / N_gal = (nmesh.prod() / N_Z * boxsize.prod())
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


def calculate_bk_sugi_box(rfield_a, rfield_b, correlation_mode, \
                          stat_attrs, comm, **kwargs):
    rank = comm.Get_rank()

    # Extract mesh attributes
    angu_config = stat_attrs['angu_config']
    boxsize, nmesh = np.array(stat_attrs['boxsize']), np.array(stat_attrs['nmesh'])
    k_min, k_max, k_bins = stat_attrs['k_min'], stat_attrs['k_max'], stat_attrs['k_bins']
    sampler, interlaced = stat_attrs['sampler'], stat_attrs['interlaced']
    tracer_type = stat_attrs['tracer_type']  # "aaa", "aab", "abb" or "abc"

    # make sure the tracer_type and correlation_mode are compatible
    validate_tracer(tracer_type, correlation_mode)



    if correlation_mode == "cross":
        NZ_a, N_gal_a = stat_attrs['NZ_a']
        NZ_b, N_gal_b = stat_attrs['NZ_b']
    else:
        NZ_a, N_gal_a = stat_attrs['NZ_a'], stat_attrs['N_gal_a']

    # Define some useful variables
    k_edge = np.linspace(k_min, k_max, k_bins + 1)
    if rank == 0:
        logging.info(f"Rank {rank}: k_edge = {k_edge}")
    comm.Barrier()

    # get the compensation mode in k-space
    compensation = get_compensation_bk_sugi(sampler, compen_mode="wmass")
    if rank == 0:
        logging.info(f"In Sugiyama estimator, the interlacing technique is not supported currently. Falling back to non-interlaced mode.")

    # Calculate the Fourier transform of the density field and compensate it
    cfield_a =  rfield_a.r2c()
    cfield_a.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    

    logging.info(f"Rank {rank}: The shape of rfield_a is {rfield_a.shape}, the shape of cfield_a is {cfield_a.shape}.")
    if rank == 0:
        logging.info(f"Rank {rank}: {compensation[0][1].__name__} applied to the density field a")

    if correlation_mode == "cross":
        cfield_b =  rfield_b.r2c()
        cfield_b.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        if rank == 0:
            logging.info(f"Rank {rank}: {compensation[0][1].__name__} applied to the density field b")


    # Some constants to be evaluated for bispectrum calculation
    """
        N_ells: N_{\ell_1 \ell_2 L}=\left(2 \ell_1+1\right)\left(2 \ell_2+1\right)(2 L+1)
        H_ells: H_{\ell_1 \ell_2 L}=\left(\begin{array}{ccc}
                                        \ell_1 & \ell_2 & L \\
                                        0 & 0 & 0
                                        \end{array}\right)
    """
    H_ells = float(wigner_3j(angu_config[0], angu_config[1], angu_config[2], 0, 0, 0))
    N_ells = (2*angu_config[0]+1) * (2*angu_config[1]+1) * (2*angu_config[2]+1)

    data_vector_mode = stat_attrs["data_vector_mode"]

    if correlation_mode == "auto":
        G_00 = cfield_a.c2r()
    else:
        G_00 = cfield_b.c2r()
    G_00 = np.real(G_00) # G_00 is strictly to be real

    kgrid, knorm = get_kgrid(cfield_a)
    cfield_a[knorm == 0.] = 0.
    if correlation_mode == "cross":
        cfield_b[knorm == 0.] = 0.
    if rank == 0:
        logging.info(f"Rank {rank}: Zero-mode in Fourier space cleared")

    # create the results container
    results = {} if rank == 0 else None
    

    if data_vector_mode == "full":
        raise NotImplementedError("Full data vector mode is not implemented yet.")
    elif data_vector_mode == "diagonal":
        total_res = np.zeros(k_bins).astype("complex128")
        if correlation_mode == "auto" or (correlation_mode == "cross" and tracer_type == "aab"):
            for m in range(0, angu_config[0]+1):
                sub_res = np.zeros(k_bins).astype("complex128")
                ylm = get_Ylm(angu_config[0], m, Racah_normalized=True)
                ylm_weighted_cfield = cfield_a * ylm(kgrid[0], kgrid[1], kgrid[2])
                # binning in k-space then ifft it to real space
                for i in range(k_bins):
                    mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i+1])
                    binned_field = (ylm_weighted_cfield * mask).c2r()
                    # the coefficient -1^{l+m} can be found in our methodology paper
                    sub_sum = (-1)**(angu_config[0]+m) * np.sum(G_00 * np.real(binned_field)* np.imag(binned_field))
                    # gather the results from all ranks
                    total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)
                    if rank == 0:
                        sub_res[i] = total_sum
                if rank == 0:
                    if m == 0:
                        total_res += sub_res
                    else:
                        total_res += 2 * sub_res
        elif correlation_mode == "cross" and tracer_type == "abb":
            for m in range(0, angu_config[0]+1):
                sub_res = np.zeros(k_bins).astype("complex128")
                ylm = get_Ylm(angu_config[0], m, Racah_normalized=True)
                ylm_weighted_cfield_a = cfield_a * ylm(kgrid[0], kgrid[1], kgrid[2])
                ylm_weighted_cfield_b = cfield_b * ylm(kgrid[0], kgrid[1], kgrid[2])
                # binning in k-space then ifft it to real space
                for i in range(k_bins):
                    mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i+1])
                    binned_field_a = (ylm_weighted_cfield_a * mask).c2r()
                    binned_field_b = (ylm_weighted_cfield_b * mask).c2r()
                    sub_sum = (-1)**(angu_config[0]+m) * np.sum(G_00 * binned_field_a * np.real(binned_field_b))
                    # gather the results from all ranks
                    total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)
                    if rank == 0:
                        sub_res[i] = total_sum
                if rank == 0:
                    if m == 0:
                        total_res += sub_res
                    else:
                        total_res += 2 * np.real(sub_res)
        

    # Now we obtain the shot-noise, we define 4 terms in EQ45 of Sugiyama 2019 as SN0, SN1, SN2, SN3
    # SN0
    S_00 = N_gal_a
    if correlation_mode == "auto":
        SN0 = 1/NZ_a**2 # normalization factor I is already included
    else:
        SN0 = 0.0
       
    """
    The logic of this part can be found in our methodology paper. The Q_0 term is necessary.
    """
    # get k_eff and k_num in one particular k_bin
    sub_count, sub_knorm_sum = get_kbin_count(k_bins, k_edge, knorm)
    # gather the results from all ranks
    k_num = comm.reduce(sub_count, op=MPI.SUM, root=0)
    total_knorm_sum = comm.reduce(sub_knorm_sum, op=MPI.SUM, root=0)
    if rank == 0:
        k_eff = total_knorm_sum / k_num   
    if correlation_mode == "auto":
        S_field = cfield_a.copy()
        S_field[:] = S_00
        cpst_bk_noise = get_compensation_bk_sugi(sampler, compen_mode="wmass")
        S_field.apply(out=Ellipsis, func=cpst_bk_noise[0][1], kind=cpst_bk_noise[0][2])
        P_field = np.real(cfield_a[:])**2 + np.imag(cfield_a[:])**2 - S_field
    elif correlation_mode == "cross":
        P_field = np.real(cfield_a[:]) * np.real(cfield_b[:]) + np.imag(cfield_a[:]) * np.imag(cfield_b[:])

    res_P0 = np.zeros(k_bins).astype('c16')
    sub_sum = radial_binning(P_field, k_bins, k_edge, knorm)
    total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)


    # then we can get SN1, SN2
    if rank == 0:
        if correlation_mode == "auto":
            if angu_config[0] == 0 and angu_config[1] == 0:
                SN1 = total_sum / k_num
                SN2 = SN1.copy()
            else:
                SN1 = 0.0
                SN2 = 0.0
        elif correlation_mode == "cross" and tracer_type == "abb":
            if angu_config[0] == 0 and angu_config[1] == 0:
                SN1 = total_sum / k_num
            else:
                SN1 = 0.0
            SN2 = 0.0
        elif correlation_mode == "cross" and tracer_type == "aab":
            SN1 = 0.0
            SN2 = 0.0

    # SN3
    if tracer_type == "abb":
        SN3 = 0.0
    else:
        SN3 = np.zeros(k_bins).astype('c16')
        k_center = 0.5 * (k_edge[1:] + k_edge[:-1])
        # note that $k_{\mathrm{c}, \min }, k_{\mathrm{c}, \max }=\left|k_1-k_2\right|+\Delta k / 2, k_1+k_2-\Delta k / 2$
        k3_min, k3_max = 2 * k_min, 2 * k_max
        k3_edge = np.linspace(k3_min, k3_max, k_bins *2 + 1)[:-1]
        k3_center = 0.5 * (k3_edge[1:] + k3_edge[:-1])

        # res_P0_extended = np.zeros(len(k3_center)).astype('c16')
        sub_sum_extended = radial_binning(P_field, len(k3_center), k3_edge, knorm)
        total_sum_extended = comm.reduce(sub_sum_extended, op=MPI.SUM, root=0)

        sub_count_extended, sub_knorm_sum_extended = get_kbin_count(len(k3_center), k3_edge, knorm)
        # gather the results from all ranks
        k_num_extended = comm.reduce(sub_count_extended, op=MPI.SUM, root=0)
        # total_knorm_sum_extended = comm.reduce(sub_knorm_sum_extended, op=MPI.SUM, root=0)

        if rank == 0:
            weighted_k_num_extended = k_num_extended / k3_center
            for i in range(k_bins):
                coeff_legendre = get_legendre_coefficients(angu_config[0], k_center[i], k_center[i], k_min, k_max, k_bins, mode="13")
                SN3[i] = np.sum(coeff_legendre * total_sum_extended / k3_center) / np.sum(weighted_k_num_extended[:(2*i+1)])

            SN3 *= 2 * angu_config[0] + 1

    if rank == 0:
        logging.info(f"Rank {rank}: Shot noise terms calculated: SN0={SN0}, SN1[0]={SN1[0] if isinstance(SN1, np.ndarray) else SN1}, SN2[0]={SN2[0] if isinstance(SN2, np.ndarray) else SN2}, SN3[0]={SN3[0] if isinstance(SN3, np.ndarray) else SN3}")

        # Combine all shot-noise terms
        total_shot_noise = SN0 + SN1 + SN2 + SN3
        logging.info(f"Rank {rank}: Total shot noise calculated.")

        # Final bispectrum result after subtracting shot-noise
        final_bk = total_res / N_ells / (H_ells **2) - total_shot_noise
        logging.info(f"Rank {rank}: Final bispectrum calculated.")

        # Store the results
        results.update({
            'B_sugi': final_bk,
            'SN_terms': {
                'SN0': SN0,
                'SN1': SN1,
                'SN2': SN2,
                'SN3': SN3
            }
        })
        logging.info(f"Rank {rank}: Results stored.")






def get_legendre_coefficients(ell, k1 ,k2 ,k_min,k_max,kbin, mode = "12"):
    """
        if mode == "13", mu is the cosine of the angle between k1 and k3
        if mode == "12", mu is the cosine of the angle between k1 and k2
    """
    k3_min, k3_max = 2 * k_min, 2 * k_max
    k3_edge = np.linspace(k3_min, k3_max, kbin *2 + 1)[:-1]
    k3_center = 0.5 * (k3_edge[1:] + k3_edge[:-1])

    res_legendre = np.zeros(len(k3_center))

    assert mode in ["12", "13"], "mode must be either '12' or '13'"

    for i in range(len(k3_center)):
        if k3_center[i] < k1 + k2:
            if mode == "13":
                res_legendre[i] = legendre_poly(ell, (k3_center[i]**2 + k1**2 - k2**2) / (2 * k1 * k3_center[i])).evalf()
            elif mode == "12":
                res_legendre[i] = legendre_poly(ell, (k1**2 + k2**2 - k3_center[i]**2) / (2 * k1 * k2)).evalf()

    res_legendre *= (-1)**ell # the mu calculated here is actually -mu in the formula

    return res_legendre

    






def get_G_ell(rfield, ell, kgrid, xgrid,compensation,boxsize,comm):
    """
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



def get_kbin_count(k_bins, k_edge, knorm):
    """
    Count the number of points as well as the sum k-distance in each k-bin 
    """
    # Initialize the result array
    sub_count = np.zeros(k_bins).astype("f8")
    sub_knorm_sum = np.zeros(k_bins).astype("f8")

    # Loop over the bins and sum the values in each bin
    for i in range(k_bins):
        mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i+1])
        sub_count[i] = np.sum(mask)
        sub_knorm_sum[i] = np.sum(knorm[mask])

    del mask
    gc.collect()

    return sub_count, sub_knorm_sum



def radial_binning(kfield, k_bins, k_edge, knorm):
    """
    Radial binning of the Fourier transform of the density field
    """
    # Initialize the result array
    sub_sum = np.zeros(k_bins).astype(kfield.dtype)

    # Loop over the bins and sum the values in each bin
    for i in range(k_bins):
        mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i+1])
        sub_sum[i] = np.sum(kfield[mask])

    del mask
    gc.collect()

    return sub_sum




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
