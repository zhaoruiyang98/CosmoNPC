import numpy as np
import time
from mpi4py import MPI
import logging
from sympy.physics.wigner import wigner_3j
from .math_evaluator import *
from .param_helper import validate_tracer, validate_poles, validate_sugi_poles
import gc


def calculate_bk_sco_box(rfield, stat_attrs, comm, **kwargs):
    rank = comm.Get_rank()

    # Extract mesh attributes
    poles = stat_attrs["poles"]
    boxsize, nmesh = np.array(stat_attrs["boxsize"]), np.array(stat_attrs["nmesh"])
    # boxcenter = np.array(stat_attrs['boxcenter'])
    k_min, k_max, k_bins = (
        stat_attrs["k_min"],
        stat_attrs["k_max"],
        stat_attrs["k_bins"],
    )
    sampler, interlaced = stat_attrs["sampler"], stat_attrs["interlaced"]
    P_shot = stat_attrs["P_shot"]
    NZ = stat_attrs["NZ"]
    rsd = np.array(stat_attrs["rsd"])

    # Validate the poles
    validate_poles(poles)

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

    # Get the kgrid, knorm for binning and Legendre polynomials
    if poles == [0]:  # For ell = 0 only, we do not need to calculate the kgrid
        knorm = np.sqrt(sum(np.real(kk) ** 2 for kk in cfield.x))
        kgrid = None
    else:
        kgrid, knorm = get_kgrid(cfield)

    # clear zero-mode in the Fourier space
    cfield[knorm == 0.0] = 0.0
    if rank == 0:
        logging.info(f"Rank {rank}: Zero-mode in Fourier space cleared")

    """ 
        Before getting the bispectrum, we firstly get the power spectrum monopole
        as well as k_eff, k_num for the bk shot noise
    """
    P_field = np.real(cfield[:]) ** 2 + np.imag(cfield[:]) ** 2

    results = {} if rank == 0 else None

    """
    Note that poles at least contain one even number...
    """
    binned_F0x_list = get_binned_ifft_field(cfield, k_bins, k_edge, knorm, 0, comm)

    F_2, F_4 = None, None

    # Loop over the poles
    for pole in poles:
        if rank == 0:
            logging.info(f"Rank {rank}: Processing pole {pole}")

        if pole & 1:
            # for ell = odd, the multipoles are 0 strictly
            res = np.zeros(k_bins).astype("float128")
            if rank == 0:
                logging.info(
                    f"Rank {rank}: ell = {pole}, Odd multipoles are automatically set to zero"
                )
        else:
            if pole == 0:
                binned_list_k1 = binned_F0x_list
                F_ell = None
            else:
                L_ells = get_legendre(pole, rsd[0], rsd[1], rsd[2])
                if rank == 0:
                    logging.info(f"Rank {rank}: L_ells = {L_ells.expr}")
                F_ell = cfield * L_ells(kgrid[0], kgrid[1], kgrid[2])
                binned_list_k1 = get_binned_ifft_field(
                    F_ell, k_bins, k_edge, knorm, pole, comm
                )
            binned_list_k2 = binned_F0x_list
            binned_list_k3 = binned_F0x_list

        # loop over all possible combinations of binned_list_k1, k2, k3
        # with k1 >= k2 >= k3

        sub_res, tri_config, N_tri = [], [], []

        for i in range(k_bins):
            for j in range(i, k_bins):
                for k in range(j, k_bins):
                    # if rank == 0:
                    #     logging.info(f"Rank {rank}: Processing bins with index: {i}, {j}, {k}")
                    # get the bispectrum
                    if i + j >= k - 1:
                        sub_res.append(
                            np.sum(
                                binned_list_k1[i]
                                * binned_list_k2[j]
                                * binned_list_k3[k]
                            )
                        )
                        tri_config.append((i, j, k))
                        # $ V_T^{ANA} = 8\pi^2k_1k_2k_3\Delta k^3 $
                        N_tri.append(
                            8
                            * np.pi**2
                            * k_center[i]
                            * k_center[j]
                            * k_center[k]
                            * dk**3
                        )

        sub_res = np.array(sub_res).astype("float128")

        # Gather the results from all ranks
        bk_res = comm.reduce(sub_res, op=MPI.SUM, root=0)

        # get NT, the number of triangles
        if rank == 0:
            N_tri = np.array(N_tri)
            bk_res *= (2 * pole + 1) / N_tri
            vol_per_cell = boxsize.prod() / nmesh.prod()
            bk_res *= vol_per_cell**3 / boxsize.prod()  # we need further check this

            # store the results
            results.update(
                {
                    f"B{pole}": bk_res,
                }
            )
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
        mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i + 1])
        if pole == 0:
            list_to_return.append(
                np.real((kfield * mask).c2r())
            )  # it is strictly to be real
        else:
            list_to_return.append(
                2 * np.real((kfield * mask).c2r())
            )  # for G_ell, we take the real part and double it
    del mask
    gc.collect()

    if rank == 0:
        logging.info(
            f"Rank {rank}: Mesh shape of binned ifft field = {list_to_return[0].shape}"
        )

    return list_to_return


def calculate_power_spectrum_survey(
    stat_attrs, rfield_a, rfield_b, correlation_mode, comm, **kwargs
):
    rank = comm.Get_rank()

    # Extract mesh attributes
    poles = stat_attrs["poles"]
    high_order_mode = stat_attrs.get("high_order_mode", "default")
    boxsize, nmesh = np.array(stat_attrs["boxsize"]), np.array(stat_attrs["nmesh"])
    boxcenter = np.array(stat_attrs["boxcenter"])
    k_min, k_max, k_bins = (
        stat_attrs["k_min"],
        stat_attrs["k_max"],
        stat_attrs["k_bins"],
    )
    sampler, interlaced = stat_attrs["sampler"], stat_attrs["interlaced"]
    N0 = stat_attrs["N0"]

    if stat_attrs["normalization_scheme"] == "particle":
        I_norm = stat_attrs["I_rand"]
        if rank == 0:
            logging.info(
                f"Rank {rank}: Using particle normalization with I_rand = {I_norm}."
            )
    else:
        I_norm = stat_attrs["I_mesh"]
        if rank == 0:
            logging.info(
                f"Rank {rank}: Using mixed-mesh normalization with I_mesh = {I_norm}."
            )

    # Define some useful variables
    k_edge = np.linspace(k_min, k_max, k_bins + 1)
    if rank == 0:
        logging.info(f"Rank {rank}: k_edge = {k_edge}")

    comm.Barrier()

    def get_high_order_strategy(poles, high_order_mode):
        supported_poles = ([0, 2, 4], [0, 2, 4, 6], [0, 2, 4, 6, 8])

        if high_order_mode == "default":
            if rank == 0:
                logging.info(
                    f"Rank {rank}: [high-order mode] using exact multipoles only."
                )
            return None

        if correlation_mode != "auto":
            if rank == 0:
                logging.info(
                    f"Rank {rank}: [high-order mode] disabled because only auto-correlation is supported."
                )
            return None

        if poles not in supported_poles:
            if rank == 0:
                logging.info(
                    f"Rank {rank}: [high-order mode] disabled because poles {poles} are unsupported. "
                    f"Supported pole sets: {list(supported_poles)}."
                )
            return None

        if rank == 0:
            logging.info(
                f"Rank {rank}: [high-order mode] activated as '{high_order_mode}' for poles {poles}."
            )

        all_high_order_poles = {pole for pole in poles if pole in (4, 6, 8)}
        requested_fast_poles = set()
        requested_exact_poles = set()
        internal_exact_poles = set()

        if high_order_mode == "default":
            # Exact-only mode:
            # [0,2,4] -> P4
            # [0,2,4,6] -> P4, P6
            # [0,2,4,6,8] -> P4, P6, P8
            requested_exact_poles = all_high_order_poles
        elif high_order_mode == "fast":
            # Fast-output mode:
            # [0,2,4] -> P4b
            # [0,2,4,6] -> P4, P6b
            # [0,2,4,6,8] -> P4, P6b, P8b
            fast_map = {
                (0, 2, 4): {4},
                (0, 2, 4, 6): {6},
                (0, 2, 4, 6, 8): {6, 8},
            }
            exact_map = {
                (0, 2, 4): set(),
                (0, 2, 4, 6): {4},
                (0, 2, 4, 6, 8): {4},
            }
            requested_fast_poles = fast_map.get(tuple(poles), set())
            requested_exact_poles = exact_map.get(tuple(poles), set())
        elif high_order_mode == "compare":
            # Compare mode keeps the exact outputs and also adds all available
            # fast estimators for the requested high-order poles:
            # [0,2,4] -> P4 + P4b
            # [0,2,4,6] -> P4 + P6 + P4b + P6b
            # [0,2,4,6,8] -> P4 + P6 + P8 + P4b + P6b + P8b
            requested_fast_poles = all_high_order_poles
            requested_exact_poles = all_high_order_poles

        if high_order_mode in ("fast", "compare") and any(
            pole in requested_fast_poles for pole in (6, 8)
        ):
            # P6b and P8b depend on the exact P4 value, so P4 may still be
            # computed internally even when it is not part of the final output.
            internal_exact_poles.add(4)

        compute_fast_poles = requested_fast_poles
        compute_exact_poles = requested_exact_poles | internal_exact_poles
        need_f2_cache = any(pole in compute_fast_poles for pole in (4, 6, 8))
        need_f4_cache = any(pole in compute_fast_poles for pole in (6, 8))

        return {
            "mode": high_order_mode,
            "compute_fast_poles": compute_fast_poles,
            "compute_exact_poles": compute_exact_poles,
            "visible_exact_poles": requested_exact_poles,
            "need_f2_cache": need_f2_cache,
            "need_f4_cache": need_f4_cache,
        }

    high_order_strategy = get_high_order_strategy(poles, high_order_mode)

    # Calculate the Fourier transform of the density fields
    cfield_a = rfield_a.r2c()

    # Compensate the cfield depending on the type of mesh
    compensation = get_compensation(interlaced, sampler)
    cfield_a.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    cfield_a[
        :
    ] *= (
        boxsize.prod()
    )  # normalize the cfield, very interesting, if the normalization is not done here, the result will be wrong,
    if rank == 0:
        logging.info(
            f"Rank {rank}: {compensation[0][1].__name__} applied to the density field a"
        )

    if correlation_mode == "auto":
        cfield_b = cfield_a
        if rank == 0:
            logging.info(
                f"Rank {rank}: Auto-correlation mode, using the same density field for b"
            )
    else:
        cfield_b = rfield_b.r2c()
        cfield_b.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        cfield_b[:] *= boxsize.prod()
        if rank == 0:
            logging.info(
                f"Rank {rank}: {compensation[0][1].__name__} applied to the density field b"
            )

    # Get the kgrid, knorm, and x_grid for binning and spherical harmonics
    xgrid = None
    if poles == [0]:  # For ell = 0 only, we do not need to calculate the kgrid
        knorm = np.sqrt(sum(np.real(kk) ** 2 for kk in cfield_a.x))
        kgrid = None
    else:
        kgrid, knorm = get_kgrid(cfield_a)
        xgrid = get_xgrid(rfield_a, boxcenter, boxsize, nmesh)

    # No need to clear zero-mode in the Fourier space, see math_evaluator.get_Ylm for explanation
    results = {} if rank == 0 else None

    # get k_eff and k_num in one particular k_bin
    sub_count, sub_knorm_sum = get_kbin_count(k_bins, k_edge, knorm)
    # gather the results from all ranks
    k_num = comm.reduce(sub_count, op=MPI.SUM, root=0)
    total_knorm_sum = comm.reduce(sub_knorm_sum, op=MPI.SUM, root=0)
    if rank == 0:
        k_eff = total_knorm_sum / k_num
    results = {"k_eff": k_eff, "k_num": k_num, "I_norm": I_norm} if rank == 0 else None

    F2_cache, F4_cache = None, None

    # Loop over the poles
    for pole in poles:
        if rank == 0:
            logging.info(f"Rank {rank}: Processing pole {pole}")

        if pole == 0:
            if correlation_mode == "auto":
                P_ell_field = np.real(cfield_a[:]) ** 2 + np.imag(cfield_a[:]) ** 2
            else:
                P_ell_field = cfield_a[:] * np.conj(cfield_b[:])
        else:
            """
            IMPORTANT NOTE:
            Here we temporarily put the rsd effect into rfield_b
            """
            should_store_exact = True
            if pole in (4, 6, 8) and high_order_strategy is not None:
                should_store_exact = pole in high_order_strategy["compute_exact_poles"]

            should_cache_f2 = (
                pole == 2
                and high_order_strategy is not None
                and high_order_strategy["need_f2_cache"]
            )
            should_cache_f4 = (
                pole == 4
                and high_order_strategy is not None
                and high_order_strategy["need_f4_cache"]
            )

            if not should_store_exact and not should_cache_f2 and not should_cache_f4:
                if rank == 0:
                    logging.info(
                        f"Rank {rank}: [high-order mode] skipping exact P{pole}; only fast outputs are requested."
                    )
                continue

            G_ell_b = get_G_ell(
                rfield_b, pole, kgrid, xgrid, compensation, boxsize, comm
            )

            r"""
            F_\ell(\bs{k}) =\mathcal{G}_\ell(\bs{k}) + (-1)^\ell \mathcal{G}_\ell^*(-\bs{k})
            """
            if should_cache_f2:
                F2_cache = G_ell_b + np.conj(
                    space_inversion_transposed_complex(G_ell_b, return_type="ndarray")
                )
                F2_cache *= 1 / 2
                if rank == 0:
                    logging.info(f"Rank {rank}: [high-order mode] cached F2.")
            if should_cache_f4:
                F4_cache = G_ell_b + np.conj(
                    space_inversion_transposed_complex(G_ell_b, return_type="ndarray")
                )
                F4_cache *= 1 / 2
                if rank == 0:
                    logging.info(f"Rank {rank}: [high-order mode] cached F4.")

            if not should_store_exact:
                if rank == 0:
                    logging.info(
                        f"Rank {rank}: [high-order mode] exact P{pole} omitted from outputs."
                    )
                continue

            P_ell_field = cfield_a[:] * np.conj(G_ell_b[:])

        # Radial binning
        sub_sum = radial_binning(P_ell_field, k_bins, k_edge, knorm)

        # Gather the results from all ranks
        total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)

        if rank == 0:
            res = total_sum / k_num
            # Add some factors, substract the shot noise and take the complex conjugate for ell > 0
            res *= (2 * pole + 1) / I_norm

            if pole == 0:
                res -= N0
                logging.info(f"Rank {rank}: Shot noise subtracted from P0")
            else:
                if pole & 1:  # ell = odd
                    res = np.imag(res)
                else: # ell = even
                    res = np.real(res)

            # Store the results
            results.update(
                {
                    f"P{pole}": res,
                }
            )
            logging.info(f"Rank {rank}: P{pole} calculated")

    # Free memory
    del cfield_a, cfield_b, P_ell_field
    gc.collect()

    if high_order_strategy is not None and high_order_strategy["compute_fast_poles"]:
        if rank == 0:
            logging.info(f"Rank {rank}: [high-order mode] completing fast multipoles.")

        if 4 in high_order_strategy["compute_fast_poles"]:
            P_22_field = F2_cache[:] * np.conj(F2_cache[:])
            sub_sum_22 = radial_binning(P_22_field, k_bins, k_edge, knorm)
            total_sum_22 = comm.reduce(sub_sum_22, op=MPI.SUM, root=0)

            if rank == 0:
                P22 = total_sum_22 / k_num
                P22 *= 9 / I_norm
                P22 -= N0 * 9 / 5
                results["P4b"] = np.real(35 / 18 * P22 - results["P2"] - 7 / 2 * results["P0"])
                logging.info(f"Rank {rank}: [high-order mode] stored fast estimator P4b.")

            del P_22_field, sub_sum_22, total_sum_22

        if 6 in high_order_strategy["compute_fast_poles"] or 8 in high_order_strategy["compute_fast_poles"]:
            P_42_field = F4_cache[:] * np.conj(F2_cache[:])
            sub_sum_42 = radial_binning(P_42_field, k_bins, k_edge, knorm)
            total_sum_42 = comm.reduce(sub_sum_42, op=MPI.SUM, root=0)

            if rank == 0 and 6 in high_order_strategy["compute_fast_poles"]:
                P42 = total_sum_42 / k_num
                P42 *= 13 / I_norm
                results["P6b"] = np.real(
                    11 / 5 * P42 - 52 / 63 * results["P4"] - 286 / 175 * results["P2"]
                )
                logging.info(f"Rank {rank}: [high-order mode] stored fast estimator P6b.")

            if 8 in high_order_strategy["compute_fast_poles"]:
                P_44_field = F4_cache[:] * np.conj(F4_cache[:])
                sub_sum_44 = radial_binning(P_44_field, k_bins, k_edge, knorm)
                total_sum_44 = comm.reduce(sub_sum_44, op=MPI.SUM, root=0)

                if rank == 0:
                    P42 = total_sum_42 / k_num
                    P42 *= 13 / I_norm
                    P44 = total_sum_44 / k_num
                    P44 *= 17 / I_norm
                    P44 -= N0 * 17 / 9
                    results["P8b"] = np.real(
                        1287 / 490 * P44
                        - 374 / 245 * P42
                        - 3553 / 15435 * results["P4"]
                        - 1326 / 8575 * results["P2"]
                        - 2431 / 490 * results["P0"]
                    )
                    logging.info(f"Rank {rank}: [high-order mode] stored fast estimator P8b.")

                del P_44_field, sub_sum_44, total_sum_44

            del P_42_field, sub_sum_42, total_sum_42

        if rank == 0 and high_order_strategy["mode"] == "fast":
            for pole in high_order_strategy["compute_fast_poles"]:
                results.pop(f"P{pole}", None)

        if F2_cache is not None:
            del F2_cache
        if F4_cache is not None:
            del F4_cache
        gc.collect()

    if xgrid is not None:
        del xgrid
    if kgrid is not None:
        del kgrid
    del knorm
    gc.collect()

    if rank == 0:
        logging.info(f"Rank {rank}: Finished processing all poles")
        return results
    else:
        return None


def calculate_power_spectrum_box(
    rfield_a, rfield_b, correlation_mode, stat_attrs, comm, **kwargs
):
    rank = comm.Get_rank()

    # Extract mesh attributes
    poles = stat_attrs["poles"]
    boxsize, nmesh = np.array(stat_attrs["boxsize"]), np.array(stat_attrs["nmesh"])
    k_min, k_max, k_bins = (
        stat_attrs["k_min"],
        stat_attrs["k_max"],
        stat_attrs["k_bins"],
    )
    sampler, interlaced = stat_attrs["sampler"], stat_attrs["interlaced"]
    rsd = np.array(stat_attrs["rsd"])
    P_shot = 1.0 / stat_attrs["NZ_a"] if correlation_mode == "auto" else 0.0

    if correlation_mode == "cross":
        NZ_a = stat_attrs["NZ_a"]
        NZ_b = stat_attrs["NZ_b"]
    else:
        NZ_a = stat_attrs["NZ_a"]

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
    cfield_a = rfield_a.r2c()
    cfield_a.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])

    # logging.info(
    #     f"Rank {rank}: The shape of rfield_a is {rfield_a.shape}, the shape of cfield_a is {cfield_a.shape}."
    # )
    if rank == 0:
        logging.info(
            f"Rank {rank}: {compensation[0][1].__name__} applied to the density field"
        )

    if correlation_mode == "cross":
        cfield_b = rfield_b.r2c()
        cfield_b.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        if rank == 0:
            logging.info(
                f"Rank {rank}: {compensation[0][1].__name__} applied to the density field"
            )
    else:
        cfield_b = cfield_a

    # Get the kgrid, knorm for binning and Legendre polynomials
    if poles == [0]:  # For ell = 0 only, we do not need to calculate the kgrid
        knorm = np.sqrt(sum(np.real(kk) ** 2 for kk in cfield_a.x))
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

    P_field = np.real(cfield_a[:]) * np.real(cfield_b[:]) + np.imag(
        cfield_a[:]
    ) * np.imag(cfield_b[:])
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

    results = {"k_eff": k_eff, "k_num": k_num} if rank == 0 else None

    # Loop over the poles
    for pole in poles:
        if rank == 0:
            logging.info(f"Rank {rank}: Processing pole {pole}")

        if pole & 1:  # for ell = odd, the multipoles are 0 strictly
            res = np.zeros(k_bins).astype("float128")
            if rank == 0:
                logging.info(
                    f"Rank {rank}: ell = {pole}, Odd multipoles are automatically set to zero"
                )
        else:
            if pole == 0:
                sub_sum = radial_binning(P_field, k_bins, k_edge, knorm)
            else:
                L_ells = get_legendre(pole, rsd[0], rsd[1], rsd[2])
                if rank == 0:
                    logging.info(f"Rank {rank}: L_ells = {L_ells.expr}")
                sub_sum = radial_binning(
                    P_field * L_ells(kgrid[0], kgrid[1], kgrid[2]),
                    k_bins,
                    k_edge,
                    knorm,
                )

            # Gather the results from all ranks
            total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)

            if rank == 0:
                res = total_sum / k_num
                if correlation_mode == "auto":
                    res *= boxsize.prod() ** (-1) * (nmesh.prod() ** 2) / NZ_a**2
                else:
                    res *= boxsize.prod() ** (-1) * nmesh.prod() ** 2 / (NZ_a * NZ_b)
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
                    res *= 2 * pole + 1

        # Store the results
        if rank == 0:
            results.update(
                {
                    f"P{pole}": res,
                    "P_shot": P_shot,
                }
            )
            logging.info(f"Rank {rank}: P{pole} calculated")

    # Free memory
    del P_field
    gc.collect()
    if rank == 0:
        logging.info(f"Rank {rank}: Finished processing all poles")
        return results
    else:
        return None


def calculate_bk_sugi_box(
    rfield_a, rfield_b, rfield_c, correlation_mode, stat_attrs, comm, **kwargs
):
    rank = comm.Get_rank()
    # Extract mesh attributes
    data_vector_mode = stat_attrs.get("data_vector_mode", "diagonal")
    shotnoise_mode = stat_attrs.get("shotnoise_mode", "ana")
    # Internal switch for S3_ana only: use precomputed effective k values on all triangle sides.
    shotnoise_ana_all_eff = kwargs.get("shotnoise_ana_all_eff", True)
    block_size = stat_attrs.get(
        "block_size", 1 if data_vector_mode == "diagonal" else "full"
    )
    [ell_1, ell_2, L] = stat_attrs["angu_config"]
    boxsize = np.array(stat_attrs["boxsize"])
    nmesh = np.array(stat_attrs["nmesh"])
    k_min, k_max, k_bins = (
        stat_attrs["k_min"],
        stat_attrs["k_max"],
        stat_attrs["k_bins"],
    )
    sampler = stat_attrs["sampler"]
    interlaced = stat_attrs["interlaced"]
    tracer_type = stat_attrs["tracer_type"]  # "aaa", "aab", "abb" or "abc"
    vol_per_cell = boxsize.prod() / nmesh.prod()

    # constants related to angular momenta

    M = 0  # magnetic quantum number for L, only M=0 is considered here
    N_ells = (2 * ell_1 + 1) * (2 * ell_2 + 1) * (2 * L + 1)
    H_ells = np.float64(wigner_3j(ell_1, ell_2, L, 0, 0, 0))
    # find all sub-configurations that satisfy the triangular condition
    magnetic_configs, three_j_values = get_magnetic_configs_box(ell_1, ell_2, L)
    if rank == 0:
        logging.info(f"Rank {rank}: Magnetic configurations found: {magnetic_configs}")

    # Extract number density and galaxy count based on correlation mode
    if correlation_mode == "cross":
        NZ_a, N_gal_a = stat_attrs["NZ_a"], stat_attrs["N_gal_a"]
        NZ_b, N_gal_b = stat_attrs["NZ_b"], stat_attrs["N_gal_b"]
        if tracer_type == "abc":
            NZ_c, N_gal_c = stat_attrs["NZ_c"], stat_attrs["N_gal_c"]
    else:
        NZ_a, N_gal_a = stat_attrs["NZ_a"], stat_attrs["N_gal_a"]

    # Define k-bin edges
    k_edge = np.linspace(k_min, k_max, k_bins + 1)
    if rank == 0:
        logging.info(f"Rank {rank}: k_edge = {k_edge}")
        logging.info(f"Rank {rank}: S3 shot-noise mode = {shotnoise_mode}")

    # Get the compensation mode in k-space
    compensation = get_compensation_bk_sugi(sampler)
    if rank == 0:
        logging.info(
            "In Sugiyama estimator, the interlacing technique is not supported currently.\
                      Falling back to non-interlaced mode."
        )

    # Recover the density field to physical overdensity
    rfield_a[:] = rfield_a[:] / vol_per_cell - NZ_a
    if correlation_mode == "cross":
        rfield_b[:] = rfield_b[:] / vol_per_cell - NZ_b
        if tracer_type == "abc":
            rfield_c[:] = rfield_c[:] / vol_per_cell - NZ_c

    # Calculate the Fourier transform of the density field and apply compensation
    cfield_a = rfield_a.r2c()
    cfield_a.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    # logging.info(f"Rank {rank}: The shape of rfield_a is {rfield_a.shape}, the shape of cfield_a is {cfield_a.shape}.")
    if rank == 0:
        logging.info(
            f"Rank {rank}: {compensation[0][1].__name__} applied to the density field a"
        )

    if correlation_mode == "cross":
        cfield_b = rfield_b.r2c()
        cfield_b.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        if rank == 0:
            logging.info(
                f"Rank {rank}: {compensation[0][1].__name__} applied to the density field b"
            )

    if correlation_mode == "cross" and tracer_type == "abc":
        cfield_c = rfield_c.r2c()
        cfield_c.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
        if rank == 0:
            logging.info(
                f"Rank {rank}: {compensation[0][1].__name__} applied to the density field c"
            )

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
    G_00 = np.real(G_00)  # G_00 is strictly to be real
    if rank == 0:
        logging.info(f"Rank {rank}: G_00 generated.")

    # create the results container
    results = {} if rank == 0 else None

    sub_count, sub_knorm_sum = get_kbin_count(k_bins, k_edge, knorm)
    k_num = comm.reduce(sub_count, op=MPI.SUM, root=0)
    total_knorm_sum = comm.reduce(sub_knorm_sum, op=MPI.SUM, root=0)
    if rank == 0:
        k_eff = total_knorm_sum / k_num
    else:
        k_eff = None
    k_eff_local = comm.bcast(k_eff, root=0)

    if data_vector_mode == "full":
        if block_size == "full":
            block_size = k_bins
        elif isinstance(block_size, int) and 1 <= block_size <= k_bins:
            pass
        else:
            raise ValueError(
                "For full mode, block_size must be 1, 'full', or an integer in [1, k_bins]."
            )
        if rank == 0:
            logging.info(
                f"Rank {rank}: Using block_size = {block_size} for full data vector mode."
            )
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

            # Symmetry can be used only when k1 and k2 are share tracer source and same ell.
            same_tracer = correlation_mode == "auto" or tracer_type == "aab"
            same_ell = ell_1 == ell_2
            can_sym_equal = same_tracer and same_ell and (m1 == m2)
            can_sym_conj = same_tracer and same_ell and (m1 == -m2)
            use_symmetry = can_sym_equal or can_sym_conj

            for bi in range(0, k_bins, block_size):
                i_end = min(k_bins, bi + block_size)
                if rank == 0:
                    logging.info("-" * 50)
                    logging.info(
                        f"Rank {rank}: Processing block rows {bi+1} to {i_end}..."
                    )

                # Reuse cache_1 across all bj blocks within this bi block.
                cache_1 = {}

                bj_start = bi if use_symmetry else 0
                for bj in range(bj_start, k_bins, block_size):
                    j_end = min(k_bins, bj + block_size)
                    block_on_diag = bi == bj
                    # cache_2 is scoped to one column block so its footprint is bounded by block_size.
                    cache_2 = {}
                    if rank == 0:
                        logging.info(
                            f"Rank {rank}: Processing block columns {bj+1} to {j_end}..."
                        )

                    for i in range(bi, i_end):
                        if i not in cache_1:
                            mask_i = np.logical_and(
                                knorm >= k_edge[i], knorm < k_edge[i + 1]
                            )
                            cache_1[i] = (ylm_weighted_cfield_1 * mask_i).c2r()
                        binned_field_1 = cache_1[i]

                        j_local_start = bj
                        if use_symmetry and block_on_diag:
                            j_local_start = max(j_local_start, i)

                        for j in range(j_local_start, j_end):
                            if can_sym_equal and block_on_diag:
                                if j not in cache_1:
                                    mask_j = np.logical_and(
                                        knorm >= k_edge[j], knorm < k_edge[j + 1]
                                    )
                                    cache_1[j] = (ylm_weighted_cfield_1 * mask_j).c2r()
                                binned_field_2 = cache_1[j]
                            elif can_sym_conj and block_on_diag:
                                if j not in cache_1:
                                    mask_j = np.logical_and(
                                        knorm >= k_edge[j], knorm < k_edge[j + 1]
                                    )
                                    cache_1[j] = (ylm_weighted_cfield_1 * mask_j).c2r()
                                # The phase factor only applies in conjugate symmetry on diagonal blocks.
                                binned_field_2 = (-1) ** (m1 + ell_1) * np.conj(
                                    cache_1[j]
                                )
                            else:
                                # Off-diagonal blocks must build cache_2 independently.
                                if j not in cache_2:
                                    mask_j = np.logical_and(
                                        knorm >= k_edge[j], knorm < k_edge[j + 1]
                                    )
                                    cache_2[j] = (ylm_weighted_cfield_2 * mask_j).c2r()
                                binned_field_2 = cache_2[j]

                            sub_sig_sum = np.sum(G_00 * binned_field_1 * binned_field_2)
                            total_sig_sum = comm.reduce(sub_sig_sum, op=MPI.SUM, root=0)
                            if rank == 0:
                                sub_res[i, j] = three_j_values[ss] * total_sig_sum

                    del cache_2
                    gc.collect()

                del cache_1
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
                logging.info(
                    f"Rank {rank}: Processing spherical harmonics Y_{ell_1}^{m1} as {ylm_1.expr} and Y_{ell_2}^{m2} as {ylm_2.expr}"
                )
            ylm_weighted_cfield_1 = cfield_a * ylm_1(kgrid[0], kgrid[1], kgrid[2])
            if tracer_type in ["aaa", "aab"]:
                if ell_1 != ell_2 or m1 != m2:
                    ylm_weighted_cfield_2 = cfield_a * ylm_2(
                        kgrid[0], kgrid[1], kgrid[2]
                    )
            else:
                ylm_weighted_cfield_2 = cfield_b * ylm_2(kgrid[0], kgrid[1], kgrid[2])

            # binning in k-space then ifft it to real space
            if rank == 0:
                logging.info(
                    f"Rank {rank}: Closing triangles in k-space by binning and iffting the weighted cfield..."
                )
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
                                logging.info(
                                    f"Rank {rank}: Using the same weighted cfield for b as for a."
                                )
                        else:
                            binned_field_2 = (-1) ** (m1 + ell_1) * np.conj(
                                binned_field_1
                            )  # the additional phase factor can be found in our methodology paper
                            if rank == 0 and i == 0:
                                logging.info(
                                    f"Rank {rank}: Using the conjugate of weighted cfield a for b."
                                )
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

    # k3 bin statistics used by S3 shot-noise analytical integration.
    k_center = 0.5 * (k_edge[1:] + k_edge[:-1])
    k3_min, k3_max = 2 * k_min, 2 * k_max
    k3_edge = np.linspace(k3_min, k3_max, k_bins * 2 + 1)[:-1]
    k3_center = 0.5 * (k3_edge[1:] + k3_edge[:-1])
    k3_bins = len(k3_center)

    sub_count_k3, sub_knorm_sum_k3 = get_kbin_count(k3_bins, k3_edge, knorm)
    k_num_k3 = comm.reduce(sub_count_k3, op=MPI.SUM, root=0)
    total_knorm_sum_k3 = comm.reduce(sub_knorm_sum_k3, op=MPI.SUM, root=0)
    if rank == 0:
        k3_eff = total_knorm_sum_k3 / k_num_k3

    """
    Now we obtain the shot-noise, we define 4 terms in EQ45 of Sugiyama 2019 as SN0, SN1, SN2, SN3
    """
    time_shotnoise_start = time.time()

    def calculate_shot_noise_s3_box_ana(total_sum):
        if rank == 0:
            if shotnoise_ana_all_eff:
                logging.info(
                    f"Rank {rank}: Calculating S3 shot noise contribution using analytical integration with effective k values on all triangle sides..."
                )
        if tracer_type == "abb":
            return 0.0
        # The analytical S3 contraction is independent for each (i, j). We only
        # exploit this in full mode, where the k_bins^2 workload is large enough
        # to benefit from distributing bin pairs over MPI ranks.
        total_sum = comm.bcast(total_sum if rank == 0 else None, root=0)
        weighted_k_num = comm.bcast(k_num_k3 / k3_center if rank == 0 else None, root=0)
        k3_eff_local = comm.bcast(k3_eff if rank == 0 else None, root=0)
        size = comm.Get_size()
        if data_vector_mode == "diagonal":
            # Diagonal mode has only O(k_bins) work, so we keep the original
            # root-only path.
            s3_local = np.zeros(k_bins, dtype="complex128")
            if rank == 0:
                for i in range(k_bins):
                    if shotnoise_ana_all_eff:
                        q_ells = get_q_ells_eff(
                            i,
                            i,
                            k_eff_local,
                            ell_1,
                            ell_2,
                            L,
                            k3_bins,
                            k3_eff_local,
                        )
                    else:
                        q_ells = get_q_ells(
                            i,
                            i,
                            k_center,
                            k_min,
                            k_max,
                            k_bins,
                            ell_1,
                            ell_2,
                            L,
                            k3_bins,
                        )
                    numer = np.sum(q_ells * total_sum / k3_center)
                    denom = np.sum(weighted_k_num[: 2 * i + 1])
                    if denom > 0:
                        s3_local[i] = numer / denom
                s3_local *= H_ells * N_ells
        elif data_vector_mode == "full":
            s3_local = np.zeros((k_bins, k_bins), dtype="complex128")
            # Flatten the (i, j) plane so the workload is shared uniformly
            # over ranks without extra bookkeeping or communication.
            for flat_idx in range(rank, k_bins * k_bins, size):
                i, j = divmod(flat_idx, k_bins)
                if shotnoise_ana_all_eff:
                    q_ells = get_q_ells_eff(
                        i,
                        j,
                        k_eff_local,
                        ell_1,
                        ell_2,
                        L,
                        k3_bins,
                        k3_eff_local,
                    )
                else:
                    q_ells = get_q_ells(
                        i,
                        j,
                        k_center,
                        k_min,
                        k_max,
                        k_bins,
                        ell_1,
                        ell_2,
                        L,
                        k3_bins,
                    )
                valid_k3_bins = get_valid_k3_bins(
                    k_center[i],
                    k_center[j],
                    k_min,
                    k_max,
                    k_bins,
                )
                numer = np.sum(q_ells * total_sum / k3_center)
                denom = np.sum(weighted_k_num * valid_k3_bins)
                if denom > 0:
                    s3_local[i, j] = numer / denom
            s3_local = comm.reduce(s3_local, op=MPI.SUM, root=0)
            if rank == 0:
                s3_local *= H_ells * N_ells
        else:
            raise ValueError(
                "data_vector_mode must be either 'diagonal' or 'full'."
            )
        # Downstream code expects every rank to see the same shot-noise tensor.
        return comm.bcast(s3_local if rank == 0 else None, root=0)

    def calculate_shot_noise_s3_box_fft():
        if tracer_type == "abb":
            if data_vector_mode == "full":
                return np.zeros((k_bins, k_bins), dtype="complex128")
            return np.zeros(k_bins, dtype="complex128")
        def build_source_field_fft():
            # In box geometry M is fixed to 0, so the k3 source field is
            # shared by all magnetic sub-configurations.
            source_cfield = cfield_a.copy()
            source_cfield[:] = cfield_a[:] * np.conj(
                cfield_a[:] if correlation_mode == "auto" else cfield_b[:]
            )
            source_cfield[:] *= boxsize.prod() ** 2

            if correlation_mode == "auto":
                shot_corr_field = cfield_a.copy()
                shot_corr_field[:] = SN0
                shot_corr_field.apply(
                    out=Ellipsis,
                    func=get_compensation_shot_sugi(sampler)[0][1],
                    kind="circular",
                )
                source_cfield[:] -= shot_corr_field[:]
                del shot_corr_field

            source_field = source_cfield.c2r()
            del source_cfield
            gc.collect()
            return source_field

        work_cfield_1 = cfield_a.copy()
        work_cfield_2 = cfield_a.copy()

        def build_shell_filtered_field(ell, m, bin_idx, work_cfield):
            ylm_k = get_Ylm(ell, m, Racah_normalized=True)
            mask = np.logical_and(knorm >= k_edge[bin_idx], knorm < k_edge[bin_idx + 1])
            work_cfield[:] = np.conj(ylm_k(kgrid[0], kgrid[1], kgrid[2])) * mask
            return work_cfield.c2r()

        same_ell_12 = ell_1 == ell_2
        source_field_fft = build_source_field_fft()
        if data_vector_mode == "diagonal":
            s3_local = np.zeros(k_bins, dtype="complex128")

            for ss, (m1, m2, M) in enumerate(magnetic_configs):
                if rank == 0:
                    logging.info(
                        f"Rank {rank}: {'='*20}Processing magnetic configuration m1={m1}, m2={m2}, M={M} for FFT-calculated S3 shot noise...{'='*20}"
                    )
                sub_res = np.zeros(k_bins, dtype="complex128")

                can_sym_equal = same_ell_12 and (m1 == m2)
                can_sym_conj = same_ell_12 and (m1 == -m2)

                for i in range(k_bins):
                    shell_field_1 = build_shell_filtered_field(
                        ell_1, m1, i, work_cfield_1
                    )
                    if can_sym_equal:
                        shell_field_2 = shell_field_1
                        if rank == 0 and i == 0:
                            logging.info(
                                f"Rank {rank}: Using the same shell-filtered Sph ifft field for k2 as for k1."
                            )
                    elif can_sym_conj:
                        shell_field_2 = ((-1) ** (m1 + ell_1)) * np.conj(shell_field_1)
                        if rank == 0 and i == 0:
                            logging.info(
                                f"Rank {rank}: Using the conjugate of shell-filtered Sph ifft field for k2 as for k1."
                            )
                    else:
                        shell_field_2 = build_shell_filtered_field(
                            ell_2, m2, i, work_cfield_2
                        )

                    sub_sig_sum = np.sum(source_field_fft * shell_field_1 * shell_field_2)
                    total_sig_sum = comm.reduce(sub_sig_sum, op=MPI.SUM, root=0)
                    if rank == 0 and k_num[i] > 0:
                        sub_res[i] = three_j_values[ss] * total_sig_sum / (k_num[i] ** 2)
                        logging.info(
                            f"Rank {rank}: Bin {i+1}/{k_bins} processed."
                        )

                    if not can_sym_equal:
                        del shell_field_2
                    del shell_field_1
                    gc.collect()

                if rank == 0:
                    if (m1, m2, M) == (0, 0, 0):
                        s3_local += sub_res
                    else:
                        s3_local += 2.0 * np.real(sub_res)
        elif data_vector_mode == "full":
            s3_local = np.zeros((k_bins, k_bins), dtype="complex128")

            for ss, (m1, m2, M) in enumerate(magnetic_configs):
                if rank == 0:
                    logging.info(
                        f"Rank {rank}: {'='*20}Processing magnetic configuration m1={m1}, m2={m2}, M={M} for FFT-calculated S3 shot noise...{'='*20}"
                    )
                sub_res = np.zeros((k_bins, k_bins), dtype="complex128")

                can_sym_equal = same_ell_12 and (m1 == m2)
                can_sym_conj = same_ell_12 and (m1 == -m2)
                use_symmetry = can_sym_equal or can_sym_conj

                for bi in range(0, k_bins, block_size):
                    i_end = min(k_bins, bi + block_size)
                    if rank == 0:
                        logging.info("-" * 50)
                        logging.info(
                            f"Rank {rank}: Processing FFT S3 block rows {bi+1} to {i_end}..."
                        )

                    cache_1 = {}
                    bj_start = bi if use_symmetry else 0
                    for bj in range(bj_start, k_bins, block_size):
                        j_end = min(k_bins, bj + block_size)
                        block_on_diag = bi == bj
                        cache_2 = {}
                        if rank == 0:
                            logging.info(
                                f"Rank {rank}: Processing FFT S3 block columns {bj+1} to {j_end}..."
                            )

                        for i in range(bi, i_end):
                            if i not in cache_1:
                                cache_1[i] = build_shell_filtered_field(
                                    ell_1, m1, i, work_cfield_1
                                )
                            shell_field_1 = cache_1[i]

                            j_local_start = bj
                            if use_symmetry and block_on_diag:
                                j_local_start = max(j_local_start, i)

                            for j in range(j_local_start, j_end):
                                if can_sym_equal and block_on_diag:
                                    if j not in cache_1:
                                        cache_1[j] = build_shell_filtered_field(
                                            ell_1, m1, j, work_cfield_1
                                        )
                                    shell_field_2 = cache_1[j]
                                elif can_sym_conj and block_on_diag:
                                    if j not in cache_1:
                                        cache_1[j] = build_shell_filtered_field(
                                            ell_1, m1, j, work_cfield_1
                                        )
                                    shell_field_2 = ((-1) ** (m1 + ell_1)) * np.conj(
                                        cache_1[j]
                                    )
                                else:
                                    if j not in cache_2:
                                        cache_2[j] = build_shell_filtered_field(
                                            ell_2, m2, j, work_cfield_2
                                        )
                                    shell_field_2 = cache_2[j]

                                sub_sig_sum = np.sum(
                                    source_field_fft * shell_field_1 * shell_field_2
                                )
                                total_sig_sum = comm.reduce(sub_sig_sum, op=MPI.SUM, root=0)
                                if rank == 0 and k_num[i] > 0 and k_num[j] > 0:
                                    sub_res[i, j] = (
                                        three_j_values[ss]
                                        * total_sig_sum
                                        / (k_num[i] * k_num[j])
                                    )

                        del cache_2
                        gc.collect()

                    del cache_1
                    gc.collect()

                if rank == 0 and use_symmetry:
                    for i in range(k_bins):
                        for j in range(i + 1, k_bins):
                            if can_sym_equal:
                                sub_res[j, i] = sub_res[i, j]
                            elif can_sym_conj:
                                sub_res[j, i] = np.conj(sub_res[i, j])

                if rank == 0:
                    if (m1, m2, M) == (0, 0, 0):
                        s3_local += sub_res
                    else:
                        s3_local += 2.0 * np.real(sub_res)
        else:
            raise ValueError(
                "data_vector_mode must be either 'diagonal' or 'full'."
            )

        if rank == 0:
            # Match the ana normalization using the FFT-based real-space contraction.
            s3_local *= H_ells * N_ells
            s3_local *= 1 / nmesh.prod()

        del source_field_fft
        del work_cfield_1, work_cfield_2
        gc.collect()
        return comm.bcast(s3_local if rank == 0 else None, root=0)

    SN3_fft = None
    if tracer_type == "abc":
        SN0, SN1, SN2, SN3 = 0.0, 0.0, 0.0, 0.0
        if rank == 0:
            logging.info(
                "Tracer type 'abc' detected, shot-noise terms are all set to zero."
            )
    else:
        # SN0
        if correlation_mode == "auto" and [ell_1, ell_2, L] == [0, 0, 0]:
            SN0 = N_gal_a  # normalization factor I is N_gal_a^3/boxsize^2, we will divide it later
        else:
            SN0 = 0.0
        if rank == 0:
            logging.info(f"Rank {rank}: SN0 calculated.")

        """
        The logic of this part can be found in our methodology paper. The Q_0 term is necessary.
        """
        P_field = cfield_a[:] * np.conj(
            cfield_a[:] if correlation_mode == "auto" else cfield_b[:]
        )
        P_field *= boxsize.prod() ** 2

        if correlation_mode == "auto":
            S_field = cfield_a.copy()
            S_field[:] = SN0
            S_field.apply(
                out=Ellipsis,
                func=get_compensation_shot_sugi(sampler)[0][1],
                kind="circular",
            )
            if rank == 0:
                logging.info(
                    f"Rank {rank}: Shot-noise compensation applied for field a."
                )
            P_field[:] -= S_field[:]

        if L != 0:  # multiply by y_L0
            y_L0 = get_Ylm(L, 0, Racah_normalized=True)
            P_field *= y_L0(kgrid[0], kgrid[1], kgrid[2])
            if rank == 0:
                logging.info(
                    f"Rank {rank}: Multiplying by spherical harmonic Y_{L}^{0} as {y_L0.expr} for SN1 and SN2 calculation."
                )

        # Perform radial binning
        sub_sum = radial_binning(P_field, k3_bins, k3_edge, knorm)
        total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)

        if rank == 0:
            SN1, SN2 = 0.0, 0.0
            if ell_1 == L and ell_2 == 0:
                if correlation_mode == "auto":
                    SN1 = (total_sum / k_num_k3)[: len(k_center)]
                    SN1 *= 2 * L + 1
                    if ell_1 == 0:
                        SN2 = SN1.copy()
                elif correlation_mode == "cross" and tracer_type == "abb":
                    SN1 = (total_sum / k_num_k3)[: len(k_center)]
                    SN1 *= 2 * L + 1

            # Convert the 1d vector to (k1,k2) plane, note that SN1 is only related to k1, SN2 is only related to k2,
            if data_vector_mode == "full":
                SN1_full = np.zeros((k_bins, k_bins)).astype("c16")
                SN2_full = np.zeros((k_bins, k_bins)).astype("c16")
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
            logging.info(
                f"Rank {rank}: SN1 and SN2 calculated." 
            )


        # SN3
        if shotnoise_mode == "ana":
            if rank == 0:
                logging.info(f"Rank {rank}: S3 shot noise calculated using analytical integration.")
            SN3 = calculate_shot_noise_s3_box_ana(total_sum)
        elif shotnoise_mode == "fft":
            if rank == 0:
                logging.info(f"Rank {rank}: Calculating S3 shot noise using FFT-based method...")
            SN3 = calculate_shot_noise_s3_box_fft()
        elif shotnoise_mode == "both":
            if rank == 0:
                logging.info(
                    f"Rank {rank}: Calculating S3 shot noise using both analytical and FFT-based methods..."
                )
            SN3 = calculate_shot_noise_s3_box_ana(total_sum)
            SN3_fft = calculate_shot_noise_s3_box_fft()
        else:
            raise ValueError("shotnoise_mode must be either 'ana', 'fft', or 'both'.")

    time_shotnoise_end = time.time()
    if rank == 0:
        logging.info(
            f"Rank{rank}: Time to compute shot noise terms: {time_shotnoise_end - time_shotnoise_start:.2f} seconds"
        )

    # Normalize the bispectrum
    if rank == 0:
        if correlation_mode == "auto":
            I_norm = (N_gal_a**3) / (boxsize.prod() ** 2)
        else:
            if tracer_type == "aab":
                I_norm = (N_gal_a**2 * N_gal_b) / (boxsize.prod() ** 2)
            elif tracer_type == "abb":
                I_norm = (N_gal_a * N_gal_b**2) / (boxsize.prod() ** 2)
            elif tracer_type == "abc":
                I_norm = (N_gal_a * N_gal_b * N_gal_c) / (boxsize.prod() ** 2)

        # Combine shot-noise terms.
        # For full mode we keep shot-noise subtraction disabled for now to validate signal first.
        # We will re-enable a mathematically consistent full-mode shot-noise model later.
        total_shot_noise = SN0 + SN1 + SN2 + SN3

        # Normalize the bispectrum result.
        total_res *= N_ells * H_ells / I_norm
        if data_vector_mode == "full":
            total_res *= (boxsize.prod()) ** 2 / np.outer(
                k_num[:k_bins], k_num[:k_bins]
            )
        else:
            total_res *= (boxsize.prod()) ** 2 / (k_num[: len(total_res)]) ** 2

        total_res *= vol_per_cell
        total_shot_noise *= 1 / I_norm
        # Final bispectrum result after subtracting shot-noise
        final_bk = total_res - total_shot_noise

        logging.info(f"Rank {rank}: Final bispectrum calculated.")

        # Store the results
        sn_terms = {
            "SN0": SN0 / I_norm,
            "SN1": SN1 / I_norm,
            "SN2": SN2 / I_norm,
            "SN3": SN3 / I_norm,
        }
        if shotnoise_mode == "both":
            sn_terms["SN3_FFT"] = SN3_fft / I_norm

        results.update(
            {
                "B_sugi": final_bk,
                "SN_terms": sn_terms,
                "I_norm": I_norm,
                "Shot_noise": total_shot_noise,
                "Bk_raw": total_res,
                "k_eff": k_eff,
                "nmodes": k_num,
            }
        )

    # broadcast results to all ranks
    results = comm.bcast(results, root=0)
    if rank == 0:
        logging.info(f"Rank {rank}: Finished processing bispectrum calculation.")

    return results


def calculate_bk_sugi_survey(
    rfield_a, rfield_b, rfield_c, correlation_mode, stat_attrs, comm, **kwargs
):
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
    shotnoise_mode = stat_attrs.get("shotnoise_mode", "ana")
    # Internal switch for S3_ana only: use precomputed effective k values on all triangle sides.
    shotnoise_ana_all_eff = kwargs.get("shotnoise_ana_all_eff", True)
    block_size = stat_attrs.get(
        "block_size", 1 if data_vector_mode == "diagonal" else "full"
    )
    [ell_1, ell_2, L] = stat_attrs["angu_config"]
    boxsize = np.array(stat_attrs["boxsize"])
    nmesh = np.array(stat_attrs["nmesh"])
    boxcenter = np.array(stat_attrs["boxcenter"])
    k_min, k_max, k_bins = (
        stat_attrs["k_min"],
        stat_attrs["k_max"],
        stat_attrs["k_bins"],
    )
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
            f"shotnoise_mode={shotnoise_mode} "
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
    else:
        k_eff = None
    k_eff_local = comm.bcast(k_eff, root=0)

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
    same_tracer_12 = correlation_mode == "auto" or tracer_type == "aab"
    same_ell_12 = ell_1 == ell_2
    total_cfg_count = len(magnetic_configs)
    skipped_cfg_count = 0
    evaluated_cfg_count = 0

    if data_vector_mode == "full":
        if block_size == "full":
            block_size = k_bins
        elif isinstance(block_size, int) and 1 <= block_size <= k_bins:
            pass
        else:
            raise ValueError(
                "For full mode, block_size must be 1, 'full', or an integer in [1, k_bins]."
            )
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
                cfield_GLM.apply(
                    out=Ellipsis, func=compensation[0][1], kind=compensation[0][2]
                )
                # cfield_GLM[:] *= boxsize.prod()
                G_LM_current = cfield_GLM.c2r()
                current_M = M
                _log(
                    1, "GLM", f"sub_cfg={ss+1}/{total_cfg_count} M={M} action=recompute"
                )
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
                    _log(
                        2,
                        "CFG",
                        "field2_source=reuse_field1 (same tracer, same ell, m1==m2)",
                    )
                else:
                    ylm_weighted_cfield_2 = cfield_a * ylm_2(
                        kgrid[0], kgrid[1], kgrid[2]
                    )
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
                bj_start = bi if use_symmetry else 0
                _log(
                    2, "BLOCK", f"sub_cfg={ss+1}/{total_cfg_count} row=[{bi},{i_end-1}]"
                )

                for bj in range(bj_start, k_bins, block_size):
                    j_end = min(k_bins, bj + block_size)
                    block_on_diag = bi == bj
                    cache_2 = {}
                    _log(
                        2,
                        "BLOCK",
                        f"sub_cfg={ss+1}/{total_cfg_count} col=[{bj},{j_end-1}] diag={block_on_diag}",
                    )
                    for i in range(bi, i_end):
                        if i not in cache_1:
                            mask_i = np.logical_and(
                                knorm >= k_edge[i], knorm < k_edge[i + 1]
                            )
                            cache_1[i] = (ylm_weighted_cfield_1 * mask_i).c2r()
                        binned_field_1 = cache_1[i]

                        j_local_start = bj
                        if use_symmetry and block_on_diag:
                            j_local_start = max(j_local_start, i)

                        for j in range(j_local_start, j_end):
                            if can_sym_equal and block_on_diag:
                                if j not in cache_1:
                                    mask_j = np.logical_and(
                                        knorm >= k_edge[j], knorm < k_edge[j + 1]
                                    )
                                    cache_1[j] = (ylm_weighted_cfield_1 * mask_j).c2r()
                                binned_field_2 = cache_1[j]
                            elif can_sym_conj and block_on_diag:
                                if j not in cache_1:
                                    mask_j = np.logical_and(
                                        knorm >= k_edge[j], knorm < k_edge[j + 1]
                                    )
                                    cache_1[j] = (ylm_weighted_cfield_1 * mask_j).c2r()
                                binned_field_2 = (-1) ** (m1 + ell_1) * np.conj(
                                    cache_1[j]
                                )
                            else:
                                if j not in cache_2:
                                    mask_j = np.logical_and(
                                        knorm >= k_edge[j], knorm < k_edge[j + 1]
                                    )
                                    cache_2[j] = (ylm_weighted_cfield_2 * mask_j).c2r()
                                binned_field_2 = cache_2[j]

                            sub_sig_sum = np.sum(
                                G_LM_current * binned_field_1 * binned_field_2
                            )
                            total_sig_sum = comm.reduce(sub_sig_sum, op=MPI.SUM, root=0)
                            if rank == 0:
                                sub_res[i, j] = three_j_values[ss] * total_sig_sum

                    del cache_2
                    gc.collect()

                del cache_1
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
            has_swap = (
                same_tracer_12
                and same_ell_12
                and (swap_cfg in cfg_to_idx)
                and (swap_cfg != (m1, m2, M))
            )
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
                    _log(
                        2,
                        "ACC",
                        f"sub_cfg={ss+1}/{total_cfg_count} action=accumulate rule=self_conjugate_once",
                    )
                else:
                    total_res += 2 * np.real(sub_res_to_add)
                    _log(
                        2,
                        "ACC",
                        f"sub_cfg={ss+1}/{total_cfg_count} action=accumulate rule=2*Re",
                    )
                _log(
                    1,
                    "CFG",
                    f"sub_cfg={ss+1}/{total_cfg_count} action=done elapsed={time.time()-cfg_t0:.2f}s",
                )

            # IMPORTANT (MPI consistency): update processed_cfg identically on all ranks.
            if add_swap_now:
                processed_cfg.add(swap_cfg)
            processed_cfg.add((m1, m2, M))

            del ylm_weighted_cfield_1, ylm_weighted_cfield_2
            gc.collect()

    elif data_vector_mode == "diagonal":
        total_res = np.zeros(k_bins).astype("complex128")
        diagonal_reuse_fixed_m2_leg = ell_2 == 0 and all(
            cfg[1] == 0 for cfg in magnetic_configs
        )
        diagonal_shared_weighted_cfield_2 = None

        if diagonal_reuse_fixed_m2_leg:
            ylm_2_shared = get_Ylm(ell_2, 0, Racah_normalized=True)
            if tracer_type in ["aaa", "aab"]:
                diagonal_shared_weighted_cfield_2 = cfield_a * ylm_2_shared(
                    kgrid[0], kgrid[1], kgrid[2]
                )
                _log(
                    1,
                    "DIAG-CACHE",
                    "reuse_enabled: fixed second leg uses tracer_a with ell_2=0, m2=0 across sub-configs.",
                )
            else:
                diagonal_shared_weighted_cfield_2 = cfield_b * ylm_2_shared(
                    kgrid[0], kgrid[1], kgrid[2]
                )
                _log(
                    1,
                    "DIAG-CACHE",
                    "reuse_enabled: fixed second leg uses tracer_b with ell_2=0, m2=0 across sub-configs.",
                )
        else:
            _log(
                1,
                "DIAG-CACHE",
                "reuse_disabled: second leg is not globally fixed to ell_2=0, m2=0.",
            )

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
                cfield_GLM.apply(
                    out=Ellipsis, func=compensation[0][1], kind=compensation[0][2]
                )
                # cfield_GLM[:] *= boxsize.prod()
                G_LM_current = cfield_GLM.c2r()
                current_M = M
                _log(
                    1, "GLM", f"sub_cfg={ss+1}/{total_cfg_count} M={M} action=recompute"
                )
                del rfield_weighted_third, cfield_GLM
                gc.collect()

            ylm_1 = get_Ylm(ell_1, m1, Racah_normalized=True)
            ylm_weighted_cfield_1 = cfield_a * ylm_1(kgrid[0], kgrid[1], kgrid[2])
            can_sym_equal = same_tracer_12 and same_ell_12 and (m1 == m2)
            can_sym_conj = same_tracer_12 and same_ell_12 and (m1 == -m2)
            ylm_weighted_cfield_2 = None
            if not (can_sym_equal or can_sym_conj):
                if diagonal_reuse_fixed_m2_leg:
                    ylm_weighted_cfield_2 = diagonal_shared_weighted_cfield_2
                else:
                    ylm_2 = get_Ylm(ell_2, m2, Racah_normalized=True)
                    if tracer_type in ["aaa", "aab"]:
                        ylm_weighted_cfield_2 = cfield_a * ylm_2(
                            kgrid[0], kgrid[1], kgrid[2]
                        )
                    else:
                        ylm_weighted_cfield_2 = cfield_b * ylm_2(
                            kgrid[0], kgrid[1], kgrid[2]
                        )
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
            has_swap = (
                same_tracer_12
                and same_ell_12
                and (swap_cfg in cfg_to_idx)
                and (swap_cfg != (m1, m2, M))
            )
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
                    _log(
                        2,
                        "ACC",
                        f"sub_cfg={ss+1}/{total_cfg_count} action=accumulate rule=self_conjugate_once",
                    )
                else:
                    total_res += 2 * np.real(sub_res_to_add)
                    _log(
                        2,
                        "ACC",
                        f"sub_cfg={ss+1}/{total_cfg_count} action=accumulate rule=2*Re",
                    )

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
    Shot-noise preparation for survey geometry.

    The survey-like shot-noise terms are built from a small set of auxiliary
    fields:
    - one "delta-like" field, denoted here by rfield_shot_*
    - one squared-weight field N_field_*
    - optionally one S_LM array for the same-tracer correction

    The tracer combination determines which of these objects are mathematically
    relevant:
    - abc: no repeated tracer, so all shot-noise terms must vanish
    - aab: only the repeated tracer "a" contributes to the contraction, but the
      delta-like leg comes from tracer_b
    - abb: symmetric to aab, with repeated tracer "b"
    - aaa/auto: all same-tracer shot-noise pieces are potentially active

    The code therefore constructs only the fields that can contribute for the
    current tracer_type, rather than building every possible auxiliary field.
    """
    from .mesh_generator import get_N_field

    catalogs = kwargs.get("catalogs", None)
    if catalogs is None:
        raise ValueError(
            "catalogs must be provided to calculate_bk_sugi_survey for N_field construction."
        )

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
    # 1) tracer_type == "abc": no repeated tracer => shot noise is exactly zero
    # 2) tracer_type == "aab": the repeated tracer is "a", so use N_field_a, but
    #    the delta-like leg entering the contraction is rfield_b
    # 3) tracer_type == "abb": symmetric case, use rfield_a together with N_field_b
    # 4) aaa/auto and any fallback case: use tracer_a for both ingredients
    if tracer_type == "abc":
        _log(
            1,
            "SHOT",
            "tracer_type=abc -> skip rfield/N_field construction (shotnoise=0).",
        )
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

    _log(
        1,
        "SUMMARY",
        f"evaluated={evaluated_cfg_count} skipped={skipped_cfg_count} total={total_cfg_count}",
    )

    k_center = 0.5 * (k_edge[1:] + k_edge[:-1])
    k3_min, k3_max = 2 * k_min, 2 * k_max
    k3_edge = np.linspace(k3_min, k3_max, k_bins * 2 + 1)[:-1]
    k3_center = 0.5 * (k3_edge[1:] + k3_edge[:-1])
    k3_bins = len(k3_center)
    sub_count_k3, sub_knorm_sum_k3 = get_kbin_count(k3_bins, k3_edge, knorm)
    k_num_k3 = comm.reduce(sub_count_k3, op=MPI.SUM, root=0)
    total_knorm_sum_k3 = comm.reduce(sub_knorm_sum_k3, op=MPI.SUM, root=0)
    if rank == 0:
        k3_eff = total_knorm_sum_k3 / k_num_k3
    else:
        k3_eff = None
    k_num_k3 = comm.bcast(k_num_k3 if rank == 0 else None, root=0)
    k3_eff = comm.bcast(k3_eff, root=0)

    # calculate shot-noise terms
    def calculate_shot_noise_S1_like(rfield_shot, N_field_shot, S_LM_shot=None):
        """
        Compute S1-like one-dimensional shot-noise term S(k1) for survey mode.
        This helper is kept inside calculate_bk_sugi_survey to reuse local variables.

        Implemented conditions:
        - survives only when ell_1 == L and ell_2 == 0
        - returns zeros when required shot-noise fields are unavailable
        - the S_LM correction term is included only for same-tracer case (aaa/auto)

        Structurally this is the survey analogue of the one-leg shot-noise term:
        one delta-like field is paired with one N_field, then projected onto the
        L channel. The S_LM subtraction removes the disconnected same-object
        piece that exists only in the pure auto / same-tracer case.
        """
        S1_local = np.zeros(k_bins, dtype="complex128")

        if not (ell_1 == L and ell_2 == 0):
            _log(
                1,
                "SHOT",
                "S1-like skipped: Kronecker condition ell_1==L and ell_2==0 not satisfied.",
            )
            return S1_local
        if (rfield_shot is None) or (N_field_shot is None):
            _log(1, "SHOT", "S1-like skipped: rfield_shot or N_field_shot is None.")
            return S1_local

        cfield_delta = rfield_shot.r2c()
        cfield_delta.apply(
            out=Ellipsis, func=compensation[0][1], kind=compensation[0][2]
        )
        cfield_delta[:] *= boxsize.prod()

        # The S_LM ("Sbar") correction exists only for pure same-tracer auto
        # statistics. For cross cases (aab/abb/abc) we intentionally skip it:
        # there is no same-tracer disconnected piece of this form to subtract.
        use_sbar_term = correlation_mode == "auto" and tracer_type == "aaa"
        if use_sbar_term:
            if S_LM_shot is None:
                S_LM_arr = np.zeros(L + 1, dtype="complex128")
                _log(
                    1,
                    "SHOT-S1",
                    "use S_LM correction with default zeros (S_LM_shot is None)",
                )
            elif np.isscalar(S_LM_shot):
                S_LM_arr = np.zeros(L + 1, dtype="complex128")
                S_LM_arr[0] = S_LM_shot
            else:
                S_LM_arr = np.asarray(S_LM_shot, dtype="complex128")
            _log(1, "SHOT-S1", f"use S_LM correction with values={S_LM_arr}")
        else:
            S_LM_arr = None
            _log(1, "SHOT-S1", "skip S_LM correction (not same-tracer auto case).")

        # Use conjugate symmetry in M: evaluating M >= 0 is sufficient because
        # the negative-M contribution is its conjugate partner. The weights
        # implement the usual "count M=0 once, M>0 twice" rule in a compact way.
        _log(1, "SHOT-S1", f"start M-loop range=[0,{L}] use_sbar_term={use_sbar_term}")
        _log(1, "SHOT-S1", "strategy=accumulate_complex_field_then_bin_then_2Re")
        accum_field = np.zeros_like(cfield_delta[:], dtype="complex128")
        for M in range(0, L + 1):
            m_weight = 0.5 if M == 0 else 1.0
            _log(2, "SHOT-S1", f"processing M={M}, weight={m_weight}")
            y_LM = get_Ylm(L, M, Racah_normalized=True)

            rfield_NLM = N_field_shot * y_LM(xgrid[0], xgrid[1], xgrid[2])
            cfield_NLM = rfield_NLM.r2c()
            cfield_NLM.apply(
                out=Ellipsis, func=compensation[0][1], kind=compensation[0][2]
            )
            cfield_NLM[:] *= boxsize.prod()

            term_field = cfield_delta[:] * np.conj(cfield_NLM[:])

            if use_sbar_term:
                Sbar_LM = S_LM_arr[M]
                S_field = cfield_delta.copy()
                S_field[:] = Sbar_LM
                S_field.apply(
                    out=Ellipsis,
                    func=compensation_shot_sugi[0][1],
                    kind=compensation_shot_sugi[0][2],
                )
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
            S1_local[valid_mask] = (
                (2 * L + 1) * total_sum[valid_mask] / k_num[valid_mask]
            )
        _log(
            1,
            "SHOT-S1",
            "finished M accumulation, single radial binning, and 2*Re on binned vector.",
        )
        S1_local = comm.bcast(S1_local if rank == 0 else None, root=0)

        del cfield_delta, accum_field, sub_sum_total
        gc.collect()
        return S1_local

    def calculate_Q_L_like(rfield_shot, N_field_shot, S_LM_shot=None):
        """
        Compute Q_L(k3) for S3-like shot-noise term.

        Q_L is the common intermediate object for the analytical S3 branch.
        Once Q_L(k3) is known, the final S3 term is obtained by integrating it
        against the k-triangle kernel q_ell. This separation keeps the expensive
        field-building step independent from the later bin-pair bookkeeping.
        """
        Q_like = np.zeros(k3_bins, dtype="complex128")
        if (rfield_shot is None) or (N_field_shot is None):
            _log(1, "SHOT-S3", "Q_L skipped: rfield_shot or N_field_shot is None.")
            return Q_like

        # As in S1-like, the S_LM subtraction is only meaningful in the
        # same-tracer auto case.
        use_sbar_term = correlation_mode == "auto" and tracer_type == "aaa"
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
            cfield_delta_lm.apply(
                out=Ellipsis, func=compensation[0][1], kind=compensation[0][2]
            )
            cfield_delta_lm[:] *= boxsize.prod()

            term_field = cfield_delta_lm[:] * np.conj(cfield_N00[:])
            if use_sbar_term:
                Sbar_lm = S_LM_arr[M]
                S_field = cfield_delta_lm.copy()
                S_field[:] = Sbar_lm
                S_field.apply(
                    out=Ellipsis,
                    func=compensation_shot_sugi[0][1],
                    kind=compensation_shot_sugi[0][2],
                )
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

        This helper takes the already-binned Q_L(k3) object from the analytical
        branch and maps it to the requested data-vector layout. The math is the
        same in diagonal and full modes; only the set of (k1, k2) entries differs.
        """
        if tracer_type == "abb":
            _log(
                1,
                "SHOT-S3",
                f"S3 forced to zero in mode={mode} because tracer_type=abb.",
            )
            if mode == "full":
                return np.zeros((k_bins, k_bins), dtype="complex128")
            return np.zeros(k_bins, dtype="complex128")

        # Each output entry is independent once Q_like is known. We only exploit
        # this in full mode, where the k_bins^2 workload is large enough to
        # benefit from distributing bin pairs over MPI ranks.
        k_num_k3_local = comm.bcast(k_num_k3 if rank == 0 else None, root=0)
        k3_eff_local = comm.bcast(k3_eff if rank == 0 else None, root=0)
        weighted_k_num = np.zeros(k3_bins, dtype="float64")
        valid_k3 = k_num_k3_local > 0
        weighted_k_num[valid_k3] = k_num_k3_local[valid_k3] / k3_center[valid_k3]
        size = comm.Get_size()

        if mode == "diagonal":
            S3_local = np.zeros(k_bins, dtype="complex128")
            # Diagonal mode has only O(k_bins) work, so we keep the original
            # root-only path.
            if rank == 0:
                for i in range(k_bins):
                    q_ells = get_q_ells_eff(
                        i,
                        i,
                        k_eff_local,
                        ell_1,
                        ell_2,
                        L,
                        k3_bins,
                        k3_eff_local,
                    )
                    numer = np.sum(q_ells * Q_like / k3_center)
                    denom = np.sum(weighted_k_num[: 2 * i + 1])
                    if denom > 0:
                        S3_local[i] = numer / denom
                S3_local *= H_ells * N_ells
                _log(1, "SHOT-S3", "diagonal analytical S3 calculation finished.")
            return comm.bcast(S3_local if rank == 0 else None, root=0)
        elif mode == "full":
            S3_local = np.zeros((k_bins, k_bins), dtype="complex128")
            # Flatten the full (i, j) plane to distribute work evenly with a
            # single rank/size stride rule.
            for flat_idx in range(rank, k_bins * k_bins, size):
                i, j = divmod(flat_idx, k_bins)
                q_ells = get_q_ells_eff(
                    i,
                    j,
                    k_eff_local,
                    ell_1,
                    ell_2,
                    L,
                    k3_bins,
                    k3_eff_local,
                )
                valid_bins = get_valid_k3_bins(
                    k_center[i],
                    k_center[j],
                    k_min,
                    k_max,
                    k_bins,
                )
                numer = np.sum(q_ells * Q_like / k3_center)
                denom = np.sum(weighted_k_num * valid_bins)
                if denom > 0:
                    S3_local[i, j] = numer / denom
            S3_local = comm.reduce(S3_local, op=MPI.SUM, root=0)
            if rank == 0:
                S3_local *= H_ells * N_ells
                _log(1, "SHOT-S3", "full analytical S3 calculation finished with MPI reduction.")
            return comm.bcast(S3_local if rank == 0 else None, root=0)
        else:
            raise ValueError(
                "mode must be either 'diagonal' or 'full' for calculate_shot_noise_S3_like."
            )

    def compute_shot_noise_S3(rfield_shot, N_field_shot, S_LM_shot, mode):
        """
        Dispatch the S3 calculation to the requested implementation.

        "ana" and "fft" are two implementations of the same physical S3 term:
        - ana: build Q_L(k3) first, then integrate against q_ell kernels
        - fft: contract the shell-filtered fields directly in configuration space

        The "both" mode is handled by the caller by invoking this helper twice,
        once per implementation, so that the two results can be compared side by
        side without changing the rest of the estimator flow.
        """
        if tracer_type == "abb":
            if mode == "full":
                return np.zeros((k_bins, k_bins), dtype="complex128")
            return np.zeros(k_bins, dtype="complex128")
        if shotnoise_mode == "ana":
            Q_like = calculate_Q_L_like(rfield_shot, N_field_shot, S_LM_shot)
            return calculate_shot_noise_S3_like(Q_like, mode=mode)
        if shotnoise_mode == "fft":
            def get_s_lm_value(s_lm_arr, m_idx):
                if s_lm_arr is None:
                    return 0.0 + 0.0j
                if m_idx >= 0:
                    return s_lm_arr[m_idx]
                return ((-1) ** m_idx) * np.conj(s_lm_arr[-m_idx])

            def build_shell_filtered_field(ell, m, bin_idx, work_cfield):
                ylm_k = get_Ylm(ell, m, Racah_normalized=True)
                mask = np.logical_and(knorm >= k_edge[bin_idx], knorm < k_edge[bin_idx + 1])
                work_cfield[:] = np.conj(ylm_k(kgrid[0], kgrid[1], kgrid[2])) * mask
                return work_cfield.c2r()

            def build_fft_source_field(M_idx, s_lm_arr):
                ylm_LM = get_Ylm(L, M_idx, Racah_normalized=True)
                rfield_delta_lm = rfield_shot * np.conj(
                    ylm_LM(xgrid[0], xgrid[1], xgrid[2])
                )
                cfield_delta_lm = rfield_delta_lm.r2c()
                cfield_delta_lm.apply(
                    out=Ellipsis, func=compensation[0][1], kind=compensation[0][2]
                )
                cfield_delta_lm[:] *= boxsize.prod()

                source_field = cfield_delta_lm.copy()
                source_field[:] = cfield_delta_lm[:] * np.conj(cfield_N00[:])

                if use_sbar_term:
                    sbar_lm = get_s_lm_value(s_lm_arr, M_idx)
                    shot_corr = cfield_delta_lm.copy()
                    shot_corr[:] = sbar_lm
                    shot_corr.apply(
                        out=Ellipsis,
                        func=compensation_shot_sugi[0][1],
                        kind=compensation_shot_sugi[0][2],
                    )
                    source_field[:] -= shot_corr[:]
                    del shot_corr

                result = source_field.c2r()
                del rfield_delta_lm, cfield_delta_lm, source_field
                gc.collect()
                return result

            # Same rule as the analytical branch: only same-tracer auto uses the
            # S_LM subtraction inside the FFT source field.
            use_sbar_term = correlation_mode == "auto" and tracer_type == "aaa"
            if use_sbar_term:
                if S_LM_shot is None:
                    S_LM_arr = np.zeros(L + 1, dtype="complex128")
                elif np.isscalar(S_LM_shot):
                    S_LM_arr = np.zeros(L + 1, dtype="complex128")
                    S_LM_arr[0] = S_LM_shot
                else:
                    S_LM_arr = np.asarray(S_LM_shot, dtype="complex128")
            else:
                S_LM_arr = None

            rfield_N00 = N_field_shot
            cfield_N00 = rfield_N00.r2c()
            cfield_N00.apply(
                out=Ellipsis, func=compensation[0][1], kind=compensation[0][2]
            )
            cfield_N00[:] *= boxsize.prod()

            work_cfield_1 = cfield_a.copy()
            work_cfield_2 = cfield_a.copy()
            current_M_fft = None
            S3_fft_source_current = None
            fft_cfg_to_idx = {cfg: ii for ii, cfg in enumerate(magnetic_configs)}
            processed_fft_cfg = set()

            if mode == "diagonal":
                S3_local = np.zeros(k_bins, dtype="complex128")
                _log(
                    1,
                    "SHOT-S3",
                    f"start FFT S3 diagonal with {len(magnetic_configs)} sub-configs.",
                )
            elif mode == "full":
                S3_local = np.zeros((k_bins, k_bins), dtype="complex128")
                _log(
                    1,
                    "SHOT-S3",
                    f"start FFT S3 full with {len(magnetic_configs)} sub-configs and block_size={block_size}.",
                )
            else:
                raise ValueError(
                    "mode must be either 'diagonal' or 'full' for FFT S3."
                )

            for ss, (m1, m2, M) in enumerate(magnetic_configs):
                if (m1, m2, M) in processed_fft_cfg:
                    continue

                if M != current_M_fft:
                    if S3_fft_source_current is not None:
                        del S3_fft_source_current
                        gc.collect()
                    S3_fft_source_current = build_fft_source_field(M, S_LM_arr)
                    current_M_fft = M

                can_sym_equal = (ell_1 == ell_2) and (m1 == m2)
                can_sym_conj = (ell_1 == ell_2) and (m1 == -m2)
                if mode == "diagonal":
                    sub_res = np.zeros(k_bins, dtype="complex128")
                    for i in range(k_bins):
                        # Diagonal mode: build only the current shell fields to keep memory bounded.
                        shell_field_1 = build_shell_filtered_field(
                            ell_1, m1, i, work_cfield_1
                        )
                        if can_sym_equal:
                            shell_field_2 = shell_field_1
                        elif can_sym_conj:
                            shell_field_2 = ((-1) ** (m1 + ell_1)) * np.conj(
                                shell_field_1
                            )
                        else:
                            shell_field_2 = build_shell_filtered_field(
                                ell_2, m2, i, work_cfield_2
                            )

                        sub_sum = np.sum(
                            shell_field_1 * shell_field_2 * S3_fft_source_current
                        )
                        total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)
                        if rank == 0 and k_num[i] > 0:
                            sub_res[i] = (
                                three_j_values[ss] * total_sum / (k_num[i] ** 2)
                            )

                        if not can_sym_equal:
                            del shell_field_2
                        del shell_field_1
                        gc.collect()
                else:
                    sub_res = np.zeros((k_bins, k_bins), dtype="complex128")
                    use_symmetry = can_sym_equal or can_sym_conj
                    for bi in range(0, k_bins, block_size):
                        i_end = min(k_bins, bi + block_size)
                        cache_1 = {}
                        bj_start = bi if use_symmetry else 0
                        _log(
                            2,
                            "SHOT-S3",
                            f"fft_full sub_cfg={ss+1}/{len(magnetic_configs)} row=[{bi},{i_end-1}]",
                        )

                        for bj in range(bj_start, k_bins, block_size):
                            j_end = min(k_bins, bj + block_size)
                            block_on_diag = bi == bj
                            cache_2 = {}
                            _log(
                                2,
                                "SHOT-S3",
                                f"fft_full sub_cfg={ss+1}/{len(magnetic_configs)} col=[{bj},{j_end-1}] diag={block_on_diag}",
                            )

                            for i in range(bi, i_end):
                                if i not in cache_1:
                                    cache_1[i] = build_shell_filtered_field(
                                        ell_1, m1, i, work_cfield_1
                                    )
                                shell_field_1 = cache_1[i]

                                j_local_start = bj
                                if use_symmetry and block_on_diag:
                                    j_local_start = max(j_local_start, i)

                                for j in range(j_local_start, j_end):
                                    if can_sym_equal and block_on_diag:
                                        if j not in cache_1:
                                            cache_1[j] = build_shell_filtered_field(
                                                ell_1, m1, j, work_cfield_1
                                            )
                                        shell_field_2 = cache_1[j]
                                    elif can_sym_conj and block_on_diag:
                                        if j not in cache_1:
                                            cache_1[j] = build_shell_filtered_field(
                                                ell_1, m1, j, work_cfield_1
                                            )
                                        shell_field_2 = ((-1) ** (m1 + ell_1)) * np.conj(
                                            cache_1[j]
                                        )
                                    else:
                                        if j not in cache_2:
                                            cache_2[j] = build_shell_filtered_field(
                                                ell_2, m2, j, work_cfield_2
                                            )
                                        shell_field_2 = cache_2[j]

                                    sub_sum = np.sum(
                                        shell_field_1
                                        * shell_field_2
                                        * S3_fft_source_current
                                    )
                                    total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)
                                    if rank == 0 and k_num[i] > 0 and k_num[j] > 0:
                                        sub_res[i, j] = (
                                            three_j_values[ss]
                                            * total_sum
                                            / (k_num[i] * k_num[j])
                                        )

                            del cache_2
                            gc.collect()

                        del cache_1
                        gc.collect()

                    if rank == 0 and use_symmetry:
                        for i in range(k_bins):
                            for j in range(i + 1, k_bins):
                                if can_sym_equal:
                                    sub_res[j, i] = sub_res[i, j]
                                elif can_sym_conj:
                                    sub_res[j, i] = np.conj(sub_res[i, j])

                swap_cfg = (m2, m1, M)
                has_swap = (
                    (ell_1 == ell_2)
                    and (swap_cfg in fft_cfg_to_idx)
                    and (swap_cfg != (m1, m2, M))
                )
                add_swap_now = has_swap and (swap_cfg not in processed_fft_cfg)

                if rank == 0:
                    sub_res_to_add = sub_res
                    if add_swap_now:
                        if mode == "diagonal":
                            # Diagonal mode: transpose relation becomes equality on diagonal bins.
                            sub_res_to_add = 2.0 * sub_res
                        else:
                            sub_res_to_add = sub_res + sub_res.T

                    if (m1, m2, M) == (0, 0, 0):
                        S3_local += sub_res_to_add
                    else:
                        S3_local += 2.0 * np.real(sub_res_to_add)

                if add_swap_now:
                    processed_fft_cfg.add(swap_cfg)
                processed_fft_cfg.add((m1, m2, M))

            if rank == 0:
                S3_local *= H_ells * N_ells
                S3_local *= 1/ nmesh.prod()  
                r"""
                Let L = boxsize.prod(), N = nmesh.prod(). The scalings for the FFT-based S3 term are:

                1. L^2: for the shell-filtered inverse FFT, same as F_{\ell}^{m}, see FFT_Norm_in_Estimators.md
                2. N^2 / L^2: the normalization difference between discrete inverse FFT and continuous integral, see FFT_Norm_in_Estimators.md
                3. N^(-2): for the C2R (complex-to-real) convention
                4. L/N: for the final summation after C2R
                5. L: for the R2C (real-to-complex) of delta_lm or N00, see FFT_Norm_in_Estimators.md

                Thus, the total scaling is: L^2 * (N^2 / L^2) * N^(-2) * (L/N) * L = L^2 / N

                Since L^2 already appears in delta_lm and N00, we need an additional factor of 1/N for correct normalization.
                """


            if S3_fft_source_current is not None:
                del S3_fft_source_current
            del cfield_N00
            del work_cfield_1, work_cfield_2
            gc.collect()
            return comm.bcast(S3_local if rank == 0 else None, root=0)
        raise ValueError("shotnoise_mode must be either 'ana', 'fft', or 'both'.")

    # Shot-noise assembly table for diagonal mode:
    # - SN0: only aaa with (ell1, ell2, L) = (0, 0, 0)
    # - SN1: aaa and abb when ell1 = L and ell2 = 0
    # - SN2: only aaa with (0, 0, 0), and in that case SN2 = SN1
    # - SN3: all cases except abb; for abc it is still forced to zero earlier
    #
    # The same logical content is repeated below for full mode, but SN1/SN2 are
    # expanded from 1d vectors onto the full (k1, k2) plane.
    if data_vector_mode == "diagonal":
        SN0 = 0.0 + 0.0j
        SN1 = np.zeros(k_bins, dtype="complex128")
        SN2 = np.zeros(k_bins, dtype="complex128")
        SN3 = np.zeros(k_bins, dtype="complex128")
        SN3_fft = None

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
                _log(
                    1,
                    "SHOT",
                    "active tracer branch: aab -> rfield_shot=rfield_b, N_field_shot=N_field_a, S_LM_shot=S_LM_a",
                )
            elif tracer_type == "abb":
                rfield_shot = rfield_shot_a
                N_field_shot = N_field_b
                S_LM_shot = stat_attrs.get("S_LM_b", None)
                _log(
                    1,
                    "SHOT",
                    "active tracer branch: abb -> rfield_shot=rfield_a, N_field_shot=N_field_b, S_LM_shot=S_LM_b",
                )
            else:
                rfield_shot = rfield_shot_a
                N_field_shot = N_field_a
                S_LM_shot = stat_attrs.get("S_LM_a", None)
                _log(
                    1,
                    "SHOT",
                    "active tracer branch: aaa/other -> rfield_shot=rfield_a, N_field_shot=N_field_a, S_LM_shot=S_LM_a",
                )

            # S0 is the pure monopole same-tracer constant term. Any cross case,
            # or any non-(0,0,0) angular configuration, must leave it at zero.
            if tracer_type == "aaa" and (ell_1 == 0 and ell_2 == 0 and L == 0):
                S_LM_a = stat_attrs.get("S_LM_a", 0.0)
                SN0 = S_LM_a[0] if isinstance(S_LM_a, np.ndarray) else S_LM_a
                _log(
                    1,
                    "SHOT-S0",
                    f"active: tracer_type=aaa and angu_config=(0,0,0), value={SN0}",
                )
            else:
                SN0 = 0.0 + 0.0j
                _log(
                    1,
                    "SHOT-S0",
                    "set_to_zero: requires tracer_type=aaa and angu_config=(0,0,0).",
                )

            # S1 survives only for the angular structure ell_1 = L, ell_2 = 0.
            # In survey mode this branch is relevant for:
            # - aaa: same-tracer auto
            # - abb: one repeated tracer on the second and third legs
            if (tracer_type in ["aaa", "abb"]) and (ell_1 == L and ell_2 == 0):
                SN1 = calculate_shot_noise_S1_like(rfield_shot, N_field_shot, S_LM_shot)
                _log(1, "SHOT", "SN1 active.")
            else:
                SN1 = np.zeros(k_bins, dtype="complex128")
                _log(
                    1,
                    "SHOT",
                    "SN1 set to 0 (requires tracer_type in {aaa,abb} and ell_1==L, ell_2==0).",
                )

            # S2 exists only in the fully same-tracer monopole case; in that
            # limit it is identical to S1 and we explicitly reuse it.
            if tracer_type == "aaa" and (ell_1 == 0 and ell_2 == 0 and L == 0):
                SN2 = SN1.copy()
                _log(
                    1,
                    "SHOT-S2",
                    "active: reuse SN1 because tracer_type=aaa and angu_config=(0,0,0).",
                )
            else:
                SN2 = np.zeros(k_bins, dtype="complex128")
                _log(
                    1,
                    "SHOT-S2",
                    "set_to_zero: requires tracer_type=aaa and angu_config=(0,0,0).",
                )

            # S3 is the only shot-noise term that can be evaluated by either the
            # analytical or FFT implementation. The tracer rules are:
            # - abb: force to zero by construction
            # - abc: already zero because no repeated tracer exists
            # - aaa/aab: evaluate normally
            if tracer_type != "abb":
                if shotnoise_mode == "both":
                    if rank == 0:
                        _log(
                            1,
                            "SHOT-S3",
                            "shotnoise_mode=both: computing both 'ana' and 'fft' versions of SN3.",
                        )   
                    shotnoise_mode_saved = shotnoise_mode
                    shotnoise_mode = "ana"
                    SN3 = compute_shot_noise_S3(
                        rfield_shot, N_field_shot, S_LM_shot, mode="diagonal"
                    )
                    shotnoise_mode = "fft"
                    SN3_fft = compute_shot_noise_S3(
                        rfield_shot, N_field_shot, S_LM_shot, mode="diagonal"
                    )
                    shotnoise_mode = shotnoise_mode_saved
                else:
                    SN3 = compute_shot_noise_S3(
                        rfield_shot, N_field_shot, S_LM_shot, mode="diagonal"
                    )
                _log(1, "SHOT", "SN3 active.")
            else:
                SN3 = np.zeros(k_bins, dtype="complex128")
                SN3_fft = np.zeros(k_bins, dtype="complex128") if shotnoise_mode == "both" else None
                _log(1, "SHOT", "SN3 set to 0 because tracer_type=abb.")
    else:
        SN0 = 0.0 + 0.0j
        SN1 = np.zeros((k_bins, k_bins), dtype="complex128")
        SN2 = np.zeros((k_bins, k_bins), dtype="complex128")
        SN3 = np.zeros((k_bins, k_bins), dtype="complex128")
        SN3_fft = None

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
                _log(
                    1,
                    "SHOT",
                    "full mode branch: aab -> rfield_shot=rfield_b, N_field_shot=N_field_a, S_LM_shot=S_LM_a",
                )
            elif tracer_type == "abb":
                rfield_shot = rfield_shot_a
                N_field_shot = N_field_b
                S_LM_shot = stat_attrs.get("S_LM_b", None)
                _log(
                    1,
                    "SHOT",
                    "full mode branch: abb -> rfield_shot=rfield_a, N_field_shot=N_field_b, S_LM_shot=S_LM_b",
                )
            else:
                rfield_shot = rfield_shot_a
                N_field_shot = N_field_a
                S_LM_shot = stat_attrs.get("S_LM_a", None)
                _log(
                    1,
                    "SHOT",
                    "full mode branch: aaa/other -> rfield_shot=rfield_a, N_field_shot=N_field_a, S_LM_shot=S_LM_a",
                )

            # S0: only for tracer_type=="aaa" and (ell1,ell2,L)==(0,0,0).
            if tracer_type == "aaa" and (ell_1 == 0 and ell_2 == 0 and L == 0):
                S_LM_a = stat_attrs.get("S_LM_a", 0.0)
                SN0 = S_LM_a[0] if isinstance(S_LM_a, np.ndarray) else S_LM_a
                _log(
                    1,
                    "SHOT-S0",
                    f"full_mode active: tracer_type=aaa and angu_config=(0,0,0), value={SN0}",
                )
            else:
                SN0 = 0.0 + 0.0j
                _log(
                    1,
                    "SHOT-S0",
                    "full_mode set_to_zero: requires tracer_type=aaa and angu_config=(0,0,0).",
                )

            # In full mode we still compute the intrinsic S1 content as a 1d
            # vector first, then broadcast it across columns because S1 depends
            # only on k1.
            if (tracer_type in ["aaa", "abb"]) and (ell_1 == L and ell_2 == 0):
                SN1_vec = calculate_shot_noise_S1_like(
                    rfield_shot, N_field_shot, S_LM_shot
                )
                _log(1, "SHOT", "full mode SN1 active (from 1D S1-like).")
            else:
                SN1_vec = np.zeros(k_bins, dtype="complex128")
                _log(
                    1,
                    "SHOT",
                    "full mode SN1 set to 0 (requires tracer_type in {aaa,abb} and ell_1==L, ell_2==0).",
                )

            for i in range(k_bins):
                SN1[i, :] = SN1_vec[i]

            # Similarly, S2 depends only on k2, so we build a 1d vector and then
            # copy it along rows.
            if tracer_type == "aaa" and (ell_1 == 0 and ell_2 == 0 and L == 0):
                SN2_vec = SN1_vec.copy()
                _log(
                    1,
                    "SHOT-S2",
                    "full_mode active: reuse SN1_vec because tracer_type=aaa and angu_config=(0,0,0).",
                )
            else:
                SN2_vec = np.zeros(k_bins, dtype="complex128")
                _log(
                    1,
                    "SHOT-S2",
                    "full_mode set_to_zero: requires tracer_type=aaa and angu_config=(0,0,0).",
                )

            for i in range(k_bins):
                SN2[:, i] = SN2_vec[i]

            # S3 in full mode follows exactly the same tracer logic as in
            # diagonal mode; the only difference is that the output is a full
            # (k1, k2) matrix.
            if tracer_type != "abb":
                if shotnoise_mode == "both":
                    if rank == 0:
                        _log(
                            1,
                            "SHOT-S3",
                            "full mode shotnoise_mode=both: computing both 'ana' and 'fft' versions of SN3.",
                        )
                    shotnoise_mode_saved = shotnoise_mode
                    shotnoise_mode = "ana"
                    SN3 = compute_shot_noise_S3(
                        rfield_shot, N_field_shot, S_LM_shot, mode="full"
                    )
                    shotnoise_mode = "fft"
                    SN3_fft = compute_shot_noise_S3(
                        rfield_shot, N_field_shot, S_LM_shot, mode="full"
                    )
                    shotnoise_mode = shotnoise_mode_saved
                else:
                    SN3 = compute_shot_noise_S3(
                        rfield_shot, N_field_shot, S_LM_shot, mode="full"
                    )
                _log(1, "SHOT", "full mode SN3 active.")
            else:
                SN3 = np.zeros((k_bins, k_bins), dtype="complex128")
                SN3_fft = (
                    np.zeros((k_bins, k_bins), dtype="complex128")
                    if shotnoise_mode == "both"
                    else None
                )
                _log(1, "SHOT", "full mode SN3 set to 0 because tracer_type=abb.")
    time_shotnoise_end = time.time()
    if rank == 0:
        logging.info(
            f"Rank {rank}: Time to compute shot noise terms: {time_shotnoise_end - time_shotnoise_start:.2f} seconds"
        )
    if rank == 0:
        # Final assembly:
        # 1) normalize the signal-only bispectrum estimate
        # 2) normalize each shot-noise term by the same I_norm
        # 3) subtract the combined shot-noise contribution from the raw signal
        #
        # When shotnoise_mode == "both", SN3 stores the analytical branch and
        # SN3_FFT is exported as an auxiliary comparison product; the estimator
        # still subtracts the analytical SN3 from the signal by default.
        total_res *= N_ells * H_ells / I_norm
        if data_vector_mode == "full":
            total_res *= 1.0 / np.outer(k_num, k_num)
        else:
            total_res *= 1.0 / (k_num**2)
            sn_shape = (k_bins,)
        total_res *= vol_per_cell

        total_shot_noise = (SN0 + SN1 + SN2 + SN3) / I_norm
        final_bk = total_res - total_shot_noise

        sn_terms = {
            "SN0": SN0 / I_norm,
            "SN1": SN1 / I_norm,
            "SN2": SN2 / I_norm,
            "SN3": SN3 / I_norm,
        }
        if shotnoise_mode == "both":
            sn_terms["SN3_FFT"] = SN3_fft / I_norm

        results.update(
            {
                "B_sugi": final_bk,
                "Bk_raw": total_res,
                "SN_terms": sn_terms,
                "Shot_noise": total_shot_noise,
                "I_norm": I_norm,
                "k_eff": k_eff,
                "nmodes": k_num,
            }
        )

    # Keep N_field/rfield construction in this stage, but do not serialize mesh objects.
    del rfield_shot_a, rfield_shot_b, rfield_shot_c
    del N_field_a, N_field_b, N_field_c
    gc.collect()

    results = comm.bcast(results, root=0)
    if rank == 0:
        logging.info(
            f"Rank {rank}: Finished survey bispectrum (signal-only) calculation."
        )
    return results


def get_G_ell(rfield, ell, kgrid, xgrid, compensation, boxsize, comm):
    r"""
    Calculate the G_ell function for the given ell.
    This function is a placeholder and should be replaced with the actual implementation.
    \mathcal{G}_\ell(\mathbf{k})= \frac{4\pi}{2\ell+1} \left[F_\ell^0(\mathbf{k})
    +2\sum_{m=1}^\ell F_\ell^m(\mathbf{k})\right]
    """
    rank = comm.Get_rank()

    Ylms = [get_Ylm(ell, m) for m in range(ell + 1)]
    rf = rfield * Ylms[0](xgrid[0], xgrid[1], xgrid[2])
    G_ell = rf.r2c()
    G_ell[:] *= Ylms[0](kgrid[0], kgrid[1], kgrid[2])

    for m in range(1, ell + 1):
        rf = rfield * np.conj(Ylms[m](xgrid[0], xgrid[1], xgrid[2]))
        cf = rf.r2c()
        cf[:] *= Ylms[m](kgrid[0], kgrid[1], kgrid[2])
        G_ell[:] += 2 * cf[:]

    # recollect the memory
    del rf, cf
    gc.collect()

    G_ell.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    if rank == 0:
        logging.info(f"Rank {rank}: {compensation[0][1].__name__} applied to G_ell")
        # print(f"rank {rank}: type of G_ell = {type(G_ell)}")
    G_ell[:] *= 4 * np.pi * boxsize.prod() / (2 * ell + 1)

    return G_ell
