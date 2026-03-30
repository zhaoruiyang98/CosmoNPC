import numpy as np
import sympy as sp
import gc
from mpi4py import MPI

"""
Get the distance of every mesh grid to the mesh-center in k-space for radial binning
as well as get the normalized position of every mesh grid in k/x-space for spherical harmonics
"""


def get_kgrid(cfield):
    kgrid = [np.real(kk) for kk in cfield.x]
    knorm = np.sqrt(sum(np.real(kk) ** 2 for kk in cfield.x))
    normalized_kgrid = []
    for k in kgrid:
        normalized_kgrid.append(k / knorm)
    # we used the property of pmesh
    if knorm[0, 0, 0] == 0.0:
        for k in normalized_kgrid:
            k[0, 0, 0] = 0.0
    return normalized_kgrid, knorm


def get_xgrid(rfield, boxcenter, boxsize, nmesh):
    off_set = 0.5 * boxsize / nmesh + boxcenter
    xgrid = [np.real(rfield.x[i]) + off_set[i] for i in range(3)]
    xnorm = np.sqrt(sum(xx**2 for xx in xgrid))
    result = [x / xnorm for x in xgrid]
    del xnorm  # Explicitly delete xnorm to free memory
    gc.collect()  # Trigger garbage collection
    return result


def get_Ylm(l, m, Racah_normalized=False):
    r"""
    Return a function that computes the complex spherical
    harmonic of order (l,m)

    Parameters
    ----------
    l : int
        the degree of the harmonic
    m : int
        the order of the harmonic; abs(m) <= l
    Racah_normalized : bool, optional
        if True, the returned function will expect Racah-normalized,
        an additional factor of sqrt(4pi/(2l+1)) will be applied.
        Default is False.

    Returns
    -------
    Ylm : callable
        a function that takes 3 arguments: (xhat, yhat, zhat)
        unit-normalized Cartesian coordinates and returns the
        specified complex Ylm

    References
    ----------
    https://en.wikipedia.org/wiki/Spherical_harmonics#Complex_form

    Warning
    -------
    Spherical harmonics can not be defined at the origin (r=0).
    The returned function will return sqrt{1/4\pi} for l=m=0 and
    return 0 for all other (l,m) at the origin.
    This should not be a big issue since uasually we only need Ylm
    with l>0

    """

    # Input validation
    l = int(l)
    m = int(m)
    if abs(m) > l:
        raise ValueError("abs(m) must be <= l")

    # the relevant cartesian and spherical symbols
    x, y, z = sp.symbols("x y z ", real=True)
    r = sp.symbols("r", real=True, positive=True)
    xhat, yhat, zhat = sp.symbols("xhat yhat zhat", real=True)
    phi, theta = sp.symbols("phi theta")

    # Define substitutions for spherical coordinates
    defs = [
        (sp.sin(phi), y / sp.sqrt(x**2 + y**2)),
        (sp.cos(phi), x / sp.sqrt(x**2 + y**2)),
        (sp.cos(theta), z / sp.sqrt(x**2 + y**2 + z**2)),
    ]

    # the normalization factor
    amp = sp.sqrt(
        (2 * l + 1) / (4 * np.pi) * sp.factorial(l - abs(m)) / sp.factorial(l + abs(m))
    )

    # the cos(theta) dependence encoded by the associated Legendre poly
    # Note: sympy's assoc_legendre already contains the Condon-Shortley phase (-1)^m
    # https://docs.sympy.org/latest/modules/functions/special.html
    expr = sp.assoc_legendre(l, abs(m), sp.cos(theta))

    # Explicitly handle phi dependence: e^(I*m*phi) = (cos(phi) + I*sin(phi))^m
    phi_term = (sp.cos(phi) + sp.I * sp.sin(phi)) ** abs(m)

    # Combine theta and phi dependence
    expr *= phi_term

    # Apply substitutions and simplify
    expr = expr.subs(defs)
    expr = sp.together(expr).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x / r, xhat), (y / r, yhat), (z / r, zhat)])

    # apply Racah normalization if requested
    if Racah_normalized:
        expr = expr * sp.sqrt(4 * np.pi / (2 * l + 1))

    # For negative m, take complex conjugate and multiply by (-1)^m
    if m < 0:
        expr = (-1) ** m * sp.conjugate(expr)

    # Further simplify to ensure no symbolic variables remain
    expr = sp.simplify(expr)

    # Convert to callable function using numexpr
    Ylm = sp.lambdify((xhat, yhat, zhat), expr, "numexpr")

    # Attach some meta-data
    Ylm.expr = expr
    Ylm.l = l
    Ylm.m = m

    return Ylm


def get_legendre(ell, r_xhat, r_yhat, r_zhat):
    """
    Return a function that computes the Legendre polynomial of degree ell
    for the cosine of the angle between a fixed unit vector (r_xhat, r_yhat, r_zhat)
    and an arbitrary unit vector (k_xhat, k_yhat, k_zhat).

    Parameters
    ----------
    ell : int
        the degree of the Legendre polynomial
    r_xhat : float
        x-component of the fixed unit vector
    r_yhat : float
        y-component of the fixed unit vector
    r_zhat : float
        z-component of the fixed unit vector

    Returns
    -------
    P_ell : callable
        a function that takes 3 arguments: (k_xhat, k_yhat, k_zhat)
        representing the components of a unit vector and returns the
        Legendre polynomial P_ell(cos(theta)) where theta is the angle
        between (r_xhat, r_yhat, r_zhat) and (k_xhat, k_yhat, k_zhat)

    References
    ----------
    https://en.wikipedia.org/wiki/Legendre_polynomials
    """

    # Input validation
    ell = int(ell)
    if ell < 0:
        raise ValueError("ell must be non-negative")
    if not np.isclose(r_xhat**2 + r_yhat**2 + r_zhat**2, 1.0):
        raise ValueError("(r_xhat, r_yhat, r_zhat) must form a unit vector")

    # Define symbolic variables
    k_xhat, k_yhat, k_zhat = sp.symbols("k_xhat k_yhat k_zhat", real=True)
    cos_theta = sp.Symbol("cos_theta", real=True)

    # Compute the cosine of the angle using the dot product
    expr = r_xhat * k_xhat + r_yhat * k_yhat + r_zhat * k_zhat

    # Compute the Legendre polynomial P_ell(cos_theta)
    legendre_expr = sp.legendre(ell, cos_theta)

    # Substitute cos_theta with the dot product expression
    expr = legendre_expr.subs(cos_theta, expr)

    # Simplify the expression
    expr = sp.simplify(expr)

    # Convert to a callable function using numexpr
    P_ell = sp.lambdify((k_xhat, k_yhat, k_zhat), expr, "numexpr")

    # Attach metadata
    P_ell.expr = expr
    P_ell.ell = ell
    P_ell.r_xhat = r_xhat
    P_ell.r_yhat = r_yhat
    P_ell.r_zhat = r_zhat

    return P_ell


"""
Taken from nbodykit.algorithms.convpower.catalogmesh
see Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240> for details
"""


def CompensateTSC(w, v):
    r"""
    Return the Fourier-space kernel that accounts for the convolution of
    the gridded field with the TSC window function in configuration space.
    Parameters
    ----------
    w : list of arrays
        the list of "circular" coordinate arrays, ranging from
        :math:`[-\pi, \pi)`.
    v : array_like
        the field array
    """
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi)) ** 3
        v = v / tmp
    return v


def CompensatePCS(w, v):
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi)) ** 4
        v = v / tmp
    return v


def CompensateCIC(w, v):
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi)) ** 2
        tmp[wi == 0.0] = 1.0
        v = v / tmp
    return v


def CompensateNGP(w, v):
    for i in range(3):
        wi = w[i]
        tmp = np.sinc(0.5 * wi / np.pi)
        tmp[wi == 0.0] = 1.0
        v = v / tmp
    return v


def CompensateTSCShotnoise(w, v):
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi) ** 2
        v = v / (1 - s + 2.0 / 15 * s**2) ** 0.5
    return v


def CompensatePCSShotnoise(w, v):
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi) ** 2
        v = v / (1 - 4.0 / 3.0 * s + 2.0 / 5.0 * s**2 - 4.0 / 315.0 * s**3) ** 0.5
    return v


def CompensateCICShotnoise(w, v):
    for i in range(3):
        wi = w[i]
        v = v / (1 - 2.0 / 3 * np.sin(0.5 * wi) ** 2) ** 0.5
    return v


def CompensateNGPShotnoise(w, v):
    return v


def Compensate_bk_noise_tsc(w, v):
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi) ** 2
        tmp = (np.sinc(0.5 * wi / np.pi)) ** 3
        tmp[wi == 0.0] = 1.0
        v = v * (1 - s + 2.0 / 15 * s**2) / tmp**2
    return v


def Compensate_bk_noise_cic(w, v):
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi) ** 2
        tmp = (np.sinc(0.5 * wi / np.pi)) ** 2
        tmp[wi == 0.0] = 1.0
        v = v * (1 - 2.0 / 3 * s) / tmp**2
    return v


def Compensate_bk_noise_ngp(w, v):
    return v


def Compensate_bk_noise_pcs(w, v):
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi) ** 2
        tmp = (np.sinc(0.5 * wi / np.pi)) ** 4
        tmp[wi == 0.0] = 1.0
        v = v * (1 - 4.0 / 3.0 * s + 2.0 / 5.0 * s**2 - 4.0 / 315.0 * s**3) / tmp**2
    return v


def get_magnetic_configs_box(ell_1, ell_2, L):
    """
    Ensured by angu_config that ell_1+ell_2+L is even, wigner_3j(ell_1, ell_2, L, 0,0,0) != 0
    """
    from sympy.physics.wigner import wigner_3j

    # for box configuration, M = 0
    M = 0
    magnetic_configs = []
    three_j_values = []

    for m1 in range(0, ell_1 + 1):
        for m2 in range(-ell_2, ell_2 + 1):
            if m1 + m2 == M:
                magnetic_configs.append((m1, m2, M))
                three_j_values.append(
                    np.float64(wigner_3j(ell_1, ell_2, L, m1, m2, M).evalf())
                )

    return magnetic_configs, three_j_values


def get_magnetic_configs_survey(ell_1, ell_2, L):
    from sympy.physics.wigner import wigner_3j

    magnetic_configs = []
    three_j_values = []
    found_zero = False

    for M in range(-L, L + 1):
        for m1 in range(-ell_1, ell_1 + 1):
            m2 = -M - m1
            if -ell_2 <= m2 <= ell_2:
                magnetic_configs.append((m1, m2, M))
                three_j_values.append(
                    np.float64(wigner_3j(ell_1, ell_2, L, m1, m2, M).evalf())
                )
                if (m1, m2, M) == (0, 0, 0):
                    found_zero = True
                    break
        if found_zero:
            break

    return magnetic_configs, three_j_values


def get_legendre_coefficients(ell, k1, k2, k_min, k_max, kbin, mode="12"):
    """
    if mode == "13", mu is the cosine of the angle between k1 and k3
    if mode == "12", mu is the cosine of the angle between k1 and k2
    """
    from sympy import legendre_poly

    k3_min, k3_max = 2 * k_min, 2 * k_max
    k3_edge = np.linspace(k3_min, k3_max, kbin * 2 + 1)[:-1]
    k3_center = 0.5 * (k3_edge[1:] + k3_edge[:-1])

    res_legendre = np.zeros(len(k3_center))

    assert mode in ["12", "13"], "mode must be either '12' or '13'"

    for i in range(len(k3_center)):
        if abs(k1 - k2) <= k3_center[i] <= (k1 + k2):
            if mode == "13":
                res_legendre[i] = legendre_poly(
                    ell, (k3_center[i] ** 2 + k1**2 - k2**2) / (2 * k1 * k3_center[i])
                ).evalf()
            elif mode == "12":
                res_legendre[i] = legendre_poly(
                    ell, (k1**2 + k2**2 - k3_center[i] ** 2) / (2 * k1 * k2)
                ).evalf()

    res_legendre *= (-1) ** ell  # the mu calculated here is actually -mu in the formula

    return res_legendre


def get_associated_legendre_coefficients(ell, m, k1, k2, k_min, k_max, kbin, mode="13"):
    """
    Get the associated Legendre polynomial coefficients P_ell^m
    if mode == "13", mu is the cosine of the angle between k1 and k3
    if mode == "23", mu is the cosine of the angle between k2 and k3
    """
    from sympy import assoc_legendre

    k3_min, k3_max = 2 * k_min, 2 * k_max
    k3_edge = np.linspace(k3_min, k3_max, kbin * 2 + 1)[:-1]
    k3_center = 0.5 * (k3_edge[1:] + k3_edge[:-1])

    res_assoc_legendre = np.zeros(len(k3_center))
    assert mode in ["23", "13"], "mode must be either '23' or '13'"

    for i in range(len(k3_center)):
        if abs(k1 - k2) <= k3_center[i] <= (k1 + k2):
            if mode == "13":
                mu = (k3_center[i] ** 2 + k1**2 - k2**2) / (2 * k1 * k3_center[i])
            elif mode == "23":
                mu = (k3_center[i] ** 2 + k2**2 - k1**2) / (2 * k2 * k3_center[i])
            res_assoc_legendre[i] = assoc_legendre(ell, m, mu).evalf()

    res_assoc_legendre *= (-1) ** (
        ell + m
    )  # the mu calculated here is actually -mu in the formula
    return res_assoc_legendre


def get_valid_k3_bins(k1, k2, k_min, k_max, kbin):
    k3_min, k3_max = 2 * k_min, 2 * k_max
    k3_edge = np.linspace(k3_min, k3_max, kbin * 2 + 1)[:-1]
    k3_center = 0.5 * (k3_edge[1:] + k3_edge[:-1])

    valid_bins = np.logical_and(k3_center >= abs(k1 - k2), k3_center <= (k1 + k2))
    return valid_bins

def get_kbin_count(k_bins, k_edge, knorm):
    """
    Count the number of points as well as the sum k-distance in each k-bin
    """
    # Initialize the result array
    sub_count = np.zeros(k_bins).astype("f8")
    sub_knorm_sum = np.zeros(k_bins).astype("f8")

    # Step 1: flatten knorm to 1d array
    knorm = np.ascontiguousarray(knorm.ravel())

    # Step 2: remove points outside the bin range first
    valid = np.logical_and(knorm >= k_edge[0], knorm < k_edge[-1])
    knorm = knorm[valid]

    # Step 3: find the bin index of each remaining k value
    # For bins [k_edge[i], k_edge[i + 1]), searchsorted(..., side="right") - 1
    # gives the corresponding 0-based bin index.
    bin_indices = np.searchsorted(k_edge, knorm, side="right") - 1

    # Step 4: count points and sum knorm in each bin
    sub_count = np.bincount(bin_indices, minlength=k_bins).astype("f8")
    sub_knorm_sum = np.bincount(
        bin_indices, weights=knorm, minlength=k_bins
    ).astype("f8")

    del bin_indices, valid
    gc.collect()

    return sub_count, sub_knorm_sum


def get_kbin_count_old(k_bins, k_edge, knorm):
    """
    Count the number of points as well as the sum k-distance in each k-bin
    """
    # Initialize the result array
    sub_count = np.zeros(k_bins).astype("f8")
    sub_knorm_sum = np.zeros(k_bins).astype("f8")

    # Loop over the bins and sum the values in each bin
    for i in range(k_bins):
        mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i + 1])
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

    # Step 1: flatten kfield and knorm to 1d arrays
    kfield = np.ascontiguousarray(kfield.ravel())
    knorm = np.ascontiguousarray(knorm.ravel())

    # Step 2: remove points outside the bin range first
    valid = np.logical_and(knorm >= k_edge[0], knorm < k_edge[-1])
    knorm = knorm[valid]
    kfield = kfield[valid]

    # Step 3: find the bin index of each remaining k value
    bin_indices = np.searchsorted(k_edge, knorm, side="right") - 1

    # Step 4: sum kfield in each bin, handling complex values manually if necessary
    if np.iscomplexobj(kfield):
        sub_sum = np.bincount(
            bin_indices, weights=np.real(kfield), minlength=k_bins
        ) + 1j * np.bincount(
            bin_indices, weights=np.imag(kfield), minlength=k_bins
        )
        sub_sum = sub_sum.astype(kfield.dtype, copy=False)
    else:
        sub_sum = np.bincount(
            bin_indices, weights=kfield, minlength=k_bins
        ).astype(kfield.dtype, copy=False)

    del bin_indices, valid
    gc.collect()

    return sub_sum


def radial_binning_old(kfield, k_bins, k_edge, knorm):
    """
    Radial binning of the Fourier transform of the density field
    """
    # Initialize the result array
    sub_sum = np.zeros(k_bins).astype(kfield.dtype)

    # Loop over the bins and sum the values in each bin
    for i in range(k_bins):
        mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i + 1])
        sub_sum[i] = np.sum(kfield[mask])

    del mask
    gc.collect()

    return sub_sum


def get_q_ells(i, j, k_center, k_min, k_max, k_bins, ell_1, ell_2, L, k3_bins):
    from sympy.physics.wigner import wigner_3j

    r"""
    $$
    q_{\ell_1 \ell_2 L}\left(k_1, k_2, k_3\right)=
    \sum_n(-1)^n \mathcal{L}_{\ell_1}^n\left(\hat{k}_1 \cdot \hat{k}_3\right) 
    \mathcal{L}_{\ell_2}^{-n}\left(\hat{k}_2 \cdot \hat{k}_3\right)
    \left(\begin{array}{ccc}
    \ell_1 & \ell_2 & L \\
    n & -n & 0
    \end{array}\right)
    $$
    """

    q_ells = np.zeros(k3_bins).astype("f8")
    ell_min = min(ell_1, ell_2)

    for xx in range(-ell_min, ell_min + 1):
        three_j = np.float64(wigner_3j(ell_1, ell_2, L, xx, -xx, 0).evalf())
        al_13 = get_associated_legendre_coefficients(
            ell_1, xx, k_center[i], k_center[j], k_min, k_max, k_bins, mode="13"
        )
        al_23 = get_associated_legendre_coefficients(
            ell_2, -xx, k_center[i], k_center[j], k_min, k_max, k_bins, mode="23"
        )
        sub_coeff = (-1) ** (xx) * three_j * al_13 * al_23
        q_ells += sub_coeff

    return q_ells


def _get_flip_partner(idx, dim_len):
    """
    Return the 1D partner of one index under the serial inversion rule.

    The rule used here is:
    - keep index 0 fixed
    - reverse all remaining indices `1:`
    """
    if idx == 0:
        return 0
    return dim_len - idx


def _build_axis_inverse_segments(local_start, local_end, dim_len, splits):
    """
    Build contiguous inverse-mapping segments for one distributed axis.

    Each segment is `(block_id, dst0, dst1, src0, src1, reverse_axis)`.

    Meaning:
    - `block_id` identifies which distributed block owns the source interval
    - `dst0:dst1` is the destination-local interval on the current rank
    - `src0:src1` is the source-local interval inside the source block
    - `reverse_axis` tells the caller whether to reverse the source interval
      before storing it into the destination interval

    The inversion rule maps a local interval back to a small number of
    contiguous source intervals. That makes block-based MPI communication
    possible and avoids point-by-point metadata exchange.
    """
    segments = []

    if local_start < 1 and local_end > 0:
        # The zero mode is fixed by the inversion rule, so it forms a trivial
        # one-element segment with no reversal.
        segments.append((0, 0 - local_start, 1 - local_start, 0, 1, False))

    start_nonzero = max(local_start, 1)
    if start_nonzero >= local_end:
        return segments

    src_low = _get_flip_partner(local_end - 1, dim_len)
    src_high = _get_flip_partner(start_nonzero, dim_len) + 1

    for block_id, (split_start, split_end) in enumerate(splits):
        # Intersect the full inverse image of this local interval with each
        # distributed source block. Each overlap becomes one transferable
        # contiguous segment.
        ov0 = max(src_low, split_start)
        ov1 = min(src_high, split_end)
        if ov0 >= ov1:
            continue

        # Convert the overlapped global source interval into:
        # - local destination coordinates on the current rank
        # - local source coordinates inside the source rank's block
        dst_global_start = _get_flip_partner(ov1 - 1, dim_len)
        dst_global_end = _get_flip_partner(ov0, dim_len) + 1
        dst0 = dst_global_start - local_start
        dst1 = dst_global_end - local_start
        src0 = ov0 - split_start
        src1 = ov1 - split_start
        segments.append((block_id, dst0, dst1, src0, src1, True))

    return segments


def _build_transposed_complex_plans(global_shape, y_splits, z_splits):
    """
    Precompute block transfer plans for a TransposedComplexField layout.

    Layout assumption:
    - x-axis is fully local on every rank
    - y- and z-axes are distributed across ranks

    Entry layout
    ------------
    `(peer_rank, pair_block_id, dy0, dy1, dz0, dz1, sy0, sy1, sz0, sz1,
    rev_y, rev_z)`

    Meaning
    -------
    - `peer_rank`: the other rank in this data transfer
    - `pair_block_id`: distinguishes multiple blocks exchanged by the same pair
    - `dy0:dy1`, `dz0:dz1`: destination-local slice on the receiver
    - `sy0:sy1`, `sz0:sz1`: source-local slice on the sender
    - `rev_y`, `rev_z`: whether the source slice must be reversed before use
    """
    nx, ny, nz = global_shape
    del nx
    a = len(y_splits)
    b = len(z_splits)
    size = a * b

    recv_entries = [[] for _ in range(size)]
    send_entries = [[] for _ in range(size)]

    for dst_rank in range(size):
        dst_yb = dst_rank // b
        dst_zb = dst_rank % b
        y0, y1 = y_splits[dst_yb]
        z0, z1 = z_splits[dst_zb]

        y_segments = _build_axis_inverse_segments(y0, y1, ny, y_splits)
        z_segments = _build_axis_inverse_segments(z0, z1, nz, z_splits)
        pair_block_counts = {}

        for src_yb, dy0, dy1, sy0, sy1, rev_y in y_segments:
            for src_zb, dz0, dz1, sz0, sz1, rev_z in z_segments:
                # Cartesian-product the y- and z-axis segments. Because x is
                # fully local in this layout, one such pair defines one
                # complete 3D transfer block.
                src_rank = src_yb * b + src_zb
                pair_block_id = pair_block_counts.get(src_rank, 0)
                pair_block_counts[src_rank] = pair_block_id + 1

                entry = (
                    src_rank,
                    pair_block_id,
                    dy0,
                    dy1,
                    dz0,
                    dz1,
                    sy0,
                    sy1,
                    sz0,
                    sz1,
                    rev_y,
                    rev_z,
                )
                recv_entries[dst_rank].append(entry)
                send_entries[src_rank].append(
                    (
                        dst_rank,
                        pair_block_id,
                        dy0,
                        dy1,
                        dz0,
                        dz1,
                        sy0,
                        sy1,
                        sz0,
                        sz1,
                        rev_y,
                        rev_z,
                    )
                )

    return tuple(tuple(x) for x in recv_entries), tuple(tuple(x) for x in send_entries)


def _infer_transposed_complex_layout(comm, local_slices):
    """
    Infer the global y/z decomposition from rank-local TransposedComplexField slices.

    pmesh already knows the correct distribution, but the inversion routine
    only sees the field object on each rank. This helper reconstructs the
    global y/z block layout by exchanging the local slices once.
    """
    gathered = comm.allgather(tuple((int(s.start), int(s.stop)) for s in local_slices))
    unique_y = sorted({item[1] for item in gathered})
    unique_z = sorted({item[2] for item in gathered})

    y_splits = tuple(unique_y)
    z_splits = tuple(unique_z)
    b = len(z_splits)

    rank_to_blocks = {}
    for rank, slices in enumerate(gathered):
        # `slices[0]` is the x-axis. It is fully local on every rank for a
        # TransposedComplexField, so only y and z matter for the rank layout.
        yb = y_splits.index(slices[1])
        zb = z_splits.index(slices[2])
        rank_to_blocks[rank] = (yb, zb)

    ordered = [None] * len(gathered)
    for rank, (yb, zb) in rank_to_blocks.items():
        ordered[yb * b + zb] = rank

    if any(x is None for x in ordered):
        raise ValueError(
            "Failed to infer a complete TransposedComplexField rank layout."
        )

    return y_splits, z_splits, tuple(ordered)


def space_inversion_transposed_complex(field, return_type="ndarray"):
    """
    Apply the serial-space-inversion rule to a distributed TransposedComplexField.

    Parameters
    ----------
    field : pmesh.pm.TransposedComplexField
        Distributed complex field produced by pmesh.
    return_type : {"field", "ndarray"}
        `"field"` returns a new distributed TransposedComplexField.
        `"ndarray"` returns the local block as a NumPy array on each rank.

    Returns
    -------
    pmesh.pm.TransposedComplexField or ndarray
        Inverted result with the same distributed layout as the input when
        `return_type="field"`, otherwise the local NumPy block.

    Notes
    -----
    For a TransposedComplexField, x is fully local on every rank while y and z
    are distributed. The implementation therefore:
    1. exchanges the required y/z blocks between ranks
    2. assembles the y/z-inverted field locally
    3. applies the x-axis inversion without MPI
    """
    if return_type not in {"field", "ndarray"}:
        raise ValueError("return_type must be 'field' or 'ndarray'")

    comm = field.pm.comm
    rank = comm.Get_rank()
    local_f = np.asarray(field)
    local_slices = field.slices

    # Recover the global y/z decomposition directly from the pmesh field layout.
    nx = local_f.shape[0]
    y_splits, z_splits, rank_order = _infer_transposed_complex_layout(
        comm, local_slices
    )
    reordered_rank = rank_order.index(rank)
    global_shape = (nx, y_splits[-1][1], z_splits[-1][1])

    # Build the communication plan for this global layout.
    recv_all, send_all = _build_transposed_complex_plans(
        global_shape, y_splits, z_splits
    )
    recv_entries = recv_all[reordered_rank]
    send_entries = send_all[reordered_rank]

    # The plan uses a rank-local block id. Fold that into the MPI tag so the
    # same source/destination rank pair can exchange multiple blocks safely.
    tag_stride = (len(y_splits) + 2) * (len(z_splits) + 2)

    # Step 1: post receives for all remote y/z blocks needed by this rank.
    #
    # Buffer shape is `(nx, local_y, local_z)` because x is fully local and
    # never partitioned across ranks for this field layout.
    recv_reqs = []
    recv_meta = []
    for (
        src_rank_reordered,
        pair_block_id,
        dy0,
        dy1,
        dz0,
        dz1,
        sy0,
        sy1,
        sz0,
        sz1,
        rev_y,
        rev_z,
    ) in recv_entries:
        actual_src = rank_order[src_rank_reordered]
        if actual_src == rank:
            continue
        buf = np.empty((nx, sy1 - sy0, sz1 - sz0), dtype=local_f.dtype)
        req = comm.Irecv(
            buf, source=actual_src, tag=actual_src * tag_stride + pair_block_id
        )
        recv_reqs.append(req)
        recv_meta.append((buf, dy0, dy1, dz0, dz1))

    # Step 2: send local y/z blocks, reversing y/z locally when required.
    #
    # We reverse before sending so the receiver can place the block directly
    # into its destination slice without any further reordering.
    send_reqs = []
    for (
        dst_rank_reordered,
        pair_block_id,
        dy0,
        dy1,
        dz0,
        dz1,
        sy0,
        sy1,
        sz0,
        sz1,
        rev_y,
        rev_z,
    ) in send_entries:
        actual_dst = rank_order[dst_rank_reordered]
        if actual_dst == rank:
            continue
        view = local_f[:, sy0:sy1, sz0:sz1]
        if rev_y:
            view = view[:, ::-1, :]
        if rev_z:
            view = view[:, :, ::-1]
        # MPI send buffers should be contiguous even if the local view is not.
        buf = np.ascontiguousarray(view)
        req = comm.Isend(buf, dest=actual_dst, tag=rank * tag_stride + pair_block_id)
        send_reqs.append(req)

    # Step 3: assemble the y/z-inverted field using local self-copies first.
    #
    # Local self-copies go through exactly the same plan structure as remote
    # transfers. That keeps the logic uniform and easier to debug.
    yz_inverted = np.empty_like(local_f)
    for (
        src_rank_reordered,
        pair_block_id,
        dy0,
        dy1,
        dz0,
        dz1,
        sy0,
        sy1,
        sz0,
        sz1,
        rev_y,
        rev_z,
    ) in recv_entries:
        actual_src = rank_order[src_rank_reordered]
        if actual_src != rank:
            continue
        view = local_f[:, sy0:sy1, sz0:sz1]
        if rev_y:
            view = view[:, ::-1, :]
        if rev_z:
            view = view[:, :, ::-1]
        yz_inverted[:, dy0:dy1, dz0:dz1] = view

    # Step 4: wait for remote blocks and place them into their destination slices.
    #
    # Once all remote transfers have completed, the y/z-inverted field is fully
    # assembled on this rank and has the same distributed layout as the input.
    if recv_reqs:
        MPI.Request.Waitall(recv_reqs)
    for buf, dy0, dy1, dz0, dz1 in recv_meta:
        yz_inverted[:, dy0:dy1, dz0:dz1] = buf

    # Step 5: x is fully local for this layout, so invert x without MPI.
    #
    # This is the last missing part of the global inversion. Index 0 stays
    # fixed and the remaining x modes are reversed locally.
    local_arr = np.empty_like(local_f)
    local_arr[0, :, :] = yz_inverted[0, :, :]
    local_arr[1:, :, :] = np.flip(yz_inverted[1:, :, :], axis=0)

    if send_reqs:
        MPI.Request.Waitall(send_reqs)

    if return_type == "ndarray":
        return local_arr

    # Rebuild a pmesh field object with the same distributed layout as the
    # input field and copy the local inverted block into it.
    out = field.pm.create(type="complex")
    out[...] = local_arr
    return out
