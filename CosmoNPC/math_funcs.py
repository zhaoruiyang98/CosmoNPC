import numpy as np
import sympy as sp
import numexpr
import gc

"""
Get the distance of every mesh grid to the mesh-center in k-space for radial binning
as well as get the normalized position of every mesh grid in k/x-space for spherical harmonics
"""
# def get_kgrid(cfield):
#     kgrid = [np.real(kk) for kk in cfield.x]
#     knorm = np.sqrt(sum(np.real(kk)**2 for kk in cfield.x))
#     knorm[knorm == 0.] = np.inf
#     normalized_kgrid = []
#     for k in kgrid:
#         normalized_kgrid.append(k / knorm)
#     # reset knorm, honestly, it's stupid...
#     # I have some new idea to avoid this but i'm lazy to validate it
#     knorm[knorm == np.inf] = 0.
#     return normalized_kgrid, knorm

def get_kgrid(cfield):
    kgrid = [np.real(kk) for kk in cfield.x]
    knorm = np.sqrt(sum(np.real(kk)**2 for kk in cfield.x))
    normalized_kgrid = []
    for k in kgrid:
        normalized_kgrid.append(k / knorm)
    if knorm[0,0,0] == 0.:
        for k in normalized_kgrid:
            k[0,0,0] = 0.
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
    """
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
    l = int(l); m = int(m)
    if abs(m) > l:
        raise ValueError("abs(m) must be <= l")

    # the relevant cartesian and spherical symbols
    x, y, z  = sp.symbols('x y z ', real=True)
    r = sp.symbols('r', real = True, positive=True)
    xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True)
    phi, theta = sp.symbols('phi theta')

    # Define substitutions for spherical coordinates
    defs = [(sp.sin(phi), y/sp.sqrt(x**2 + y**2)),
            (sp.cos(phi), x/sp.sqrt(x**2 + y**2)),
            (sp.cos(theta), z/sp.sqrt(x**2 + y**2 + z**2))]

    # the normalization factor
    amp = sp.sqrt((2*l + 1)/(4 * np.pi) * sp.factorial(l - abs(m))/sp.factorial(l + abs(m)))

    # the cos(theta) dependence encoded by the associated Legendre poly
    # Note: sympy's assoc_legendre already contains the Condon-Shortley phase (-1)^m
    # https://docs.sympy.org/latest/modules/functions/special.html
    expr = sp.assoc_legendre(l, abs(m), sp.cos(theta))

    # Explicitly handle phi dependence: e^(I*m*phi) = (cos(phi) + I*sin(phi))^m
    phi_term = (sp.cos(phi) + sp.I * sp.sin(phi))**abs(m)

    # Combine theta and phi dependence
    expr *= phi_term

    # Apply substitutions and simplify
    expr = expr.subs(defs)
    expr = sp.together(expr).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x/r, xhat), (y/r, yhat), (z/r, zhat)])

    # apply Racah normalization if requested
    if Racah_normalized:
        expr = expr * sp.sqrt(4 * np.pi / (2 * l + 1))

    # For negative m, take complex conjugate and multiply by (-1)^m
    if m < 0:
        expr = (-1)**m * sp.conjugate(expr)

    # Further simplify to ensure no symbolic variables remain
    expr = sp.simplify(expr)

    # Convert to callable function using numexpr
    Ylm = sp.lambdify((xhat, yhat, zhat), expr, 'numexpr')

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
    k_xhat, k_yhat, k_zhat = sp.symbols('k_xhat k_yhat k_zhat', real=True)
    cos_theta = sp.Symbol('cos_theta', real=True)

    # Compute the cosine of the angle using the dot product
    expr = r_xhat * k_xhat + r_yhat * k_yhat + r_zhat * k_zhat

    # Compute the Legendre polynomial P_ell(cos_theta)
    legendre_expr = sp.legendre(ell, cos_theta)

    # Substitute cos_theta with the dot product expression
    expr = legendre_expr.subs(cos_theta, expr)


    # Simplify the expression
    expr = sp.simplify(expr)

    # Convert to a callable function using numexpr
    P_ell = sp.lambdify((k_xhat, k_yhat, k_zhat), expr, 'numexpr')

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
    """
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
        tmp[wi == 0.] = 1.
        v = v / tmp
    return v

def CompensateNGP(w, v):
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi)) 
        tmp[wi == 0.] = 1.
        v = v / tmp
    return v

def CompensateTSCShotnoise(w, v):
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi)**2
        v = v / (1 - s + 2./15 * s**2) ** 0.5
    return v

def CompensatePCSShotnoise(w, v):
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi)**2
        v = v / (1 - 4./3. * s + 2./5. * s**2 - 4./315. * s**3) ** 0.5
    return v

def CompensateCICShotnoise(w, v):
    for i in range(3):
        wi = w[i]
        v = v / (1 - 2. / 3 * np.sin(0.5 * wi) ** 2) ** 0.5
    return v

def CompensateNGPShotnoise(w, v):
    return v


def Compensate_bk_noise_tsc(w,v):
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi)**2
        tmp = (np.sinc(0.5 * wi / np.pi)) ** 3
        tmp[wi == 0.] = 1.
        v = v * (1 - s + 2./15 * s**2) / tmp**2
    return v

def Compensate_bk_noise_cic(w,v):
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi)**2
        tmp = (np.sinc(0.5 * wi / np.pi)) ** 2
        tmp[wi == 0.] = 1.
        v = v * (1 - 2./3 * s) / tmp**2
    return v

def Compensate_bk_noise_ngp(w,v):
    return v


def Compensate_bk_noise_pcs(w,v):
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi)**2
        tmp = (np.sinc(0.5 * wi / np.pi)) ** 4
        tmp[wi == 0.] = 1.
        v = v * (1 - 4./3. * s + 2./5. * s**2 - 4./315. * s**3) / tmp**2
    return v
        
        
