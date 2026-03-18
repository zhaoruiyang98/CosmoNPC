# Configuration for the analysis of galaxy clustering

CONFIG = {
    # Statistic
    "statistic": "pk",  # ["pk", "bk_sco", "bk_sugi"]: Type of statistics to compute

    # Correlation
    "correlation_mode": "auto",  # ["auto", "cross"]: Auto or cross correlation

    # Geometry
    "geometry": "survey-like",  # "survey-like" or "box-like"

    # Catalogs
    "catalogs": {
        "data_a": "/Users/xieyunchen/Downloads/BOSS_fits/galaxy_DR12v5_LOWZ_South.fits",
        "randoms_a": "/Users/xieyunchen/Downloads/BOSS_fits/random0_DR12v5_LOWZ_South.fits",
        "data_b": "/Users/xieyunchen/Downloads/BOSS_fits/galaxy_DR12v5_LOWZ_South.fits",
        "randoms_b": "/Users/xieyunchen/Downloads/BOSS_fits/random0_DR12v5_LOWZ_South.fits",
    },

    # Column names, VERY IMPORTANT
    # "column_names": ["x", "y", "z"],  # For box-like catalog; "w" is optional
    # Warning: for box-like catalog, the order of columns here must match that in the .npy files

    "column_names": ["RA", "DEC", "Z", "WEIGHT_FKP", "WEIGHT_SYSTOT", "WEIGHT_NOZ","WEIGHT_CP","NZ"], # For survey-like catalog

    # RSD
    # "rsd": [0, 0, 1],  # Unit 3-vector for redshift space distortion (RSD), box-like only

    # Mesh
    # nmesh: Grid points (nx, ny, nz)
    "nmesh": [256, 256, 256],
    # boxsize: Grid size (Mpc/h)
    "boxsize": [1800., 1800., 1800.],
    # sampler: Mesh sampling: "tsc", "cic", "pcs"
    "sampler": "tsc",
    # interlaced: Interlaced mesh sampling
    "interlaced": False,

    # Para_task
    # k_min: Min k (h/Mpc)
    "k_min": 0.0,
    # k_max: Max k (h/Mpc)
    "k_max": 0.3,
    # k_bins: k-space bins
    "k_bins": 30,
    # poles: Multipole orders
    "poles": [0,2,4],
    # compensation: Mesh compensation
    "compensation": True,
    # normalization_scheme: Pk normalization
    "normalization_scheme": "particle",  # "particle" or "mixed-mesh"
    # alpha_scheme: how to compute alpha (pypower or nbodykit)
    "alpha_scheme": "nbodykit",
    # "fast_estimation_mode": "closed", # "off", "replace", "coexist", default "closed"
    # "fast_estimation_mode": "closed",

    # Para_cosmo (for .npy/box-like catalogs, cosmology parameters not used)
    "cosmology": {
        "h": 0.676,           # Hubble parameter
        "Omega0": 0.31,     # Matter density
        "Omega0_b": 0.048,     # Baryon density
        "Omega0_cdm": 0.259115,# CDM density
        "n_s": 0.96,           # Scalar spectral index
        "sigma8": 0.8288,      # Matter fluctuation amplitude
    },

    # Redshift range
    "z_range": [0.15, 0.43],

    # Comp_weight_plan
    # scheme: None, "boss", "eboss" or "desi", this is important for correctly reading the necessary columns
    # name_alias: Weight column alias
    "comp_weight_plan": {
        "scheme": "boss",
        "name_alias": None,
    },

    # Output
    # output_dir: Output directory
    "output_dir": "/Users/xieyunchen/Library/CloudStorage/OneDrive-个人/jupyter files/2025/05Mar/CosmoNPC/results/pk_alpha_compare/nbodykit",
}
