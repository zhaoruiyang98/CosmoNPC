# Configuration for the analysis of galaxy clustering

CONFIG = {
    # Statistic
    "statistic": "pk",  # ["pk", "bk_sco", "bk_sugi"]: Type of statistics to compute

    # Correlation
    "correlation_mode": "auto",  # ["auto", "cross"]: Auto or cross correlation

    # Geometry
    "geometry": "box-like",  # "survey-like" or "box-like"

    # Catalogs
    "catalogs": {
        "data_a": "/Users/xieyunchen/Downloads/abacus_HF_ELG_0p950_DR2_v2.0_AbacusSummit_base_c000_ph000_base_conf_nfwexp_clustering.dat.h5",
        "randoms_a": None,
        "data_b": None,
        "randoms_b": None,
    },

    # Column names
    # "column_names": ["x", "y", "z"],  # For box-like catalog; "w" is optional
    # Warning: for box-like catalog, the order of columns here must match that in the .npy files
    "column_names": ["X", "Y", "Z", "VX", "VY", "VZ"],  # For box-like catalog with RSD; velocity columns optional
    # "column_names": ["x", "y", "z", "w_comp", "w_fkp", "nz"], # For survey-like catalog

    # RSD
    "rsd": [0, 0, 1],  # Unit 3-vector for redshift space distortion (RSD), box-like only
    "apply_rsd": True,  # Whether to apply RSD, box-like only


    # Mesh
    # nmesh: Grid points (nx, ny, nz)
    "nmesh": [256, 256, 256],
    # boxsize: Grid size (Mpc/h)
    "boxsize": [2000., 2000., 2000.],
    # sampler: Mesh sampling: "tsc", "cic", "pcs"
    "sampler": "tsc",
    # interlaced: Interlaced mesh sampling
    "interlaced": True,

    # Para_task
    # k_min: Min k (h/Mpc)
    "k_min": 0.0,
    # k_max: Max k (h/Mpc)
    "k_max": 0.3,
    # k_bins: k-space bins
    "k_bins": 30,
    # poles: Multipole orders
    "poles": [0, 2],
    # compensation: Mesh compensation
    "compensation": True,
    # normalization_scheme: Pk normalization, particle or mixed-mesh
    "normalization_scheme": "particle",


    # Para_cosmo (for .npy/box-like catalogs, cosmology parameters not used)
    "cosmology": {
        "h": 0.6766,           # Hubble parameter
        "Omega0": 0.30966,     # Matter density
        # "Omega_b": 0.048     # Baryon density
    },

    # Redshift range
    # "z_range": [0.15, 0.43], # redshift range for survey-like catalog
    "redshift_box": 0.95,  # redshift for RSD calculation in box-like catalog

    # Comp_weight_plan
    # scheme: "boss", "eboss" or "desi"
    # "scheme": "boss",
    # name_alias: Weight column alias
    # "name_alias": "sssssss",

    # Output
    # output_dir: Output directory
    "output_dir": "/Users/xieyunchen/Downloads/res_pk_output",
}
