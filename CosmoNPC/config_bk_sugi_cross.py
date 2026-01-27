# Configuration for the analysis of galaxy clustering

CONFIG = {
    # Statistic
    "statistic": "bk_sugi",  # ["pk", "bk_sco", "bk_sugi"]: Type of statistics to compute

    # Correlation
    "correlation_mode": "cross",  # ["auto" or "cross"]: Auto or cross correlation 

    # Tracer type
    "tracer_type": "abc",  # ["aaa","aab","abb","abc"] 

    # Geometry
    "geometry": "box-like",  # "survey-like" or "box-like"

    # Catalogs, maximum 3 tracers for cross-correlation
    "catalogs": {
        "data_a": "/Users/xieyunchen/Downloads/cut_fits/molino.z0.0.fiducial.nbody1.hod0.npy",
        "randoms_a": None,
        "data_b": "/Users/xieyunchen/Downloads/cut_fits/molino.z0.0.fiducial.nbody1.hod0.npy",
        "randoms_b": None,
        "data_c": "/Users/xieyunchen/Downloads/cut_fits/molino.z0.0.fiducial.nbody1.hod0.npy",
        "randoms_c": None,
    },

    # Column names
    "column_names": ["x", "y", "z"],  # For box-like catalog; "w" is optional
    # Warning: for box-like catalog, the order of columns here must match that in the .npy files

    # "column_names": ["x", "y", "z", "w_comp", "w_fkp", "nz"], # For survey-like catalog

    # RSD, only for box-like catalog
    # "rsd": [0, 0, 1],  # Unit 3-vector for redshift space distortion (RSD), box-like only

    # Mesh
    # nmesh: Grid points (nx, ny, nz)
    "nmesh": [256, 256, 256],
    # boxsize: Grid size (Mpc/h)
    "boxsize": [1000., 1000., 1000.],
    # sampler: Mesh sampling: "tsc", "cic", "pcs"
    "sampler": "tsc",
    # interlaced: Interlaced mesh sampling, for now this operation won't actually be performed only if I test its reliability
    "interlaced": False,

    # Para_task
    # k_min: Min k (h/Mpc)
    "k_min": 0.0,
    # k_max: Max k (h/Mpc)
    "k_max": 0.2,
    # k_bins: k-space bins
    "k_bins": 20,
    # poles: Multipole orders
    "angu_config": [2, 2, 0], # for bk_sugi, this parameter indicates one single angular momenta configuration l_1,l_2,L
    # compensation: Mesh compensation
    "compensation": True,

    "data_vector_mode": "diagonal", #"diagonal" or "full"
    # normalization_scheme: Pk normalization, particle or mixed-mesh
    "normalization_scheme": "particle", # only work for survey-like measurement

    # Para_cosmo (for .npy/box-like catalogs, cosmology parameters not used)
    "cosmology": {
        "h": 0.6777,           # Hubble parameter
        "Omega0": 0.31377,     # Matter density
        "Omega_b": 0.048     # Baryon density
    },

    # Redshift range, only for survey-like catalog
    "z_range": [0.15, 0.43],

    # Comp_weight_plan, only for survey-like catalog
    # scheme: "boss", "eboss" or "desi"
    "scheme": "boss",
    # name_alias: Weight column alias
    # "name_alias": "sssssss",

    # Output
    # output_dir: Output directory
    "output_dir": "/Users/xieyunchen/Downloads/res_pk_output",
}
