# Configuration for the analysis of galaxy clustering

CONFIG = {
    # Statistic
    "statistic": "bk_sugi",  # ["pk", "bk_sco", "bk_sugi"]: Type of statistics to compute

    # Correlation
    "correlation_mode": "auto",  # ["auto" or "cross"]: Auto or cross correlation 

    # Tracer type
    "tracer_type": "aaa",  # ["aaa","aab","abb","abc"] 

    # Geometry
    "geometry": "survey-like",  # "survey-like" or "box-like"

    # Catalogs, maximum 3 tracers for cross-correlation
    "catalogs": {
        "data_a": "/Users/xieyunchen/Downloads/BOSS_fits/galaxy_DR12v5_LOWZ_South.fits",
        "randoms_a": "/Users/xieyunchen/Downloads/BOSS_fits/random0_DR12v5_LOWZ_South.fits",
        "data_b": "/Users/xieyunchen/Downloads/BOSS_fits/galaxy_DR12v5_LOWZ_South.fits",
        "randoms_b": "/Users/xieyunchen/Downloads/BOSS_fits/random0_DR12v5_LOWZ_South.fits",
        "data_c": "/Users/xieyunchen/Downloads/BOSS_fits/galaxy_DR12v5_LOWZ_South.fits",
        "randoms_c": "/Users/xieyunchen/Downloads/BOSS_fits/random0_DR12v5_LOWZ_South.fits",
    },

    # Column names
    # "column_names": ["x", "y", "z"],  # For box-like catalog; "w" is optional
    # Warning: for box-like catalog, the order of columns here must match that in the .npy files
    "column_names": ["RA", "DEC", "Z", "WEIGHT_FKP", "WEIGHT_SYSTOT", "WEIGHT_NOZ","WEIGHT_CP","NZ"],  


    # "column_names": ["x", "y", "z", "w_comp", "w_fkp", "nz"], # For survey-like catalog

    # RSD, only for box-like catalog
    # "rsd": [0, 0, 1],  # Unit 3-vector for redshift space distortion (RSD), box-like only
    # "apply_rsd": True,  # Whether to apply RSD, box-like only

    # Mesh
    # nmesh: Grid points (nx, ny, nz)
    "nmesh": [256, 256, 256],
    # "nmesh": [512, 512, 512],
    
    # boxsize: Grid size (Mpc/h)
    "boxsize": [1800., 1800., 1800.],
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
    "k_bins": 10,
    # poles: Multipole orders
    "angu_config": [2, 0, 2], # for bk_sugi, this parameter indicates one single angular momenta configuration l_1,l_2,L
    # compensation: Mesh compensation
    "compensation": True,

    "data_vector_mode": "diagonal", #"diagonal" or "full"
    "block_size": "full", # 1,"full" or an integer between 1 and k_bins 
                        # (only for bk_sugi with data_vector_mode="full")

    # normalization_scheme: bk normalization, particle, mesh or mixed-mesh
    "normalization_scheme": "mesh", # only work for survey-like measurement

    # alpha_scheme: how to compute alpha (pypower or nbodykit)
    "alpha_scheme": "nbodykit",

    # Para_cosmo (for .npy/box-like catalogs, it can be used to apply RSD)
    "cosmology": { #Planck18 parameters with TT,TE,EE+lowE+lensing
        "h": 0.6736,           # Hubble parameter
        "Omega0": 0.3153 ,     # Matter density
        # "Omega_b": 0.048     # Baryon density
    },

    
    "z_range": [0.15, 0.43], # Redshift range, only for survey-like catalog
    # "redshift_box": 0.95,  # redshift for RSD calculation in box-like catalog

    # Comp_weight_plan
    # scheme: None, "boss", "eboss" or "desi", this is important for correctly reading the necessary columns
    # name_alias: Weight column alias
    "comp_weight_plan": {
        "scheme": "boss",
        "name_alias": None,
    },

    # Output
    # output_dir: Output directory
    "output_dir": "/Users/xieyunchen/Library/CloudStorage/OneDrive-个人/jupyter files/2025/05Mar/CosmoNPC/results/bk_alpha_compare/nbodykit",
}
