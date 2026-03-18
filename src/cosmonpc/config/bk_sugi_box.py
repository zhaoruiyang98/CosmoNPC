# Configuration for the analysis of galaxy clustering

CONFIG = {
    # Statistic
    "statistic": "bk_sugi",  # ["pk", "bk_sco", "bk_sugi"]: Type of statistics to compute

    # Correlation
    "correlation_mode": "auto",  # ["auto" or "cross"]: Auto or cross correlation 

    # Tracer type
    "tracer_type": "aaa",  # ["aaa","aab","abb","abc"] 

    # Geometry
    "geometry": "box-like",  # "survey-like" or "box-like"

    # Catalogs, maximum 3 tracers for cross-correlation
    "catalogs": {
        "data_a": "/Users/xieyunchen/Downloads/abacus_HF_ELG_0p950_DR2_v2.0_AbacusSummit_base_c000_ph000_base_conf_nfwexp_clustering.dat.h5",
        "randoms_a": None,
        "data_b":"/Users/xieyunchen/Downloads/abacus_HF_ELG_0p950_DR2_v2.0_AbacusSummit_base_c000_ph000_base_conf_nfwexp_clustering.dat.h5",
        "randoms_b": None,
        "data_c": "/Users/xieyunchen/Downloads/abacus_HF_ELG_0p950_DR2_v2.0_AbacusSummit_base_c000_ph000_base_conf_nfwexp_clustering.dat.h5",
        "randoms_c": None,
    },

    # Column names
    # "column_names": ["x", "y", "z"],  # For box-like catalog; "w" is optional
    # Warning: for box-like catalog, the order of columns here must match that in the .npy files
    "column_names": ["X", "Y", "Z", "VX", "VY", "VZ"],  # For box-like catalog with RSD; velocity columns optional


    # "column_names": ["x", "y", "z", "w_comp", "w_fkp", "nz"], # For survey-like catalog

    # RSD, only for box-like catalog
    "rsd": [0, 0, 1],  # Unit 3-vector for redshift space distortion (RSD), box-like only
    "apply_rsd": True,  # Whether to apply RSD, box-like only

    # Mesh
    # nmesh: Grid points (nx, ny, nz)
    "nmesh": [256, 256, 256],
    # "nmesh": [512, 512, 512],
    
    # boxsize: Grid size (Mpc/h)
    "boxsize": [2000., 2000., 2000.],
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
    "angu_config": [0, 0, 0], # for bk_sugi, this parameter indicates one single angular momenta configuration l_1,l_2,L
    # compensation: Mesh compensation
    "compensation": True,

    "data_vector_mode": "full", #"diagonal" or "full"
    "block_size": "full", # 1,"full" or an integer between 1 and k_bins 
                        # (only for bk_sugi with data_vector_mode="full")

    # normalization_scheme: Pk normalization, particle or mixed-mesh
    "normalization_scheme": "particle", 

    # Para_cosmo (for .npy/box-like catalogs, it can be used to apply RSD)
    "cosmology": { #Planck18 parameters with TT,TE,EE+lowE+lensing
        "h": 0.6736,           # Hubble parameter
        "Omega0": 0.3153 ,     # Matter density
        # "Omega_b": 0.048     # Baryon density
    },

    
    # "z_range": [0.15, 0.43], # Redshift range, only for survey-like catalog
    "redshift_box": 0.95,  # redshift for RSD calculation in box-like catalog

    # Comp_weight_plan, only for survey-like catalog
    # scheme: "boss", "eboss" or "desi"
    # "scheme": "boss",
    # name_alias: Weight column alias
    # "name_alias": "sssssss",

    # Output
    # output_dir: Output directory
    "output_dir": "/Users/xieyunchen/Downloads/res_pk_output",
}
