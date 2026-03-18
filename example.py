from copy import deepcopy
from cosmonpc import run_stats
from cosmonpc.config import pk_box, bk_sugi_box, pk_survey, bk_sugi_survey

if __name__ == "__main__":
    # config = deepcopy(pk_box.CONFIG)
    config = deepcopy(bk_sugi_box.CONFIG)
    config["tracer_type"] = "abb"
    config["catalogs"] = {
        "data_a": "data/abacus_HF/DR2_v2.0/AbacusSummit_base_c000_ph000/Boxes/LRG/abacus_HF_LRG_0p950_DR2_v2.0_AbacusSummit_base_c000_ph000_base_clustering.dat.h5",
        "randoms_a": None,
        "data_b": "data/abacus_HF/DR2_v2.0/AbacusSummit_base_c000_ph000/Boxes/ELG/abacus_HF_ELG_0p950_DR2_v2.0_AbacusSummit_base_c000_ph000_base_conf_nfwexp_clustering.dat.h5",
        "randoms_b": None,
    }
    config["redshift_box"] = 0.95
    config["rsd"] = [0, 0, 1]
    config["nmesh"] = [512, 512, 512]
    config["data_vector_mode"] = "diagonal"
    config["output_dir"] = "data/2pt3pt"
    config["correlation_mode"] = "cross"
    run_stats(config)
