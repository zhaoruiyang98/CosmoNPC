import os
import yaml
import numpy as np
from mpi4py import MPI
from CosmoNPC import run_task
from CosmoNPC.config_bk_sugi import CONFIG
# from CosmoNPC.config_bk_sugi_cross import CONFIG
# from CosmoNPC.config_pk import CONFIG
# from CosmoNPC.config_pk_survey import CONFIG





# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def main():
    # check the configuration and create output directory
    validate_config(CONFIG)
    catalog_check(CONFIG['catalogs'], CONFIG['geometry'], \
                  CONFIG['correlation_mode'], CONFIG['statistic'])
    if rank == 0:
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        
    run_task(**CONFIG)



def validate_config(config):
    assert config['sampler'] in ['ngp', 'cic', 'pcs', 'tsc'], \
        "sampler should be 'ngp', 'cic', 'pcs', or 'tsc'"
    Cubic_Check(config['nmesh'], "nmesh", int)
    Cubic_Check(config['boxsize'], "boxsize", (float, int))

# Function to check the catalogs based on geometry and correlation mode
def catalog_check(catalogs, geometry, correlation_mode, statistic, tracer_type=None):
    assert catalogs['data_a'] is not None, "data_a catalog must be provided"
    if statistic in ['pk', 'bk_sco', 'bk_sugi']:
        if correlation_mode == "auto":
            if geometry == "survey-like":
                assert catalogs['randoms_a'] is not None, \
                    "randoms_a catalog must be provided for survey-like auto-correlation"
        elif correlation_mode == "cross":
            assert catalogs['data_b'] is not None, "data_b catalog must be provided for cross-correlation"
            if geometry == "survey-like":
                assert catalogs['randoms_a'] is not None, \
                    "randoms_a catalog must be provided for survey-like cross-correlation"
                assert catalogs['randoms_b'] is not None, \
                    "randoms_b catalog must be provided for survey-like cross-correlation"
    # Extra check for tracer_type == "abc" and statistic is bispectrum
    if tracer_type == "abc" and statistic in ['bk_sco', 'bk_sugi']:
        assert catalogs.get('data_c') is not None, "data_c catalog must be provided for tracer_type 'abc' in bispectrum"
        if geometry == "survey-like":
            assert catalogs.get('randoms_c') is not None, \
                "randoms_c catalog must be provided for tracer_type 'abc' in survey-like bispectrum"

# Function to check if a value is a cubic number
def Cubic_Check(value, value_name, value_type):
    assert isinstance(value, (list, tuple)) and len(value) == 3, \
        f"{value_name} must be a list or tuple of three elements"
    assert all(isinstance(x, value_type) and x > 0 for x in value), \
        f"All elements in {value_name} must be positive {value_type.__name__}s"
    assert len(set(value)) == 1, \
        f"All elements in {value_name} must be equal"

if __name__ == '__main__':
    main()

