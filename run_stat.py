import os
import yaml
import numpy as np
from mpi4py import MPI
from CosmoNPC import run_task

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def main():
    config = load_config('config_box_npy.yaml')
    
    geometry = config['geometry']
    column_names = config['column_names']
    tasks = config['tasks']
    catalogs = config['catalogs']
    rsd = config['rsd']
    nmesh = config['mesh']['nmesh']
    sampler = config['mesh']['sampler']
    interlaced = config['mesh']['interlaced']
    boxsize = config['mesh']['boxsize']
    statistic = config['statistic']
    para_cosmo = config['para_cosmo']
    z_range = config['z_range']
    comp_weight_plan = config['comp_weight_plan']
    output_dir = config['output']['directory']
    
    
    validate_config(config)

    

    create_output_dir(output_dir)

    task = next((t for t in tasks if t['name'] == statistic), None)
    if task is None:
        print(f"Rank {rank}: No task found for statistic '{statistic}'")
    else:
        task_name = task['name']
        task_params = {k: v for k, v in task.items() if k != 'name'}
        result = run_task(
            task_name,
            geometry=geometry,
            catalogs=catalogs,
            rsd=rsd,
            column_names=column_names,
            nmesh=nmesh,
            boxsize=boxsize,
            sampler=sampler,
            interlaced=interlaced,
            comp_weight_plan=comp_weight_plan,
            z_range=z_range,
            para_cosmo=para_cosmo,
            output_dir=output_dir,
            **task_params
        )


# Load the YAML configuration file
def load_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def validate_config(config):
    assert config['mesh']['sampler'] in ['cic', 'pcs', 'tsc'], \
        "sampler should be 'cic', 'pcs', or 'tsc'"
    Cubic_Check(config['mesh']['nmesh'], "nmesh", int)
    Cubic_Check(config['mesh']['boxsize'], "boxsize", (float, int))

def create_output_dir(output_dir):
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)


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