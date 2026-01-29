import numpy as np
from mpi4py import MPI
import logging
import time 

from .mesh_generator import *
from .stat_algorithm import *
import os


def run_task(statistic,correlation_mode, geometry,catalogs,**kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')

    if rank == 0:
        logging.info(
            f"Running {correlation_mode} {statistic} task with catalogs:{catalogs},\n"
        )

    k_max, nmesh, boxsize = kwargs['k_max'], kwargs['nmesh'], kwargs['boxsize']
    nyquist_freq = np.pi * nmesh[0] / boxsize[0]
    if rank == 0:
        logging.info(f"Nyquist frequency: {nyquist_freq}")

    time_start = time.time()

    if statistic == "pk":
        # check if the kmax is larger than the Nyquist frequency
        if k_max >= nyquist_freq:
            raise ValueError(f"k_max {k_max} is larger than the Nyquist frequency {nyquist_freq}. \
                                Please choose a smaller k_max.")
        
        # validate poles configuration
        validate_poles(kwargs['poles'])

        if geometry == "survey-like":
            if rank == 0:
                logging.info("Using survey-like geometry...")

            stat_attrs, rfield_a, rfield_b = get_mesh_pk_survey(catalogs, 
                        correlation_mode, 
                        nmesh=nmesh, 
                        geometry=geometry,
                        column_names=kwargs['column_names'],
                        boxsize=boxsize, 
                        sampler=kwargs['sampler'], 
                        interlaced=kwargs['interlaced'],
                        z_range=kwargs['z_range'],
                        comp_weight_plan=kwargs['comp_weight_plan'],
                        para_cosmo=kwargs['cosmology'],
                        normalization_scheme=kwargs['normalization_scheme'],
                        comm=comm)
        else:
            if rank == 0:
                logging.info("Using box geometry...")
            # check if all pole in poles are odd, if true, raise an error. Because for box geometry, odd poles are defaultly set to 0
            poles = kwargs['poles']
            if all(p % 2 == 1 for p in poles):
                raise ValueError(f"All poles in {poles} are odd. Please choose at least one even pole, \
                                 since odd poles are defaultly set to 0 for periodic boxes.")

            stat_attrs, rfield_a, rfield_b = get_mesh_box(catalogs,
                        correlation_mode,
                        nmesh=kwargs['nmesh'],
                        geometry=geometry, 
                        boxsize=kwargs['boxsize'], 
                        sampler=kwargs['sampler'], 
                        interlaced=kwargs['interlaced'],
                        column_names=kwargs['column_names'],
                        comm=comm,
                        statistic = statistic,
                        apply_rsd = kwargs['apply_rsd'],
                        para_cosmo=kwargs['cosmology'],
                        redshift_box=kwargs['redshift_box'],
                        los=kwargs['rsd'],
                        )
        
        # add more information
        stat_attrs.update(kwargs)
        stat_attrs['nyquist_freq'] = nyquist_freq

        # make sure all ranks have finished before performing FFTs
        comm.Barrier()
        time_rfield = time.time()
        if rank == 0:
            logging.info(f"Time to create (FKP) overdensity field(s): {time_rfield - time_start:.2f} seconds")
            logging.info(f"{'$' * 60} Start to compute power spectrum. {'$' * 60}")
        
        # Compute power spectrum
        if geometry == "survey-like":
            pk_res = calculate_power_spectrum_survey(stat_attrs, rfield_a, rfield_b,correlation_mode,\
                                                      comm = comm, **kwargs)
        else:
            pk_res = calculate_power_spectrum_box(rfield_a, rfield_b, correlation_mode,\
                                                   stat_attrs, comm = comm, **kwargs)

        # Save the power spectrum results
        if rank == 0:
            pk_res.update(stat_attrs)
            # delete unnecessary keys and values in pk_res
            if geometry == "box-like":
                keys_to_remove = ['z_range', "scheme"]
                for key in keys_to_remove:
                    if key in pk_res:
                        del pk_res[key]
            logging.info(f"Power spectrum result: {pk_res}")
            # save the result to a file
            output_dir = kwargs.get('output_dir')
            catalog_name_a = os.path.splitext(os.path.basename(catalogs['data_a']))[0]
            if correlation_mode == "cross":
                catalog_name_b = os.path.splitext(os.path.basename(catalogs['data_b']))[0]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if correlation_mode == "auto":
                output_path = os.path.join(output_dir, f"pk_res_{correlation_mode}_{catalog_name_a}.npy")
            elif correlation_mode == "cross":
                output_path = os.path.join(output_dir, f"pk_res_{correlation_mode}_{catalog_name_a}_{catalog_name_b}.npy")
            np.save(output_path, pk_res)
            logging.info(f"Power spectrum result saved to {output_path}")
        

        time_pk = time.time()
        if rank == 0:
            logging.info(f"Time to compute power spectrum: {time_pk - time_rfield:.2f} seconds")
            logging.info(f"Total time for pk task: {time_pk - time_start:.2f} seconds")


    if statistic == "bk_sugi":
        tracer_type = kwargs['tracer_type']
        angu_config = kwargs['angu_config']

        # make sure the tracer_type and correlation_mode are compatible
        validate_tracer(tracer_type, correlation_mode)


        # check if the 2 * kmax is larger than the Nyquist frequency
        if 2 * k_max >= nyquist_freq:
            raise ValueError(f"2 * k_max = {2 * k_max} is larger than the Nyquist frequency {nyquist_freq}. \
                             Please choose a smaller k_max.")
        
        if geometry == "box-like":
            if rank == 0:
                logging.info("Using box geometry...")
            angu_config = kwargs['angu_config']
            validate_sugi_poles(angu_config, geometry)
            stat_attrs, rfield_a, rfield_b, rfield_c = get_mesh_box(catalogs,
                        correlation_mode,
                        statistic=statistic,
                        nmesh=kwargs['nmesh'],
                        geometry=geometry, 
                        boxsize=kwargs['boxsize'], 
                        sampler=kwargs['sampler'], 
                        interlaced=kwargs['interlaced'],
                        column_names=kwargs['column_names'],
                        comm=comm,
                        tracer_type=tracer_type,
                        apply_rsd = kwargs['apply_rsd'],
                        para_cosmo=kwargs['cosmology'],
                        redshift_box=kwargs['redshift_box'],
                        los=kwargs['rsd'],
                        )
        else:
            raise NotImplementedError("Survey-like geometry for bk_sugi is not implemented yet.")
            
        # add more information
        stat_attrs.update(kwargs)
        stat_attrs['nyquist_freq'] = nyquist_freq

        # make sure all ranks have finished before performing FFTs
        comm.Barrier()
        time_start_rfield = time.time()
        if rank == 0:
            logging.info(f"Time to create overdensity field(s): {time_start_rfield - time_start:.2f} seconds")
            logging.info(f"{'$' * 60} Start to compute bispectrum using Sugiyama estimator. {'$' * 60}")

        time_rfield = time.time()
        if rank == 0:
            logging.info(f"Time to create (FKP) overdensity field(s): {time_rfield - time_start:.2f} seconds")

        # Compute bispectrum using Sugiyama estimator

        if geometry == "box-like":
            bk_res = calculate_bk_sugi_box(rfield_a, rfield_b, rfield_c, correlation_mode,
                                           stat_attrs, comm = comm, **kwargs)
        else:
            raise NotImplementedError("Survey-like geometry for bk_sugi is not implemented yet.")
            

        # Save the bispectrum results
        if rank == 0:
            bk_res.update(stat_attrs)
            # delete unnecessary keys and values in bk_res
            if geometry == "box-like":
                keys_to_remove = ['z_range', 'comp_weight_plan', "scheme"]
                for key in keys_to_remove:
                    if key in bk_res:
                        del bk_res[key]


            logging.info(f"Bispectrum result: {bk_res}")
            # save the result to a file
            output_dir = kwargs.get('output_dir')
            catalog_name_a = os.path.splitext(os.path.basename(catalogs['data_a']))[0]
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"bk_sugi_res_{angu_config}_{correlation_mode}_{tracer_type}_{catalog_name_a}.npy")

            np.save(output_path, bk_res)
            logging.info(f"Bispectrum result saved to {output_path}")

        time_bk = time.time()
        if rank == 0:
            logging.info(f"Time to compute bispectrum: {time_bk - time_rfield:.2f} seconds")
            logging.info(f"Total time for bk_sugi task: {time_bk - time_start:.2f} seconds")
            












