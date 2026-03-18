import numpy as np
import h5py
from numpy.lib import format
from mpi4py import MPI
import logging
# from astropy import cosmology
from astropy.cosmology import Planck18
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import warnings
import gc
import fitsio


def h5_reader(comm, files, column_names):
    """
    Read HDF5 files in parallel across all MPI ranks.
    
    Args:
        comm: MPI communicator
        files: List of HDF5 file paths
        column_names: List of dataset names to read (it will not raise an error if some columns are missing, just skip them instead)
        
    Returns:
        numpy.ndarray: Combined data from all files for this rank's row ranges
    """
    rank, size = comm.Get_rank(), comm.Get_size()

    # Initialize empty array with correct dtype
    dtype = [(col, 'f8') for col in column_names]
    result = np.array([], dtype=dtype)
    
    for f in [files] if isinstance(files, str) else files:
        # print(f"Rank {rank} processing file: {f}")
        with h5py.File(f, 'r') as h5f:
            # Get number of rows from the first available column (assume at least one exists)
            # Note: This mimics fits_reader's hdu.get_nrows(), which returns total table rows.
            # We assume all datasets have same length; use first column in column_names that exists.
            nrows = None
            for col in column_names:
                if col in h5f:
                    nrows = len(h5f[col])
                    break
            if nrows is None:
                nrows = 0  # No requested columns present

            if rank == 0:
                logging.info(f"File: {f}, Total Rows: {nrows}, All Columns: {list(h5f.keys())}")

            all_columns = list(h5f.keys())
            columns_to_read = [col for col in column_names if col in all_columns]
            columns_to_read = comm.bcast(columns_to_read, root=0)
            if rank == 0:
                logging.info(f"Columns to read: {columns_to_read}")

            # Calculate row distribution
            base, extra = nrows // size, nrows % size
            start = rank * base + min(rank, extra)
            end = start + base + (1 if rank < extra else 0)
            
            # Read data if this rank has rows
            if start < nrows:
                # Build structured array for this slice
                local_len = end - start
                local_data = np.empty(local_len, dtype=dtype)
                for col in columns_to_read:
                    local_data[col] = h5f[col][start:end]
                
                # Concatenate directly
                result = np.concatenate([result, local_data]) if len(result) > 0 else local_data

    # Log the number of rows read by this rank
    local_row_count = len(result)
    total_row_count = comm.reduce(local_row_count, op=MPI.SUM, root=0)
    
    if rank == 0:
        logging.info(f"Total rows across all ranks: {total_row_count}")
    # Log local row count on every rank (like original fits_reader implicitly does via print/debug)
    logging.info(f"Rank {rank} read {local_row_count} rows")  # or use info if preferred

    return result


def npy_reader(data_path,comm):
    """
    Read npy files in parallel using MPI by automatically handle the file splitting.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        arr = format.open_memmap(data_path, mode='r')
        shape = arr.shape
        dtype = arr.dtype
    else:
        shape = None
        dtype = None

    # Broadcast shape and dtype to all ranks
    shape = comm.bcast(shape, root=0)
    dtype = comm.bcast(dtype, root=0)

    # Calculate the number of rows for each rank
    total_rows = shape[0]
    rows_per_rank = total_rows // size
    remainder = total_rows % size
    start_row = rank * rows_per_rank + min(rank, remainder)
    end_row = start_row + rows_per_rank + (1 if rank < remainder else 0)

    # Each rank reads its own slice of the data
    local_data = format.open_memmap(data_path, mode='r')[start_row:end_row]

    # logging the shape of the data read by each rank
    logging.info(f"Reading .npy data from disk, Rank {rank}: Rows {start_row} to {end_row-1}, shape {local_data.shape}")

    return local_data



def fits_reader(comm, files, column_names):
    """
    Read FITS files in parallel across all MPI ranks.
    
    Args:
        comm: MPI communicator
        files: List of FITS file paths
        column_names: List of column names to read (raises error if any are missing)
        
    Returns:
        numpy.ndarray: Combined data from all files for this rank's row ranges
    """
    rank, size = comm.Get_rank(), comm.Get_size()

    # Initialize empty array with correct dtype
    dtype = [(col, 'f8') for col in column_names]
    result = np.array([], dtype=dtype)
    
    for f in [files] if isinstance(files, str) else files:
        # print(f"Rank {rank} processing file: {f}")
        with fitsio.FITS(f) as fits:
            # Select appropriate HDU (prefer binary table)
            hdu = fits[1] if len(fits) > 1 else fits[0]
            nrows = hdu.get_nrows() 

            if rank == 0:
                logging.info(f"File: {f}, Total Rows: {nrows}, All Columns: {hdu.get_colnames()}")

            all_columns = hdu.get_colnames()
            columns_to_read = [col for col in column_names if col in all_columns]
            columns_to_read = comm.bcast(columns_to_read, root=0)
            if rank == 0:
                logging.info(f"Columns to read: {columns_to_read}")

            # Calculate row distribution
            base, extra = nrows // size, nrows % size
            start = rank * base + min(rank, extra)
            end = start + base + (1 if rank < extra else 0)
            
            # Read data if this rank has rows
            if start < nrows:
                data = hdu.read(rows=range(start, end), columns=columns_to_read)
                
                # Concatenate directly
                result = np.concatenate([result, data]) if len(result) > 0 else data

    # Log the number of rows read by this rank
    local_row_count = len(result)
    total_row_count = comm.reduce(local_row_count, op=MPI.SUM, root=0)
    
    if rank == 0:
        logging.info(f"Total rows across all ranks: {total_row_count}")

    return result


def add_completeness_weight(dr, comp_weight_plan, catalog_type, comm):
    """
    Add or update the 'WEIGHT' column in the catalog array to account for completeness weights.

    Parameters
    ----------
    dr : numpy.recarray
        Input catalog as a record array. May be modified in-place or a new array returned
        if the 'WEIGHT' column is added.
    comp_weight_plan : dict
        Completeness weight plan, must contain:
            - "scheme": str or None. Allowed: None, "boss", "eboss", "desi".
            - "name_alias": str or None. Name of an existing column to use as the weight.
    catalog_type : str
        Catalog type, must be "data" or "randoms".
    comm : MPI.Comm
        MPI communicator for logging.

    Returns
    -------
    dr : numpy.recarray
        Catalog array with the 'WEIGHT' column added or updated.

    Raises
    ------
    AssertionError
        If catalog_type is not "data" or "randoms", or if scheme is not allowed.
    ValueError
        If both scheme and alias are specified, or if alias is not found, or if "desi" scheme is requested.

    Logic
    -----
    - If 'WEIGHT' column exists, use it as is and log this.
    - If 'WEIGHT' column does not exist:
        - Add a new 'WEIGHT' column initialized to zeros.
        - If both scheme and alias are None, set all weights to 1.0 and log this.
        - If alias is given and scheme is None:
            - Use the specified alias column as weights if it exists, else raise ValueError.
        - If alias is None and scheme is given:
            - For "boss" scheme: for "data", compute weights as WEIGHT_SYSTOT * (WEIGHT_NOZ + WEIGHT_CP - 1.0); for "randoms", set to 1.0.
            - For "eboss" scheme: for "data", compute weights as WEIGHT_SYSTOT * WEIGHT_NOZ * WEIGHT_CP; for "randoms", set to 1.0.
            - For "desi" scheme: raise ValueError (to be implemented).
        - If both scheme and alias are given, raise ValueError.
    - Only rank 0 logs info messages.
    """

    rank = comm.Get_rank()

    scheme = comp_weight_plan["scheme"]
    alias = comp_weight_plan["name_alias"]
    assert catalog_type in ["data", "randoms"], "catalog_type must be 'data' or 'randoms'"
    assert scheme in [None, "boss", "eboss", "desi"]


    if "WEIGHT" in dr.dtype.names:
        if rank == 0:
            logging.info("Using existing WEIGHT column")
    else:
        dr = np.lib.recfunctions.append_fields(dr, 'WEIGHT', data=np.zeros(len(dr)), usemask=False)
        if alias == None and scheme is None:
            dr['WEIGHT'] = 1.0
            if rank == 0:
                logging.info("No completeness weight scheme or alias provided. Setting completeness weight to 1.0")
        elif alias is not None and scheme is None:
            if alias in dr.dtype.names:
                dr['WEIGHT'] = dr[alias]
                if rank == 0:
                    logging.info(f"Using {alias} as the completeness weight")
            else:
                raise ValueError(f"Alias {alias} not found in catalog columns.")
        elif alias is None and scheme is not None:
            if scheme == "boss":
                dr['WEIGHT'] = dr['WEIGHT_SYSTOT'] * (dr['WEIGHT_NOZ'] + dr['WEIGHT_CP'] - 1.0) if catalog_type == "data" else 1.0
                if rank == 0:
                    logging.info(f"Using BOSS-like completeness weight for {catalog_type}")
            elif scheme == "eboss":
                dr['WEIGHT'] = dr['WEIGHT_SYSTOT'] * dr['WEIGHT_NOZ'] * dr['WEIGHT_CP'] if catalog_type == "data" else 1.0
                if rank == 0:
                    logging.info(f"Using eBOSS-like completeness weight for {catalog_type}")
            elif scheme == "desi":
                raise ValueError("DESI completeness weight scheme not yet implemented.")
        else:
            raise ValueError("Cannot specify both a scheme and an alias for completeness weight.")

    return dr



def catalog_reader(catalog, geometry, column_names, z_range, comp_weight_plan, \
                    comm, para_cosmo=None, boxcenter=None, catalog_type=None, \
                    normalization_scheme="particle", apply_rsd = False, boxsize=None, \
                    redshift_box=None, los=None):
    """
    Reads the data/randoms catalog and applies necessary preprocessing steps.
    Args:
        catalog (str or list): Path(s) to the catalog file(s).
        geometry (str): The geometry type ("box-like" or "survey-like").
        column_names (list): List of column names for position and weight which only works in some cases.
        z_range (tuple): The redshift range to filter the data and randoms.
        comp_weight_plan (dict): A dictionary containing the completeness weight plan.
        para_cosmo (dict): A dictionary containing cosmological parameters.
        comm (MPI.Comm): The MPI communicator.
        boxcenter (array-like, optional): The center of the box for box-like geometry.
        catalog_type (str, optional): Type of catalog ("data" or "randoms") for survey-like geometry.
        apply_rsd (bool, optional): Whether to apply redshift-space distortions.
        boxsize (array-like, optional): The size of the box for box-like geometry.
        redshift_box (float, optional): The redshift value for the catalog.
        los (array-like, optional): The line-of-sight direction for redshift-space distortions.
    Returns:
        DataFrame: The processed catalog.
    """
    # Initialize MPI
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        logging.info(f"{'*' * 80}\nStart to read catalog: {catalog}")

    supported_types = {"npy", "fits","h5", "hdf5"}  # Supported file types for catalogs
    data_path = catalog # could be a single file or a list of files

    # Determine the file extension
    if isinstance(data_path, list):
        data_ext = data_path[0].split('.')[-1].lower() if data_path and data_path[0] else None
    else:
        data_ext = data_path.split('.')[-1].lower() if data_path else None

    # Check if the file type is supported
    if data_ext not in supported_types:
        raise ValueError(f"Unsupported data file type: {data_ext}")

    if geometry == "box-like":
        """
            Handle box-like geometry, currently only single .npy file is supported
        """
        if data_ext == "npy":

            assert column_names is not None and len(column_names) >= 3, \
                "For box-like geometry with .npy file, column_names must be provided \
                    with at least 3 elements for x,y,z."

            # Read .npy file and process it
            data_arr = npy_reader(data_path, comm)
            position_indices = [column_names.index(axis) for axis in ['x', 'y', 'z']]
            if rank == 0:
                logging.info(f"Using {column_names} as the position columns")
            # Create a structured ndarray directly without intermediate arrays
            data_cat = np.zeros(len(data_arr), dtype=[('Position', 'f8', (3,)), ('WEIGHT', 'f8')])
            data_cat['Position'] = np.column_stack([data_arr[:, idx] for idx in position_indices])

            # Check for weight column and assign directly
            if 'w' in column_names:
                data_cat['WEIGHT'] = data_arr[:, column_names.index('w')]
                if rank == 0:
                    logging.info(f"Using {column_names[column_names.index('w')]} as the WEIGHT column")
            else:
                data_cat['WEIGHT'] = 1.0
                if rank == 0:
                    logging.info("WEIGHT column does not exist in the list. Setting WEIGHT to 1.0")


        elif data_ext == "fits" or data_ext in ["h5", "hdf5"]:
            # raise NotImplementedError("Box-like geometry with .fits file is not yet implemented.")
            if data_ext == "fits":
                data_arr = fits_reader(comm, catalog, column_names)
            elif data_ext in ["h5", "hdf5"]:
                data_arr = h5_reader(comm, catalog, column_names)

            # Create a structured ndarray directly using the keys from data_arr
            data_cat = np.zeros(len(data_arr), dtype=[('Position', 'f8', (3,)), ('WEIGHT', 'f8')])

            # Check for position columns
            for axes in [('X', 'Y', 'Z'), ('x', 'y', 'z')]:
                if all(axis in data_arr.dtype.names for axis in axes):
                    data_cat['Position'] = np.column_stack([data_arr[axis] for axis in axes])
                    if rank == 0:
                        logging.info(f"Using {axes} as the position columns")
                    break
            # Check if velocity columns exist for RSD
            if apply_rsd:
                for vel_axes in [('VX', 'VY', 'VZ'), ('vx', 'vy', 'vz')]:
                    if all(axis in data_arr.dtype.names for axis in vel_axes):
                        velocities = np.column_stack([data_arr[axis] for axis in vel_axes])
                        if rank == 0:
                            logging.info(f"Using {vel_axes} as the velocity columns for RSD")
                        break
                data_cat['Position'] = add_rsd(comm, data_cat['Position'], velocities, \
                                              para_cosmo['h'], para_cosmo['Omega0'], \
                                                LOS=los, geometry='box-like', \
                                                box_length=boxsize[0], redshift_box=redshift_box)
                if rank == 0:
                    logging.info("Applied RSD to positions.")

            # Assign the weight column
            if 'w' in data_arr.dtype.names:
                data_cat['WEIGHT'] = data_arr['w']
                if rank == 0:
                    logging.info("Using 'w' as the WEIGHT column")
            else:
                data_cat['WEIGHT'] = 1.0
                if rank == 0:
                    logging.info("WEIGHT column does not exist in the data. Setting WEIGHT to 1.0")

        # Free memory
        del data_arr
        gc.collect()

        return data_cat
    
    elif geometry == "survey-like":
        """
        Handle survey-like geometry, note that the catalog could be either data or randoms
        """

        if data_ext == "npy":
            raise NotImplementedError("Survey-like geometry with .npy file is not yet implemented.")
            # assert column_names is not None and len(column_names) >= 5, \
            #     "For survey-like geometry with .npy file, column_names must be provided \
            #         with at least 5 elements for x, y, z, w_fkp, w_comp."
            
            # # Read .npy file and process it
            # data_arr = npy_reader(data_path, comm)

            # #Create ArrayCatalogs
            # position_indices = [column_names.index(axis) for axis in ['x', 'y', 'z']]
            # data_cat = ArrayCatalog({"Position": np.column_stack([data_arr[:, idx] for idx in position_indices])})

            # # Check for weight and nz columns
            # if 'w_comp' in column_names and 'w_fkp' in column_names :
            #     data_cat['WEIGHT'] = data_arr[:, column_names.index('w_comp')]
            #     data_cat['WEIGHT_FKP'] = data_arr[:, column_names.index('w_fkp')]
            #     if rank == 0:
            #         logging.info(f"Using {column_names[column_names.index('w_comp')]} as the WEIGHT column, \
            #                      {column_names[column_names.index('w_fkp')]} as the WEIGHT_FKP column.")
            # if 'nz' in column_names:
            #     data_cat['NZ'] = data_arr[:, column_names.index('nz')]
            #     if rank == 0:
            #         logging.info(f"Using {column_names[column_names.index('nz')]} as the NZ column.")
            # else:
            #     data_cat['NZ'] = 1.0
            #     if rank == 0:
            #         logging.info("NZ column does not exist in the list. Setting NZ to 1.0")

            # # Free memory
            # del data_arr
            # gc.collect()

        elif data_ext == "fits":
            if boxcenter is None:
                return_boxcenter = True
            else:
                return_boxcenter = False

            assert "Z" in column_names, \
                "For survey-like geometry with .fits file, column_names must include 'Z' for redshift slice."

            data_arr = fits_reader(comm, catalog, column_names)

            # slice the data_arr based on z_range
            if z_range is not None:
                z_min, z_max = z_range
                data_arr = data_arr[(data_arr["Z"] > z_min) & (data_arr["Z"] < z_max)]
            
            # logging the number of objects after slicing from all ranks
            sub_num_objects = len(data_arr)
            total_num_objects = comm.allreduce(sub_num_objects, op=MPI.SUM)
            if rank == 0:
                logging.info(f"Total objects after applying z_range cut: {total_num_objects}")

            # add completeness weight
            data_arr = add_completeness_weight(data_arr, comp_weight_plan, catalog_type, comm)

            # convert (RA, DEC, Z) to (x, y, z)
            posi = ra_dec_z_to_xyz(data_arr, para_cosmo, comm)

            # Create a structured ndarray directly using the keys from data_arr
            data_cat = np.zeros(len(data_arr), dtype=[('Position', 'f8', (3,)), 
                                                     ('WEIGHT', 'f8'), 
                                                     ('WEIGHT_FKP', 'f8'), 
                                                     ('NZ', 'f8'),
                                                     ])
            
            data_cat['Position'] = posi
            data_cat['WEIGHT'] = data_arr['WEIGHT']
            if 'WEIGHT_FKP' in data_arr.dtype.names:
                data_cat['WEIGHT_FKP'] = data_arr['WEIGHT_FKP']
            else:
                raise ValueError("WEIGHT_FKP column is missing in the catalog.")

            
            # NZ is necessary for particle normalization
            # for desi like catalog, NZ could be NX
            if 'NZ' in data_arr.dtype.names:
                data_cat['NZ'] = data_arr['NZ']
            elif 'NX' in data_arr.dtype.names:
                data_cat['NZ'] = data_arr['NX']
                if rank == 0:
                    logging.info("Using NX column as NZ for DESI-like catalog.")
            else:
                if normalization_scheme == "particle":
                    raise ValueError("NZ column is missing in the catalog, which is required for particle normalization.")
                else:
                    data_cat['NZ'] = 1.0
                    if rank == 0:
                        logging.info("NZ column does not exist in the catalog, Setting NZ to 1.0")

            # find the boxcenter in all ranks if not provided
            if boxcenter is None:
                local_min = np.min(data_cat['Position'], axis=0)
                local_max = np.max(data_cat['Position'], axis=0)
                global_min = np.empty(3)
                global_max = np.empty(3)
                comm.Allreduce(local_min, global_min, op=MPI.MIN)
                comm.Allreduce(local_max, global_max, op=MPI.MAX)
                boxcenter = 0.5 * (global_min + global_max)
                if rank == 0:
                    logging.info(f"Calculated boxcenter: {boxcenter}")
            else:
                if rank == 0:
                    logging.info(f"Using provided boxcenter: {boxcenter}")
            
            # free memory
            del data_arr, posi
            gc.collect()

            if not return_boxcenter:
                return data_cat
            else:
                return data_cat, boxcenter



def ra_dec_z_to_xyz(data_arr, para_cosmo=None, comm=None):
    """
    Convert (RA, DEC, Z) to (x_1, x_2, x_3) using the specified cosmology.
    Parameters
    ----------
    coords_array : numpy.ndarray
        Input array with columns 'RA', 'DEC', 'Z'.
    para_cosmo : dict, optional
        Cosmological parameters to modify the fiducial cosmology. If None, use default Planck18.
    z_range : tuple, optional
        Redshift range (z_min, z_max) to filter the data. If None, no filtering is applied.
    comm : MPI.Comm, optional
        MPI communicator for logging.  
    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3) with columns (x_1, x_2, x_3).
    """ 
    rank = comm.Get_rank()

    # Set up the fiducial cosmology.
    # Check if para_cosmo provides at least 'h' and 'Omega0' for customizing Planck18; otherwise use default Planck18.
    if para_cosmo is not None and "h" in para_cosmo and "Omega0" in para_cosmo:
        # my_cosmo = Planck18.clone(
        #     H0= 100 * para_cosmo["h"] * u.km / u.s / u.Mpc,  
        #     Om0=para_cosmo["Omega0"],                  
        #     Ob0=para_cosmo["Omega_b"] if "Omega_b" in para_cosmo else None
        # )
        my_cosmo = FlatLambdaCDM(H0=100 * para_cosmo["h"] * u.km / u.s / u.Mpc, Om0=para_cosmo["Omega0"])
        if rank == 0:
            logging.info(f"Using modified Planck18 cosmology: {my_cosmo}")
    else:
        my_cosmo = Planck18
        if rank == 0:
            logging.info(f"Using default Planck18 cosmology: {my_cosmo}")

    # Convert (RA, DEC, Z) to (x, y, z)
    ra = np.deg2rad(data_arr["RA"])  
    dec = np.deg2rad(data_arr["DEC"])  
    z = data_arr["Z"]  

    comoving_dist = my_cosmo.comoving_distance(z).value * para_cosmo["h"] # in Mpc/h

    # to avoid mix the variable names between redshift and z-axis, use x_1, x_2, x_3 for final coordinates
    x_1 = comoving_dist * np.cos(dec) * np.cos(ra)
    x_2 = comoving_dist * np.cos(dec) * np.sin(ra)
    x_3 = comoving_dist * np.sin(dec)

    to_return = np.column_stack([x_1, x_2, x_3])

    # memory cleanup
    del data_arr, ra, dec, z, comoving_dist, x_1, x_2, x_3
    gc.collect()

    return to_return


def add_rsd(comm, posi_arr, vel_arr, h, om0, redshift_box=None,LOS=None, geometry='box-like', box_length=None):
    rank = comm.Get_rank()

    # compute rsd factor
    # my_cosmo  = Planck18.clone(H0=h*100*u.km/u.s/u.Mpc, Om0=om0)
    my_cosmo = FlatLambdaCDM(H0=h*100*u.km/u.s/u.Mpc, Om0=om0)
    Ez = my_cosmo.efunc(redshift_box)
    rsd_factor = (1 + redshift_box) / (Ez * 100)  # in (Mpc/h)/(km/s)
    if rank ==0:
        logging.info(f"Applying RSD with factor: {rsd_factor}")

    # for box-like geometry, simply shift the positions along the provided LOS 
    if geometry == 'box-like':
        los = np.array(LOS)
        if np.linalg.norm(los) != 1:
            raise ValueError("LOS must be a 3d unit vector")
        if rank ==0:
            logging.info(f"Using box-like geometry for RSD with LOS = {los}")
    if geometry == 'survey-like':
        posi_norm = np.linalg.norm(posi_arr, axis=1)
        los = posi_arr / posi_norm
        if rank ==0:
            logging.info("Using survey-like geometry for RSD")

    # apply RSD shift
    posi_arr += rsd_factor * los * vel_arr 

    # apply periodic boundary conditions
    if geometry == 'box-like':
        if box_length is None:
            raise ValueError("For box-like geometry, box_length must be provided")
        posi_arr = posi_arr % box_length
        if rank == 0:
            logging.info("Boundary conditions applied for box-like catalogue")
            
    return posi_arr
