import os
import pandas as pd
from astropy.io.votable import parse, from_table, writeto
from astropy.table import Table
from astropy.io import fits
import numpy as np
from scipy.spatial import cKDTree

# Example zero points for each filter in AB magnitudes
zero_points_abmag = {
    'f090w': 27.4525,
    'f115w': 27.5681,
    'f150w': 27.814,
    'f200w': 27.9973,
    'f277w': 27.8803,
    'f356w': 28.0068,
    'f410m': 27.1848,
    'f444w': 28.0647
}

# Mapping of zspec column names to main DataFrame column names
column_mapping = {
    'ra': 'ALPHA_J2000',
    'dec': 'DELTA_J2000',
    'z_pec': 'zspec'
}

def read_votable_to_dataframe(file_path):
    print(f"Reading VOTable file: {file_path}")
    votable = parse(file_path)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    df = table.to_pandas()
    print(f"Columns in VOTable file {file_path}: {df.columns.tolist()}")
    return df

def read_fits_to_dataframe(file_path):
    print(f"Reading FITS file: {file_path}")
    with fits.open(file_path) as hdul:
        df = Table(hdul[1].data).to_pandas()
    print(f"Columns in FITS file {file_path}: {df.columns.tolist()}")
    return df

def read_csv_to_dataframe(file_path):
    print(f"Reading CSV file: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Columns in CSV file {file_path}: {df.columns.tolist()}")
    return df

def convert_mag_to_microjanskys(df, filter_name):
    flux_counts = 'FLUX_ISO'
    fluxerr_counts = 'FLUXERR_ISO'
    flux_AUTO = 'FLUX_AUTO'
    fluxerr_counts = 'FLUXERR_ISO'
    flux_col = f'{filter_name}_flux'
    fluxerr_col = f'{filter_name}_fluxerr'

    #apply the aperature correction at the end
    if flux_counts in df.columns:
        df[flux_col] = 3631 * 10**(-0.4 * (-2.5 * np.log10(np.abs(df[flux_counts])) + 28.08)) * 1e6 * (df[flux_AUTO]/df[flux_counts])
    else:
        print(f"Column FLUX_ISO not found in DataFrame for filter {filter_name}.")
        
    if fluxerr_counts in df.columns:
        mag_error = (2.5 / (np.log(10) * df[flux_counts])) * df[fluxerr_counts]
        df[fluxerr_col] = df[flux_col] * np.log(10) * mag_error * 0.4 * (df[flux_AUTO]/df[flux_counts])
        
        # Apply the condition: if fluxerr < 5% of flux, set it to 5% of flux
        #flux_threshold = 0.05 * df[flux_col]
        #df[fluxerr_col] = np.maximum(df[fluxerr_col], flux_threshold)
    else:
        print(f"Column FLUXERR_ISO not found in DataFrame for filter {filter_name}.")

    return df


def extract_and_convert(file_paths, zero_points):
    all_data = pd.DataFrame()
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_identifier = os.path.splitext(file_name)[0][-5:].lower()
        print(f"Processing file: {file_name}, Filter identifier: {file_identifier}")
        
        if file_identifier not in zero_points:
            print(f"Warning: Zero point for filter {file_identifier} not found in zero_points dictionary.")
            continue
        
        df = read_votable_to_dataframe(file_path)
        df = convert_mag_to_microjanskys(df, file_identifier)
        
        flux_col = f'{file_identifier}_flux'
        fluxerr_col = f'{file_identifier}_fluxerr'
        
        if 'ALPHA_J2000' not in df.columns or 'DELTA_J2000' not in df.columns:
            print(f"Required columns ALPHA_J2000 and DELTA_J2000 not found in DataFrame for filter {file_identifier}.")
            continue
        
        df = df[['ALPHA_J2000', 'DELTA_J2000', flux_col, fluxerr_col]]
        
        if all_data.empty:
            all_data = df
        else:
            all_data = pd.merge(all_data, df, on=['ALPHA_J2000', 'DELTA_J2000'], how='outer')
    
    print("Final DataFrame preview:")
    print(all_data.head())
    
    return all_data

def load_zspec_data(zspec_file_paths):
    zspec_data = pd.DataFrame()
    
    for file_path in zspec_file_paths:
        if file_path.endswith('.csv'):
            df = read_csv_to_dataframe(file_path)
        elif file_path.endswith('.fits'):
            df = read_fits_to_dataframe(file_path)
        elif file_path.endswith('.cat'):
            df = read_votable_to_dataframe(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            continue
        
        # Rename columns according to the mapping
        df = df.rename(columns=column_mapping)
        
        # Check required columns after renaming
        if 'ALPHA_J2000' not in df.columns or 'DELTA_J2000' not in df.columns or 'zspec' not in df.columns:
            print(f"Required columns ALPHA_J2000, DELTA_J2000, and zspec not found in DataFrame from {file_path}.")
            continue
        
        zspec_data = pd.concat([zspec_data, df[['ALPHA_J2000', 'DELTA_J2000', 'zspec']]])
    
    return zspec_data

def spatial_tolerance_match(main_data, zspec_data, tolerance=0.0001):
    # Convert coordinates to numpy arrays
    main_coords = main_data[['ALPHA_J2000', 'DELTA_J2000']].values
    zspec_coords = zspec_data[['ALPHA_J2000', 'DELTA_J2000']].values
    
    # Create KDTree for fast spatial queries
    tree = cKDTree(zspec_coords)
    
    # Query the tree for nearest neighbors within the tolerance
    distances, indices = tree.query(main_coords, distance_upper_bound=tolerance)
    
    # Prepare a DataFrame for zspec results
    zspec_matches = pd.DataFrame({
        'ALPHA_J2000': main_data['ALPHA_J2000'],
        'DELTA_J2000': main_data['DELTA_J2000'],
        'zspec': -99.0  # Default value as float
    })
    
    for i, index in enumerate(indices):
        if index < len(zspec_data) and distances[i] <= tolerance:
            zspec_matches.at[i, 'zspec'] = float(zspec_data.iloc[index]['zspec'])
    
    return pd.concat([main_data.reset_index(drop=True), zspec_matches.reset_index(drop=True)], axis=1)

def save_dataframe_to_votable(df, output_file):
    if not df.empty:
        # Convert all columns to appropriate types
        df = df.convert_dtypes()

        # Handle any specific column type conversions if needed
        if 'zspec' in df.columns:
            df['zspec'] = df['zspec'].astype(float)
        
        # Drop duplicate columns if present
        df = df.loc[:, ~df.columns.duplicated()]

        # Debug: Print DataFrame types
        print(f"DataFrame dtypes after conversion: {df.dtypes}")

        # Create the table
        try:
            table = Table.from_pandas(df)
            votable = from_table(table)
            writeto(votable, output_file)
            print(f"Merged catalog saved to {output_file}")
        except Exception as e:
            print(f"Error saving DataFrame to VOTable: {e}")
    else:
        print("No data to save.")

# Directories
input_directory = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\JWST_DATA_PRIMER\PRIMER_COSMOS_EAST'
zspec_directory = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\UDS_zspec_cats'
output_file = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\JWST_DATA_PRIMER\PRIMER_COSMOS_EAST\converted_flux_catalog_apercor.cat'

# Ensure the output directory exists
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

# File paths
file_paths = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.cat')]
zspec_file_paths = [os.path.join(zspec_directory, f) for f in os.listdir(zspec_directory) if f.endswith('.csv') or f.endswith('.fits') or f.endswith('.cat')]

print(f"Files to process: {file_paths}")
print(f"ZSPEC files to process: {zspec_file_paths}")

# Process and merge data
merged_df = extract_and_convert(file_paths, zero_points_abmag)
zspec_data = load_zspec_data(zspec_file_paths)
final_df = spatial_tolerance_match(merged_df, zspec_data)

# Save the final DataFrame to a VOTable file
save_dataframe_to_votable(final_df, output_file)
