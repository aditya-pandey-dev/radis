import pandas as pd
import numpy as np

def load_chianti_iron12(wmin=193.0, wmax=197.0):
    """Load CHIANTI Fe XII spectral lines (193-197 Angstrom)"""
    
    # EXACT path from your screenshot
    path = '/Users/jaiswaljaishankar05/Downloads/radis/radis/db/chianti/Fe_12/Fe_12.wgfa'
    
    print(f"Loading CHIANTI from: {path}")
    
    # Read fixed-width format file, skip header lines
    try:
        df = pd.read_fwf(path, skiprows=12, header=None)
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Trying alternative method...")
        df = pd.read_csv(path, skiprows=12, header=None, delim_whitespace=True)
    
    print(f"Raw data shape: {df.shape}, columns: {len(df.columns)}")
    
    # Assign CHIANTI standard columns
    if len(df.columns) >= 6:
        df.columns = ['wl_ang', 'loggf', 'A_einstein', 'g_lower', 'E_lower', 'E_upper']
    else:
        print(f"Warning: Expected 6+ columns, got {len(df.columns)}")
        for i in range(len(df.columns)):
            df.rename(columns={i: f'col_{i}'}, inplace=True)
    
    # Clean and convert data
    df['wl_ang'] = pd.to_numeric(df['wl_ang'].astype(str).str.strip(), errors='coerce')
    df['loggf'] = pd.to_numeric(df['loggf'].astype(str).str.strip(), errors='coerce')
    df['A_einstein'] = pd.to_numeric(df['A_einstein'].astype(str).str.strip(), errors='coerce')
    df['g_lower'] = pd.to_numeric(df['g_lower'].astype(str).str.strip(), errors='coerce')
    
    # Create RADIS-compatible DataFrame
    radis_df = pd.DataFrame({
        'wav': df['wl_ang'] * 0.1,           # Convert Angstrom to nm
        'int': 10**df['loggf'],              # Convert loggf to gf (line strength)
        'A': df['A_einstein'],               # Einstein A coefficient
        'gpp': df['g_lower']                 # Lower level statistical weight
    }).dropna()
    
    print(f"Valid lines after cleaning: {len(radis_df)}")
    
    # Filter by wavelength range
    mask = (radis_df['wav'] >= wmin) & (radis_df['wav'] <= wmax)
    result = radis_df[mask].reset_index(drop=True)
    
    print(f"Lines in {wmin}-{wmax} nm range: {len(result)}")
    print(result.head())
    
    return result
