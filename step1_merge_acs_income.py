"""
Step 1: Merge ACS Income Data (B19013) with Block-Group Shapefiles
Key: GEOID field
Output: Neighborhood income map

Target Counties (Massachusetts):
- Suffolk County (FIPS: 25025)
- Middlesex County (FIPS: 25017)
- Norfolk County (FIPS: 25021)

GEOID Format: 12 digits = State(2) + County(3) + Tract(6) + BlockGroup(1)
Example: 250250001001 = MA(25) + Suffolk(025) + Tract(000100) + BG(1)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path
import requests
import zipfile
import os
import sys
from io import BytesIO
from tqdm import tqdm
from shapely.geometry import Point
from shapely.ops import unary_union

def load_acs_income_data(file_path, counties=['25025', '25017', '25021']):
    """
    Load ACS income data (B19013 - Median Household Income).
    Filters for specific Massachusetts counties:
    - Suffolk County (25025)
    - Middlesex County (25017)
    - Norfolk County (25021)
    
    Parameters:
    -----------
    file_path : str
        Path to ACS income CSV file
    counties : list
        List of county FIPS codes to filter (default: Suffolk, Middlesex, Norfolk)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with GEOID and income data
    """
    print("Loading ACS income data...")
    print(f"Filtering for counties: {counties}")
    print("  - Suffolk County (25025)")
    print("  - Middlesex County (25017)")
    print("  - Norfolk County (25021)")
    
    df = pd.read_csv(file_path)
    
    # ACS data typically has GEOID in format like "14000US250250001001"
    # We need to extract the block group GEOID (last 12 digits)
    # Format: State (2) + County (3) + Tract (6) + Block Group (1) = 12 digits
    
    if 'GEOID' not in df.columns:
        # Try common ACS column names
        if 'GEO_ID' in df.columns:
            df['GEOID'] = df['GEO_ID'].str.replace('14000US', '')
        elif 'geoid' in df.columns:
            df['GEOID'] = df['geoid']
        else:
            raise ValueError("Could not find GEOID column. Available columns:", df.columns.tolist())
    
    # Extract block group GEOID (12 digits) - ensure it's a string
    df['GEOID'] = df['GEOID'].astype(str).str.zfill(12)
    
    # Extract county code (positions 2-5, 0-indexed: positions 2-5)
    # GEOID format: State(2) + County(3) + Tract(6) + BlockGroup(1) = 12 digits
    # County code is positions 2-5 (e.g., "25025" for Suffolk)
    df['county_code'] = df['GEOID'].str[:5]
    
    # Filter for specified counties (5-digit format: 25025, 25017, 25021)
    df = df[df['county_code'].isin(counties)]
    print(f"Filtered to {len(df)} records in target counties")
    
    # Find income column (B19013_001E is typical for median household income)
    income_cols = [col for col in df.columns if 'B19013' in col or 'income' in col.lower() or 'median' in col.lower()]
    if not income_cols:
        raise ValueError("Could not find income column. Available columns:", df.columns.tolist())
    
    income_col = income_cols[0]
    df = df[['GEOID', income_col]].copy()
    df.columns = ['GEOID', 'median_household_income']
    
    # Remove rows with missing or invalid income data
    df = df[df['median_household_income'].notna()]
    df = df[df['median_household_income'] > 0]
    
    print(f"Loaded {len(df)} records with income data")
    return df


def load_block_group_shapefile(shapefile_path, counties=['25025', '25017', '25021']):
    """
    Load block group shapefile and filter for specific counties.
    Filters for:
    - Suffolk County (25025)
    - Middlesex County (25017)
    - Norfolk County (25021)
    
    Parameters:
    -----------
    shapefile_path : str
        Path to block group shapefile (.shp)
    counties : list
        List of county FIPS codes to filter (default: Suffolk, Middlesex, Norfolk)
        
    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame with block group geometries
    """
    print("Loading block group shapefile...")
    print(f"Filtering for counties: {counties}")
    # Use fiona directly to avoid geopandas/fiona.path compatibility issues
    try:
        import fiona
        shapefile_path_abs = str(Path(shapefile_path).resolve())
        # Try geopandas first
        gdf = gpd.read_file(shapefile_path_abs)
    except AttributeError:
        # Fallback: use fiona directly
        import fiona
        from shapely.geometry import shape
        shapefile_path_abs = str(Path(shapefile_path).resolve())
        features = []
        with fiona.open(shapefile_path_abs) as src:
            for feature in src:
                features.append({
                    'geometry': shape(feature['geometry']),
                    **feature['properties']
                })
        gdf = gpd.GeoDataFrame(features, crs=src.crs)
    
    # Ensure GEOID is in the correct format
    if 'GEOID' in gdf.columns:
        gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(12)
    elif 'GEOID10' in gdf.columns:
        gdf['GEOID'] = gdf['GEOID10'].astype(str).str.zfill(12)
    elif 'GEO_ID' in gdf.columns:
        gdf['GEOID'] = gdf['GEO_ID'].astype(str).str.replace('14000US', '').str.zfill(12)
    else:
        raise ValueError("Could not find GEOID column. Available columns:", gdf.columns.tolist())
    
    # Extract county code (first 5 digits: state + county)
    # GEOID format: State(2) + County(3) + Tract(6) + BlockGroup(1) = 12 digits
    # County code is first 5 digits (e.g., "25025" for Suffolk)
    gdf['county_code'] = gdf['GEOID'].str[:5]
    
    # Filter for specified counties (5-digit format: 25025, 25017, 25021)
    gdf = gdf[gdf['county_code'].isin(counties)]
    
    print(f"Loaded {len(gdf)} block groups in target counties")
    return gdf[['GEOID', 'geometry']]


def load_train_stations_and_create_buffer(gtfs_stops_file='data/gtfs/stops.txt', 
                                         gtfs_routes_file='data/gtfs/routes.txt',
                                         gtfs_trips_file=None,
                                         gtfs_stop_times_file=None,
                                         buffer_miles=1.0):
    """
    Load MBTA train stations and create a 1-mile buffer around them.
    
    Parameters:
    -----------
    gtfs_stops_file : str
        Path to GTFS stops.txt file
    gtfs_routes_file : str
        Path to GTFS routes.txt file (to filter by route_type)
    gtfs_trips_file : str, optional
        Path to GTFS trips.txt (to link routes to stops)
    gtfs_stop_times_file : str, optional
        Path to GTFS stop_times.txt (to link stops to trips)
    buffer_miles : float
        Buffer distance in miles (default: 1.0)
        
    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame with train station buffers
    """
    print(f"\nLoading train stations and creating {buffer_miles}-mile buffer...")
    
    if not Path(gtfs_stops_file).exists():
        print(f"⚠️  GTFS stops file not found: {gtfs_stops_file}")
        print("   Skipping train station filtering. Using all block groups.")
        return None
    
    # Load stops
    stops_df = pd.read_csv(gtfs_stops_file)
    
    # Filter for valid coordinates
    stops_df = stops_df[stops_df['stop_lat'].notna() & stops_df['stop_lon'].notna()]
    
    # Filter for train stations using route_type
    # Route types: 0=Light rail, 1=Subway, 2=Rail (commuter rail)
    train_route_types = [0, 1, 2]
    
    if gtfs_routes_file and Path(gtfs_routes_file).exists():
        try:
            routes_df = pd.read_csv(gtfs_routes_file)
            
            # Filter routes for train types
            train_routes = routes_df[routes_df['route_type'].isin(train_route_types)]['route_id'].unique()
            
            # If we have trips and stop_times, join to get train stops
            if (gtfs_trips_file and Path(gtfs_trips_file).exists() and 
                gtfs_stop_times_file and Path(gtfs_stop_times_file).exists()):
                trips_df = pd.read_csv(gtfs_trips_file)
                stop_times_df = pd.read_csv(gtfs_stop_times_file)
                
                # Join: routes -> trips -> stop_times -> stops
                train_trips = trips_df[trips_df['route_id'].isin(train_routes)]['trip_id'].unique()
                train_stop_times = stop_times_df[stop_times_df['trip_id'].isin(train_trips)]
                train_stop_ids = train_stop_times['stop_id'].unique()
                
                stops_df = stops_df[stops_df['stop_id'].isin(train_stop_ids)]
                print(f"   Filtered to {len(stops_df)} train stations using route_type")
            else:
                # Fallback: filter by stop name patterns
                train_keywords = ['station', 'subway', 'red line', 'orange line', 'blue line', 
                                'green line', 'commuter rail', 'mbta', 't', 'line']
                if 'stop_name' in stops_df.columns:
                    stops_df['is_train'] = stops_df['stop_name'].str.lower().str.contains(
                        '|'.join(train_keywords), na=False
                    )
                    stops_df = stops_df[stops_df['is_train']]
                    print(f"   Filtered to {len(stops_df)} train stations using name patterns")
        except Exception as e:
            print(f"   Warning: Could not filter by routes ({e}). Using all stops.")
    
    if len(stops_df) == 0:
        print("   ⚠️  No train stations found. Using all stops.")
        # Reload all stops
        stops_df = pd.read_csv(gtfs_stops_file)
        stops_df = stops_df[stops_df['stop_lat'].notna() & stops_df['stop_lon'].notna()]
    
    # Create Point geometries
    geometry = [Point(lon, lat) for lon, lat in zip(stops_df['stop_lon'], stops_df['stop_lat'])]
    stations_gdf = gpd.GeoDataFrame(stops_df, geometry=geometry, crs='EPSG:4326')
    
    # Convert to projected CRS for accurate buffer (meters)
    # Use Massachusetts State Plane (meters)
    stations_gdf = stations_gdf.to_crs('EPSG:26986')  # MA State Plane Mainland (meters)
    
    # Create buffer (1 mile = 1609.34 meters)
    buffer_meters = buffer_miles * 1609.34
    stations_gdf['buffer'] = stations_gdf.geometry.buffer(buffer_meters)
    
    # Union all buffers into one polygon
    all_buffers = unary_union(stations_gdf['buffer'].tolist())
    
    # Create a GeoDataFrame with the combined buffer
    buffer_gdf = gpd.GeoDataFrame([1], geometry=[all_buffers], crs='EPSG:26986')
    
    print(f"   Loaded {len(stations_gdf)} train stations")
    print(f"   Created {buffer_miles}-mile buffer around stations")
    
    return buffer_gdf


def filter_by_train_buffer(merged_gdf, buffer_gdf):
    """
    Filter merged income data to only include block groups within train station buffer.
    
    Parameters:
    -----------
    merged_gdf : gpd.GeoDataFrame
        Merged income and block group data
    buffer_gdf : gpd.GeoDataFrame
        Train station buffer polygon
        
    Returns:
    --------
    gpd.GeoDataFrame
        Filtered merged data
    """
    if buffer_gdf is None:
        return merged_gdf
    
    print("\nFiltering block groups to train station area (1-mile buffer)...")
    
    # Ensure merged_gdf has a CRS (set to 4326 if missing)
    if merged_gdf.crs is None:
        merged_gdf.set_crs('EPSG:4326', inplace=True)
    
    # Convert merged_gdf to buffer CRS for intersection
    merged_gdf_proj = merged_gdf.to_crs(buffer_gdf.crs)
    
    # Get the buffer polygon
    buffer_polygon = buffer_gdf.geometry.iloc[0]
    
    # Filter block groups that intersect with buffer
    merged_gdf_proj['within_buffer'] = merged_gdf_proj.geometry.intersects(buffer_polygon)
    filtered_gdf_proj = merged_gdf_proj[merged_gdf_proj['within_buffer']].copy()
    filtered_gdf_proj = filtered_gdf_proj.drop(columns='within_buffer')
    
    # Convert back to original CRS (4326)
    filtered_gdf = filtered_gdf_proj.to_crs('EPSG:4326')
    
    print(f"   Original block groups: {len(merged_gdf)}")
    print(f"   Filtered block groups (within 1 mile of train stations): {len(filtered_gdf)}")
    print(f"   Reduction: {len(merged_gdf) - len(filtered_gdf)} block groups removed ({100*(len(merged_gdf)-len(filtered_gdf))/len(merged_gdf):.1f}%)")
    
    return filtered_gdf


def merge_income_with_shapefile(income_df, shapefile_gdf):
    """
    Merge ACS income data with block group shapefile on GEOID.
    
    Parameters:
    -----------
    income_df : pd.DataFrame
        DataFrame with GEOID and income data
    shapefile_gdf : gpd.GeoDataFrame
        GeoDataFrame with block group geometries
        
    Returns:
    --------
    gpd.GeoDataFrame
        Merged GeoDataFrame with income and geometry
    """
    print("Merging income data with shapefile...")
    
    # Merge on GEOID
    merged = shapefile_gdf.merge(income_df, on='GEOID', how='inner')
    
    print(f"Merged {len(merged)} block groups with income data")
    print(f"Income range: ${merged['median_household_income'].min():,.0f} - ${merged['median_household_income'].max():,.0f}")
    
    return merged


def create_income_map(merged_gdf, output_path='neighborhood_income_map.png', 
                     title='Neighborhood Income Map (Median Household Income)',
                     show_plot=True):
    """
    Create a choropleth map of neighborhood income.
    
    Parameters:
    -----------
    merged_gdf : gpd.GeoDataFrame
        GeoDataFrame with income and geometry
    output_path : str
        Path to save the map
    title : str
        Title for the map
    """
    print("Creating income map...")
    
    # Set CRS to Web Mercator for basemap (if not already)
    if merged_gdf.crs is None:
        merged_gdf.set_crs(epsg=4326, inplace=True)
    
    # Convert to Web Mercator for basemap
    merged_gdf = merged_gdf.to_crs(epsg=3857)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot choropleth
    merged_gdf.plot(column='median_household_income', 
                    cmap='YlOrRd', 
                    legend=True,
                    ax=ax,
                    edgecolor='white',
                    linewidth=0.3,
                    legend_kwds={'label': 'Median Household Income ($)',
                                'orientation': 'vertical',
                                'shrink': 0.8})
    
    # Add basemap (optional - requires internet)
    try:
        ctx.add_basemap(ax, crs=merged_gdf.crs, source=ctx.providers.CartoDB.Positron)
    except:
        print("Note: Could not add basemap. Map will show without background.")
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def download_acs_income_data(output_path='data/acs_income_b19013.csv', year=2023):
    """
    Download ACS income data using Census API (no API key needed for public data).
    Falls back to manual download instructions if API fails.
    """
    print("Downloading ACS income data from Census API...")
    
    # Census API endpoint for ACS 5-year estimates (public, no key needed for small requests)
    # Using 2023 ACS 5-year (2019-2023)
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    
    # Variables: B19013_001E = Median Household Income
    variables = "B19013_001E"
    
    # Geography: Block Groups in Massachusetts (state=25)
    # Counties: Suffolk=025, Middlesex=017, Norfolk=021
    # Format: state:25&county:025,017,021&tract:*&block%20group:*
    
    # Try downloading each county separately (more reliable)
    try:
        all_data = []
        counties = {'025': 'Suffolk', '017': 'Middlesex', '021': 'Norfolk'}
        
        for county_fips, county_name in counties.items():
            print(f"  Downloading {county_name} County ({county_fips})...")
            url = f"{base_url}?get=NAME,GEO_ID,{variables}&for=block%20group:*&in=state:25&in=county:{county_fips}"
            
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                data = response.json()
                
                if len(data) > 1:
                    df_county = pd.DataFrame(data[1:], columns=data[0])
                    all_data.append(df_county)
                    print(f"    ✓ {len(df_county)} block groups")
                else:
                    print(f"    ⚠ No data for {county_name}")
            except Exception as e:
                print(f"    ⚠ Error downloading {county_name}: {e}")
        
        if not all_data:
            raise Exception("No data downloaded from any county")
        
        # Combine all counties
        df = pd.concat(all_data, ignore_index=True)
        
        # Process GEOID - Census API returns GEO_ID as "1500000US250250001001"
        # We need 12-digit GEOID: 250250001001
        if 'GEO_ID' in df.columns:
            df['GEOID'] = df['GEO_ID'].str.replace('1500000US', '')
        
        # Rename income column
        if 'B19013_001E' in df.columns:
            df = df.rename(columns={'B19013_001E': 'median_household_income'})
            # Convert to numeric, handling -666666666 (null) values
            df['median_household_income'] = pd.to_numeric(df['median_household_income'], errors='coerce')
            # Replace null codes with NaN
            df['median_household_income'] = df['median_household_income'].replace(-666666666, pd.NA)
        
        # Filter to only include rows with valid GEOIDs (12 digits)
        df = df[df['GEOID'].str.len() == 12]
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Downloaded and saved to {output_path}")
        print(f"   Total records: {len(df)}")
        print(f"   Counties: Suffolk, Middlesex, Norfolk")
        return True
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("⚠️  Rate limit exceeded. Trying with API key...")
            # Try with API key if available
            api_key = os.getenv('CENSUS_API_KEY')
            if api_key:
                url_with_key = f"{url}&key={api_key}"
                try:
                    response = requests.get(url_with_key, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    df = pd.DataFrame(data[1:], columns=data[0])
                    if 'GEO_ID' in df.columns:
                        df['GEOID'] = df['GEO_ID'].str.replace('1500000US', '')
                    if 'B19013_001E' in df.columns:
                        df = df.rename(columns={'B19013_001E': 'median_household_income'})
                        df['median_household_income'] = pd.to_numeric(df['median_household_income'], errors='coerce')
                        df['median_household_income'] = df['median_household_income'].replace(-666666666, pd.NA)
                    df = df[df['GEOID'].str.len() == 12]
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(output_path, index=False)
                    print(f"✅ Downloaded with API key and saved to {output_path}")
                    return True
                except:
                    pass
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error downloading from API: {e}")
    
    # Fallback: Manual download instructions
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("="*70)
    print("Please download ACS income data manually:")
    print("\n1. Go to: https://data.census.gov/cedsci/")
    print("2. Click 'Advanced Search'")
    print("3. Search for: B19013 (Median Household Income)")
    print("4. Select: '5-Year Estimates' (most recent, e.g., 2019-2023)")
    print("5. Geography:")
    print("   - Select 'Block Groups'")
    print("   - Select 'Massachusetts'")
    print("   - Select counties: Suffolk, Middlesex, Norfolk")
    print("6. Download as CSV")
    print(f"7. Save as: {output_path}")
    print("\nOr get a free Census API key and set:")
    print("  export CENSUS_API_KEY='your_key'")
    print("  (Get key: https://api.census.gov/data/key_signup.html)")
    print("="*70)
    return False


def download_block_group_shapefile(output_dir='data', year=2024):
    """
    Download Massachusetts block group shapefile from Census TIGER/Line files.
    """
    print("Downloading block group shapefile...")
    
    # Census TIGER/Line file URL
    # Format: https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{state}_bg.zip
    state_fips = "25"  # Massachusetts
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{state_fips}_bg.zip"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"tl_{year}_{state_fips}_bg.zip"
    shapefile_path = output_dir / "block_groups.shp"
    
    # Check if already exists
    if shapefile_path.exists():
        print(f"Shapefile already exists: {shapefile_path}")
        return str(shapefile_path)
    
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(zip_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print("Extracting shapefile...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Find the .shp file
        shp_files = list(output_dir.glob(f"tl_{year}_{state_fips}_bg.shp"))
        if shp_files:
            # Rename to block_groups.shp
            old_path = shp_files[0]
            new_path = output_dir / "block_groups.shp"
            
            # Rename all related files
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                old_file = old_path.with_suffix(ext)
                new_file = new_path.with_suffix(ext)
                if old_file.exists():
                    old_file.rename(new_file)
            
            print(f"Shapefile extracted to {new_path}")
            
            # Clean up zip file
            zip_path.unlink()
            
            return str(new_path)
        else:
            print("ERROR: Could not find .shp file in downloaded zip")
            return None
            
    except Exception as e:
        print(f"Error downloading shapefile: {e}")
        print("\nPlease download manually from:")
        print("https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html")
        return None


def main():
    """
    Main function to execute Step 1.
    Filters data for three Massachusetts counties:
    - Suffolk County (25025)
    - Middlesex County (25017)
    - Norfolk County (25021)
    """
    # Massachusetts county FIPS codes
    target_counties = ['25025', '25017', '25021']  # Suffolk, Middlesex, Norfolk
    
    # File paths
    acs_income_file = 'data/acs_income_b19013.csv'
    block_group_shapefile = 'data/block_groups.shp'
    
    # Ensure data directory exists
    Path('data').mkdir(exist_ok=True)
    
    # Download ACS income data if missing
    if not Path(acs_income_file).exists():
        print("ACS income file not found. Attempting to download...")
        success = download_acs_income_data(acs_income_file)
        if not success:
            print("\nPlease download manually and place at:", acs_income_file)
            print("Or set CENSUS_API_KEY environment variable and try again.")
            return
    else:
        print(f"Using existing ACS income file: {acs_income_file}")
    
    # Download block group shapefile if missing
    if not Path(block_group_shapefile).exists():
        print("Block group shapefile not found. Attempting to download...")
        downloaded_path = download_block_group_shapefile('data')
        if not downloaded_path:
            print("\nPlease download manually and place at:", block_group_shapefile)
            return
    else:
        print(f"Using existing shapefile: {block_group_shapefile}")
    
    # Load data (filtered for target counties)
    income_df = load_acs_income_data(acs_income_file, counties=target_counties)
    shapefile_gdf = load_block_group_shapefile(block_group_shapefile, counties=target_counties)
    
    # Merge
    merged_gdf = merge_income_with_shapefile(income_df, shapefile_gdf)
    
    # Filter to train station area (1-mile buffer)
    gtfs_stops_file = 'data/gtfs/stops.txt'
    gtfs_routes_file = 'data/gtfs/routes.txt'
    gtfs_trips_file = 'data/gtfs/trips.txt' if Path('data/gtfs/trips.txt').exists() else None
    gtfs_stop_times_file = 'data/gtfs/stop_times.txt' if Path('data/gtfs/stop_times.txt').exists() else None
    
    buffer_gdf = load_train_stations_and_create_buffer(
        gtfs_stops_file=gtfs_stops_file,
        gtfs_routes_file=gtfs_routes_file,
        gtfs_trips_file=gtfs_trips_file,
        gtfs_stop_times_file=gtfs_stop_times_file,
        buffer_miles=1.0
    )
    
    if buffer_gdf is not None:
        merged_gdf = filter_by_train_buffer(merged_gdf, buffer_gdf)
    
    # Final verification: Ensure only target counties are included
    print("\n" + "="*60)
    print("FINAL COUNTY VERIFICATION")
    print("="*60)
    merged_gdf['county_code'] = merged_gdf['GEOID'].str[:5]
    valid_counties = set(target_counties)
    merged_counties = set(merged_gdf['county_code'].unique())
    
    # Filter to only include target counties (remove any outliers)
    before_count = len(merged_gdf)
    merged_gdf = merged_gdf[merged_gdf['county_code'].isin(target_counties)]
    after_count = len(merged_gdf)
    
    if before_count != after_count:
        print(f"⚠️  Removed {before_count - after_count} block groups from other counties")
    
    print(f"✅ Final data contains only:")
    for county_code in sorted(merged_counties & valid_counties):
        county_name = {'25025': 'Suffolk', '25017': 'Middlesex', '25021': 'Norfolk'}.get(county_code, county_code)
        count = len(merged_gdf[merged_gdf['county_code'] == county_code])
        print(f"   - {county_name} County ({county_code}): {count} block groups")
    
    if merged_counties - valid_counties:
        print(f"\n⚠️  Removed counties: {sorted(merged_counties - valid_counties)}")
    
    print(f"\nTotal block groups: {len(merged_gdf)}")
    print("="*60)
    
    # Create map
    title = 'Neighborhood Income Map - Train Station Area (1-mile buffer)\nSuffolk, Middlesex, and Norfolk Counties, MA'
    create_income_map(merged_gdf, 
                     title=title)
    
    # Save merged data
    output_file = 'data/neighborhood_income_merged.geojson'
    merged_gdf.to_file(output_file, driver='GeoJSON')
    print(f"\nMerged data saved to {output_file}")
    
    # Also save as CSV (without geometry)
    csv_output = 'data/neighborhood_income_merged.csv'
    merged_gdf.drop(columns='geometry').to_csv(csv_output, index=False)
    print(f"Data (without geometry) saved to {csv_output}")
    
    # Display sample data rows
    print("\n" + "="*60)
    print("SAMPLE DATA ROWS (First 10)")
    print("="*60)
    sample_data = merged_gdf[['GEOID', 'median_household_income']].head(10)
    print(sample_data.to_string(index=False))
    
    # Print summary statistics by county
    print("\n" + "="*60)
    print("SUMMARY BY COUNTY")
    print("="*60)
    merged_gdf['county_code'] = merged_gdf['GEOID'].str[:5]  # Fix: use first 5 digits
    county_names = {'25025': 'Suffolk County', '25017': 'Middlesex County', '25021': 'Norfolk County'}
    merged_gdf['county_name'] = merged_gdf['county_code'].map(county_names)
    
    for county_code, county_name in county_names.items():
        county_data = merged_gdf[merged_gdf['county_code'] == county_code]
        if len(county_data) > 0:
            print(f"\n{county_name} ({county_code}):")
            print(f"  Block groups: {len(county_data)}")
            print(f"  Median income: ${county_data['median_household_income'].median():,.0f}")
            print(f"  Mean income: ${county_data['median_household_income'].mean():,.0f}")
            print(f"  Income range: ${county_data['median_household_income'].min():,.0f} - ${county_data['median_household_income'].max():,.0f}")


if __name__ == '__main__':
    main()

