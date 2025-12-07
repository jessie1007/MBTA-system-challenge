"""
Step 2: Transit Station Accessibility Analysis

1. Load MBTA GTFS stops.txt
2. Extract train stations (filter by route_type)
3. Compute distance from block group centroids to nearest train station
4. Categorize into zones:
   - 0-0.5 miles: walk-zone
   - 0.5-2 miles: bike-zone
   - 2+ miles: far-zone
5. Merge with income data from Step 1
6. Add population (B01003) and vehicle availability (B08201)
7. Analyze: "X% of low-income vs Y% of high-income residents live in bike-but-not-walk zone"
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Route types in GTFS (for filtering train stations)
# 0 = Light rail, streetcar, tram
# 1 = Subway, metro
# 2 = Rail
# 3 = Bus
# 4 = Ferry
# 11 = Trolleybus
# 12 = Monorail

TRAIN_ROUTE_TYPES = [0, 1, 2]  # Light rail, subway, rail


def load_gtfs_stops(gtfs_stops_file):
    """
    Load MBTA GTFS stops.txt file.
    
    Parameters:
    -----------
    gtfs_stops_file : str
        Path to GTFS stops.txt file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with stop_id, stop_lat, stop_lon, route_type
    """
    print("Loading MBTA GTFS stops...")
    stops_df = pd.read_csv(gtfs_stops_file)
    
    print(f"Loaded {len(stops_df)} stops")
    print(f"Columns: {stops_df.columns.tolist()}")
    
    # GTFS stops.txt typically has: stop_id, stop_name, stop_lat, stop_lon
    # Route type might be in a separate file (routes.txt or stop_times.txt)
    # We may need to join with routes.txt to get route_type
    
    required_cols = ['stop_id', 'stop_lat', 'stop_lon']
    missing_cols = [col for col in required_cols if col not in stops_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean data
    stops_df = stops_df[stops_df['stop_lat'].notna()]
    stops_df = stops_df[stops_df['stop_lon'].notna()]
    
    print(f"Valid stops: {len(stops_df)}")
    return stops_df


def load_gtfs_routes(gtfs_routes_file):
    """
    Load GTFS routes.txt to get route_type for each route.
    
    Parameters:
    -----------
    gtfs_routes_file : str
        Path to GTFS routes.txt file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with route_id and route_type
    """
    print("Loading GTFS routes...")
    routes_df = pd.read_csv(gtfs_routes_file)
    
    if 'route_id' not in routes_df.columns or 'route_type' not in routes_df.columns:
        raise ValueError("routes.txt must contain route_id and route_type columns")
    
    return routes_df[['route_id', 'route_type']]


def get_train_stations(stops_df, routes_df=None, trips_df=None, stop_times_df=None):
    """
    Filter stops to get only train stations.
    
    If route_type is not directly in stops, we need to join through:
    stops -> stop_times -> trips -> routes
    
    Parameters:
    -----------
    stops_df : pd.DataFrame
        Stops data
    routes_df : pd.DataFrame, optional
        Routes data with route_type
    trips_df : pd.DataFrame, optional
        Trips data linking routes and stop_times
    stop_times_df : pd.DataFrame, optional
        Stop times linking stops and trips
        
    Returns:
    --------
    pd.DataFrame
        Filtered stops for train stations only
    """
    print("Filtering for train stations...")
    
    # Check if vehicle_type is in stops (newer GTFS format)
    if 'vehicle_type' in stops_df.columns:
        # Vehicle type: 0=Light rail, 1=Subway, 2=Rail (trains)
        train_stations = stops_df[stops_df['vehicle_type'].isin(TRAIN_ROUTE_TYPES)]
        print(f"Found {len(train_stations)} train stations (using vehicle_type)")
        return train_stations[['stop_id', 'stop_lat', 'stop_lon']]
    
    # If route_type is already in stops_df
    if 'route_type' in stops_df.columns:
        train_stations = stops_df[stops_df['route_type'].isin(TRAIN_ROUTE_TYPES)]
        print(f"Found {len(train_stations)} train stations")
        return train_stations[['stop_id', 'stop_lat', 'stop_lon']]
    
    # Otherwise, join through GTFS tables
    if routes_df is not None and trips_df is not None and stop_times_df is not None:
        # Join: stop_times -> trips -> routes
        stop_times_with_trips = stop_times_df.merge(trips_df, on='trip_id', how='inner')
        stop_times_with_routes = stop_times_with_trips.merge(routes_df, on='route_id', how='inner')
        
        # Filter for train routes
        train_stop_times = stop_times_with_routes[
            stop_times_with_routes['route_type'].isin(TRAIN_ROUTE_TYPES)
        ]
        
        # Get unique train station stop_ids
        train_stop_ids = train_stop_times['stop_id'].unique()
        
        # Filter stops
        train_stations = stops_df[stops_df['stop_id'].isin(train_stop_ids)]
        print(f"Found {len(train_stations)} train stations")
        return train_stations[['stop_id', 'stop_lat', 'stop_lon']]
    
    # If we can't determine route_type, return all stops with a warning
    print("WARNING: Could not determine route_type. Using all stops.")
    print("To filter for trains, provide routes.txt, trips.txt, and stop_times.txt")
    return stops_df[['stop_id', 'stop_lat', 'stop_lon']]


def compute_block_group_centroids(block_groups_gdf):
    """
    Compute centroids for each block group.
    
    Parameters:
    -----------
    block_groups_gdf : gpd.GeoDataFrame
        Block groups with geometry
        
    Returns:
    --------
    gpd.GeoDataFrame
        Block groups with centroid points
    """
    print("Computing block group centroids...")
    block_groups_gdf = block_groups_gdf.copy()
    block_groups_gdf['centroid'] = block_groups_gdf.geometry.centroid
    block_groups_gdf['centroid_lat'] = block_groups_gdf['centroid'].y
    block_groups_gdf['centroid_lon'] = block_groups_gdf['centroid'].x
    return block_groups_gdf


def calculate_distance_to_nearest_station(block_groups_gdf, train_stations_df):
    """
    Calculate distance from each block group centroid to nearest train station.
    
    Parameters:
    -----------
    block_groups_gdf : gpd.GeoDataFrame
        Block groups with centroids
    train_stations_df : pd.DataFrame
        Train stations with stop_lat, stop_lon
        
    Returns:
    --------
    gpd.GeoDataFrame
        Block groups with distance to nearest station
    """
    print("Calculating distances to nearest train stations...")
    
    block_groups_gdf = block_groups_gdf.copy()
    
    # Create list of station coordinates
    stations = [(row['stop_lat'], row['stop_lon']) for _, row in train_stations_df.iterrows()]
    
    # Calculate distance to nearest station for each block group
    distances = []
    nearest_station_ids = []
    
    for idx, row in block_groups_gdf.iterrows():
        centroid = (row['centroid_lat'], row['centroid_lon'])
        
        # Calculate distance to all stations
        dists = [geodesic(centroid, station).miles for station in stations]
        
        # Find nearest
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        distances.append(min_dist)
        nearest_station_ids.append(train_stations_df.iloc[min_idx]['stop_id'])
    
    block_groups_gdf['distance_to_nearest_station_miles'] = distances
    block_groups_gdf['nearest_station_id'] = nearest_station_ids
    
    print(f"Distance range: {min(distances):.2f} - {max(distances):.2f} miles")
    
    return block_groups_gdf


def categorize_accessibility_zones(block_groups_gdf):
    """
    Categorize block groups into accessibility zones:
    - walk-zone: 0-0.5 miles
    - bike-zone: 0.5-2 miles
    - far-zone: 2+ miles
    
    Parameters:
    -----------
    block_groups_gdf : gpd.GeoDataFrame
        Block groups with distance_to_nearest_station_miles
        
    Returns:
    --------
    gpd.GeoDataFrame
        Block groups with accessibility_zone column
    """
    print("Categorizing accessibility zones...")
    
    block_groups_gdf = block_groups_gdf.copy()
    
    def categorize_distance(dist):
        if dist <= 0.5:
            return 'walk-zone'
        elif dist <= 2.0:
            return 'bike-zone'
        else:
            return 'far-zone'
    
    block_groups_gdf['accessibility_zone'] = block_groups_gdf['distance_to_nearest_station_miles'].apply(
        categorize_distance
    )
    
    # Print distribution
    zone_counts = block_groups_gdf['accessibility_zone'].value_counts()
    print("\nZone distribution:")
    for zone, count in zone_counts.items():
        print(f"  {zone}: {count} block groups ({count/len(block_groups_gdf)*100:.1f}%)")
    
    return block_groups_gdf


def load_acs_population(file_path, counties=['25025', '25017', '25021']):
    """
    Load ACS population data (B01003 - Total Population).
    
    Parameters:
    -----------
    file_path : str
        Path to ACS population CSV file
    counties : list
        County FIPS codes to filter
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with GEOID and total_population
    """
    print("Loading ACS population data (B01003)...")
    df = pd.read_csv(file_path)
    
    # Process GEOID (same as income data)
    if 'GEOID' not in df.columns:
        if 'GEO_ID' in df.columns:
            df['GEOID'] = df['GEO_ID'].str.replace('14000US', '')
        elif 'geoid' in df.columns:
            df['GEOID'] = df['geoid']
        else:
            raise ValueError("Could not find GEOID column")
    
    df['GEOID'] = df['GEOID'].astype(str).str.zfill(12)
    df['county_code'] = df['GEOID'].str[2:5]
    df = df[df['county_code'].isin(counties)]
    
    # Find population column (B01003_001E is total population)
    pop_cols = [col for col in df.columns if 'B01003' in col or 'population' in col.lower() or 'total' in col.lower()]
    if not pop_cols:
        raise ValueError("Could not find population column")
    
    pop_col = pop_cols[0]
    df = df[['GEOID', pop_col]].copy()
    df.columns = ['GEOID', 'total_population']
    
    df = df[df['total_population'].notna()]
    df = df[df['total_population'] > 0]
    
    print(f"Loaded {len(df)} records with population data")
    return df


def load_acs_vehicle_availability(file_path, counties=['25025', '25017', '25021']):
    """
    Load ACS vehicle availability data (B08201 - Households by Vehicles Available).
    
    We want households with 0 vehicles available.
    
    Parameters:
    -----------
    file_path : str
        Path to ACS vehicle availability CSV file
    counties : list
        County FIPS codes to filter
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with GEOID and households_without_car
    """
    print("Loading ACS vehicle availability data (B08201)...")
    df = pd.read_csv(file_path)
    
    # Process GEOID
    if 'GEOID' not in df.columns:
        if 'GEO_ID' in df.columns:
            df['GEOID'] = df['GEO_ID'].str.replace('14000US', '')
        elif 'geoid' in df.columns:
            df['GEOID'] = df['geoid']
        else:
            raise ValueError("Could not find GEOID column")
    
    df['GEOID'] = df['GEOID'].astype(str).str.zfill(12)
    df['county_code'] = df['GEOID'].str[2:5]
    df = df[df['county_code'].isin(counties)]
    
    # B08201_002E is "Total households - No vehicle available"
    vehicle_cols = [col for col in df.columns if 'B08201_002' in col or ('vehicle' in col.lower() and 'no' in col.lower())]
    if not vehicle_cols:
        # Try to find total households and calculate
        total_cols = [col for col in df.columns if 'B08201_001' in col]
        if total_cols:
            vehicle_cols = total_cols  # We'll use total and calculate percentage
        else:
            raise ValueError("Could not find vehicle availability column")
    
    vehicle_col = vehicle_cols[0]
    
    # If we have "no vehicle" column, use it directly
    if '002' in vehicle_col or 'no' in vehicle_col.lower():
        df = df[['GEOID', vehicle_col]].copy()
        df.columns = ['GEOID', 'households_without_car']
    else:
        # Use total households (we'll need to calculate percentage separately)
        df = df[['GEOID', vehicle_col]].copy()
        df.columns = ['GEOID', 'total_households']
        # For now, set to 0 - user can update with actual no-vehicle data
        df['households_without_car'] = 0
    
    df = df[df['households_without_car'].notna()]
    
    print(f"Loaded {len(df)} records with vehicle availability data")
    return df


def define_income_categories(income_series, low_threshold=None, high_threshold=None):
    """
    Define income categories (low, medium, high).
    
    Parameters:
    -----------
    income_series : pd.Series
        Median household income values
    low_threshold : float, optional
        Threshold for low income (default: 33rd percentile)
    high_threshold : float, optional
        Threshold for high income (default: 67th percentile)
        
    Returns:
    --------
    pd.Series
        Income category labels
    """
    if low_threshold is None:
        low_threshold = income_series.quantile(0.33)
    if high_threshold is None:
        high_threshold = income_series.quantile(0.67)
    
    def categorize(income):
        if income <= low_threshold:
            return 'low-income'
        elif income <= high_threshold:
            return 'medium-income'
        else:
            return 'high-income'
    
    return income_series.apply(categorize)


def analyze_income_by_zone(merged_gdf):
    """
    Analyze income distribution by accessibility zone.
    Calculate: "X% of low-income vs Y% of high-income residents live in bike-but-not-walk zone"
    
    Parameters:
    -----------
    merged_gdf : gpd.GeoDataFrame
        Block groups with income, population, and accessibility zones
        
    Returns:
    --------
    dict
        Analysis results
    """
    print("\n" + "="*60)
    print("INCOME DISTRIBUTION BY ACCESSIBILITY ZONE")
    print("="*60)
    
    # Define income categories
    merged_gdf['income_category'] = define_income_categories(
        merged_gdf['median_household_income']
    )
    
    # Calculate weighted statistics (by population)
    if 'total_population' in merged_gdf.columns:
        # Weight by population
        total_pop = merged_gdf['total_population'].sum()
        
        results = {}
        for income_cat in ['low-income', 'medium-income', 'high-income']:
            cat_data = merged_gdf[merged_gdf['income_category'] == income_cat]
            cat_pop = cat_data['total_population'].sum()
            
            print(f"\n{income_cat.upper()}:")
            print(f"  Total population: {cat_pop:,.0f}")
            
            for zone in ['walk-zone', 'bike-zone', 'far-zone']:
                zone_data = cat_data[cat_data['accessibility_zone'] == zone]
                zone_pop = zone_data['total_population'].sum()
                zone_pct = (zone_pop / cat_pop * 100) if cat_pop > 0 else 0
                
                print(f"    {zone}: {zone_pop:,.0f} residents ({zone_pct:.1f}%)")
                
                # Store key result: bike-but-not-walk zone
                if zone == 'bike-zone':
                    results[f'{income_cat}_bike_zone_pct'] = zone_pct
        
        # Key finding
        print("\n" + "="*60)
        print("KEY FINDING:")
        print("="*60)
        low_bike_pct = results.get('low-income_bike_zone_pct', 0)
        high_bike_pct = results.get('high-income_bike_zone_pct', 0)
        
        print(f"{low_bike_pct:.1f}% of low-income residents live in the bike-but-not-walk zone")
        print(f"{high_bike_pct:.1f}% of high-income residents live in the bike-but-not-walk zone")
        
        if low_bike_pct > high_bike_pct:
            diff = low_bike_pct - high_bike_pct
            print(f"\nLow-income residents are {diff:.1f} percentage points MORE likely")
            print("to live in the bike-but-not-walk zone than high-income residents.")
        else:
            diff = high_bike_pct - low_bike_pct
            print(f"\nHigh-income residents are {diff:.1f} percentage points MORE likely")
            print("to live in the bike-but-not-walk zone than low-income residents.")
        
        return results
    else:
        # Unweighted analysis (by block group count)
        print("\nNote: Population data not available. Using block group counts.")
        
        for income_cat in ['low-income', 'medium-income', 'high-income']:
            cat_data = merged_gdf[merged_gdf['income_category'] == income_cat]
            total_bgs = len(cat_data)
            
            print(f"\n{income_cat.upper()}: {total_bgs} block groups")
            
            for zone in ['walk-zone', 'bike-zone', 'far-zone']:
                zone_count = len(cat_data[cat_data['accessibility_zone'] == zone])
                zone_pct = (zone_count / total_bgs * 100) if total_bgs > 0 else 0
                print(f"  {zone}: {zone_count} block groups ({zone_pct:.1f}%)")
        
        return {}


def main():
    """
    Main function to execute Step 2.
    """
    # File paths - UPDATE THESE
    gtfs_stops_file = 'data/gtfs/stops.txt'
    gtfs_routes_file = 'data/gtfs/routes.txt'  # Optional if route_type in stops
    step1_output = 'data/neighborhood_income_merged.geojson'  # From Step 1
    acs_population_file = 'data/acs_population_b01003.csv'  # Optional
    acs_vehicle_file = 'data/acs_vehicles_b08201.csv'  # Optional
    
    target_counties = ['25025', '25017', '25021']  # Suffolk, Middlesex, Norfolk
    
    # Load Step 1 output (block groups with income)
    print("Loading Step 1 output...")
    if not Path(step1_output).exists():
        print(f"ERROR: Step 1 output not found: {step1_output}")
        print("Please run step1_merge_acs_income.py first")
        return
    
    block_groups_gdf = gpd.read_file(step1_output)
    print(f"Loaded {len(block_groups_gdf)} block groups with income data")
    
    # Load GTFS stops
    if not Path(gtfs_stops_file).exists():
        print(f"ERROR: GTFS stops file not found: {gtfs_stops_file}")
        print("Download MBTA GTFS data from: https://www.mbta.com/developers/gtfs")
        return
    
    stops_df = load_gtfs_stops(gtfs_stops_file)
    
    # Try to get train stations
    routes_df = None
    if Path(gtfs_routes_file).exists():
        routes_df = load_gtfs_routes(gtfs_routes_file)
    
    train_stations_df = get_train_stations(stops_df, routes_df=routes_df)
    
    # Compute centroids
    block_groups_gdf = compute_block_group_centroids(block_groups_gdf)
    
    # Calculate distances
    block_groups_gdf = calculate_distance_to_nearest_station(block_groups_gdf, train_stations_df)
    
    # Categorize zones
    block_groups_gdf = categorize_accessibility_zones(block_groups_gdf)
    
    # Load optional population data
    if Path(acs_population_file).exists():
        pop_df = load_acs_population(acs_population_file, counties=target_counties)
        block_groups_gdf = block_groups_gdf.merge(pop_df, on='GEOID', how='left')
        block_groups_gdf['total_population'] = block_groups_gdf['total_population'].fillna(0)
    else:
        print("Note: Population data not found. Analysis will use block group counts.")
        block_groups_gdf['total_population'] = 1  # Default weight
    
    # Load optional vehicle availability data
    if Path(acs_vehicle_file).exists():
        vehicle_df = load_acs_vehicle_availability(acs_vehicle_file, counties=target_counties)
        block_groups_gdf = block_groups_gdf.merge(vehicle_df, on='GEOID', how='left')
    
    # Analyze income by zone
    results = analyze_income_by_zone(block_groups_gdf)
    
    # Save results
    output_file = 'data/transit_accessibility_analysis.geojson'
    block_groups_gdf.to_file(output_file, driver='GeoJSON')
    print(f"\nResults saved to {output_file}")
    
    # Save summary CSV
    csv_output = 'data/transit_accessibility_summary.csv'
    summary_cols = ['GEOID', 'median_household_income', 'income_category', 
                   'accessibility_zone', 'distance_to_nearest_station_miles',
                   'total_population', 'nearest_station_id']
    if 'households_without_car' in block_groups_gdf.columns:
        summary_cols.append('households_without_car')
    
    block_groups_gdf[summary_cols].to_csv(csv_output, index=False)
    print(f"Summary saved to {csv_output}")


if __name__ == '__main__':
    main()

