#!/usr/bin/env python3
"""
Load MBTA Rail Stations from GTFS Data

Loads GTFS stops.txt and filters for:
- Heavy Rail (Blue, Red, Orange) - route_type 1
- Light Rail (Green Line) - route_type 0
- Commuter Rail (optional) - route_type 2

Creates GeoPandas points for each station.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# GTFS Route Types
# 0 = Light rail, streetcar, tram (Green Line)
# 1 = Subway, metro (Heavy Rail: Blue, Red, Orange)
# 2 = Rail (Commuter Rail)
# 3 = Bus
# 4 = Ferry

HEAVY_RAIL = 1  # Blue, Red, Orange Lines
LIGHT_RAIL = 0  # Green Line
COMMUTER_RAIL = 2  # Commuter Rail (optional)

def load_gtfs_stops(gtfs_stops_file):
    """
    Load GTFS stops.txt file.
    
    Parameters:
    -----------
    gtfs_stops_file : str
        Path to GTFS stops.txt file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with stop information
    """
    print(f"Loading GTFS stops from: {gtfs_stops_file}")
    
    if not Path(gtfs_stops_file).exists():
        raise FileNotFoundError(f"GTFS stops file not found: {gtfs_stops_file}")
    
    stops_df = pd.read_csv(gtfs_stops_file)
    
    # Clean data - remove stops without coordinates
    stops_df = stops_df[stops_df['stop_lat'].notna() & stops_df['stop_lon'].notna()]
    
    print(f"✅ Loaded {len(stops_df)} stops with coordinates")
    print(f"   Columns: {list(stops_df.columns)}")
    
    return stops_df

def load_gtfs_routes(gtfs_routes_file):
    """
    Load GTFS routes.txt file to get route_type information.
    
    Parameters:
    -----------
    gtfs_routes_file : str
        Path to GTFS routes.txt file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with route_id and route_type
    """
    print(f"Loading GTFS routes from: {gtfs_routes_file}")
    
    if not Path(gtfs_routes_file).exists():
        print("⚠️  Routes file not found, will use alternative filtering method")
        return None
    
    routes_df = pd.read_csv(gtfs_routes_file)
    
    if 'route_id' not in routes_df.columns or 'route_type' not in routes_df.columns:
        print("⚠️  Routes file missing required columns (route_id, route_type)")
        return None
    
    print(f"✅ Loaded {len(routes_df)} routes")
    
    return routes_df[['route_id', 'route_type', 'route_short_name', 'route_long_name']]

def filter_train_stations(stops_df, routes_df=None, include_commuter_rail=False):
    """
    Filter stops to get only train stations.
    
    Parameters:
    -----------
    stops_df : pd.DataFrame
        Stops data from stops.txt
    routes_df : pd.DataFrame, optional
        Routes data with route_type
    include_commuter_rail : bool
        Whether to include commuter rail stations (default: False)
        
    Returns:
    --------
    pd.DataFrame
        Filtered stops for train stations only
    """
    print("\nFiltering for train stations...")
    
    # Define route types to include
    train_route_types = [HEAVY_RAIL, LIGHT_RAIL]  # Heavy Rail and Light Rail
    if include_commuter_rail:
        train_route_types.append(COMMUTER_RAIL)
        print("  Including: Heavy Rail, Light Rail, Commuter Rail")
    else:
        print("  Including: Heavy Rail, Light Rail")
    
    # Method 1: If we have routes data, filter through route matching
    if routes_df is not None:
        # Get train route IDs
        train_routes = routes_df[routes_df['route_type'].isin(train_route_types)]
        train_route_ids = train_routes['route_id'].unique()
        
        print(f"  Found {len(train_route_ids)} train routes")
        
        # Try to match stops to routes through stop_times and trips
        train_station_ids = set()
        
        # Check for stop_times and trips files
        gtfs_locations = ['data/gtfs', 'MBTA_GTFS']
        
        stop_times_df = None
        trips_df = None
        
        for base_path in gtfs_locations:
            stop_times_file = Path(base_path) / 'stop_times.txt'
            trips_file = Path(base_path) / 'trips.txt'
            
            if stop_times_file.exists() and stop_times_df is None:
                print(f"  Loading stop_times from: {stop_times_file}")
                stop_times_df = pd.read_csv(stop_times_file, low_memory=False)
            
            if trips_file.exists() and trips_df is None:
                print(f"  Loading trips from: {trips_file}")
                trips_df = pd.read_csv(trips_file, low_memory=False)
        
        # Join: stop_times -> trips -> routes
        if stop_times_df is not None and trips_df is not None:
            if 'trip_id' in stop_times_df.columns and 'trip_id' in trips_df.columns:
                if 'route_id' in trips_df.columns and 'stop_id' in stop_times_df.columns:
                    print("  Matching stops to routes via stop_times and trips...")
                    merged = stop_times_df.merge(
                        trips_df[['trip_id', 'route_id']], 
                        on='trip_id', 
                        how='left'
                    )
                    train_station_ids = set(
                        merged[merged['route_id'].isin(train_route_ids)]['stop_id'].unique()
                    )
                    print(f"  ✅ Found {len(train_station_ids)} train stations via route matching")
        
        # Filter stops
        if train_station_ids:
            train_stations = stops_df[stops_df['stop_id'].isin(train_station_ids)].copy()
        else:
            print("  ⚠️  Could not match via routes, using name filtering...")
            train_stations = filter_by_name(stops_df, include_commuter_rail)
    else:
        # Method 2: Filter by stop name patterns
        print("  Using name-based filtering (routes.txt not available)...")
        train_stations = filter_by_name(stops_df, include_commuter_rail)
    
    print(f"\n✅ Filtered to {len(train_stations)} train stations")
    
    return train_stations

def filter_by_name(stops_df, include_commuter_rail=False):
    """Filter stations by name patterns."""
    # Heavy Rail keywords (Red, Orange, Blue Lines)
    heavy_rail_keywords = [
        'red line', 'orange line', 'blue line',
        'red', 'orange', 'blue',
        'subway', 'station'
    ]
    
    # Light Rail keywords (Green Line)
    light_rail_keywords = [
        'green line', 'green',
        'trolley', 'streetcar'
    ]
    
    # Commuter Rail keywords
    commuter_keywords = [
        'commuter rail', 'commuter',
        'south station', 'north station', 'back bay'
    ]
    
    all_keywords = heavy_rail_keywords + light_rail_keywords
    if include_commuter_rail:
        all_keywords += commuter_keywords
    
    if 'stop_name' in stops_df.columns:
        pattern = '|'.join(all_keywords)
        train_stations = stops_df[
            stops_df['stop_name'].str.lower().str.contains(pattern, na=False, regex=True)
        ].copy()
    else:
        # If no stop_name, return all stops (not ideal)
        print("  ⚠️  No stop_name column, returning all stops")
        train_stations = stops_df.copy()
    
    return train_stations

def create_station_points(stations_df):
    """
    Create GeoPandas GeoDataFrame with Point geometries for each station.
    
    Parameters:
    -----------
    stations_df : pd.DataFrame
        DataFrame with stop_id, stop_lat, stop_lon
        
    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame with Point geometries
    """
    print("\nCreating GeoPandas points for stations...")
    
    # Create Point geometries from lat/lon
    geometry = [
        Point(lon, lat) 
        for lon, lat in zip(stations_df['stop_lon'], stations_df['stop_lat'])
    ]
    
    # Create GeoDataFrame
    stations_gdf = gpd.GeoDataFrame(stations_df, geometry=geometry, crs='EPSG:4326')
    
    print(f"✅ Created {len(stations_gdf)} station points")
    print(f"   CRS: {stations_gdf.crs}")
    
    return stations_gdf

def main():
    """Main function."""
    import sys
    
    # File paths - try multiple locations
    gtfs_stops_locations = [
        'data/gtfs/stops.txt',
        'MBTA_GTFS/stops.txt'
    ]
    
    gtfs_routes_locations = [
        'data/gtfs/routes.txt',
        'MBTA_GTFS/routes.txt'
    ]
    
    # Parse command line arguments
    include_commuter_rail = '--commuter-rail' in sys.argv or '-c' in sys.argv
    
    print("="*70)
    print("Load MBTA Rail Stations from GTFS")
    print("="*70)
    
    if include_commuter_rail:
        print("Including Commuter Rail stations")
    else:
        print("Excluding Commuter Rail stations (use --commuter-rail to include)")
    
    # Load stops
    stops_df = None
    for loc in gtfs_stops_locations:
        if Path(loc).exists():
            stops_df = load_gtfs_stops(loc)
            break
    
    if stops_df is None:
        print("ERROR: Could not find GTFS stops.txt")
        print("Please download from: https://cdn.mbta.com/MBTA_GTFS.zip")
        return
    
    # Load routes
    routes_df = None
    for loc in gtfs_routes_locations:
        if Path(loc).exists():
            routes_df = load_gtfs_routes(loc)
            break
    
    # Filter for train stations
    train_stations_df = filter_train_stations(stops_df, routes_df, include_commuter_rail)
    
    # Create GeoPandas points
    train_stations_gdf = create_station_points(train_stations_df)
    
    # Display summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Total train stations: {len(train_stations_gdf)}")
    
    if 'stop_name' in train_stations_gdf.columns:
        print("\nSample stations:")
        for i, row in train_stations_gdf.head(10).iterrows():
            print(f"  {row['stop_name']} ({row['stop_id']})")
    
    # Save to file
    output_file = 'data/mbta_train_stations.geojson'
    print(f"\nSaving to: {output_file}")
    try:
        train_stations_gdf.to_file(output_file, driver='GeoJSON')
        print(f"✅ Saved {len(train_stations_gdf)} stations to {output_file}")
    except Exception as e:
        print(f"⚠️  Could not save as GeoJSON: {e}")
        # Save as CSV instead
        csv_file = 'data/mbta_train_stations.csv'
        train_stations_gdf.drop(columns='geometry').to_csv(csv_file, index=False)
        print(f"✅ Saved to {csv_file} (without geometry)")
    
    # Also save as shapefile
    shp_file = 'data/mbta_train_stations.shp'
    try:
        train_stations_gdf.to_file(shp_file, driver='ESRI Shapefile')
        print(f"✅ Saved to {shp_file}")
    except Exception as e:
        print(f"⚠️  Could not save as shapefile: {e}")
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)
    print("\nUsage:")
    print("  python3 load_mbta_stations.py              # Heavy + Light Rail only")
    print("  python3 load_mbta_stations.py --commuter-rail  # Include Commuter Rail")
    
    return train_stations_gdf

if __name__ == '__main__':
    stations_gdf = main()

