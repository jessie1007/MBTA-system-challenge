#!/usr/bin/env python3
"""
Create income heat map with MBTA train stations overlaid.

- Colored polygons show median household income by block group
- Blue dots mark MBTA train stations (Heavy Rail, Light Rail, Commuter Rail)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
import contextily as ctx

# Set plotting style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

def load_block_groups_with_income():
    """Load block groups with geometry and income data."""
    shapefile_path = 'data/block_groups.shp'
    income_file = 'data/neighborhood_income_merged.geojson'
    
    if not Path(shapefile_path).exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    print(f"Loading block group geometry from: {shapefile_path}")
    try:
        import fiona
        from shapely.geometry import shape
        features = []
        with fiona.open(shapefile_path) as src:
            crs = src.crs
            for feature in src:
                features.append({
                    'geometry': shape(feature['geometry']),
                    **feature['properties']
                })
        block_groups_gdf = gpd.GeoDataFrame(features, crs=crs)
        print(f"✅ Loaded {len(block_groups_gdf)} block groups with geometry")
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        raise
    
    # Load income data
    if Path(income_file).exists():
        print(f"Loading income data from: {income_file}")
        try:
            with open(income_file, 'r') as f:
                geojson_data = json.load(f)
            features = geojson_data.get('features', [])
            income_data = []
            for feature in features:
                props = feature.get('properties', {})
                income_data.append(props)
            income_df = pd.DataFrame(income_data)
            print(f"✅ Loaded income data for {len(income_df)} block groups")
            
            # Merge with geometry
            if 'GEOID' in block_groups_gdf.columns and 'GEOID' in income_df.columns:
                # Ensure GEOID format matches
                block_groups_gdf['GEOID'] = block_groups_gdf['GEOID'].astype(str).str.zfill(12)
                income_df['GEOID'] = income_df['GEOID'].astype(str).str.zfill(12)
                
                # Merge
                gdf = block_groups_gdf.merge(income_df, on='GEOID', how='inner')
                print(f"✅ Merged: {len(gdf)} block groups with geometry and income data")
                return gdf
        except Exception as e:
            print(f"Warning: Could not load income data: {e}")
    
    return block_groups_gdf

def load_train_stations_from_gtfs():
    """
    Load train stations from GTFS data.
    Filters for Heavy Rail, Light Rail, and Commuter Rail routes.
    """
    print("\n" + "="*70)
    print("Loading MBTA Train Stations from GTFS")
    print("="*70)
    
    # GTFS file locations
    gtfs_locations = ['data/gtfs', 'MBTA_GTFS']
    
    stops_file = None
    routes_file = None
    trips_file = None
    stop_times_file = None
    
    # Find GTFS files
    for base_path in gtfs_locations:
        base = Path(base_path)
        if (base / 'stops.txt').exists() and stops_file is None:
            stops_file = base / 'stops.txt'
        if (base / 'routes.txt').exists() and routes_file is None:
            routes_file = base / 'routes.txt'
        if (base / 'trips.txt').exists() and trips_file is None:
            trips_file = base / 'trips.txt'
        if (base / 'stop_times.txt').exists() and stop_times_file is None:
            stop_times_file = base / 'stop_times.txt'
    
    if stops_file is None:
        raise FileNotFoundError("Could not find GTFS stops.txt file")
    
    print(f"Loading stops from: {stops_file}")
    stops_df = pd.read_csv(stops_file)
    stops_df = stops_df[stops_df['stop_lat'].notna() & stops_df['stop_lon'].notna()]
    print(f"✅ Loaded {len(stops_df)} stops")
    
    # Route types: 0 = Light Rail, 1 = Heavy Rail, 2 = Commuter Rail
    train_route_types = [0, 1, 2]
    
    train_station_ids = set()
    
    if routes_file and trips_file and stop_times_file:
        print(f"Loading routes from: {routes_file}")
        routes_df = pd.read_csv(routes_file)
        
        # Get train route IDs
        train_routes = routes_df[routes_df['route_type'].isin(train_route_types)]
        train_route_ids = train_routes['route_id'].unique()
        print(f"✅ Found {len(train_route_ids)} train routes (Heavy Rail, Light Rail, Commuter Rail)")
        
        # Match stops to routes through stop_times and trips
        print(f"Loading trips from: {trips_file}")
        trips_df = pd.read_csv(trips_file, low_memory=False)
        
        print(f"Loading stop_times from: {stop_times_file}")
        stop_times_df = pd.read_csv(stop_times_file, low_memory=False)
        
        # Join: stop_times -> trips -> routes
        if 'trip_id' in stop_times_df.columns and 'trip_id' in trips_df.columns:
            if 'route_id' in trips_df.columns and 'stop_id' in stop_times_df.columns:
                print("Matching stops to routes via stop_times and trips...")
                merged = stop_times_df.merge(
                    trips_df[['trip_id', 'route_id']], 
                    on='trip_id', 
                    how='left'
                )
                train_station_ids = set(
                    merged[merged['route_id'].isin(train_route_ids)]['stop_id'].unique()
                )
                print(f"✅ Found {len(train_station_ids)} train stations via route matching")
    else:
        print("⚠️  Warning: Could not find routes.txt, trips.txt, or stop_times.txt")
        print("   Will use all stops (not ideal)")
        train_station_ids = set(stops_df['stop_id'].unique())
    
    # Filter stops to only train stations
    if train_station_ids:
        train_stations = stops_df[stops_df['stop_id'].isin(train_station_ids)].copy()
    else:
        train_stations = stops_df.copy()
    
    print(f"✅ Filtered to {len(train_stations)} train stations")
    
    # Create GeoDataFrame
    from shapely.geometry import Point
    geometry = [Point(lon, lat) for lon, lat in zip(train_stations['stop_lon'], train_stations['stop_lat'])]
    train_stations_gdf = gpd.GeoDataFrame(train_stations, geometry=geometry, crs='EPSG:4326')
    
    return train_stations_gdf

def create_income_map_with_stations(block_groups_gdf, train_stations_gdf, 
                                    output_file='neighborhood_income_map_with_stations.png'):
    """
    Create income heat map with MBTA stations overlaid.
    
    Parameters:
    -----------
    block_groups_gdf : gpd.GeoDataFrame
        Block groups with income data and geometry
    train_stations_gdf : gpd.GeoDataFrame
        Train stations with geometry
    output_file : str
        Output file path
    """
    print("\n" + "="*70)
    print("Creating Income Map with MBTA Stations")
    print("="*70)
    
    # Ensure CRS is set
    if block_groups_gdf.crs is None:
        block_groups_gdf.set_crs('EPSG:4326', inplace=True)
    if train_stations_gdf.crs is None:
        train_stations_gdf.set_crs('EPSG:4326', inplace=True)
    
    # Convert to Web Mercator for basemap
    print("Converting to Web Mercator projection...")
    block_groups_mercator = block_groups_gdf.to_crs(epsg=3857)
    train_stations_mercator = train_stations_gdf.to_crs(epsg=3857)
    print("✅ Converted to Web Mercator")
    
    # Create figure
    print("Creating map...")
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Plot choropleth (block groups by income)
    if 'median_household_income' in block_groups_mercator.columns:
        print("Plotting income choropleth...")
        block_groups_mercator.plot(
            column='median_household_income', 
            cmap='YlOrRd', 
            legend=True,
            ax=ax,
            edgecolor='white',
            linewidth=0.2,
            alpha=0.85,
            legend_kwds={
                'label': 'Median Household Income ($)',
                'orientation': 'vertical',
                'shrink': 0.7,
                'pad': 0.02
            }
        )
    else:
        print("⚠️  Warning: No income data found. Plotting without color scale.")
        block_groups_mercator.plot(
            ax=ax,
            color='lightgray',
            edgecolor='white',
            linewidth=0.2,
            alpha=0.7
        )
    
    # Overlay train stations as blue dots
    print(f"Overlaying {len(train_stations_mercator)} train stations...")
    train_stations_mercator.plot(
        ax=ax, 
        color='blue', 
        markersize=40,
        marker='o',
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8,
        label='MBTA Train Station'
    )
    
    # Add basemap
    try:
        print("Adding basemap...")
        ctx.add_basemap(
            ax, 
            crs=block_groups_mercator.crs, 
            source=ctx.providers.CartoDB.Positron,
            alpha=0.5
        )
        print("✅ Basemap added")
    except Exception as e:
        print(f"⚠️  Could not add basemap: {e}")
    
    # Add title
    ax.set_title(
        'Income Distribution by Block Group with MBTA Train Stations\n' + 
        'Suffolk, Middlesex, and Norfolk Counties, MA',
        fontsize=18, 
        fontweight='bold', 
        pad=20
    )
    
    # Add legend for stations
    station_legend = mpatches.Patch(
        facecolor='blue', 
        edgecolor='white',
        label='MBTA Train Station (Heavy Rail, Light Rail, Commuter Rail)',
        alpha=0.8
    )
    ax.legend(handles=[station_legend], loc='lower right', fontsize=11, framealpha=0.9)
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Map saved to: {output_file}")
    plt.show()
    
    return fig

def main():
    """Main function."""
    print("="*70)
    print("Income Map with MBTA Train Stations")
    print("="*70)
    
    # Load block groups with income
    print("\nStep 1: Loading block groups with income data...")
    block_groups_gdf = load_block_groups_with_income()
    
    # Load train stations
    print("\nStep 2: Loading train stations from GTFS...")
    train_stations_gdf = load_train_stations_from_gtfs()
    
    # Create map
    print("\nStep 3: Creating map...")
    create_income_map_with_stations(block_groups_gdf, train_stations_gdf)
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)

if __name__ == '__main__':
    main()




