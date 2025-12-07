#!/usr/bin/env python3
"""
Complete Transit Accessibility Analysis
Following the methodology:
1. Load MBTA rail stations from GTFS
2. Calculate distances using proper CRS and spatial indexing
3. Assign access zones
4. Join population and income data
5. Create all requested outputs
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from shapely.geometry import Point
from shapely.ops import nearest_points
import json

# Set plotting style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
        print("Using default matplotlib style")

sns.set_palette("husl")

# Massachusetts State Plane CRS (meters)
MA_CRS = 'EPSG:26986'  # Massachusetts State Plane Mainland

def load_train_stations_from_gtfs():
    """
    Load MBTA rail stations from GTFS stops.txt
    Filter for route_type: 0 (light rail), 1 (heavy rail), 2 (commuter rail)
    """
    # Try multiple locations
    gtfs_stops_locations = [
        'data/gtfs/stops.txt',
        'MBTA_GTFS/stops.txt'
    ]
    
    gtfs_routes_locations = [
        'data/gtfs/routes.txt',
        'MBTA_GTFS/routes.txt'
    ]
    
    gtfs_stop_times_locations = [
        'data/gtfs/stop_times.txt',
        'MBTA_GTFS/stop_times.txt'
    ]
    
    gtfs_trips_locations = [
        'data/gtfs/trips.txt',
        'MBTA_GTFS/trips.txt'
    ]
    
    # Load stops
    stops_df = None
    for loc in gtfs_stops_locations:
        if Path(loc).exists():
            print(f"Loading GTFS stops from: {loc}")
            stops_df = pd.read_csv(loc)
            break
    
    if stops_df is None:
        raise FileNotFoundError("GTFS stops.txt not found")
    
    # Clean stops
    stops_df = stops_df[stops_df['stop_lat'].notna() & stops_df['stop_lon'].notna()]
    print(f"Found {len(stops_df)} stops with coordinates")
    
    # Load routes to filter by route_type
    routes_df = None
    for loc in gtfs_routes_locations:
        if Path(loc).exists():
            routes_df = pd.read_csv(loc)
            break
    
    TRAIN_ROUTE_TYPES = [0, 1, 2]  # Light rail, subway, rail
    
    if routes_df is not None and 'route_type' in routes_df.columns:
        train_route_ids = routes_df[routes_df['route_type'].isin(TRAIN_ROUTE_TYPES)]['route_id'].unique()
        print(f"Found {len(train_route_ids)} train routes")
        
        # Try to match stops to routes through stop_times and trips
        train_station_ids = set()
        
        # Load stop_times and trips if available
        stop_times_df = None
        trips_df = None
        
        for loc in gtfs_stop_times_locations:
            if Path(loc).exists():
                stop_times_df = pd.read_csv(loc)
                break
        
        for loc in gtfs_trips_locations:
            if Path(loc).exists():
                trips_df = pd.read_csv(loc)
                break
        
        if stop_times_df is not None and trips_df is not None:
            # Join: stop_times -> trips -> routes
            if 'trip_id' in stop_times_df.columns and 'trip_id' in trips_df.columns:
                if 'route_id' in trips_df.columns:
                    merged = stop_times_df.merge(trips_df[['trip_id', 'route_id']], on='trip_id', how='left')
                    train_station_ids = set(merged[merged['route_id'].isin(train_route_ids)]['stop_id'].unique())
                    print(f"Found {len(train_station_ids)} train stations via route matching")
        
        # If we have train station IDs, filter stops
        if train_station_ids:
            train_stations = stops_df[stops_df['stop_id'].isin(train_station_ids)].copy()
        else:
            # Fallback: filter by stop name patterns
            train_keywords = ['station', 'subway', 'red line', 'orange line', 'blue line', 
                             'green line', 'commuter rail', 'mbta', 'park', 'square', 'center']
            if 'stop_name' in stops_df.columns:
                train_stations = stops_df[
                    stops_df['stop_name'].str.lower().str.contains('|'.join(train_keywords), na=False, regex=True)
                ].copy()
                print(f"Filtered to {len(train_stations)} train stations by name pattern")
            else:
                train_stations = stops_df
                print("Using all stops (cannot filter without route matching)")
    else:
        # No routes file, use name filtering
        train_keywords = ['station', 'subway', 'red line', 'orange line', 'blue line', 
                         'green line', 'commuter rail', 'mbta', 'park', 'square']
        if 'stop_name' in stops_df.columns:
            train_stations = stops_df[
                stops_df['stop_name'].str.lower().str.contains('|'.join(train_keywords), na=False, regex=True)
            ].copy()
        else:
            train_stations = stops_df
    
    # Create GeoDataFrame with Point geometries
    geometry = [Point(lon, lat) for lon, lat in zip(train_stations['stop_lon'], train_stations['stop_lat'])]
    train_stations_gdf = gpd.GeoDataFrame(train_stations, geometry=geometry, crs='EPSG:4326')
    
    print(f"✅ Created {len(train_stations_gdf)} train station points")
    return train_stations_gdf

def load_block_groups_with_geometry():
    """Load block group polygons with geometry."""
    shapefile_path = 'data/block_groups.shp'
    
    if not Path(shapefile_path).exists():
        raise FileNotFoundError(f"Block group shapefile not found: {shapefile_path}")
    
    print(f"Loading block group shapefile: {shapefile_path}")
    
    try:
        gdf = gpd.read_file(shapefile_path)
        print(f"✅ Loaded {len(gdf)} block groups")
        return gdf
    except Exception as e:
        print(f"Error loading with geopandas: {e}")
        # Try fiona directly
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
            gdf = gpd.GeoDataFrame(features, crs=crs)
            print(f"✅ Loaded {len(gdf)} block groups using fiona directly")
            print(f"   CRS: {crs}")
            return gdf
        except Exception as e2:
            print(f"Error with fiona: {e2}")
            raise

def calculate_distances_with_spatial_index(block_groups_gdf, train_stations_gdf, use_walking_distance=True):
    """
    Calculate distances using proper CRS and spatial indexing.
    Can use either straight-line distance or actual walking distance along street network.
    
    Parameters:
    -----------
    block_groups_gdf : gpd.GeoDataFrame
        Block groups with geometry
    train_stations_gdf : gpd.GeoDataFrame
        Train stations with geometry
    use_walking_distance : bool
        If True, use actual walking distance along street network (requires osmnx)
        If False, use straight-line distance
        
    Returns:
    --------
    gpd.GeoDataFrame
        Block groups with distance to nearest station
    """
    if use_walking_distance:
        try:
            from calculate_walking_distances import calculate_walking_distances_to_stations
            print("\n" + "="*60)
            print("Calculating WALKING Distances to Train Stations")
            print("(Using actual street network)")
            print("="*60)
            
            # Calculate bounding box for OSM download
            bounds = block_groups_gdf.total_bounds
            bbox = (bounds[3], bounds[1], bounds[2], bounds[0])  # (north, south, east, west)
            
            result = calculate_walking_distances_to_stations(
                block_groups_gdf, train_stations_gdf,
                bbox=bbox,
                place_name="Boston, Massachusetts, USA",
                cache_file='data/boston_street_network.graphml'
            )
            
            # Rename columns for compatibility
            if 'walking_distance_miles' in result.columns:
                result['distance_meters'] = result['walking_distance_meters']
                result['distance_to_nearest_station_miles'] = result['walking_distance_miles']
            
            return result
        except ImportError:
            print("⚠️  osmnx not available. Falling back to straight-line distance.")
            use_walking_distance = False
    
    # Fallback to straight-line distance
    print("\n" + "="*60)
    print("Calculating STRAIGHT-LINE Distances to Train Stations")
    print("="*60)
    
    # Step A: Convert to Massachusetts State Plane CRS (meters)
    print("Step A: Converting to Massachusetts State Plane CRS (EPSG:26986)...")
    block_groups_proj = block_groups_gdf.to_crs(MA_CRS)
    train_stations_proj = train_stations_gdf.to_crs(MA_CRS)
    print("✅ Converted to projected CRS")
    
    # Step B: Compute block group centroids
    print("\nStep B: Computing block group centroids...")
    block_groups_proj['centroid'] = block_groups_proj.geometry.centroid
    print("✅ Computed centroids")
    
    # Step C: Build spatial index for stations (using sindex)
    print("\nStep C: Building spatial index for stations...")
    train_stations_proj_sindex = train_stations_proj.sindex
    print("✅ Built spatial index")
    
    # Step D: Calculate shortest distance to nearest station
    print("\nStep D: Calculating distances to nearest stations...")
    distances_meters = []
    nearest_station_ids = []
    
    for idx, row in block_groups_proj.iterrows():
        centroid = row['centroid']
        
        # Use spatial index to find nearby stations first
        possible_matches_index = list(train_stations_proj_sindex.intersection(centroid.bounds))
        possible_matches = train_stations_proj.iloc[possible_matches_index]
        
        if len(possible_matches) > 0:
            # Calculate distance to all possible matches
            dists = [centroid.distance(station.geometry) for _, station in possible_matches.iterrows()]
            min_idx = np.argmin(dists)
            min_dist_meters = dists[min_idx]
            nearest_station_id = possible_matches.iloc[min_idx]['stop_id']
        else:
            # Fallback: calculate to all stations (shouldn't happen with spatial index)
            dists = [centroid.distance(station.geometry) for _, station in train_stations_proj.iterrows()]
            min_idx = np.argmin(dists)
            min_dist_meters = dists[min_idx]
            nearest_station_id = train_stations_proj.iloc[min_idx]['stop_id']
        
        distances_meters.append(min_dist_meters)
        nearest_station_ids.append(nearest_station_id)
        
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(block_groups_proj)} block groups...")
    
    # Convert meters to miles
    distances_miles = [d / 1609.34 for d in distances_meters]
    
    block_groups_proj['distance_meters'] = distances_meters
    block_groups_proj['distance_to_nearest_station_miles'] = distances_miles
    block_groups_proj['nearest_station_id'] = nearest_station_ids
    
    valid_distances = [d for d in distances_miles if not pd.isna(d)]
    if valid_distances:
        print(f"\n✅ Distance calculation complete!")
        print(f"   Range: {min(valid_distances):.2f} - {max(valid_distances):.2f} miles")
        print(f"   {len(valid_distances)}/{len(block_groups_proj)} block groups have valid distances")
    
    return block_groups_proj

def assign_access_zones(gdf):
    """Assign access zones based on distance."""
    print("\nAssigning access zones...")
    
    def categorize_zone(distance):
        if pd.isna(distance):
            return 'unknown'
        elif distance < 0.5:
            return 'walk'
        elif distance <= 2.0:
            return 'bike'
        else:
            return 'far'
    
    gdf['access_zone'] = gdf['distance_to_nearest_station_miles'].apply(categorize_zone)
    
    zone_counts = gdf['access_zone'].value_counts()
    print("\nZone distribution:")
    for zone, count in zone_counts.items():
        pct = (count / len(gdf)) * 100
        print(f"  {zone}: {count} block groups ({pct:.1f}%)")
    
    return gdf

def join_population_and_income(gdf):
    """Join population and income data to block groups."""
    print("\n" + "="*60)
    print("Joining Population and Income Data")
    print("="*60)
    
    # Ensure GEOID is in correct format
    if 'GEOID' in gdf.columns:
        gdf['GEOID'] = gdf['GEOID'].astype(str).str.zfill(12)
    elif 'GEOID10' in gdf.columns:
        gdf['GEOID'] = gdf['GEOID10'].astype(str).str.zfill(12)
    else:
        print("Warning: Could not find GEOID column")
        return gdf
    
    # Load income data
    income_csv = 'data/neighborhood_income_merged.csv'
    if Path(income_csv).exists():
        print(f"\nLoading income data: {income_csv}")
        income_df = pd.read_csv(income_csv)
        income_df['GEOID'] = income_df['GEOID'].astype(str).str.zfill(12)
        gdf = gdf.merge(income_df[['GEOID', 'median_household_income']], on='GEOID', how='left')
        print(f"✅ Joined income data: {gdf['median_household_income'].notna().sum()} block groups have income")
    else:
        print("⚠️  Income CSV not found")
        gdf['median_household_income'] = np.nan
    
    # Load population data (if available)
    pop_csv = 'data/acs_population_b01003.csv'
    if Path(pop_csv).exists():
        print(f"\nLoading population data: {pop_csv}")
        pop_df = pd.read_csv(pop_csv)
        # Process GEOID
        if 'GEOID' not in pop_df.columns:
            if 'GEO_ID' in pop_df.columns:
                pop_df['GEOID'] = pop_df['GEO_ID'].str.replace('14000US', '')
        pop_df['GEOID'] = pop_df['GEOID'].astype(str).str.zfill(12)
        
        # Find population column
        pop_cols = [col for col in pop_df.columns if 'B01003' in col or 'population' in col.lower() or 'total' in col.lower()]
        if pop_cols:
            pop_col = pop_cols[0]
            pop_df = pop_df[['GEOID', pop_col]].copy()
            pop_df.columns = ['GEOID', 'total_population']
            gdf = gdf.merge(pop_df, on='GEOID', how='left')
            print(f"✅ Joined population data: {gdf['total_population'].notna().sum()} block groups have population")
        else:
            print("⚠️  Could not find population column")
    else:
        print("⚠️  Population CSV not found, using block group count as proxy")
        gdf['total_population'] = 1
    
    # Fill missing population with 1 (for counting)
    gdf['total_population'] = gdf['total_population'].fillna(1)
    
    return gdf

def categorize_income_groups(gdf):
    """Categorize block groups into income groups (bottom 30%, middle 40%, top 30%)."""
    print("\nCategorizing income groups...")
    
    if 'median_household_income' not in gdf.columns:
        print("⚠️  No income data available")
        return gdf
    
    # Calculate percentiles
    income_30th = gdf['median_household_income'].quantile(0.30)
    income_70th = gdf['median_household_income'].quantile(0.70)
    
    print(f"  Bottom 30%: < ${income_30th:,.0f}")
    print(f"  Middle 40%: ${income_30th:,.0f} - ${income_70th:,.0f}")
    print(f"  Top 30%: > ${income_70th:,.0f}")
    
    def categorize(income):
        if pd.isna(income):
            return 'Unknown'
        elif income < income_30th:
            return 'Low Income (Bottom 30%)'
        elif income <= income_70th:
            return 'Middle Income (Middle 40%)'
        else:
            return 'High Income (Top 30%)'
    
    gdf['income_group'] = gdf['median_household_income'].apply(categorize)
    
    return gdf

def create_income_zone_analysis(gdf):
    """(a) % of low-income population in walk/bike/far zones"""
    print("\n" + "="*60)
    print("(a) Income Group Analysis by Access Zone")
    print("="*60)
    
    if 'income_group' not in gdf.columns or 'total_population' not in gdf.columns:
        print("⚠️  Missing income or population data")
        return None
    
    # Filter out unknown
    gdf_clean = gdf[(gdf['income_group'] != 'Unknown') & (gdf['access_zone'] != 'unknown')].copy()
    
    results = []
    for income_group in ['Low Income (Bottom 30%)', 'Middle Income (Middle 40%)', 'High Income (Top 30%)']:
        group_data = gdf_clean[gdf_clean['income_group'] == income_group]
        total_pop = group_data['total_population'].sum()
        
        for zone in ['walk', 'bike', 'far']:
            zone_pop = group_data[group_data['access_zone'] == zone]['total_population'].sum()
            zone_pct = (zone_pop / total_pop * 100) if total_pop > 0 else 0
            
            results.append({
                'Income Group': income_group,
                'Zone': zone,
                'Population': zone_pop,
                'Percentage': zone_pct
            })
    
    results_df = pd.DataFrame(results)
    
    # Create pivot table
    pivot = results_df.pivot_table(index='Income Group', columns='Zone', values='Percentage', aggfunc='sum')
    print("\n% of Population by Income Group and Zone:")
    print(pivot.to_string())
    
    return results_df

def create_zone_income_chart(results_df, output_file='income_zone_distribution.png'):
    """(d) Bar chart showing population distribution by zone + income group"""
    if results_df is None or len(results_df) == 0:
        print("⚠️  No data for chart")
        return None
    
    print(f"\nCreating bar chart: {output_file}")
    
    # Prepare data
    pivot = results_df.pivot_table(index='Income Group', columns='Zone', values='Percentage', aggfunc='sum')
    
    # Reindex to ensure correct order
    income_order = ['Low Income (Bottom 30%)', 'Middle Income (Middle 40%)', 'High Income (Top 30%)']
    zone_order = ['walk', 'bike', 'far']
    
    # Only reindex rows/columns that exist
    existing_rows = [r for r in income_order if r in pivot.index]
    existing_cols = [c for c in zone_order if c in pivot.columns]
    
    if existing_rows:
        pivot = pivot.reindex(existing_rows)
    if existing_cols:
        pivot = pivot.reindex(columns=existing_cols)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Population Distribution by Income Group and Access Zone', fontsize=16, fontweight='bold', y=1.02)
    
    # Chart 1: Stacked Bar
    ax1 = axes[0]
    pivot.plot(kind='bar', stacked=True, ax=ax1,
               color=['#2ecc71', '#f39c12', '#e74c3c'],
               alpha=0.85, edgecolor='white', linewidth=1)
    ax1.set_xlabel('Income Group', fontsize=11, fontweight='bold')
    ax1.set_ylabel('% of Population', fontsize=11, fontweight='bold')
    ax1.set_title('Stacked: Population by Zone and Income Group', fontsize=12, fontweight='bold')
    ax1.legend(title='Access Zone', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Chart 2: Grouped Bar
    ax2 = axes[1]
    pivot.plot(kind='bar', ax=ax2,
               color=['#2ecc71', '#f39c12', '#e74c3c'],
               alpha=0.85, edgecolor='white', linewidth=1)
    ax2.set_xlabel('Income Group', fontsize=11, fontweight='bold')
    ax2.set_ylabel('% of Population', fontsize=11, fontweight='bold')
    ax2.set_title('Grouped: Population by Zone and Income Group', fontsize=12, fontweight='bold')
    ax2.legend(title='Access Zone', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved as '{output_file}'")
    plt.show()
    
    return pivot

def create_zone_map(gdf, train_stations_gdf, output_file='access_zone_map.png'):
    """(c) Map of walk/bike/far zones"""
    print(f"\nCreating zone map: {output_file}")
    
    # Convert back to WGS84 for mapping
    gdf_map = gdf.to_crs('EPSG:4326')
    train_stations_map = train_stations_gdf.to_crs('EPSG:4326')
    
    # Create map
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot zones with colors
    zone_colors = {
        'walk': '#2ecc71',  # Green
        'bike': '#f39c12',  # Orange
        'far': '#e74c3c',   # Red
        'unknown': '#95a5a6'  # Gray
    }
    
    for zone, color in zone_colors.items():
        zone_data = gdf_map[gdf_map['access_zone'] == zone]
        if len(zone_data) > 0:
            zone_data.plot(ax=ax, color=color, alpha=0.6, edgecolor='white', linewidth=0.1, label=zone)
    
    # Plot train stations
    train_stations_map.plot(ax=ax, color='blue', markersize=20, marker='s', 
                           edgecolor='white', linewidth=0.5, alpha=0.8, label='Train Stations', zorder=5)
    
    ax.set_title('Transit Access Zones by Block Group\n(Walk: <0.5mi, Bike: 0.5-2mi, Far: >2mi)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Map saved as '{output_file}'")
    plt.show()

def main():
    """Main function to run complete analysis."""
    print("="*70)
    print("Complete Transit Accessibility Analysis")
    print("="*70)
    
    # Step 1: Load train stations from GTFS
    print("\n" + "="*70)
    print("Step 1: Loading MBTA Rail Stations from GTFS")
    print("="*70)
    train_stations_gdf = load_train_stations_from_gtfs()
    
    # Step 2: Load block groups
    print("\n" + "="*70)
    print("Step 2: Loading Block Group Polygons")
    print("="*70)
    block_groups_gdf = load_block_groups_with_geometry()
    
    # Step 3: Calculate distances
    print("\n" + "="*70)
    print("Step 3: Calculating Distances")
    print("="*70)
    # Use walking distances (actual street network paths)
    block_groups_gdf = calculate_distances_with_spatial_index(block_groups_gdf, train_stations_gdf, use_walking_distance=True)
    
    # Step 4: Assign zones
    print("\n" + "="*70)
    print("Step 4: Assigning Access Zones")
    print("="*70)
    block_groups_gdf = assign_access_zones(block_groups_gdf)
    
    # Step 5: Join population and income
    print("\n" + "="*70)
    print("Step 5: Joining Population and Income Data")
    print("="*70)
    block_groups_gdf = join_population_and_income(block_groups_gdf)
    
    # Step 6: Categorize income groups
    print("\n" + "="*70)
    print("Step 6: Categorizing Income Groups")
    print("="*70)
    block_groups_gdf = categorize_income_groups(block_groups_gdf)
    
    # Step 7: Create outputs
    print("\n" + "="*70)
    print("Step 7: Creating Analysis Outputs")
    print("="*70)
    
    # (a) Income group analysis
    results_df = create_income_zone_analysis(block_groups_gdf)
    
    # (d) Bar chart
    create_zone_income_chart(results_df)
    
    # (c) Zone map
    create_zone_map(block_groups_gdf, train_stations_gdf)
    
    # Save complete dataset
    output_file = 'data/complete_accessibility_analysis.geojson'
    print(f"\nSaving complete dataset to: {output_file}")
    try:
        block_groups_gdf.to_file(output_file, driver='GeoJSON')
        print(f"✅ Saved to {output_file}")
    except Exception as e:
        print(f"⚠️  Could not save as GeoJSON: {e}")
        # Save as CSV instead
        csv_file = 'data/complete_accessibility_analysis.csv'
        block_groups_gdf.drop(columns='geometry').to_csv(csv_file, index=False)
        print(f"✅ Saved to {csv_file} (without geometry)")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nGenerated outputs:")
    print("  (a) Income group analysis by zone (printed above)")
    print("  (d) Bar chart: income_zone_distribution.png")
    print("  (c) Zone map: access_zone_map.png")
    print("  Complete dataset: data/complete_accessibility_analysis.geojson")
    
    return block_groups_gdf

if __name__ == '__main__':
    main()

