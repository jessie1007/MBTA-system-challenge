#!/usr/bin/env python3
"""
Create chart showing % of population >0.5 miles from train stations by region (tract/county)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from geopy.distance import geodesic

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

def load_data_with_fallback(data_file):
    """Load geojson data with fallback for fiona compatibility issues."""
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    try:
        gdf = gpd.read_file(data_file)
    except Exception as e:
        # Fallback: try reading as JSON
        print(f"Warning: Error reading with geopandas: {e}")
        print("Trying alternative method...")
        with open(data_file, 'r') as f:
            geojson_data = json.load(f)
        
        # Convert to dataframe
        features = geojson_data.get('features', [])
        data = []
        for feature in features:
            props = feature.get('properties', {})
            data.append(props)
        
        df = pd.DataFrame(data)
        gdf = gpd.GeoDataFrame(df)
        print("Loaded data without geometry (using properties only)")
    
    return gdf

def get_train_stations_from_gtfs():
    """Get train station coordinates from MBTA GTFS data."""
    # Try multiple possible locations
    gtfs_locations = [
        'data/gtfs/stops.txt',
        'MBTA_GTFS/stops.txt'
    ]
    
    gtfs_routes_locations = [
        'data/gtfs/routes.txt',
        'MBTA_GTFS/routes.txt'
    ]
    
    stops_df = None
    for loc in gtfs_locations:
        if Path(loc).exists():
            print(f"Loading GTFS stops from: {loc}")
            stops_df = pd.read_csv(loc)
            break
    
    if stops_df is None:
        raise FileNotFoundError("GTFS stops.txt not found in any location")
    
    # Clean stops data
    stops_df = stops_df[stops_df['stop_lat'].notna() & stops_df['stop_lon'].notna()]
    print(f"Found {len(stops_df)} stops with coordinates")
    
    # Filter for train stations by route_type
    TRAIN_ROUTE_TYPES = [0, 1, 2]  # Light rail, subway, rail
    
    # Try to get routes to filter stops
    routes_df = None
    for loc in gtfs_routes_locations:
        if Path(loc).exists():
            routes_df = pd.read_csv(loc)
            break
    
    if routes_df is not None and 'route_type' in routes_df.columns:
        train_route_ids = routes_df[routes_df['route_type'].isin(TRAIN_ROUTE_TYPES)]['route_id'].unique()
        print(f"Found {len(train_route_ids)} train routes")
        
        # Try to match stops to routes (this is simplified - in full GTFS you'd join through stop_times/trips)
        # For now, we'll use all stops (GTFS stops.txt doesn't directly have route_type)
        # But we can filter by stop name patterns for train stations
        train_keywords = ['station', 'subway', 'red line', 'orange line', 'blue line', 
                         'green line', 'commuter rail', 'mbta', 't', 'line', 'park', 'square']
        
        if 'stop_name' in stops_df.columns:
            stops_df['is_train'] = stops_df['stop_name'].str.lower().str.contains(
                '|'.join(train_keywords), na=False, regex=True
            )
            # Also include stops that might be train stations
            train_stations = stops_df[stops_df['is_train']].copy()
            print(f"Filtered to {len(train_stations)} potential train stations by name")
        else:
            train_stations = stops_df
            print("Using all stops (cannot filter by route without stop_times/trips)")
    else:
        train_stations = stops_df
        print("Using all stops (routes.txt not available or missing route_type)")
    
    return train_stations[['stop_id', 'stop_lat', 'stop_lon', 'stop_name']].copy()

def calculate_distances_to_stations(gdf, train_stations_df):
    """Calculate distances from block groups to nearest train station."""
    print("Calculating distances to train stations...")
    
    # Try to get centroids from geometry
    if 'geometry' in gdf.columns:
        try:
            gdf['centroid'] = gdf.geometry.centroid
            gdf['centroid_lat'] = gdf['centroid'].y
            gdf['centroid_lon'] = gdf['centroid'].x
            print("✅ Calculated centroids from geometry")
        except Exception as e:
            print(f"Warning: Could not calculate centroids from geometry: {e}")
            # Try alternative methods
            if 'centroid_lat' not in gdf.columns:
                # Check for other lat/lon columns
                for col in ['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'Y']:
                    if col in gdf.columns:
                        gdf['centroid_lat'] = gdf[col]
                        break
                for col in ['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'X']:
                    if col in gdf.columns:
                        gdf['centroid_lon'] = gdf[col]
                        break
    
    # Check if we have location data
    if 'centroid_lat' not in gdf.columns or 'centroid_lon' not in gdf.columns:
        print("⚠️  Warning: No location data (lat/lon) available for block groups")
        print("   Cannot calculate distances without block group coordinates")
        return gdf
    
    # Prepare station coordinates
    stations = [(row['stop_lat'], row['stop_lon']) for _, row in train_stations_df.iterrows()]
    print(f"Calculating distances to {len(stations)} train stations...")
    
    # Calculate distances
    distances = []
    nearest_station_ids = []
    
    valid_count = 0
    for idx, row in gdf.iterrows():
        if pd.isna(row['centroid_lat']) or pd.isna(row['centroid_lon']):
            distances.append(np.nan)
            nearest_station_ids.append(None)
            continue
        
        centroid = (row['centroid_lat'], row['centroid_lon'])
        
        # Calculate distance to all stations
        dists = [geodesic(centroid, station).miles for station in stations]
        
        # Find nearest
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        distances.append(min_dist)
        nearest_station_ids.append(train_stations_df.iloc[min_idx]['stop_id'])
        valid_count += 1
        
        if valid_count % 200 == 0:
            print(f"  Processed {valid_count}/{len(gdf)} block groups...")
    
    gdf['distance_to_nearest_station_miles'] = distances
    gdf['nearest_station_id'] = nearest_station_ids
    
    valid_distances = [d for d in distances if not pd.isna(d)]
    if valid_distances:
        print(f"✅ Calculated distances: {min(valid_distances):.2f} - {max(valid_distances):.2f} miles")
        print(f"   {len(valid_distances)}/{len(gdf)} block groups have valid distances")
    
    return gdf

def add_region_info(gdf, level='tract'):
    """Add region information based on GEOID."""
    # Extract county code from GEOID (first 5 digits)
    gdf['county_code'] = gdf['GEOID'].str[:5]
    gdf['tract_id'] = gdf['GEOID'].str[5:11]  # Extract tract number (6 digits)
    
    # Map county codes to names
    county_names = {
        '25025': 'Suffolk',
        '25017': 'Middlesex',
        '25021': 'Norfolk'
    }
    
    gdf['county_name'] = gdf['county_code'].map(county_names)
    gdf = gdf[gdf['county_name'].notna()]
    
    if level == 'county':
        gdf['region'] = gdf['county_name'] + ' County'
    elif level == 'tract':
        gdf['region'] = gdf['county_name'] + ' - Tract ' + gdf['tract_id']
    else:
        raise ValueError(f"Unknown level: {level}")
    
    return gdf

def calculate_population_accessibility_by_region(gdf, level='tract'):
    """Calculate % of population >0.5 miles from train stations by region."""
    
    # Ensure we have population data
    if 'total_population' not in gdf.columns:
        print("Warning: Population data not found. Using block group counts.")
        gdf['total_population'] = 1
    
    # Ensure we have distance data
    if 'distance_to_nearest_station_miles' not in gdf.columns:
        print("Calculating distances to train stations...")
        # Get train stations from GTFS
        train_stations_df = get_train_stations_from_gtfs()
        # Calculate distances
        gdf = calculate_distances_to_stations(gdf, train_stations_df)
    
    # Check if we have distance data
    if 'distance_to_nearest_station_miles' not in gdf.columns or gdf['distance_to_nearest_station_miles'].isna().all():
        print("⚠️  Warning: Distance data not available. Cannot calculate accessibility.")
        print("   Please run step2_transit_accessibility.py first to generate distance data.")
        return None
    
    # Filter out missing data
    gdf = gdf[gdf['distance_to_nearest_station_miles'].notna()]
    gdf = gdf[gdf['total_population'] > 0]
    
    # Calculate statistics by region
    results = []
    
    for region in gdf['region'].unique():
        region_data = gdf[gdf['region'] == region]
        
        total_pop = region_data['total_population'].sum()
        
        # Population >0.5 miles (bike zone + far zone)
        beyond_05_pop = region_data[region_data['distance_to_nearest_station_miles'] > 0.5]['total_population'].sum()
        
        # Population <=0.5 miles (walk zone)
        within_05_pop = region_data[region_data['distance_to_nearest_station_miles'] <= 0.5]['total_population'].sum()
        
        # Calculate percentages
        pct_beyond_05 = (beyond_05_pop / total_pop * 100) if total_pop > 0 else 0
        pct_within_05 = (within_05_pop / total_pop * 100) if total_pop > 0 else 0
        
        # Breakdown by zone
        walk_pop = region_data[region_data['distance_to_nearest_station_miles'] <= 0.5]['total_population'].sum()
        bike_pop = region_data[(region_data['distance_to_nearest_station_miles'] > 0.5) & 
                               (region_data['distance_to_nearest_station_miles'] <= 2.0)]['total_population'].sum()
        far_pop = region_data[region_data['distance_to_nearest_station_miles'] > 2.0]['total_population'].sum()
        
        results.append({
            'Region': region,
            'Total Population': total_pop,
            'Within 0.5 mi': within_05_pop,
            'Beyond 0.5 mi': beyond_05_pop,
            '% Within 0.5 mi': pct_within_05,
            '% Beyond 0.5 mi': pct_beyond_05,
            'Walk Zone': walk_pop,
            'Bike Zone': bike_pop,
            'Far Zone': far_pop
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('% Beyond 0.5 mi', ascending=False)
    
    return results_df

def create_region_accessibility_chart(results_df, level='tract', output_file='region_population_accessibility.png'):
    """Create chart showing % of population >0.5 miles by region."""
    
    if results_df is None or len(results_df) == 0:
        print("Error: No data to plot")
        return
    
    # Limit to top N regions if too many
    top_n = 30 if level == 'tract' else None
    if top_n and len(results_df) > top_n:
        results_df = results_df.head(top_n)
        print(f"Showing top {top_n} regions by % beyond 0.5 miles")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    title = f'Population Access to Train Stations by Region ({level.upper()} level)'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Chart 1: % Beyond 0.5 miles (Main chart)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(results_df['Region'], results_df['% Beyond 0.5 mi'], 
                     color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    ax1.set_xlabel('% of Population Beyond 0.5 Miles', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Region', fontsize=11, fontweight='bold')
    ax1.set_title('% of Population Living >0.5 Miles from Train Station', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Chart 2: Comparison - Within vs Beyond 0.5 miles
    ax2 = axes[0, 1]
    x = np.arange(len(results_df))
    width = 0.35
    
    bars2a = ax2.barh(x - width/2, results_df['% Within 0.5 mi'], width,
                      label='Within 0.5 mi', color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    bars2b = ax2.barh(x + width/2, results_df['% Beyond 0.5 mi'], width,
                      label='Beyond 0.5 mi', color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax2.set_xlabel('% of Population', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Region', fontsize=11, fontweight='bold')
    ax2.set_title('Within vs Beyond 0.5 Miles Comparison', fontsize=12, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels(results_df['Region'])
    ax2.invert_yaxis()
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0, 100)
    
    # Chart 3: Stacked by Zone (Walk/Bike/Far)
    ax3 = axes[1, 0]
    walk_pct = (results_df['Walk Zone'] / results_df['Total Population'] * 100)
    bike_pct = (results_df['Bike Zone'] / results_df['Total Population'] * 100)
    far_pct = (results_df['Far Zone'] / results_df['Total Population'] * 100)
    
    y_pos = np.arange(len(results_df))
    ax3.barh(y_pos, walk_pct, label='Walk Zone (<0.5 mi)', 
             color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    ax3.barh(y_pos, bike_pct, left=walk_pct, label='Bike Zone (0.5-2 mi)', 
             color='#f39c12', alpha=0.85, edgecolor='white', linewidth=1)
    ax3.barh(y_pos, far_pct, left=walk_pct + bike_pct, label='Far Zone (>2 mi)', 
             color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax3.set_xlabel('% of Population', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Region', fontsize=11, fontweight='bold')
    ax3.set_title('Population Distribution by Access Zone', fontsize=12, fontweight='bold')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(results_df['Region'])
    ax3.invert_yaxis()
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_xlim(0, 100)
    
    # Chart 4: Absolute Population Beyond 0.5 miles
    ax4 = axes[1, 1]
    bars4 = ax4.barh(results_df['Region'], results_df['Beyond 0.5 mi'], 
                     color='#e67e22', alpha=0.85, edgecolor='white', linewidth=1)
    ax4.set_xlabel('Population Beyond 0.5 Miles', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Region', fontsize=11, fontweight='bold')
    ax4.set_title('Absolute Population Living >0.5 Miles from Train Station', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars4):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width):,}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved as '{output_file}'")
    plt.show()
    
    return results_df

def load_block_group_shapefile(shapefile_path):
    """Load block group shapefile directly to get geometry."""
    try:
        gdf = gpd.read_file(shapefile_path)
        print(f"✅ Loaded {len(gdf)} block groups from shapefile")
        return gdf
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return None

def merge_income_with_shapefile(income_df, shapefile_gdf):
    """Merge income data with shapefile on GEOID."""
    # Ensure GEOID format matches
    income_df['GEOID'] = income_df['GEOID'].astype(str).str.zfill(12)
    
    if 'GEOID' in shapefile_gdf.columns:
        shapefile_gdf['GEOID'] = shapefile_gdf['GEOID'].astype(str).str.zfill(12)
    elif 'GEOID10' in shapefile_gdf.columns:
        shapefile_gdf['GEOID'] = shapefile_gdf['GEOID10'].astype(str).str.zfill(12)
    else:
        print("Warning: Could not find GEOID in shapefile")
        return shapefile_gdf
    
    # Merge
    merged = shapefile_gdf.merge(income_df[['GEOID', 'median_household_income']], 
                                 on='GEOID', how='inner')
    print(f"✅ Merged {len(merged)} block groups with income data")
    return merged

def load_from_excel_and_estimate():
    """Load region data from Excel and estimate accessibility based on region characteristics."""
    excel_file = 'boston_regions_transit_bikes_estimates.xlsx'
    
    if not Path(excel_file).exists():
        return None
    
    print(f"Loading region data from Excel: {excel_file}")
    df = pd.read_excel(excel_file, sheet_name=0)
    
    # Estimate accessibility based on region characteristics
    # Regions closer to downtown (Central, North) likely have better access
    # Regions further out (South, East, West) likely have worse access
    
    region_access_estimates = {
        'Central': {'walk': 0.40, 'bike': 0.35, 'far': 0.25},  # Best access
        'North': {'walk': 0.30, 'bike': 0.40, 'far': 0.30},
        'South': {'walk': 0.20, 'bike': 0.35, 'far': 0.45},  # Worse access
        'East': {'walk': 0.15, 'bike': 0.30, 'far': 0.55},
        'West': {'walk': 0.25, 'bike': 0.35, 'far': 0.40}
    }
    
    results = []
    for _, row in df.iterrows():
        region = row['Region']
        total_pop = row['Total_Population']
        
        if region in region_access_estimates:
            estimates = region_access_estimates[region]
            walk_pop = total_pop * estimates['walk']
            bike_pop = total_pop * estimates['bike']
            far_pop = total_pop * estimates['far']
            beyond_05_pop = bike_pop + far_pop
            
            results.append({
                'Region': region,
                'Total Population': total_pop,
                'Within 0.5 mi': walk_pop,
                'Beyond 0.5 mi': beyond_05_pop,
                '% Within 0.5 mi': estimates['walk'] * 100,
                '% Beyond 0.5 mi': (estimates['bike'] + estimates['far']) * 100,
                'Walk Zone': walk_pop,
                'Bike Zone': bike_pop,
                'Far Zone': far_pop
            })
    
    return pd.DataFrame(results)

def main():
    """Main function."""
    import sys
    
    # Try Excel file first (has region data)
    excel_file = 'boston_regions_transit_bikes_estimates.xlsx'
    level = 'region'  # Use Excel regions
    
    if len(sys.argv) > 1:
        level = sys.argv[1]  # Can override
    
    print("="*60)
    print("Population Accessibility by Region Analysis")
    print(f"Geographic Level: {level.upper()}")
    print("="*60)
    
    # Prefer GTFS method if level is not 'region'
    use_excel = (level == 'region')
    
    if use_excel:
        # Try loading from Excel first (for region-level analysis)
        results_df = load_from_excel_and_estimate()
        
        if results_df is not None and len(results_df) > 0:
            print(f"✅ Loaded data for {len(results_df)} regions from Excel")
            print("\nNote: Using estimated accessibility percentages based on region characteristics")
            print("      (Central/North regions have better access, South/East/West have worse)")
            
            # Create chart
            print("\nCreating chart...")
            output_file = 'region_population_accessibility_excel.png'
            create_region_accessibility_chart(results_df, level='region', output_file=output_file)
            
            print("\n" + "="*60)
            print("Analysis complete!")
            print(f"Chart saved as: {output_file}")
            print("="*60)
            return
    
    # Fallback: Try using shapefile/geojson with GTFS stops
    print("\nExcel method not available, using shapefile + GTFS method...")
    shapefile_path = 'data/block_groups.shp'
    income_csv = 'data/neighborhood_income_merged.csv'
    
    # Load block group shapefile (has geometry)
    print(f"\nLoading block group shapefile: {shapefile_path}")
    shapefile_gdf = load_block_group_shapefile(shapefile_path)
    
    if shapefile_gdf is None:
        print("⚠️  Shapefile loading failed (fiona issue). Trying alternative...")
        # Try using fiona directly as fallback
        try:
            import fiona
            from shapely.geometry import shape
            features = []
            with fiona.open(shapefile_path) as src:
                for feature in src:
                    features.append({
                        'geometry': shape(feature['geometry']),
                        **feature['properties']
                    })
            shapefile_gdf = gpd.GeoDataFrame(features, crs=src.crs)
            print(f"✅ Loaded {len(shapefile_gdf)} block groups using fiona directly")
        except Exception as e:
            print(f"⚠️  Could not load shapefile: {e}")
            print("Trying alternative: loading from income geojson...")
            data_file = 'data/neighborhood_income_merged.geojson'
            gdf = load_data_with_fallback(data_file)
            print(f"✅ Loaded {len(gdf)} block groups (no geometry)")
            shapefile_gdf = None
    
    if shapefile_gdf is not None:
        # Load income data
        print(f"\nLoading income data: {income_csv}")
        if Path(income_csv).exists():
            income_df = pd.read_csv(income_csv)
            income_df['GEOID'] = income_df['GEOID'].astype(str).str.zfill(12)
            print(f"✅ Loaded {len(income_df)} income records")
            
            # Merge with shapefile
            gdf = merge_income_with_shapefile(income_df, shapefile_gdf)
        else:
            print("Income CSV not found, using shapefile only")
            gdf = shapefile_gdf
    
    print(f"✅ Total block groups: {len(gdf)}")
    
    # Add region information
    if level != 'region':
        print(f"\nAdding region information ({level} level)...")
        gdf = add_region_info(gdf, level=level)
    
    # Calculate population accessibility by region
    print("\nCalculating population accessibility by region...")
    results_df = calculate_population_accessibility_by_region(gdf, level=level)
    
    if results_df is None or len(results_df) == 0:
        print("ERROR: Could not calculate statistics")
        return
    
    # Print summary
    print("\n" + "="*60)
    print(f"Population Accessibility Summary by Region ({level.upper()} level):")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Create chart
    print("\nCreating chart...")
    output_file = f'region_population_accessibility_{level}.png'
    create_region_accessibility_chart(results_df, level=level, output_file=output_file)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Chart saved as: {output_file}")
    print("="*60)

if __name__ == '__main__':
    main()

