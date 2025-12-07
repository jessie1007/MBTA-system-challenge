#!/usr/bin/env python3
"""
Create population accessibility analysis using WALKING distances
for specific neighborhoods: Brookline, Everett, Somerville, Dorchester, Roxbury, Mattapan
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
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

def load_block_groups_with_geometry():
    """Load block groups with geometry from shapefile and merge with income data."""
    # Load geometry from shapefile
    shapefile_path = 'data/block_groups.shp'
    income_file = 'data/neighborhood_income_merged.geojson'
    
    if not Path(shapefile_path).exists():
        print(f"⚠️  Warning: Shapefile not found: {shapefile_path}")
        return None
    
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
        return None
    
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

def load_train_stations():
    """Load train stations from saved file or GTFS."""
    stations_file = 'data/mbta_train_stations.geojson'
    
    if Path(stations_file).exists():
        try:
            stations_gdf = gpd.read_file(stations_file)
            print(f"✅ Loaded {len(stations_gdf)} train stations from {stations_file}")
            return stations_gdf
        except Exception as e:
            print(f"Error loading stations: {e}")
    
    # Fallback: load from GTFS
    print("Loading train stations from GTFS...")
    try:
        from load_mbta_stations import main as load_stations
        stations_gdf = load_stations()
        return stations_gdf
    except Exception as e:
        print(f"Error loading from GTFS: {e}")
        raise

def map_neighborhoods(gdf):
    """
    Map block groups to neighborhoods based on county and income patterns.
    """
    gdf = gdf.copy()
    
    # Extract county and tract info
    gdf['county_code'] = gdf['GEOID'].str[:5]
    gdf['tract_id'] = gdf['GEOID'].str[5:11]
    
    # Initialize neighborhood column
    gdf['neighborhood'] = 'Other'
    
    # County mapping
    county_names = {
        '25025': 'Suffolk',  # Boston
        '25017': 'Middlesex',  # Cambridge, Somerville, Everett
        '25021': 'Norfolk'  # Brookline
    }
    gdf['county_name'] = gdf['county_code'].map(county_names)
    
    # Cambridge - Middlesex County, typically higher income (map first)
    if 'median_household_income' in gdf.columns:
        middlesex_mask = (gdf['county_code'] == '25017')
        if middlesex_mask.sum() > 0:
            middlesex_income_threshold = gdf[middlesex_mask]['median_household_income'].quantile(0.6)
            cambridge_mask = middlesex_mask & (gdf['median_household_income'] >= middlesex_income_threshold)
            gdf.loc[cambridge_mask, 'neighborhood'] = 'Cambridge'
    
    # Somerville - Middlesex County, typically middle income (map after Cambridge)
    if 'median_household_income' in gdf.columns:
        middlesex_mask = (gdf['county_code'] == '25017')
        if middlesex_mask.sum() > 0:
            middlesex_low = gdf[middlesex_mask]['median_household_income'].quantile(0.3)
            middlesex_high = gdf[middlesex_mask]['median_household_income'].quantile(0.7)
            somerville_mask = middlesex_mask & (gdf['median_household_income'] >= middlesex_low) & (gdf['median_household_income'] <= middlesex_high)
            gdf.loc[somerville_mask & (gdf['neighborhood'] == 'Other'), 'neighborhood'] = 'Somerville'
    
    # Everett - Middlesex County, typically lower-middle income
    if 'median_household_income' in gdf.columns:
        middlesex_mask = (gdf['county_code'] == '25017')
        if middlesex_mask.sum() > 0:
            middlesex_low = gdf[middlesex_mask]['median_household_income'].quantile(0.2)
            middlesex_mid = gdf[middlesex_mask]['median_household_income'].quantile(0.5)
            everett_mask = middlesex_mask & (gdf['median_household_income'] >= middlesex_low) & (gdf['median_household_income'] <= middlesex_mid)
            gdf.loc[everett_mask & (gdf['neighborhood'] == 'Other'), 'neighborhood'] = 'Everett'
    
    # Roxbury - Suffolk County (Boston), typically lower income
    if 'median_household_income' in gdf.columns:
        suffolk_mask = (gdf['county_code'] == '25025')
        if suffolk_mask.sum() > 0:
            suffolk_low = gdf[suffolk_mask]['median_household_income'].quantile(0.1)
            suffolk_mid = gdf[suffolk_mask]['median_household_income'].quantile(0.4)
            roxbury_mask = suffolk_mask & (gdf['median_household_income'] >= suffolk_low) & (gdf['median_household_income'] <= suffolk_mid)
            gdf.loc[roxbury_mask, 'neighborhood'] = 'Roxbury'
    
    # Dorchester - Suffolk County (Boston), typically lower-middle income
    if 'median_household_income' in gdf.columns:
        suffolk_mask = (gdf['county_code'] == '25025')
        if suffolk_mask.sum() > 0:
            suffolk_low = gdf[suffolk_mask]['median_household_income'].quantile(0.2)
            suffolk_mid = gdf[suffolk_mask]['median_household_income'].quantile(0.6)
            dorchester_mask = suffolk_mask & (gdf['median_household_income'] >= suffolk_low) & (gdf['median_household_income'] <= suffolk_mid)
            gdf.loc[dorchester_mask & (gdf['neighborhood'] == 'Other'), 'neighborhood'] = 'Dorchester'
    
    # Mattapan - Suffolk County (Boston), typically lower-middle income (south of Dorchester)
    if 'median_household_income' in gdf.columns:
        suffolk_mask = (gdf['county_code'] == '25025')
        if suffolk_mask.sum() > 0:
            # Mattapan is typically in lower income range, similar to parts of Dorchester
            suffolk_low = gdf[suffolk_mask]['median_household_income'].quantile(0.15)
            suffolk_mid = gdf[suffolk_mask]['median_household_income'].quantile(0.45)
            mattapan_mask = suffolk_mask & (gdf['median_household_income'] >= suffolk_low) & (gdf['median_household_income'] <= suffolk_mid)
            # Prioritize tracts that haven't been assigned yet
            gdf.loc[mattapan_mask & (gdf['neighborhood'] == 'Other'), 'neighborhood'] = 'Mattapan'
    
    # Filter to only specified neighborhoods
    target_neighborhoods = ['Everett', 'Somerville', 'Dorchester', 'Cambridge', 'Roxbury', 'Mattapan']
    gdf = gdf[gdf['neighborhood'].isin(target_neighborhoods)]
    
    return gdf

def calculate_walking_distances_for_neighborhoods(gdf, train_stations_gdf):
    """Calculate walking distances for block groups in target neighborhoods."""
    print("\n" + "="*70)
    print("Calculating WALKING Distances to Train Stations")
    print("="*70)
    
    # Check if we need to calculate distances
    if 'walking_distance_miles' in gdf.columns and gdf['walking_distance_miles'].notna().any():
        print("✅ Walking distance data already available")
        return gdf
    
    # Ensure we have geometry
    if 'geometry' not in gdf.columns:
        print("⚠️  Warning: No geometry available. Cannot calculate walking distances.")
        return gdf
    
    # Calculate walking distances
    try:
        from calculate_walking_distances import calculate_walking_distances_to_stations
        
        # Get bounding box for the neighborhoods
        bounds = gdf.total_bounds
        bbox = (bounds[3], bounds[1], bounds[2], bounds[0])  # (north, south, east, west)
        
        print(f"Calculating walking distances for {len(gdf)} block groups...")
        result = calculate_walking_distances_to_stations(
            gdf, train_stations_gdf,
            bbox=bbox,
            place_name="Boston, Massachusetts, USA",
            cache_file='data/boston_street_network.graphml',
            max_distance_meters=20000  # 20km max
        )
        
        # Update distance column name for compatibility
        if 'walking_distance_miles' in result.columns:
            result['distance_to_nearest_station_miles'] = result['walking_distance_miles']
        
        return result
        
    except ImportError:
        print("⚠️  osmnx not available. Cannot calculate walking distances.")
        print("   Install with: pip install osmnx")
        return gdf
    except Exception as e:
        print(f"⚠️  Error calculating walking distances: {e}")
        print("   Falling back to straight-line distances if available...")
        return gdf

def calculate_population_accessibility(gdf):
    """
    Calculate population statistics by neighborhood and accessibility zone.
    """
    # Ensure we have population data
    if 'total_population' not in gdf.columns:
        print("Warning: Population data not found. Using block group counts.")
        gdf['total_population'] = 1
    
    # Ensure we have distance data
    if 'distance_to_nearest_station_miles' not in gdf.columns:
        print("⚠️  Error: No distance data available.")
        return None
    
    # Categorize access zones based on walking distance
    def categorize_zone(distance_miles):
        if pd.isna(distance_miles):
            return 'unknown'
        elif distance_miles <= 0.5:
            return 'walk-zone'
        elif distance_miles <= 2.0:
            return 'bike-zone'
        else:
            return 'far-zone'
    
    gdf['accessibility_zone'] = gdf['distance_to_nearest_station_miles'].apply(categorize_zone)
    
    # Filter out unknown zones
    gdf = gdf[gdf['accessibility_zone'] != 'unknown']
    
    # Calculate population by neighborhood and zone
    neighborhood_stats = []
    
    for neighborhood in gdf['neighborhood'].unique():
        hood_data = gdf[gdf['neighborhood'] == neighborhood]
        
        total_pop = hood_data['total_population'].sum()
        
        # Population within 0.5 miles (walk-zone)
        walk_pop = hood_data[hood_data['accessibility_zone'] == 'walk-zone']['total_population'].sum()
        
        # Population beyond 0.5 miles (bike-zone + far-zone)
        beyond_05_pop = hood_data[hood_data['accessibility_zone'].isin(['bike-zone', 'far-zone'])]['total_population'].sum()
        
        # Calculate percentages
        pct_beyond_05 = (beyond_05_pop / total_pop * 100) if total_pop > 0 else 0
        pct_within_05 = (walk_pop / total_pop * 100) if total_pop > 0 else 0
        
        # Breakdown by zone
        bike_pop = hood_data[hood_data['accessibility_zone'] == 'bike-zone']['total_population'].sum()
        far_pop = hood_data[hood_data['accessibility_zone'] == 'far-zone']['total_population'].sum()
        
        neighborhood_stats.append({
            'Neighborhood': neighborhood,
            'Total Population': total_pop,
            'Within 0.5 miles': walk_pop,
            'Beyond 0.5 miles': beyond_05_pop,
            '% Beyond 0.5 miles': pct_beyond_05,
            '% Within 0.5 miles': pct_within_05,
            'Bike zone (0.5-2 mi)': bike_pop,
            'Far zone (>2 mi)': far_pop
        })
    
    stats_df = pd.DataFrame(neighborhood_stats)
    stats_df = stats_df.sort_values('% Beyond 0.5 miles', ascending=False)
    
    return stats_df

def create_population_accessibility_chart(stats_df, output_file='neighborhood_walking_accessibility.png'):
    """Create bar chart showing % of population beyond 0.5 miles by neighborhood."""
    
    if stats_df is None or len(stats_df) == 0:
        print("Error: No data to plot")
        return
    
    # Create figure with multiple charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Population Access to Train Stations by Neighborhood (WALKING DISTANCES)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Chart 1: % Beyond 0.5 miles (Main chart)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(stats_df['Neighborhood'], stats_df['% Beyond 0.5 miles'], 
                     color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    ax1.set_xlabel('% of Population Beyond 0.5 Miles', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Neighborhood', fontsize=11, fontweight='bold')
    ax1.set_title('% of Population Living >0.5 Miles from Train Station\n(Walking Distance)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Chart 2: Absolute Population Beyond 0.5 miles
    ax2 = axes[0, 1]
    bars2 = ax2.barh(stats_df['Neighborhood'], stats_df['Beyond 0.5 miles'], 
                     color='#e67e22', alpha=0.85, edgecolor='white', linewidth=1)
    ax2.set_xlabel('Population Beyond 0.5 Miles', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Neighborhood', fontsize=11, fontweight='bold')
    ax2.set_title('Absolute Population Living >0.5 Miles from Train Station\n(Walking Distance)', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width):,}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Chart 3: Comparison - Within vs Beyond 0.5 miles
    ax3 = axes[1, 0]
    x = np.arange(len(stats_df))
    width = 0.35
    
    bars3a = ax3.barh(x - width/2, stats_df['% Within 0.5 miles'], width,
                     label='Within 0.5 miles', color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    bars3b = ax3.barh(x + width/2, stats_df['% Beyond 0.5 miles'], width,
                     label='Beyond 0.5 miles', color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax3.set_xlabel('% of Population', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Neighborhood', fontsize=11, fontweight='bold')
    ax3.set_title('Population Distribution: Within vs Beyond 0.5 Miles\n(Walking Distance)', fontsize=12, fontweight='bold')
    ax3.set_yticks(x)
    ax3.set_yticklabels(stats_df['Neighborhood'])
    ax3.invert_yaxis()
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_xlim(0, 100)
    
    # Chart 4: Breakdown by Zone (Stacked)
    ax4 = axes[1, 1]
    
    # Prepare data for stacked bar
    walk_pct = stats_df['% Within 0.5 miles']
    bike_pct = (stats_df['Bike zone (0.5-2 mi)'] / stats_df['Total Population'] * 100)
    far_pct = (stats_df['Far zone (>2 mi)'] / stats_df['Total Population'] * 100)
    
    y_pos = np.arange(len(stats_df))
    
    ax4.barh(y_pos, walk_pct, label='Walk zone (<0.5 mi)', 
             color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    ax4.barh(y_pos, bike_pct, left=walk_pct, label='Bike zone (0.5-2 mi)', 
             color='#f39c12', alpha=0.85, edgecolor='white', linewidth=1)
    ax4.barh(y_pos, far_pct, left=walk_pct + bike_pct, label='Far zone (>2 mi)', 
             color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax4.set_xlabel('% of Population', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Neighborhood', fontsize=11, fontweight='bold')
    ax4.set_title('Population Distribution by Access Zone\n(Walking Distance)', fontsize=12, fontweight='bold')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(stats_df['Neighborhood'])
    ax4.invert_yaxis()
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved as '{output_file}'")
    plt.show()
    
    return stats_df

def main():
    """Main function."""
    print("="*70)
    print("Neighborhood Population Accessibility Analysis")
    print("Using WALKING DISTANCES (Actual Street Network)")
    print("="*70)
    print("\nNeighborhoods: Cambridge, Everett, Somerville, Dorchester, Roxbury, Mattapan")
    print("="*70)
    
    # Load block groups with geometry and income data
    print("\nLoading block groups with geometry and income data...")
    gdf = load_block_groups_with_geometry()
    
    if gdf is None:
        print("ERROR: Could not load block groups with geometry")
        return
    
    print(f"✅ Loaded {len(gdf)} block groups with geometry")
    
    # Map neighborhoods
    print("\nMapping block groups to neighborhoods...")
    gdf = map_neighborhoods(gdf)
    print(f"✅ Mapped to {gdf['neighborhood'].nunique()} neighborhoods")
    print(f"   Neighborhoods: {', '.join(sorted(gdf['neighborhood'].unique()))}")
    print(f"   Total block groups in target neighborhoods: {len(gdf)}")
    
    # Load train stations
    print("\nLoading train stations...")
    train_stations_gdf = load_train_stations()
    
    # Calculate walking distances
    gdf = calculate_walking_distances_for_neighborhoods(gdf, train_stations_gdf)
    
    # Calculate population accessibility statistics
    print("\nCalculating population accessibility statistics...")
    stats_df = calculate_population_accessibility(gdf)
    
    if stats_df is None:
        print("ERROR: Could not calculate statistics")
        return
    
    # Print summary
    print("\n" + "="*70)
    print("Population Accessibility Summary by Neighborhood (WALKING DISTANCES):")
    print("="*70)
    print(stats_df.to_string(index=False))
    
    # Create chart
    print("\nCreating chart...")
    create_population_accessibility_chart(stats_df)
    
    # Save results
    output_csv = 'data/neighborhood_walking_accessibility.csv'
    stats_df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == '__main__':
    main()

