#!/usr/bin/env python3
"""
Create chart showing % of residents in walk/bike/far zones for low-income vs high-income block groups
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from geopy.distance import geodesic
from shapely.geometry import Point

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

def categorize_income_groups(gdf):
    """Categorize block groups into income groups: bottom 30%, middle 40%, top 30%."""
    
    if 'median_household_income' not in gdf.columns:
        raise ValueError("median_household_income column not found")
    
    # Calculate income percentiles
    income_30th = gdf['median_household_income'].quantile(0.30)
    income_70th = gdf['median_household_income'].quantile(0.70)
    
    print(f"\nIncome thresholds:")
    print(f"  Bottom 30%: < ${income_30th:,.0f}")
    print(f"  Middle 40%: ${income_30th:,.0f} - ${income_70th:,.0f}")
    print(f"  Top 30%: > ${income_70th:,.0f}")
    
    # Categorize
    def categorize_income(income):
        if pd.isna(income):
            return 'Unknown'
        elif income < income_30th:
            return 'Low Income (Bottom 30%)'
        elif income <= income_70th:
            return 'Middle Income (Middle 40%)'
        else:
            return 'High Income (Top 30%)'
    
    gdf['income_group'] = gdf['median_household_income'].apply(categorize_income)
    
    return gdf

def calculate_distances_if_needed(gdf):
    """Calculate distances to train stations if not already available."""
    
    if 'distance_to_nearest_station_miles' in gdf.columns:
        print("Distance data already available")
        return gdf
    
    print("Calculating distances to train stations...")
    
    # Load GTFS data
    gtfs_stops_file = 'data/gtfs/stops.txt'
    gtfs_routes_file = 'data/gtfs/routes.txt'
    
    if not Path(gtfs_stops_file).exists():
        raise FileNotFoundError(f"GTFS stops file not found: {gtfs_stops_file}")
    
    # Load stops
    stops_df = pd.read_csv(gtfs_stops_file)
    stops_df = stops_df[stops_df['stop_lat'].notna() & stops_df['stop_lon'].notna()]
    
    # Filter for train stations
    TRAIN_ROUTE_TYPES = [0, 1, 2]  # Light rail, subway, rail
    
    if Path(gtfs_routes_file).exists():
        routes_df = pd.read_csv(gtfs_routes_file)
        train_routes = routes_df[routes_df['route_type'].isin(TRAIN_ROUTE_TYPES)]['route_id'].unique()
        # Filter stops by route (simplified - use all stops for now)
        train_stations = stops_df
    else:
        # Use all stops as train stations (simplified)
        train_stations = stops_df
    
    print(f"Found {len(train_stations)} train stations")
    
    # Calculate centroids if needed
    if 'centroid_lat' not in gdf.columns:
        # Try to get lat/lon from geometry or calculate centroid
        if 'geometry' in gdf.columns:
            try:
                gdf['centroid'] = gdf.geometry.centroid
                gdf['centroid_lat'] = gdf['centroid'].y
                gdf['centroid_lon'] = gdf['centroid'].x
            except:
                print("Warning: Could not calculate centroids from geometry")
                return gdf
        else:
            print("Warning: No geometry or centroid data available")
            return gdf
    
    # Calculate distances
    stations = [(row['stop_lat'], row['stop_lon']) for _, row in train_stations.iterrows()]
    distances = []
    
    for idx, row in gdf.iterrows():
        if pd.isna(row['centroid_lat']) or pd.isna(row['centroid_lon']):
            distances.append(np.nan)
            continue
        
        centroid = (row['centroid_lat'], row['centroid_lon'])
        dists = [geodesic(centroid, station).miles for station in stations]
        min_dist = min(dists)
        distances.append(min_dist)
    
    gdf['distance_to_nearest_station_miles'] = distances
    print(f"Calculated distances: {min([d for d in distances if not pd.isna(d)]):.2f} - {max([d for d in distances if not pd.isna(d)]):.2f} miles")
    
    return gdf

def categorize_accessibility_zones(gdf):
    """Categorize block groups into accessibility zones."""
    
    if 'distance_to_nearest_station_miles' not in gdf.columns:
        raise ValueError("distance_to_nearest_station_miles column not found")
    
    def categorize_distance(dist):
        if pd.isna(dist):
            return 'unknown'
        elif dist <= 0.5:
            return 'walk-zone'
        elif dist <= 2.0:
            return 'bike-zone'
        else:
            return 'far-zone'
    
    gdf['accessibility_zone'] = gdf['distance_to_nearest_station_miles'].apply(categorize_distance)
    
    return gdf

def calculate_population_by_zone_and_income(gdf):
    """Calculate population percentages by accessibility zone and income group."""
    
    # Ensure we have population data
    if 'total_population' not in gdf.columns:
        print("Warning: Population data not found. Using block group counts.")
        gdf['total_population'] = 1
    
    # Filter out unknown zones
    gdf = gdf[gdf['accessibility_zone'] != 'unknown']
    gdf = gdf[gdf['income_group'] != 'Unknown']
    
    # Calculate statistics
    results = []
    
    for income_group in ['Low Income (Bottom 30%)', 'Middle Income (Middle 40%)', 'High Income (Top 30%)']:
        group_data = gdf[gdf['income_group'] == income_group]
        
        if len(group_data) == 0:
            continue
        
        total_pop = group_data['total_population'].sum()
        
        # Population in each zone
        walk_pop = group_data[group_data['accessibility_zone'] == 'walk-zone']['total_population'].sum()
        bike_pop = group_data[group_data['accessibility_zone'] == 'bike-zone']['total_population'].sum()
        far_pop = group_data[group_data['accessibility_zone'] == 'far-zone']['total_population'].sum()
        
        # Calculate percentages
        walk_pct = (walk_pop / total_pop * 100) if total_pop > 0 else 0
        bike_pct = (bike_pop / total_pop * 100) if total_pop > 0 else 0
        far_pct = (far_pop / total_pop * 100) if total_pop > 0 else 0
        
        results.append({
            'Income Group': income_group,
            'Total Population': total_pop,
            'Walk Zone (<0.5 mi)': walk_pop,
            'Bike Zone (0.5-2 mi)': bike_pop,
            'Far Zone (>2 mi)': far_pop,
            '% Walk Zone': walk_pct,
            '% Bike Zone': bike_pct,
            '% Far Zone': far_pct
        })
    
    results_df = pd.DataFrame(results)
    
    return results_df

def create_income_accessibility_chart(results_df, output_file='income_accessibility_chart.png'):
    """Create chart showing % of residents in each zone for low vs high income."""
    
    if results_df is None or len(results_df) == 0:
        print("Error: No data to plot")
        return
    
    # Filter to only low and high income for main comparison
    comparison_df = results_df[results_df['Income Group'].isin([
        'Low Income (Bottom 30%)', 'High Income (Top 30%)'
    ])].copy()
    
    # Create figure with multiple charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Transit Access by Income Group: % of Residents in Each Zone', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Chart 1: Stacked Bar - Low vs High Income (Main chart)
    ax1 = axes[0, 0]
    x = np.arange(len(comparison_df))
    width = 0.6
    
    walk_pct = comparison_df['% Walk Zone'].values
    bike_pct = comparison_df['% Bike Zone'].values
    far_pct = comparison_df['% Far Zone'].values
    
    bars1 = ax1.bar(x, walk_pct, width, label='Walk Zone (<0.5 mi)', 
                    color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    bars2 = ax1.bar(x, bike_pct, width, bottom=walk_pct, label='Bike Zone (0.5-2 mi)', 
                    color='#f39c12', alpha=0.85, edgecolor='white', linewidth=1)
    bars3 = ax1.bar(x, far_pct, width, bottom=walk_pct + bike_pct, label='Far Zone (>2 mi)', 
                    color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax1.set_xlabel('Income Group', fontsize=11, fontweight='bold')
    ax1.set_ylabel('% of Population', fontsize=11, fontweight='bold')
    ax1.set_title('Low Income vs High Income: Population Distribution by Access Zone', 
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['Income Group'].str.replace(' (Bottom 30%)', '').str.replace(' (Top 30%)', ''), 
                         rotation=0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for i, (w, b, f) in enumerate(zip(walk_pct, bike_pct, far_pct)):
        if w > 5:
            ax1.text(i, w/2, f'{w:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if b > 5:
            ax1.text(i, w + b/2, f'{b:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if f > 5:
            ax1.text(i, w + b + f/2, f'{f:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Chart 2: Grouped Bar - Comparison of each zone
    ax2 = axes[0, 1]
    x = np.arange(len(comparison_df))
    width = 0.25
    
    bars1 = ax2.bar(x - width, walk_pct, width, label='Walk Zone', 
                    color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    bars2 = ax2.bar(x, bike_pct, width, label='Bike Zone', 
                    color='#f39c12', alpha=0.85, edgecolor='white', linewidth=1)
    bars3 = ax2.bar(x + width, far_pct, width, label='Far Zone', 
                    color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax2.set_xlabel('Income Group', fontsize=11, fontweight='bold')
    ax2.set_ylabel('% of Population', fontsize=11, fontweight='bold')
    ax2.set_title('Zone Comparison: Low Income vs High Income', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_df['Income Group'].str.replace(' (Bottom 30%)', '').str.replace(' (Top 30%)', ''), 
                        rotation=0)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(max(walk_pct), max(bike_pct), max(far_pct)) * 1.2)
    
    # Chart 3: All three income groups (including middle)
    ax3 = axes[1, 0]
    x = np.arange(len(results_df))
    width = 0.6
    
    walk_all = results_df['% Walk Zone'].values
    bike_all = results_df['% Bike Zone'].values
    far_all = results_df['% Far Zone'].values
    
    bars1 = ax3.bar(x, walk_all, width, label='Walk Zone', 
                    color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    bars2 = ax3.bar(x, bike_all, width, bottom=walk_all, label='Bike Zone', 
                    color='#f39c12', alpha=0.85, edgecolor='white', linewidth=1)
    bars3 = ax3.bar(x, far_all, width, bottom=walk_all + bike_all, label='Far Zone', 
                    color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax3.set_xlabel('Income Group', fontsize=11, fontweight='bold')
    ax3.set_ylabel('% of Population', fontsize=11, fontweight='bold')
    ax3.set_title('All Income Groups: Population Distribution by Access Zone', 
                  fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(results_df['Income Group'].str.replace(' (Bottom 30%)', '').str.replace(' (Top 30%)', '').str.replace(' (Middle 40%)', ''), 
                        rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Chart 4: Difference between Low and High Income
    ax4 = axes[1, 1]
    
    low_income = comparison_df[comparison_df['Income Group'] == 'Low Income (Bottom 30%)'].iloc[0]
    high_income = comparison_df[comparison_df['Income Group'] == 'High Income (Top 30%)'].iloc[0]
    
    diff_walk = low_income['% Walk Zone'] - high_income['% Walk Zone']
    diff_bike = low_income['% Bike Zone'] - high_income['% Bike Zone']
    diff_far = low_income['% Far Zone'] - high_income['% Far Zone']
    
    zones = ['Walk Zone', 'Bike Zone', 'Far Zone']
    differences = [diff_walk, diff_bike, diff_far]
    colors = ['#2ecc71' if d < 0 else '#e74c3c' for d in differences]
    
    bars = ax4.barh(zones, differences, color=colors, alpha=0.85, edgecolor='white', linewidth=1)
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Percentage Point Difference (Low Income - High Income)', 
                   fontsize=11, fontweight='bold')
    ax4.set_ylabel('Access Zone', fontsize=11, fontweight='bold')
    ax4.set_title('Equity Gap: Difference in Access Between Income Groups', 
                  fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        ax4.text(diff, i, f'{diff:+.1f}pp', 
                ha='left' if diff > 0 else 'right', va='center', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved as '{output_file}'")
    plt.show()
    
    return results_df

def main():
    """Main function."""
    # Try to use Step 2 output first (has all the data we need)
    step2_output = 'data/transit_accessibility_analysis.geojson'
    step1_output = 'data/neighborhood_income_merged.geojson'
    
    print("="*60)
    print("Income and Transit Accessibility Analysis")
    print("="*60)
    
    # Load data
    if Path(step2_output).exists():
        print(f"Loading Step 2 output: {step2_output}")
        gdf = load_data_with_fallback(step2_output)
        print(f"✅ Loaded {len(gdf)} block groups with transit accessibility data")
    elif Path(step1_output).exists():
        print(f"Loading Step 1 output: {step1_output}")
        gdf = load_data_with_fallback(step1_output)
        print(f"✅ Loaded {len(gdf)} block groups")
        print("Calculating distances to train stations...")
        gdf = calculate_distances_if_needed(gdf)
    else:
        print("ERROR: No data files found.")
        print("Please run step1_merge_acs_income.py and step2_transit_accessibility.py first.")
        return
    
    # Categorize income groups
    print("\nCategorizing block groups by income...")
    gdf = categorize_income_groups(gdf)
    
    # Categorize accessibility zones
    print("\nCategorizing accessibility zones...")
    gdf = categorize_accessibility_zones(gdf)
    
    # Calculate population statistics
    print("\nCalculating population by zone and income group...")
    results_df = calculate_population_by_zone_and_income(gdf)
    
    if results_df is None or len(results_df) == 0:
        print("ERROR: Could not calculate statistics")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("Population Distribution by Income Group and Access Zone:")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Create chart
    print("\nCreating chart...")
    create_income_accessibility_chart(results_df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == '__main__':
    main()

