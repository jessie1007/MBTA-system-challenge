#!/usr/bin/env python3
"""
Create a chart showing % of population living >0.5 miles from train stations by neighborhood

IMPORTANT NOTE ON DISTANCE CALCULATION:
---------------------------------------
This analysis uses STRAIGHT-LINE (Euclidean) distance from block group centroids 
to train stations. This does NOT account for:
- Actual walking paths along streets
- Street network constraints
- Barriers (rivers, highways, railroads)
- Pedestrian infrastructure (sidewalks, crosswalks)

For more accurate analysis, network-based distance calculation would be needed using:
- OpenStreetMap (OSM) street network data
- Network analysis libraries (osmnx, networkx)
- Shortest path algorithms along actual road networks

Straight-line distance typically UNDERESTIMATES actual walking distance by 20-40%,
meaning more people may be beyond 0.5 miles than this analysis shows.
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

def map_neighborhoods(gdf):
    """
    Map block groups to neighborhoods based on county and income patterns.
    This is an approximation - for production, use official neighborhood boundaries.
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
    
    # Brookline - Norfolk County, typically higher income areas
    if 'median_household_income' in gdf.columns:
        norfolk_mask = (gdf['county_code'] == '25021')
        if norfolk_mask.sum() > 0:
            norfolk_income_threshold = gdf[norfolk_mask]['median_household_income'].quantile(0.7)
            brookline_mask = norfolk_mask & (gdf['median_household_income'] >= norfolk_income_threshold)
            gdf.loc[brookline_mask, 'neighborhood'] = 'Brookline'
    
    # Cambridge - Middlesex County, typically higher income
    if 'median_household_income' in gdf.columns:
        middlesex_mask = (gdf['county_code'] == '25017')
        if middlesex_mask.sum() > 0:
            middlesex_income_threshold = gdf[middlesex_mask]['median_household_income'].quantile(0.6)
            cambridge_mask = middlesex_mask & (gdf['median_household_income'] >= middlesex_income_threshold)
            gdf.loc[cambridge_mask, 'neighborhood'] = 'Cambridge'
    
    # Somerville - Middlesex County, typically middle income
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
            # Use tract IDs that might correspond to Mattapan area (approximate)
            # Mattapan is generally in southern Boston, we'll use income range similar to lower Dorchester
            mattapan_mask = suffolk_mask & (gdf['median_household_income'] >= suffolk_low) & (gdf['median_household_income'] <= suffolk_mid)
            # Prioritize tracts that haven't been assigned yet
            gdf.loc[mattapan_mask & (gdf['neighborhood'] == 'Other'), 'neighborhood'] = 'Mattapan'
    
    # Filter to only specified neighborhoods
    target_neighborhoods = ['Everett', 'Somerville', 'Dorchester', 'Brookline', 'Mattapan']
    gdf = gdf[gdf['neighborhood'].isin(target_neighborhoods)]
    
    return gdf

def calculate_population_accessibility(gdf):
    """
    Calculate population statistics by neighborhood and accessibility zone.
    """
    # Ensure we have population data
    if 'total_population' not in gdf.columns:
        print("Warning: Population data not found. Using block group counts.")
        gdf['total_population'] = 1
    
    # Ensure we have accessibility zone
    if 'accessibility_zone' not in gdf.columns:
        if 'distance_to_nearest_station_miles' in gdf.columns:
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
        else:
            print("Error: No accessibility zone or distance data found.")
            return None
    
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

def create_population_accessibility_chart(stats_df, output_file='population_accessibility_by_neighborhood.png'):
    """Create bar chart showing % of population beyond 0.5 miles by neighborhood."""
    
    if stats_df is None or len(stats_df) == 0:
        print("Error: No data to plot")
        return
    
    # Create figure with multiple charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Population Access to Train Stations by Neighborhood', fontsize=16, fontweight='bold', y=0.995)
    
    # Chart 1: % Beyond 0.5 miles (Main chart)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(stats_df['Neighborhood'], stats_df['% Beyond 0.5 miles'], 
                     color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=1)
    ax1.set_xlabel('% of Population Beyond 0.5 Miles', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Neighborhood', fontsize=11, fontweight='bold')
    ax1.set_title('% of Population Living >0.5 Miles from Train Station', fontsize=12, fontweight='bold')
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
    ax2.set_title('Absolute Population Living >0.5 Miles from Train Station', fontsize=12, fontweight='bold')
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
    ax3.set_title('Population Distribution: Within vs Beyond 0.5 Miles', fontsize=12, fontweight='bold')
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
    ax4.set_title('Population Distribution by Access Zone', fontsize=12, fontweight='bold')
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
    # Try to use Step 2 output first (has all the data we need)
    step2_output = 'data/transit_accessibility_analysis.geojson'
    step1_output = 'data/neighborhood_income_merged.geojson'
    
    print("="*60)
    print("Population Accessibility Analysis by Neighborhood")
    print("="*60)
    print("\n⚠️  IMPORTANT: Distance calculations use STRAIGHT-LINE distance")
    print("   (from block group centroids to train stations).")
    print("   This does NOT account for actual walking paths along streets.")
    print("   Actual walking distances may be 20-40% longer than shown here.")
    print("="*60)
    
    # Load data
    if Path(step2_output).exists():
        print(f"Loading Step 2 output: {step2_output}")
        gdf = load_data_with_fallback(step2_output)
        print(f"✅ Loaded {len(gdf)} block groups with transit accessibility data")
    elif Path(step1_output).exists():
        print(f"Loading Step 1 output: {step1_output}")
        print("Note: Will need to calculate distances to train stations...")
        gdf = load_data_with_fallback(step1_output)
        print(f"✅ Loaded {len(gdf)} block groups")
        print("⚠️  Warning: Distance calculations not available. Please run step2_transit_accessibility.py first.")
        print("   Using placeholder data for demonstration.")
        # Add placeholder accessibility data
        gdf['distance_to_nearest_station_miles'] = np.random.uniform(0, 3, len(gdf))
        gdf['accessibility_zone'] = gdf['distance_to_nearest_station_miles'].apply(
            lambda x: 'walk-zone' if x <= 0.5 else ('bike-zone' if x <= 2.0 else 'far-zone')
        )
        gdf['total_population'] = 1  # Placeholder
    else:
        print("ERROR: No data files found.")
        print("Please run step1_merge_acs_income.py and step2_transit_accessibility.py first.")
        return
    
    # Map neighborhoods
    print("\nMapping block groups to neighborhoods...")
    gdf = map_neighborhoods(gdf)
    print(f"✅ Mapped to {gdf['neighborhood'].nunique()} neighborhoods")
    print(f"   Neighborhoods: {', '.join(sorted(gdf['neighborhood'].unique()))}")
    
    # Calculate population accessibility statistics
    print("\nCalculating population accessibility statistics...")
    stats_df = calculate_population_accessibility(gdf)
    
    if stats_df is None:
        print("ERROR: Could not calculate statistics")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("Population Accessibility Summary by Neighborhood:")
    print("="*60)
    print(stats_df.to_string(index=False))
    
    # Create chart
    print("\nCreating chart...")
    create_population_accessibility_chart(stats_df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == '__main__':
    main()

