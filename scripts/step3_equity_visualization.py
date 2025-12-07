"""
Step 3: Equity Visualization Maps

Creates visualizations to demonstrate transportation equity:
1. Income distribution heatmap (top 20% vs bottom 20% wealth)
2. Transit stop density map
3. Combined map showing income and transit access
4. Analysis to support hypothesis: less wealthy people have less access
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import contextily as ctx
from pathlib import Path
from scipy.spatial import cKDTree

def load_step1_data(step1_output):
    """Load merged income data from Step 1."""
    print("Loading Step 1 income data...")
    gdf = gpd.read_file(step1_output)
    print(f"Loaded {len(gdf)} block groups")
    return gdf


def load_step2_data(step2_output):
    """Load transit accessibility data from Step 2."""
    print("Loading Step 2 transit accessibility data...")
    gdf = gpd.read_file(step2_output)
    print(f"Loaded {len(gdf)} block groups with transit data")
    return gdf


def categorize_income_quintiles(income_series):
    """
    Categorize income into quintiles.
    Returns: 'bottom_20', 'middle_60', 'top_20'
    """
    bottom_20_threshold = income_series.quantile(0.20)
    top_20_threshold = income_series.quantile(0.80)
    
    def categorize(income):
        if income <= bottom_20_threshold:
            return 'bottom_20'
        elif income >= top_20_threshold:
            return 'top_20'
        else:
            return 'middle_60'
    
    return income_series.apply(categorize)


def create_income_distribution_map(gdf, output_path='income_distribution_map.png'):
    """
    Create heatmap showing top 20% and bottom 20% income distribution.
    """
    print("Creating income distribution heatmap...")
    
    gdf = gdf.copy()
    
    # Categorize income
    gdf['income_quintile'] = categorize_income_quintiles(gdf['median_household_income'])
    
    # Create color mapping
    # Bottom 20% = Red shades
    # Middle 60% = Yellow/Gray
    # Top 20% = Green shades
    color_map = {
        'bottom_20': '#d73027',  # Red
        'middle_60': '#fee08b',  # Yellow
        'top_20': '#1a9850'      # Green
    }
    
    # Convert to Web Mercator for basemap
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    gdf_mercator = gdf.to_crs(epsg=3857)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot by income quintile
    for quintile, color in color_map.items():
        subset = gdf_mercator[gdf_mercator['income_quintile'] == quintile]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.2, alpha=0.7)
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=gdf_mercator.crs, source=ctx.providers.CartoDB.Positron, alpha=0.5)
    except:
        pass
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#d73027', label='Bottom 20% Income (Lowest Wealth)', alpha=0.7),
        mpatches.Patch(facecolor='#fee08b', label='Middle 60% Income', alpha=0.7),
        mpatches.Patch(facecolor='#1a9850', label='Top 20% Income (Highest Wealth)', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)
    
    ax.set_title('Income Distribution by Block Group\n' + 
                 'Suffolk, Middlesex, and Norfolk Counties, MA',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Income distribution map saved to {output_path}")
    plt.close()
    
    return gdf


def create_transit_density_map(gdf, train_stations_gdf, output_path='transit_density_map.png'):
    """
    Create map showing density of transit stops.
    """
    print("Creating transit stop density map...")
    
    gdf = gdf.copy()
    
    # Convert to Web Mercator
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    gdf_mercator = gdf.to_crs(epsg=3857)
    
    # Convert train stations to GeoDataFrame if needed
    if isinstance(train_stations_gdf, pd.DataFrame):
        train_stations_gdf = gpd.GeoDataFrame(
            train_stations_gdf,
            geometry=gpd.points_from_xy(train_stations_gdf['stop_lon'], train_stations_gdf['stop_lat']),
            crs='EPSG:4326'
        )
    
    train_stations_mercator = train_stations_gdf.to_crs(epsg=3857)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot block groups colored by distance to nearest station
    gdf_mercator.plot(
        column='distance_to_nearest_station_miles',
        cmap='YlOrRd',
        legend=True,
        ax=ax,
        edgecolor='white',
        linewidth=0.2,
        legend_kwds={'label': 'Distance to Nearest Train Station (miles)',
                    'orientation': 'vertical',
                    'shrink': 0.8}
    )
    
    # Overlay train stations
    train_stations_mercator.plot(
        ax=ax,
        color='blue',
        markersize=30,
        marker='s',
        edgecolor='white',
        linewidth=1,
        label='Train Stations',
        alpha=0.8
    )
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=gdf_mercator.crs, source=ctx.providers.CartoDB.Positron, alpha=0.5)
    except:
        pass
    
    # Add legend for stations
    station_legend = mpatches.Patch(facecolor='blue', edgecolor='white', label='Train Stations')
    ax.legend(handles=[station_legend], loc='upper right', fontsize=12, framealpha=0.9)
    
    ax.set_title('Transit Station Accessibility\n' +
                 'Distance from Block Groups to Nearest Train Station',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Transit density map saved to {output_path}")
    plt.close()


def create_combined_equity_map(gdf, train_stations_gdf, output_path='equity_combined_map.png'):
    """
    Create combined map showing income distribution and transit access.
    Highlights the equity gap.
    """
    print("Creating combined equity map...")
    
    gdf = gdf.copy()
    
    # Ensure income quintile is set
    if 'income_quintile' not in gdf.columns:
        gdf['income_quintile'] = categorize_income_quintiles(gdf['median_household_income'])
    
    # Convert to Web Mercator
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    gdf_mercator = gdf.to_crs(epsg=3857)
    
    # Convert train stations
    if isinstance(train_stations_gdf, pd.DataFrame):
        train_stations_gdf = gpd.GeoDataFrame(
            train_stations_gdf,
            geometry=gpd.points_from_xy(train_stations_gdf['stop_lon'], train_stations_gdf['stop_lat']),
            crs='EPSG:4326'
        )
    train_stations_mercator = train_stations_gdf.to_crs(epsg=3857)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # Left plot: Bottom 20% income + transit stations
    ax1 = axes[0]
    bottom_20 = gdf_mercator[gdf_mercator['income_quintile'] == 'bottom_20']
    if len(bottom_20) > 0:
        bottom_20.plot(ax=ax1, color='#d73027', edgecolor='white', linewidth=0.3, alpha=0.7, label='Bottom 20% Income')
    
    # Add other income groups with transparency
    middle = gdf_mercator[gdf_mercator['income_quintile'] == 'middle_60']
    if len(middle) > 0:
        middle.plot(ax=ax1, color='lightgray', edgecolor='white', linewidth=0.1, alpha=0.3)
    
    top_20 = gdf_mercator[gdf_mercator['income_quintile'] == 'top_20']
    if len(top_20) > 0:
        top_20.plot(ax=ax1, color='lightgray', edgecolor='white', linewidth=0.1, alpha=0.3)
    
    # Add transit stations
    train_stations_mercator.plot(ax=ax1, color='blue', markersize=25, marker='s', 
                               edgecolor='white', linewidth=1, label='Train Stations', alpha=0.8)
    
    try:
        ctx.add_basemap(ax1, crs=gdf_mercator.crs, source=ctx.providers.CartoDB.Positron, alpha=0.5)
    except:
        pass
    
    ax1.set_title('Bottom 20% Income Areas\n+ Train Station Locations',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.axis('off')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Right plot: Top 20% income + transit stations
    ax2 = axes[1]
    if len(top_20) > 0:
        top_20.plot(ax=ax2, color='#1a9850', edgecolor='white', linewidth=0.3, alpha=0.7, label='Top 20% Income')
    
    # Add other income groups with transparency
    if len(middle) > 0:
        middle.plot(ax=ax2, color='lightgray', edgecolor='white', linewidth=0.1, alpha=0.3)
    
    if len(bottom_20) > 0:
        bottom_20.plot(ax=ax2, color='lightgray', edgecolor='white', linewidth=0.1, alpha=0.3)
    
    # Add transit stations
    train_stations_mercator.plot(ax=ax2, color='blue', markersize=25, marker='s',
                               edgecolor='white', linewidth=1, label='Train Stations', alpha=0.8)
    
    try:
        ctx.add_basemap(ax2, crs=gdf_mercator.crs, source=ctx.providers.CartoDB.Positron, alpha=0.5)
    except:
        pass
    
    ax2.set_title('Top 20% Income Areas\n+ Train Station Locations',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.axis('off')
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    plt.suptitle('Transportation Equity Analysis: Income Distribution vs Transit Access\n' +
                'Suffolk, Middlesex, and Norfolk Counties, MA',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined equity map saved to {output_path}")
    plt.close()


def create_accessibility_by_income_chart(gdf, output_path='accessibility_by_income_chart.png'):
    """
    Create bar chart showing accessibility zone distribution by income quintile.
    """
    print("Creating accessibility by income chart...")
    
    gdf = gdf.copy()
    
    # Ensure categories are set
    if 'income_quintile' not in gdf.columns:
        gdf['income_quintile'] = categorize_income_quintiles(gdf['median_household_income'])
    
    if 'accessibility_zone' not in gdf.columns:
        print("Warning: accessibility_zone not found. Run Step 2 first.")
        return
    
    # Calculate percentages by income quintile
    if 'total_population' in gdf.columns:
        # Weight by population
        results = []
        for quintile in ['bottom_20', 'middle_60', 'top_20']:
            quintile_data = gdf[gdf['income_quintile'] == quintile]
            total_pop = quintile_data['total_population'].sum()
            
            for zone in ['walk-zone', 'bike-zone', 'far-zone']:
                zone_data = quintile_data[quintile_data['accessibility_zone'] == zone]
                zone_pop = zone_data['total_population'].sum()
                pct = (zone_pop / total_pop * 100) if total_pop > 0 else 0
                
                results.append({
                    'income_quintile': quintile.replace('_', ' ').title(),
                    'zone': zone.replace('-', ' ').title(),
                    'percentage': pct
                })
    else:
        # Unweighted (by block group count)
        results = []
        for quintile in ['bottom_20', 'middle_60', 'top_20']:
            quintile_data = gdf[gdf['income_quintile'] == quintile]
            total_bgs = len(quintile_data)
            
            for zone in ['walk-zone', 'bike-zone', 'far-zone']:
                zone_count = len(quintile_data[quintile_data['accessibility_zone'] == zone])
                pct = (zone_count / total_bgs * 100) if total_bgs > 0 else 0
                
                results.append({
                    'income_quintile': quintile.replace('_', ' ').title(),
                    'zone': zone.replace('-', ' ').title(),
                    'percentage': pct
                })
    
    results_df = pd.DataFrame(results)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(['Bottom 20', 'Middle 60', 'Top 20']))
    width = 0.25
    
    zones = ['Walk Zone', 'Bike Zone', 'Far Zone']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    for i, zone in enumerate(zones):
        zone_data = results_df[results_df['zone'] == zone]
        values = [
            zone_data[zone_data['income_quintile'] == 'Bottom 20']['percentage'].values[0] if len(zone_data[zone_data['income_quintile'] == 'Bottom 20']) > 0 else 0,
            zone_data[zone_data['income_quintile'] == 'Middle 60']['percentage'].values[0] if len(zone_data[zone_data['income_quintile'] == 'Middle 60']) > 0 else 0,
            zone_data[zone_data['income_quintile'] == 'Top 20']['percentage'].values[0] if len(zone_data[zone_data['income_quintile'] == 'Top 20']) > 0 else 0
        ]
        ax.bar(x + i*width, values, width, label=zone, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Income Quintile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Population/Block Groups (%)', fontsize=12, fontweight='bold')
    ax.set_title('Transit Accessibility by Income Level\n' +
                'Distribution Across Walk, Bike, and Far Zones',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Bottom 20%', 'Middle 60%', 'Top 20%'])
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accessibility by income chart saved to {output_path}")
    plt.close()
    
    return results_df


def analyze_equity_gap(gdf):
    """
    Analyze and print equity gap statistics.
    """
    print("\n" + "="*70)
    print("TRANSPORTATION EQUITY ANALYSIS")
    print("="*70)
    
    gdf = gdf.copy()
    
    if 'income_quintile' not in gdf.columns:
        gdf['income_quintile'] = categorize_income_quintiles(gdf['median_household_income'])
    
    if 'accessibility_zone' not in gdf.columns:
        print("Warning: Run Step 2 first to get accessibility zones")
        return
    
    # Calculate statistics
    if 'total_population' in gdf.columns:
        print("\nPOPULATION-WEIGHTED ANALYSIS:\n")
        
        for quintile in ['bottom_20', 'top_20']:
            quintile_name = quintile.replace('_', ' ').title()
            quintile_data = gdf[gdf['income_quintile'] == quintile]
            total_pop = quintile_data['total_population'].sum()
            
            print(f"{quintile_name} Income:")
            print(f"  Total Population: {total_pop:,.0f}")
            
            for zone in ['walk-zone', 'bike-zone', 'far-zone']:
                zone_data = quintile_data[quintile_data['accessibility_zone'] == zone]
                zone_pop = zone_data['total_population'].sum()
                zone_pct = (zone_pop / total_pop * 100) if total_pop > 0 else 0
                avg_dist = zone_data['distance_to_nearest_station_miles'].mean() if len(zone_data) > 0 else 0
                
                print(f"    {zone.replace('-', ' ').title()}: {zone_pop:,.0f} ({zone_pct:.1f}%) - Avg distance: {avg_dist:.2f} miles")
        
        # Key finding: Bike-but-not-walk zone
        print("\n" + "-"*70)
        print("KEY FINDING: BIKE-BUT-NOT-WALK ZONE")
        print("-"*70)
        
        bottom_20_data = gdf[gdf['income_quintile'] == 'bottom_20']
        top_20_data = gdf[gdf['income_quintile'] == 'top_20']
        
        bottom_20_pop = bottom_20_data['total_population'].sum()
        top_20_pop = top_20_data['total_population'].sum()
        
        bottom_20_bike = bottom_20_data[bottom_20_data['accessibility_zone'] == 'bike-zone']['total_population'].sum()
        top_20_bike = top_20_data[top_20_data['accessibility_zone'] == 'bike-zone']['total_population'].sum()
        
        bottom_20_bike_pct = (bottom_20_bike / bottom_20_pop * 100) if bottom_20_pop > 0 else 0
        top_20_bike_pct = (top_20_bike / top_20_pop * 100) if top_20_pop > 0 else 0
        
        print(f"\nBottom 20% Income: {bottom_20_bike_pct:.1f}% live in bike-but-not-walk zone")
        print(f"Top 20% Income: {top_20_bike_pct:.1f}% live in bike-but-not-walk zone")
        
        if bottom_20_bike_pct > top_20_bike_pct:
            diff = bottom_20_bike_pct - top_20_bike_pct
            print(f"\n✓ HYPOTHESIS SUPPORTED:")
            print(f"  Low-income residents are {diff:.1f} percentage points MORE likely")
            print(f"  to live in bike-but-not-walk zones, indicating LESS access to transit.")
            print(f"\n  RECOMMENDATION: Encourage bike infrastructure and bike-share programs")
            print(f"  (like Bluebikes) to bridge the first-mile gap for low-income communities.")
        else:
            diff = top_20_bike_pct - bottom_20_bike_pct
            print(f"\n  High-income residents are {diff:.1f} percentage points more likely")
            print(f"  to live in bike-but-not-walk zones.")


def main():
    """
    Main function to create all equity visualizations.
    """
    # File paths
    step1_output = 'data/neighborhood_income_merged.geojson'
    step2_output = 'data/transit_accessibility_analysis.geojson'
    gtfs_stops_file = 'data/gtfs/stops.txt'
    
    # Check if Step 2 output exists (preferred - has all data)
    if Path(step2_output).exists():
        print("Using Step 2 output (includes transit accessibility)...")
        gdf = load_step2_data(step2_output)
    elif Path(step1_output).exists():
        print("Using Step 1 output (income data only)...")
        gdf = load_step1_data(step1_output)
        # Add dummy accessibility data for visualization
        gdf['distance_to_nearest_station_miles'] = np.nan
        gdf['accessibility_zone'] = 'unknown'
    else:
        print("ERROR: No data files found. Please run Step 1 first.")
        return
    
    # Load train stations for visualization
    train_stations_df = None
    if Path(gtfs_stops_file).exists():
        train_stations_df = pd.read_csv(gtfs_stops_file)
        # Filter for train stations if vehicle_type available
        if 'vehicle_type' in train_stations_df.columns:
            # Vehicle type: 0=Light rail, 1=Subway, 2=Rail (trains)
            train_stations_df = train_stations_df[train_stations_df['vehicle_type'].isin([0, 1, 2])]
        print(f"Loaded {len(train_stations_df)} train stations for visualization")
    
    # Create visualizations
    gdf = create_income_distribution_map(gdf, 'income_distribution_map.png')
    
    if train_stations_df is not None and 'distance_to_nearest_station_miles' in gdf.columns:
        create_transit_density_map(gdf, train_stations_df, 'transit_density_map.png')
        create_combined_equity_map(gdf, train_stations_df, 'equity_combined_map.png')
    
    if 'accessibility_zone' in gdf.columns:
        create_accessibility_by_income_chart(gdf, 'accessibility_by_income_chart.png')
        analyze_equity_gap(gdf)
    
    print("\n" + "="*70)
    print("All visualizations created successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - income_distribution_map.png")
    if train_stations_df is not None:
        print("  - transit_density_map.png")
        print("  - equity_combined_map.png")
    if 'accessibility_zone' in gdf.columns:
        print("  - accessibility_by_income_chart.png")


if __name__ == '__main__':
    main()

