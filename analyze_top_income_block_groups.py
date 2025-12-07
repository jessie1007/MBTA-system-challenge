#!/usr/bin/env python3
"""
Analyze Top 20 Block Groups by Median Income
- Compute centroids
- Measure distance from centroid to nearest train station
- Convert to miles
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from shapely.geometry import Point
from geopy.distance import geodesic
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

# Massachusetts State Plane CRS (meters) for accurate distance calculations
MA_CRS = 'EPSG:26986'

def load_income_data():
    """Load income data for block groups."""
    # Try multiple sources
    data_sources = [
        'data/neighborhood_income_merged.geojson',
        'data/neighborhood_income_merged.csv'
    ]
    
    for source in data_sources:
        if Path(source).exists():
            print(f"Loading from: {source}")
            if source.endswith('.geojson'):
                try:
                    gdf = gpd.read_file(source)
                    return gdf
                except Exception as e:
                    print(f"Error with geopandas: {e}")
                    # Try JSON fallback
                    try:
                        with open(source, 'r') as f:
                            geojson_data = json.load(f)
                        features = geojson_data.get('features', [])
                        data = []
                        for feature in features:
                            props = feature.get('properties', {})
                            data.append(props)
                        df = pd.DataFrame(data)
                        gdf = gpd.GeoDataFrame(df)
                        print("Loaded without geometry")
                        return gdf
                    except:
                        continue
            else:
                df = pd.read_csv(source)
                return df
    
    raise FileNotFoundError("Could not find income data file")

def load_block_groups_with_geometry():
    """Load block group shapefile with geometry."""
    shapefile_path = 'data/block_groups.shp'
    
    if not Path(shapefile_path).exists():
        return None
    
    try:
        gdf = gpd.read_file(shapefile_path)
        return gdf
    except Exception as e:
        print(f"Error with geopandas: {e}")
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
            return gdf
        except Exception as e2:
            print(f"Error with fiona: {e2}")
            return None

def load_train_stations():
    """Load train stations (from our previous script or GTFS)."""
    # Try to load from our saved file first
    stations_file = 'data/mbta_train_stations.geojson'
    
    if Path(stations_file).exists():
        print(f"Loading train stations from: {stations_file}")
        try:
            stations_gdf = gpd.read_file(stations_file)
            print(f"✅ Loaded {len(stations_gdf)} train stations")
            return stations_gdf
        except Exception as e:
            print(f"Error loading GeoJSON: {e}")
    
    # Fallback: load from GTFS directly
    print("Loading train stations from GTFS...")
    try:
        import sys
        # Import the module and call main
        sys.path.insert(0, str(Path(__file__).parent))
        from load_mbta_stations import main as load_stations
        stations_gdf = load_stations()
        if stations_gdf is not None:
            return stations_gdf
    except Exception as e:
        print(f"Error loading from GTFS: {e}")
    
    raise FileNotFoundError("Could not load train stations from any source")

def get_income_block_groups(gdf, n=None, percentile=None, bottom=False):
    """
    Get top or bottom block groups by median income.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Block groups with median_household_income column
    n : int, optional
        Number of block groups to return (e.g., 20)
    percentile : float, optional
        Percentile threshold (e.g., 0.70 for top 30% or bottom 30%)
    bottom : bool
        If True, get bottom percentile/n; if False, get top percentile/n
        
    Returns:
    --------
    GeoDataFrame with selected block groups
    """
    if 'median_household_income' not in gdf.columns:
        raise ValueError("median_household_income column not found")
    
    # Remove any rows with missing income data
    gdf_clean = gdf[gdf['median_household_income'].notna()].copy()
    
    if percentile is not None:
        if bottom:
            # Get bottom percentile (e.g., bottom 30% means below 30th percentile)
            threshold = gdf_clean['median_household_income'].quantile(percentile)
            selected_groups = gdf_clean[gdf_clean['median_household_income'] <= threshold].copy()
            selected_groups = selected_groups.sort_values('median_household_income', ascending=True)
            pct_label = f"{percentile*100:.0f}%"
            direction = "Bottom"
        else:
            # Get top percentile (e.g., top 30% means above 70th percentile)
            threshold = gdf_clean['median_household_income'].quantile(percentile)
            selected_groups = gdf_clean[gdf_clean['median_household_income'] >= threshold].copy()
            selected_groups = selected_groups.sort_values('median_household_income', ascending=False)
            pct_label = f"{(1-percentile)*100:.0f}%"
            direction = "Top"
        
        print(f"\n{direction} {pct_label} Block Groups by Median Income (threshold: ${threshold:,.0f}):")
        print("="*70)
        print(f"  Total block groups: {len(gdf_clean)}")
        print(f"  {direction} {pct_label} count: {len(selected_groups)}")
        print(f"  Income range: ${selected_groups['median_household_income'].min():,.0f} - ${selected_groups['median_household_income'].max():,.0f}")
        
    elif n is not None:
        if bottom:
            # Get bottom N
            gdf_sorted = gdf_clean.sort_values('median_household_income', ascending=True)
            selected_groups = gdf_sorted.head(n).copy()
            direction = "Bottom"
        else:
            # Get top N
            gdf_sorted = gdf_clean.sort_values('median_household_income', ascending=False)
            selected_groups = gdf_sorted.head(n).copy()
            direction = "Top"
        
        print(f"\n{direction} {n} Block Groups by Median Income:")
        print("="*70)
        for idx, row in selected_groups.head(10).iterrows():
            income = row['median_household_income']
            geoid = row.get('GEOID', 'N/A')
            print(f"  GEOID: {geoid}, Income: ${income:,.0f}")
        if len(selected_groups) > 10:
            print(f"  ... and {len(selected_groups) - 10} more")
    else:
        raise ValueError("Must specify either 'n' (number) or 'percentile' (e.g., 0.70 for top 30%)")
    
    return selected_groups

def compute_centroids_and_distances(block_groups_gdf, train_stations_gdf):
    """
    Compute centroids for block groups and calculate distances to nearest station.
    Uses proper CRS projection for accurate distance calculations.
    """
    print("\n" + "="*70)
    print("Computing Centroids and Distances")
    print("="*70)
    
    # Step 1: Ensure we have geometry
    if 'geometry' not in block_groups_gdf.columns:
        print("⚠️  No geometry available, cannot compute centroids")
        return block_groups_gdf
    
    # Step 2: Convert to projected CRS (Massachusetts State Plane)
    print("\nStep 1: Converting to Massachusetts State Plane CRS (EPSG:26986)...")
    block_groups_proj = block_groups_gdf.to_crs(MA_CRS)
    train_stations_proj = train_stations_gdf.to_crs(MA_CRS)
    print("✅ Converted to projected CRS")
    
    # Step 3: Compute centroids
    print("\nStep 2: Computing block group centroids...")
    block_groups_proj['centroid'] = block_groups_proj.geometry.centroid
    block_groups_proj['centroid_lat'] = block_groups_proj['centroid'].y
    block_groups_proj['centroid_lon'] = block_groups_proj['centroid'].x
    print("✅ Computed centroids")
    
    # Step 4: Calculate distances using spatial index
    print("\nStep 3: Calculating distances to nearest train stations...")
    
    # Build spatial index for stations
    stations_sindex = train_stations_proj.sindex
    
    distances_meters = []
    distances_miles = []
    nearest_station_ids = []
    nearest_station_names = []
    
    for idx, row in block_groups_proj.iterrows():
        centroid = row['centroid']
        
        # Use spatial index to find nearby stations
        possible_matches_index = list(stations_sindex.intersection(centroid.bounds))
        possible_matches = train_stations_proj.iloc[possible_matches_index]
        
        if len(possible_matches) > 0:
            # Calculate distance to all possible matches
            dists = [centroid.distance(station.geometry) for _, station in possible_matches.iterrows()]
            min_idx = np.argmin(dists)
            min_dist_meters = dists[min_idx]
            nearest_station = possible_matches.iloc[min_idx]
            nearest_station_id = nearest_station['stop_id']
            nearest_station_name = nearest_station.get('stop_name', 'Unknown')
        else:
            # Fallback: calculate to all stations
            dists = [centroid.distance(station.geometry) for _, station in train_stations_proj.iterrows()]
            min_idx = np.argmin(dists)
            min_dist_meters = dists[min_idx]
            nearest_station = train_stations_proj.iloc[min_idx]
            nearest_station_id = nearest_station['stop_id']
            nearest_station_name = nearest_station.get('stop_name', 'Unknown')
        
        distances_meters.append(min_dist_meters)
        distances_miles.append(min_dist_meters / 1609.34)  # Convert meters to miles
        nearest_station_ids.append(nearest_station_id)
        nearest_station_names.append(nearest_station_name)
    
    # Add to dataframe
    block_groups_proj['distance_meters'] = distances_meters
    block_groups_proj['distance_miles'] = distances_miles
    block_groups_proj['nearest_station_id'] = nearest_station_ids
    block_groups_proj['nearest_station_name'] = nearest_station_names
    
    # Assign access zones
    def categorize_zone(distance_miles):
        if pd.isna(distance_miles):
            return 'unknown'
        elif distance_miles < 0.5:
            return '<0.5 miles'
        elif distance_miles < 1.0:
            return 'between 0.5-1miles'
        else:  # distance_miles >= 1.0
            return 'beyond_1_mile'
    
    block_groups_proj['access_zone'] = block_groups_proj['distance_miles'].apply(categorize_zone)
    
    print(f"\n✅ Distance calculation complete!")
    print(f"   Range: {min(distances_miles):.2f} - {max(distances_miles):.2f} miles")
    
    # Convert back to WGS84 for compatibility
    block_groups_wgs84 = block_groups_proj.to_crs('EPSG:4326')
    
    return block_groups_wgs84

def create_visualization(top_block_groups_gdf, output_file='top_income_block_groups_analysis.png', title_suffix=''):
    """Create visualization of top block groups with distances."""
    print(f"\nCreating visualization: {output_file}")
    
    n_groups = len(top_block_groups_gdf)
    title = f'Top Income Block Groups: Distance to Nearest Train Station{title_suffix}'
    
    # Create figure with multiple charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Chart 1: Distance by Income (scatter)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(top_block_groups_gdf['median_household_income'], 
                         top_block_groups_gdf['distance_miles'],
                         c=top_block_groups_gdf['distance_miles'],
                         cmap='RdYlGn_r', s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax1.set_xlabel('Median Household Income ($)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Distance to Nearest Station (miles)', fontsize=11, fontweight='bold')
    ax1.set_title('Income vs Distance to Train Station', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Distance (miles)')
    
    # Add zone lines
    ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='<0.5 miles')
    ax1.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='1.0 mile')
    ax1.legend()
    
    # Chart 2: Histogram of distances
    ax2 = axes[0, 1]
    distances = top_block_groups_gdf['distance_miles']
    
    # Create histogram with zone colors
    bins = np.linspace(0, max(distances.max(), 10), 30)
    colors_hist = ['#2ecc71' if d < 0.5 else ('#f39c12' if d < 1.0 else '#e74c3c') 
                   for d in distances]
    
    n, bins_edges, patches = ax2.hist(distances, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color bars by zone
    for i, (patch, bin_left) in enumerate(zip(patches, bins_edges[:-1])):
        if bin_left < 0.5:
            patch.set_facecolor('#2ecc71')
        elif bin_left < 1.0:
            patch.set_facecolor('#f39c12')
        else:
            patch.set_facecolor('#e74c3c')
    
    ax2.set_xlabel('Distance to Nearest Station (miles)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Block Groups', fontsize=11, fontweight='bold')
    ax2.set_title(f'Distance Distribution (n={n_groups} block groups)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axvline(x=1.0, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    
    # Chart 3: Access Zone Distribution
    ax3 = axes[1, 0]
    zone_counts = top_block_groups_gdf['access_zone'].value_counts()
    zone_colors = {'<0.5 miles': '#2ecc71', 'between 0.5-1miles': '#f39c12', 'beyond_1_mile': '#e74c3c'}
    colors_list = [zone_colors.get(zone, '#95a5a6') for zone in zone_counts.index]
    
    bars3 = ax3.bar(zone_counts.index, zone_counts.values,
                   color=colors_list, alpha=0.85, edgecolor='black', linewidth=1)
    ax3.set_xlabel('Access Zone', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Block Groups', fontsize=11, fontweight='bold')
    ax3.set_title(f'Access Zone Distribution (n={n_groups} block groups)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Chart 4: Income vs Distance with Zone Colors
    ax4 = axes[1, 1]
    for zone in ['<0.5 miles', 'between 0.5-1miles', 'beyond_1_mile']:
        zone_data = top_block_groups_gdf[top_block_groups_gdf['access_zone'] == zone]
        if len(zone_data) > 0:
            ax4.scatter(zone_data['median_household_income'], 
                       zone_data['distance_miles'],
                       label=zone,
                       s=100, alpha=0.7, edgecolors='black', linewidth=1,
                       color=zone_colors.get(zone, '#95a5a6'))
    
    ax4.set_xlabel('Median Household Income ($)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Distance to Nearest Station (miles)', fontsize=11, fontweight='bold')
    ax4.set_title('Income vs Distance by Access Zone', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
    ax4.axhline(y=1.0, color='orange', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved as '{output_file}'")
    plt.show()
    
    return top_block_groups_gdf

def run_income_analysis(income_data, block_groups_shapefile, train_stations_gdf, 
                        percentile=0.30, bottom=False):
    """
    Run distance analysis for a given income percentile group.
    
    Parameters:
    -----------
    income_data : DataFrame
        Income data for all block groups
    block_groups_shapefile : GeoDataFrame
        Block group geometries
    train_stations_gdf : GeoDataFrame
        Train station locations
    percentile : float
        Percentile threshold (0.30 for bottom 30%, 0.70 for top 30%)
    bottom : bool
        If True, analyze bottom percentile; if False, analyze top percentile
        
    Returns:
    --------
    DataFrame with results
    """
    direction = "Bottom" if bottom else "Top"
    pct_label = f"{percentile*100:.0f}%" if bottom else f"{(1-percentile)*100:.0f}%"
    
    print(f"\n{'='*70}")
    print(f"{direction} {pct_label} Block Groups by Income: Distance Analysis")
    print("="*70)
    
    # Get income group block groups
    print(f"\nIdentifying {direction.lower()} {pct_label} block groups by median income...")
    income_groups = get_income_block_groups(income_data, percentile=percentile, bottom=bottom)
    
    if block_groups_shapefile is not None:
        # Ensure GEOID format matches
        if 'GEOID' in block_groups_shapefile.columns:
            block_groups_shapefile['GEOID'] = block_groups_shapefile['GEOID'].astype(str).str.zfill(12)
        elif 'GEOID10' in block_groups_shapefile.columns:
            block_groups_shapefile['GEOID'] = block_groups_shapefile['GEOID10'].astype(str).str.zfill(12)
        
        if 'GEOID' in income_groups.columns:
            income_groups['GEOID'] = income_groups['GEOID'].astype(str).str.zfill(12)
        
        # Merge with shapefile to get geometry
        income_groups_with_geometry = block_groups_shapefile.merge(
            income_groups[['GEOID', 'median_household_income']], 
            on='GEOID', 
            how='inner'
        )
        print(f"✅ Merged {len(income_groups_with_geometry)} block groups with geometry")
    else:
        print("⚠️  Could not load shapefile, using income data only")
        income_groups_with_geometry = income_groups
    
    # Compute centroids and distances
    if 'geometry' in income_groups_with_geometry.columns:
        income_groups_with_distances = compute_centroids_and_distances(income_groups_with_geometry, train_stations_gdf)
    else:
        print("⚠️  No geometry available, cannot calculate distances")
        return None
    
    # Display results
    print("\n" + "="*70)
    print(f"Results: {direction} {pct_label} Block Groups with Distances (n={len(income_groups_with_distances)})")
    print("="*70)
    
    results_summary = income_groups_with_distances[[
        'GEOID', 'median_household_income', 'distance_miles', 
        'nearest_station_name', 'access_zone'
    ]].copy()
    
    # Sort by income (descending for top, ascending for bottom)
    sort_ascending = bottom
    results_summary = results_summary.sort_values('median_household_income', ascending=sort_ascending)
    
    # Show first 20 and summary
    print("\nBlock Group | Income      | Distance (mi) | Zone | Nearest Station")
    print("-" * 80)
    for _, row in results_summary.head(20).iterrows():
        geoid_short = row['GEOID'][-4:] if len(str(row['GEOID'])) >= 4 else row['GEOID']
        income = row['median_household_income']
        dist = row['distance_miles']
        zone = row['access_zone']
        station = row['nearest_station_name'][:25] if pd.notna(row['nearest_station_name']) else 'Unknown'
        print(f"BG {geoid_short:4s} | ${income:>10,.0f} | {dist:>11.2f} | {zone:20s} | {station}")
    
    if len(results_summary) > 20:
        print(f"\n... and {len(results_summary) - 20} more block groups")
    
    # Statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"Average distance: {results_summary['distance_miles'].mean():.2f} miles")
    print(f"Median distance: {results_summary['distance_miles'].median():.2f} miles")
    print(f"Min distance: {results_summary['distance_miles'].min():.2f} miles")
    print(f"Max distance: {results_summary['distance_miles'].max():.2f} miles")
    
    print(f"\nAccess zone distribution:")
    zone_dist = results_summary['access_zone'].value_counts()
    for zone, count in zone_dist.items():
        print(f"  {zone}: {count} block groups ({count/len(results_summary)*100:.1f}%)")
    
    # Create visualization
    output_file = f'{direction.lower()}_30pct_income_block_groups_analysis.png'
    title_suffix = f' ({direction} 30% by Income)'
    create_visualization(income_groups_with_distances, 
                        output_file=output_file,
                        title_suffix=title_suffix)
    
    # Save results
    output_csv = f'data/{direction.lower()}_30pct_income_block_groups_distances.csv'
    results_summary.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    return results_summary

def main():
    """Main function."""
    print("="*70)
    print("Income-Based Transit Accessibility Analysis")
    print("Top 30% vs Bottom 30% Block Groups")
    print("="*70)
    
    # Load income data
    print("\nLoading income data...")
    income_data = load_income_data()
    
    # Load block group shapefile (for geometry)
    print("\nLoading block group geometry...")
    block_groups_shapefile = load_block_groups_with_geometry()
    
    # Load train stations
    print("\nLoading train stations...")
    train_stations_gdf = load_train_stations()
    
    # Run analysis for top 30%
    top_results = run_income_analysis(
        income_data, block_groups_shapefile, train_stations_gdf,
        percentile=0.70, bottom=False
    )
    
    print("\n\n" + "="*70)
    print("="*70)
    print("\n")
    
    # Run analysis for bottom 30%
    bottom_results = run_income_analysis(
        income_data, block_groups_shapefile, train_stations_gdf,
        percentile=0.30, bottom=True
    )
    
    # Compare results
    if top_results is not None and bottom_results is not None:
        print("\n\n" + "="*70)
        print("Comparison: Top 30% vs Bottom 30%")
        print("="*70)
        print(f"\nTop 30% Average Distance:    {top_results['distance_miles'].mean():.2f} miles")
        print(f"Bottom 30% Average Distance: {bottom_results['distance_miles'].mean():.2f} miles")
        print(f"Difference:                  {abs(top_results['distance_miles'].mean() - bottom_results['distance_miles'].mean()):.2f} miles")
        
        print(f"\nTop 30% Median Distance:     {top_results['distance_miles'].median():.2f} miles")
        print(f"Bottom 30% Median Distance:  {bottom_results['distance_miles'].median():.2f} miles")
        
        print(f"\nTop 30% - <0.5 miles:        {(top_results['access_zone'] == '<0.5 miles').sum()} ({(top_results['access_zone'] == '<0.5 miles').sum()/len(top_results)*100:.1f}%)")
        print(f"Bottom 30% - <0.5 miles:     {(bottom_results['access_zone'] == '<0.5 miles').sum()} ({(bottom_results['access_zone'] == '<0.5 miles').sum()/len(bottom_results)*100:.1f}%)")
        
        print(f"\nTop 30% - beyond_1_mile:     {(top_results['access_zone'] == 'beyond_1_mile').sum()} ({(top_results['access_zone'] == 'beyond_1_mile').sum()/len(top_results)*100:.1f}%)")
        print(f"Bottom 30% - beyond_1_mile:  {(bottom_results['access_zone'] == 'beyond_1_mile').sum()} ({(bottom_results['access_zone'] == 'beyond_1_mile').sum()/len(bottom_results)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

if __name__ == '__main__':
    main()

