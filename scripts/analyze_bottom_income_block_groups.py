#!/usr/bin/env python3
"""
Analyze Bottom 30% Block Groups by Median Income
- Compute centroids
- Measure distance from centroid to nearest train station
- Convert to miles
- Show distance to nearest train station by income
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
        sys.path.insert(0, str(Path(__file__).parent))
        from load_mbta_stations import main as load_stations
        stations_gdf = load_stations()
        if stations_gdf is not None:
            return stations_gdf
    except Exception as e:
        print(f"Error loading from GTFS: {e}")
    
    raise FileNotFoundError("Could not load train stations from any source")

def get_bottom_income_block_groups(gdf, percentile=0.30):
    """
    Get bottom percentile block groups by median income.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Block groups with median_household_income column
    percentile : float
        Percentile threshold (e.g., 0.30 for bottom 30%)
        
    Returns:
    --------
    GeoDataFrame with bottom block groups
    """
    if 'median_household_income' not in gdf.columns:
        raise ValueError("median_household_income column not found")
    
    # Remove any rows with missing income data
    gdf_clean = gdf[gdf['median_household_income'].notna()].copy()
    
    # Get bottom percentile (e.g., bottom 30% means below 30th percentile)
    threshold = gdf_clean['median_household_income'].quantile(percentile)
    bottom_groups = gdf_clean[gdf_clean['median_household_income'] <= threshold].copy()
    bottom_groups = bottom_groups.sort_values('median_household_income', ascending=True)
    
    pct_label = f"{percentile*100:.0f}%"
    print(f"\nBottom {pct_label} Block Groups by Median Income (threshold: ${threshold:,.0f}):")
    print("="*70)
    print(f"  Total block groups: {len(gdf_clean)}")
    print(f"  Bottom {pct_label} count: {len(bottom_groups)}")
    print(f"  Income range: ${bottom_groups['median_household_income'].min():,.0f} - ${bottom_groups['median_household_income'].max():,.0f}")
    
    return bottom_groups

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

def create_visualization(bottom_block_groups_gdf, output_file='bottom_30pct_income_analysis.png'):
    """Create visualization showing distance to nearest train station by income."""
    print(f"\nCreating visualization: {output_file}")
    
    n_groups = len(bottom_block_groups_gdf)
    
    # Create figure with multiple charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bottom 30% Income Block Groups: Distance to Nearest Train Station by Income', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Chart 1: Distance by Income (scatter) - MAIN CHART
    ax1 = axes[0, 0]
    scatter = ax1.scatter(bottom_block_groups_gdf['median_household_income'], 
                         bottom_block_groups_gdf['distance_miles'],
                         c=bottom_block_groups_gdf['distance_miles'],
                         cmap='RdYlGn_r', s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax1.set_xlabel('Median Household Income ($)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Distance to Nearest Station (miles)', fontsize=11, fontweight='bold')
    ax1.set_title('Distance to Nearest Train Station by Income', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Distance (miles)')
    
    # Add zone lines
    ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='<0.5 miles')
    ax1.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='1.0 mile')
    ax1.legend()
    
    # Chart 2: Histogram of distances
    ax2 = axes[0, 1]
    distances = bottom_block_groups_gdf['distance_miles']
    
    # Create histogram with zone colors
    bins = np.linspace(0, max(distances.max(), 10), 30)
    
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
    zone_counts = bottom_block_groups_gdf['access_zone'].value_counts()
    zone_colors = {'<0.5 miles': '#2ecc71', 'between 0.5-1miles': '#f39c12', 'beyond_1_mile': '#e74c3c'}
    colors_list = [zone_colors.get(zone, '#95a5a6') for zone in zone_counts.index]
    
    bars3 = ax3.bar(zone_counts.index, zone_counts.values,
                   color=colors_list, alpha=0.85, edgecolor='black', linewidth=1)
    ax3.set_xlabel('Access Zone', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Block Groups', fontsize=11, fontweight='bold')
    ax3.set_title(f'Access Zone Distribution (n={n_groups} block groups)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Chart 4: Income vs Distance with Zone Colors
    ax4 = axes[1, 1]
    for zone in ['<0.5 miles', 'between 0.5-1miles', 'beyond_1_mile']:
        zone_data = bottom_block_groups_gdf[bottom_block_groups_gdf['access_zone'] == zone]
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
    
    return bottom_block_groups_gdf

def main():
    """Main function."""
    print("="*70)
    print("Bottom 30% Income Block Groups: Distance Analysis")
    print("="*70)
    
    # Load income data
    print("\nLoading income data...")
    income_data = load_income_data()
    
    # Get bottom 30% block groups (percentile 0.30 means bottom 30%)
    print("\nIdentifying bottom 30% block groups by median income...")
    bottom_30_pct = get_bottom_income_block_groups(income_data, percentile=0.30)
    
    # Load block group shapefile (for geometry)
    print("\nLoading block group geometry...")
    block_groups_shapefile = load_block_groups_with_geometry()
    
    if block_groups_shapefile is not None:
        # Ensure GEOID format matches
        if 'GEOID' in block_groups_shapefile.columns:
            block_groups_shapefile['GEOID'] = block_groups_shapefile['GEOID'].astype(str).str.zfill(12)
        elif 'GEOID10' in block_groups_shapefile.columns:
            block_groups_shapefile['GEOID'] = block_groups_shapefile['GEOID10'].astype(str).str.zfill(12)
        
        if 'GEOID' in bottom_30_pct.columns:
            bottom_30_pct['GEOID'] = bottom_30_pct['GEOID'].astype(str).str.zfill(12)
        
        # Merge with shapefile to get geometry
        bottom_30_pct_with_geometry = block_groups_shapefile.merge(
            bottom_30_pct[['GEOID', 'median_household_income']], 
            on='GEOID', 
            how='inner'
        )
        print(f"✅ Merged {len(bottom_30_pct_with_geometry)} block groups with geometry")
    else:
        print("⚠️  Could not load shapefile, using income data only")
        bottom_30_pct_with_geometry = bottom_30_pct
    
    # Load train stations
    print("\nLoading train stations...")
    train_stations_gdf = load_train_stations()
    
    # Compute centroids and distances
    if 'geometry' in bottom_30_pct_with_geometry.columns:
        bottom_30_pct_with_distances = compute_centroids_and_distances(bottom_30_pct_with_geometry, train_stations_gdf)
    else:
        print("⚠️  No geometry available, cannot calculate distances")
        return
    
    # Display results
    print("\n" + "="*70)
    print(f"Results: Bottom 30% Block Groups with Distances (n={len(bottom_30_pct_with_distances)})")
    print("="*70)
    
    results_summary = bottom_30_pct_with_distances[[
        'GEOID', 'median_household_income', 'distance_miles', 
        'nearest_station_name', 'access_zone'
    ]].copy()
    results_summary = results_summary.sort_values('median_household_income', ascending=True)
    
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
    create_visualization(bottom_30_pct_with_distances)
    
    # Save results
    output_csv = 'data/bottom_30pct_income_block_groups_distances.csv'
    results_summary.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

if __name__ == '__main__':
    main()

