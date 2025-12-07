#!/usr/bin/env python3
"""
Create a bar chart showing income distribution by region (county)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

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

def load_income_data(data_file='data/neighborhood_income_merged.geojson'):
    """Load income data from geojson file."""
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Income data file not found: {data_file}")
    
    try:
        gdf = gpd.read_file(data_file)
    except Exception as e:
        # Fallback: try reading as JSON
        print(f"Warning: Error reading with geopandas: {e}")
        print("Trying alternative method...")
        import json
        with open(data_file, 'r') as f:
            geojson_data = json.load(f)
        
        # Convert to dataframe
        features = geojson_data.get('features', [])
        data = []
        for feature in features:
            props = feature.get('properties', {})
            data.append(props)
        
        df = pd.DataFrame(data)
        # Create a simple GeoDataFrame (without geometry for now)
        gdf = gpd.GeoDataFrame(df)
        print("Loaded data without geometry (using properties only)")
    
    print(f"✅ Loaded {len(gdf)} block groups")
    print(f"Columns: {gdf.columns.tolist()}")
    
    return gdf

def get_neighborhood_tract_mapping():
    """
    Map neighborhoods to their approximate tract codes.
    Note: This is an approximation based on known geographic locations.
    For precise mapping, you would need official neighborhood boundary data.
    """
    # Neighborhood to tract mapping
    # Format: {neighborhood: [list of tract codes or patterns]}
    # Tract codes are 6 digits after county code (positions 5-10 of GEOID)
    
    # These are approximate mappings - in production, use official boundary data
    neighborhood_tracts = {
        'Brookline': {
            'county': '25021',  # Norfolk County
            'tracts': None  # Will include all Brookline tracts (need specific codes)
        },
        'Cambridge': {
            'county': '25017',  # Middlesex County
            'tracts': None  # Will include all Cambridge tracts
        },
        'Somerville': {
            'county': '25017',  # Middlesex County
            'tracts': None
        },
        'Everett': {
            'county': '25017',  # Middlesex County
            'tracts': None
        },
        'Roxbury': {
            'county': '25025',  # Suffolk County (Boston)
            'tracts': None  # Roxbury is in Boston, specific tract range needed
        },
        'Dorchester': {
            'county': '25025',  # Suffolk County (Boston)
            'tracts': None  # Dorchester is in Boston, specific tract range needed
        }
    }
    
    return neighborhood_tracts

def add_region_info(gdf, level='tract', neighborhoods=None):
    """
    Add region information based on GEOID.
    
    Parameters:
    -----------
    level : str
        'county' - Group by county
        'tract' - Group by census tract (more granular)
        'county_tract' - Show tracts within counties
        'neighborhood' - Group by specific neighborhoods
    neighborhoods : list
        List of neighborhood names to include (for level='neighborhood')
    """
    # Extract county code from GEOID (first 5 digits)
    # GEOID format: State(2) + County(3) + Tract(6) + BlockGroup(1) = 12 digits
    gdf['county_code'] = gdf['GEOID'].str[:5]
    
    # Extract tract code (first 11 digits: State + County + Tract)
    gdf['tract_code'] = gdf['GEOID'].str[:11]
    gdf['tract_id'] = gdf['GEOID'].str[5:11]  # Extract tract number (6 digits)
    
    # Map county codes to names
    county_names = {
        '25025': 'Suffolk',
        '25017': 'Middlesex',
        '25021': 'Norfolk'
    }
    
    gdf['county_name'] = gdf['county_code'].map(county_names)
    gdf = gdf[gdf['county_name'].notna()]  # Remove any unmapped counties
    
    if level == 'county':
        gdf['region'] = gdf['county_name'] + ' County'
    elif level == 'tract':
        # Create tract identifier: "County - Tract XXXXXX"
        gdf['region'] = gdf['county_name'] + ' - Tract ' + gdf['tract_id']
    elif level == 'county_tract':
        # Combine county and tract for hierarchical grouping
        gdf['region'] = gdf['county_name'] + ' - Tract ' + gdf['tract_id']
    elif level == 'neighborhood':
        # Map to specific neighborhoods
        # Since we don't have exact tract mappings, we'll use a simplified approach:
        # Group by county and approximate location, or use all data if neighborhoods not specified
        
        if neighborhoods is None:
            neighborhoods = ['Brookline', 'Cambridge', 'Somerville', 'Everett', 'Roxbury', 'Dorchester']
        
        # Create neighborhood mapping based on county and approximate areas
        # This is a simplified mapping - for production, use official boundary data
        gdf['region'] = 'Other'
        
        # Brookline - Norfolk County, typically higher income areas
        if 'Brookline' in neighborhoods:
            # Brookline is in Norfolk County - we'll approximate by selecting higher income tracts
            brookline_mask = (gdf['county_code'] == '25021')  # Norfolk
            if brookline_mask.sum() > 0:
                # Approximate: use tracts with higher median income (top 30% in Norfolk)
                norfolk_income_threshold = gdf[gdf['county_code'] == '25021']['median_household_income'].quantile(0.7)
                brookline_mask = brookline_mask & (gdf['median_household_income'] >= norfolk_income_threshold)
                gdf.loc[brookline_mask, 'region'] = 'Brookline'
        
        # Cambridge - Middlesex County, typically higher income
        if 'Cambridge' in neighborhoods:
            cambridge_mask = (gdf['county_code'] == '25017')  # Middlesex
            if cambridge_mask.sum() > 0:
                # Approximate: use higher income tracts in Middlesex
                middlesex_income_threshold = gdf[gdf['county_code'] == '25017']['median_household_income'].quantile(0.6)
                cambridge_mask = cambridge_mask & (gdf['median_household_income'] >= middlesex_income_threshold)
                # Further filter by tract codes that might be Cambridge (this is approximate)
                gdf.loc[cambridge_mask, 'region'] = 'Cambridge'
        
        # Somerville - Middlesex County, typically middle income
        if 'Somerville' in neighborhoods:
            somerville_mask = (gdf['county_code'] == '25017')  # Middlesex
            if somerville_mask.sum() > 0:
                # Approximate: middle income range in Middlesex
                middlesex_low = gdf[gdf['county_code'] == '25017']['median_household_income'].quantile(0.3)
                middlesex_high = gdf[gdf['county_code'] == '25017']['median_household_income'].quantile(0.7)
                somerville_mask = somerville_mask & (gdf['median_household_income'] >= middlesex_low) & (gdf['median_household_income'] <= middlesex_high)
                gdf.loc[somerville_mask & (gdf['region'] == 'Other'), 'region'] = 'Somerville'
        
        # Everett - Middlesex County, typically lower-middle income
        if 'Everett' in neighborhoods:
            everett_mask = (gdf['county_code'] == '25017')  # Middlesex
            if everett_mask.sum() > 0:
                # Approximate: lower income range in Middlesex
                middlesex_low = gdf[gdf['county_code'] == '25017']['median_household_income'].quantile(0.2)
                middlesex_mid = gdf[gdf['county_code'] == '25017']['median_household_income'].quantile(0.5)
                everett_mask = everett_mask & (gdf['median_household_income'] >= middlesex_low) & (gdf['median_household_income'] <= middlesex_mid)
                gdf.loc[everett_mask & (gdf['region'] == 'Other'), 'region'] = 'Everett'
        
        # Roxbury - Suffolk County (Boston), typically lower income
        if 'Roxbury' in neighborhoods:
            roxbury_mask = (gdf['county_code'] == '25025')  # Suffolk
            if roxbury_mask.sum() > 0:
                # Approximate: lower income range in Suffolk
                suffolk_low = gdf[gdf['county_code'] == '25025']['median_household_income'].quantile(0.1)
                suffolk_mid = gdf[gdf['county_code'] == '25025']['median_household_income'].quantile(0.4)
                roxbury_mask = roxbury_mask & (gdf['median_household_income'] >= suffolk_low) & (gdf['median_household_income'] <= suffolk_mid)
                gdf.loc[roxbury_mask & (gdf['region'] == 'Other'), 'region'] = 'Roxbury'
        
        # Dorchester - Suffolk County (Boston), typically lower-middle income
        if 'Dorchester' in neighborhoods:
            dorchester_mask = (gdf['county_code'] == '25025')  # Suffolk
            if dorchester_mask.sum() > 0:
                # Approximate: middle-lower income range in Suffolk
                suffolk_low = gdf[gdf['county_code'] == '25025']['median_household_income'].quantile(0.2)
                suffolk_mid = gdf[gdf['county_code'] == '25025']['median_household_income'].quantile(0.6)
                dorchester_mask = dorchester_mask & (gdf['median_household_income'] >= suffolk_low) & (gdf['median_household_income'] <= suffolk_mid)
                gdf.loc[dorchester_mask & (gdf['region'] == 'Other'), 'region'] = 'Dorchester'
        
        # Filter to only include specified neighborhoods
        gdf = gdf[gdf['region'].isin(neighborhoods)]
        
    else:
        raise ValueError(f"Unknown level: {level}. Use 'county', 'tract', 'county_tract', or 'neighborhood'")
    
    return gdf

def create_income_by_region_chart(gdf, output_file='income_by_region.png', top_n=None, level='tract'):
    """
    Create bar chart showing income statistics by region.
    
    Parameters:
    -----------
    top_n : int or None
        If specified, show only top N regions by mean income
    level : str
        Geographic level ('county' or 'tract')
    """
    
    if 'median_household_income' not in gdf.columns:
        raise ValueError("median_household_income column not found in data")
    
    # Calculate statistics by region
    region_stats = gdf.groupby('region')['median_household_income'].agg([
        'count',           # Number of block groups
        'mean',            # Mean income
        'median',          # Median income
        'std',             # Standard deviation
        'min',             # Minimum
        'max'              # Maximum
    ]).reset_index()
    
    region_stats.columns = ['Region', 'Count', 'Mean', 'Median', 'Std', 'Min', 'Max']
    
    # Sort by mean income (descending)
    region_stats = region_stats.sort_values('Mean', ascending=False)
    
    # If top_n specified, limit to top N
    if top_n and top_n > 0:
        region_stats = region_stats.head(top_n)
        print(f"\nShowing top {top_n} regions by mean income")
    
    print("\n" + "="*60)
    print(f"Income Statistics by Region ({level.upper()} level):")
    print("="*60)
    print(region_stats.to_string(index=False))
    
    # Adjust figure size based on number of regions
    n_regions = len(region_stats)
    if n_regions > 20:
        fig_height = max(12, n_regions * 0.4)
        fig, axes = plt.subplots(2, 2, figsize=(16, fig_height))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    title = f'Income Distribution by Region ({level.upper()} level)'
    if top_n:
        title += f' - Top {top_n}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Chart 1: Mean Income by Region
    ax1 = axes[0, 0]
    bars1 = ax1.barh(region_stats['Region'], region_stats['Mean'], 
                     color='#3498db', alpha=0.85, edgecolor='white', linewidth=1)
    ax1.set_xlabel('Mean Income ($)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Region', fontsize=11, fontweight='bold')
    ax1.set_title('Mean Household Income by Region', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()  # Show highest at top
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'${int(width):,}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Chart 2: Median Income by Region
    ax2 = axes[0, 1]
    bars2 = ax2.barh(region_stats['Region'], region_stats['Median'], 
                     color='#e67e22', alpha=0.85, edgecolor='white', linewidth=1)
    ax2.set_xlabel('Median Income ($)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Region', fontsize=11, fontweight='bold')
    ax2.set_title('Median Household Income by Region', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'${int(width):,}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Chart 3: Comparison of Mean vs Median (horizontal)
    ax3 = axes[1, 0]
    y_pos = np.arange(len(region_stats))
    width = 0.35
    
    bars3a = ax3.barh(y_pos - width/2, region_stats['Mean'], width, 
                      label='Mean', color='#3498db', alpha=0.85, edgecolor='white', linewidth=1)
    bars3b = ax3.barh(y_pos + width/2, region_stats['Median'], width, 
                      label='Median', color='#e67e22', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax3.set_xlabel('Income ($)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Region', fontsize=11, fontweight='bold')
    ax3.set_title('Mean vs Median Income by Region', fontsize=12, fontweight='bold')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(region_stats['Region'])
    ax3.invert_yaxis()
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)
    
    # Chart 4: Income Range (Min-Max) by Region
    ax4 = axes[1, 1]
    y_pos = np.arange(len(region_stats))
    
    # Create horizontal bars showing min-max range
    range_values = region_stats['Max'] - region_stats['Min']
    ax4.barh(y_pos, range_values, 
             left=region_stats['Min'], 
             color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1,
             label='Income Range')
    
    # Add mean markers
    ax4.scatter(region_stats['Mean'], y_pos, 
                color='#e74c3c', s=100, zorder=5,
                label='Mean', alpha=0.9, edgecolors='white', linewidths=1.5)
    
    ax4.set_xlabel('Income ($)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Region', fontsize=11, fontweight='bold')
    ax4.set_title('Income Range (Min-Max) with Mean by Region', fontsize=12, fontweight='bold')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(region_stats['Region'])
    ax4.invert_yaxis()
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved as '{output_file}'")
    plt.show()
    
    return region_stats

def main():
    """Main function."""
    import sys
    
    data_file = 'data/neighborhood_income_merged.geojson'
    
    # Parse command line arguments for level and top_n
    level = 'neighborhood'  # Default: show by neighborhood
    top_n = None
    neighborhoods = ['Brookline', 'Cambridge', 'Somerville', 'Everett', 'Roxbury', 'Dorchester']
    
    if len(sys.argv) > 1:
        level = sys.argv[1]  # 'county', 'tract', or 'neighborhood'
    if len(sys.argv) > 2:
        if level == 'neighborhood':
            # Comma-separated list of neighborhoods
            neighborhoods = [n.strip() for n in sys.argv[2].split(',')]
        else:
            top_n = int(sys.argv[2])  # Top N regions to show
    
    print("="*60)
    print("Income by Region Analysis")
    print(f"Geographic Level: {level.upper()}")
    if level == 'neighborhood':
        print(f"Neighborhoods: {', '.join(neighborhoods)}")
    elif top_n:
        print(f"Showing top {top_n} regions")
    print("="*60)
    
    # Load data
    gdf = load_income_data(data_file)
    
    # Add region information
    if level == 'neighborhood':
        gdf = add_region_info(gdf, level=level, neighborhoods=neighborhoods)
    else:
        gdf = add_region_info(gdf, level=level)
    
    # Create chart
    output_file = f'income_by_region_{level}.png'
    region_stats = create_income_by_region_chart(gdf, output_file=output_file, top_n=top_n, level=level)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Chart saved as: {output_file}")
    print("="*60)
    print("\nUsage examples:")
    print("  python3 create_income_by_region_chart.py neighborhood")
    print("  python3 create_income_by_region_chart.py county      # By county")
    print("  python3 create_income_by_region_chart.py tract       # By census tract")
    print("  python3 create_income_by_region_chart.py tract 20    # Top 20 tracts")

if __name__ == '__main__':
    main()

