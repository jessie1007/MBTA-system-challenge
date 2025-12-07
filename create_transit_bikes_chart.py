#!/usr/bin/env python3
"""
Create a chart showing distribution of Transit vs Bikes by Region
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

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

def load_excel_data(excel_file):
    """Load Excel file and return dataframe."""
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Excel file not found: {excel_file}")
    
    # Try reading all sheets to see what's available
    try:
        excel_file_obj = pd.ExcelFile(excel_file)
        print(f"Available sheets: {excel_file_obj.sheet_names}")
        
        # Read first sheet
        df = pd.read_excel(excel_file, sheet_name=0)
        print(f"\n✅ Loaded {excel_file}")
        print(f"   Shape: {df.shape} (rows, columns)")
        print(f"   Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        raise

def identify_columns(df):
    """Identify region, transit, bike, and both columns from dataframe."""
    columns_lower = [col.lower() for col in df.columns]
    
    # Find region column
    region_col = None
    region_keywords = ['region', 'area', 'county', 'neighborhood', 'zone']
    for keyword in region_keywords:
        for i, col_lower in enumerate(columns_lower):
            if keyword in col_lower:
                region_col = df.columns[i]
                break
        if region_col:
            break
    
    # Find transit column
    transit_col = None
    transit_keywords = ['transit', 'public transport', 'train', 'subway', 'bus', 'mbta', 'localtransit']
    for keyword in transit_keywords:
        for i, col_lower in enumerate(columns_lower):
            if keyword in col_lower and 'both' not in col_lower:
                transit_col = df.columns[i]
                break
        if transit_col:
            break
    
    # Find bike column
    bike_col = None
    bike_keywords = ['bike', 'bicycle', 'cycling', 'bikeshare', 'bluebike']
    for keyword in bike_keywords:
        for i, col_lower in enumerate(columns_lower):
            if keyword in col_lower and 'both' not in col_lower:
                bike_col = df.columns[i]
                break
        if bike_col:
            break
    
    # Find both column
    both_col = None
    both_keywords = ['both', 'users_both', 'bluebikes_and_localtransit']
    for keyword in both_keywords:
        for i, col_lower in enumerate(columns_lower):
            if keyword in col_lower:
                both_col = df.columns[i]
                break
        if both_col:
            break
    
    # Also check for exact match
    if both_col is None:
        for col in df.columns:
            if 'Users_Both_BlueBikes_and_LocalTransit' in col or 'users_both_bluebikes_and_localtransit' in col.lower():
                both_col = col
                break
    
    print(f"\nIdentified columns:")
    print(f"  Region: {region_col}")
    print(f"  Transit: {transit_col}")
    print(f"  Bike: {bike_col}")
    print(f"  Both: {both_col}")
    
    return region_col, transit_col, bike_col, both_col

def create_transit_bikes_chart(df, region_col, transit_col, bike_col, both_col=None, output_file='transit_bikes_by_region.png'):
    """Create chart showing distribution of transit vs bikes by region."""
    
    if not all([region_col, transit_col, bike_col]):
        print("Error: Could not identify all required columns")
        print(f"  Region: {region_col}")
        print(f"  Transit: {transit_col}")
        print(f"  Bike: {bike_col}")
        return
    
    # Prepare data - include both_col if available
    columns_to_use = [region_col, transit_col, bike_col]
    if both_col:
        columns_to_use.append(both_col)
    
    chart_data = df[columns_to_use].copy()
    chart_data = chart_data.dropna()
    
    # Group by region and calculate totals/averages
    agg_dict = {
        transit_col: 'sum',
        bike_col: 'sum'
    }
    if both_col:
        agg_dict[both_col] = 'sum'
    
    if all(chart_data[col].dtype in ['int64', 'float64'] for col in [transit_col, bike_col] + ([both_col] if both_col else [])):
        # If numeric, sum by region
        region_stats = chart_data.groupby(region_col).agg(agg_dict).reset_index()
    else:
        # If categorical, count occurrences
        region_stats = chart_data.groupby(region_col).agg({col: 'count' for col in agg_dict.keys()}).reset_index()
        region_stats.columns = [region_col] + list(agg_dict.keys())
    
    # Create figure with multiple chart types
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    title = 'Distribution of Transit vs Bikes by Region'
    if both_col:
        title += ' (Including Both Modes)'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Chart 1: Grouped Bar Chart
    ax1 = axes[0, 0]
    x = np.arange(len(region_stats))
    if both_col:
        width = 0.25  # Narrower bars for 3 modes
        offset = width
    else:
        width = 0.35
        offset = width
    
    bars1 = ax1.bar(x - offset, region_stats[transit_col], width, 
                    label='Transit', color='#3498db', alpha=0.85, edgecolor='white', linewidth=1)
    bars2 = ax1.bar(x, region_stats[bike_col], width, 
                    label='Bikes', color='#e67e22', alpha=0.85, edgecolor='white', linewidth=1)
    if both_col:
        bars3 = ax1.bar(x + offset, region_stats[both_col], width, 
                        label='Both', color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax1.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count/Value', fontsize=11, fontweight='bold')
    ax1.set_title('Grouped Bar Chart: Transit vs Bikes by Region', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(region_stats[region_col], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    all_bars = [bars1, bars2]
    if both_col:
        all_bars.append(bars3)
    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label if value is positive
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height) if height == int(height) else f"{height:.1f}"}',
                        ha='center', va='bottom', fontsize=7)
    
    # Chart 2: Stacked Bar Chart
    ax2 = axes[0, 1]
    bottom_transit = region_stats[transit_col]
    ax2.bar(region_stats[region_col], region_stats[transit_col], 
            label='Transit', color='#3498db', alpha=0.85, edgecolor='white', linewidth=1)
    bottom_bikes = bottom_transit + region_stats[bike_col]
    ax2.bar(region_stats[region_col], region_stats[bike_col], 
            bottom=bottom_transit, label='Bikes', color='#e67e22', alpha=0.85, edgecolor='white', linewidth=1)
    if both_col:
        ax2.bar(region_stats[region_col], region_stats[both_col], 
                bottom=bottom_bikes, label='Both', color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax2.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count/Value', fontsize=11, fontweight='bold')
    ax2.set_title('Stacked Bar Chart: Transit vs Bikes by Region', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Chart 3: Percentage Stacked Bar Chart
    ax3 = axes[1, 0]
    total = region_stats[transit_col] + region_stats[bike_col]
    if both_col:
        total = total + region_stats[both_col]
    transit_pct = (region_stats[transit_col] / total * 100).fillna(0)
    bike_pct = (region_stats[bike_col] / total * 100).fillna(0)
    bottom_pct = transit_pct
    
    ax3.bar(region_stats[region_col], transit_pct, 
            label='Transit %', color='#3498db', alpha=0.85, edgecolor='white', linewidth=1)
    ax3.bar(region_stats[region_col], bike_pct, 
            bottom=bottom_pct, label='Bikes %', color='#e67e22', alpha=0.85, edgecolor='white', linewidth=1)
    if both_col:
        both_pct = (region_stats[both_col] / total * 100).fillna(0)
        ax3.bar(region_stats[region_col], both_pct, 
                bottom=bottom_pct + bike_pct, label='Both %', color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax3.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Percentage Distribution: Transit vs Bikes by Region', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Chart 4: Side-by-side Comparison (Horizontal)
    ax4 = axes[1, 1]
    y_pos = np.arange(len(region_stats))
    
    if both_col:
        bar_width = 0.25
        offset = bar_width
    else:
        bar_width = 0.35
        offset = bar_width
    
    ax4.barh(y_pos - offset, region_stats[transit_col], bar_width, 
             label='Transit', color='#3498db', alpha=0.85, edgecolor='white', linewidth=1)
    ax4.barh(y_pos, region_stats[bike_col], bar_width, 
             label='Bikes', color='#e67e22', alpha=0.85, edgecolor='white', linewidth=1)
    if both_col:
        ax4.barh(y_pos + offset, region_stats[both_col], bar_width, 
                 label='Both', color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=1)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(region_stats[region_col])
    ax4.set_xlabel('Count/Value', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Region', fontsize=11, fontweight='bold')
    ax4.set_title('Horizontal Bar Chart: Transit vs Bikes by Region', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved as '{output_file}'")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics by Region:")
    print("="*60)
    summary = region_stats.copy()
    summary['Total'] = summary[transit_col] + summary[bike_col]
    if both_col:
        summary['Total'] = summary['Total'] + summary[both_col]
    summary['Transit %'] = (summary[transit_col] / summary['Total'] * 100).round(1)
    summary['Bikes %'] = (summary[bike_col] / summary['Total'] * 100).round(1)
    if both_col:
        summary['Both %'] = (summary[both_col] / summary['Total'] * 100).round(1)
    print(summary.to_string(index=False))
    
    return region_stats

def main():
    """Main function."""
    excel_file = 'boston_regions_transit_bikes_estimates.xlsx'
    
    print("="*60)
    print("Transit vs Bikes Distribution by Region")
    print("="*60)
    
    # Load data
    df = load_excel_data(excel_file)
    
    # Identify columns
    region_col, transit_col, bike_col, both_col = identify_columns(df)
    
    # If columns not found automatically, try to use first few columns
    if not region_col:
        print("\n⚠️  Could not auto-detect region column. Please check column names:")
        print(f"   Available columns: {list(df.columns)}")
        # Try using first column as region if it looks categorical
        if df.iloc[:, 0].dtype == 'object' or len(df.iloc[:, 0].unique()) < 20:
            region_col = df.columns[0]
            print(f"   Using first column as region: {region_col}")
    
    if not transit_col:
        print("\n⚠️  Could not auto-detect transit column. Please check column names.")
        # Try to find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 1:
            transit_col = numeric_cols[0]
            print(f"   Using first numeric column as transit: {transit_col}")
    
    if not bike_col:
        print("\n⚠️  Could not auto-detect bike column. Please check column names.")
        # Try to find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            bike_col = numeric_cols[1]
            print(f"   Using second numeric column as bike: {bike_col}")
    
    # Create chart
    if all([region_col, transit_col, bike_col]):
        create_transit_bikes_chart(df, region_col, transit_col, bike_col, both_col)
    else:
        print("\n❌ Cannot create chart: Missing required columns")
        print("\nPlease manually specify columns:")
        print(f"   Region column: {region_col}")
        print(f"   Transit column: {transit_col}")
        print(f"   Bike column: {bike_col}")
        print(f"   Both column: {both_col}")

if __name__ == '__main__':
    main()

