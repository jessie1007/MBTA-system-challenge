#!/usr/bin/env python3
"""
Data Analysis - Excel Files

This script reads and analyzes data from two Excel files:
- boston_regions_transit_bikes_estimates.xlsx
- boston_mode_costs_by_region_estimates.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

# Set plotting style (try different style names if one doesn't work)
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
        print("Using default matplotlib style")

sns.set_palette("husl")

def main():
    """Main function to read and analyze Excel files."""
    
    # Excel file paths (found in current directory)
    excel_file1 = 'boston_regions_transit_bikes_estimates.xlsx'
    excel_file2 = 'boston_mode_costs_by_region_estimates.xlsx'
    
    # Read first Excel file
    if os.path.exists(excel_file1):
        try:
            # Read first sheet (change sheet_name if needed, or use None to read all sheets)
            df1 = pd.read_excel(excel_file1, sheet_name=0)
            
            print(f"✅ Loaded {excel_file1}")
            print(f"   Shape: {df1.shape} (rows, columns)")
            print(f"   Columns: {list(df1.columns)}")
            print(f"\nFirst few rows:")
            print(df1.head())
            
            print(f"\nData types:")
            print(df1.dtypes)
            print(f"\nBasic statistics:")
            print(df1.describe())
            
            print(f"\nMissing values:")
            print(df1.isnull().sum())
            
        except Exception as e:
            print(f"❌ Error loading {excel_file1}: {e}")
            import traceback
            traceback.print_exc()
            df1 = None
    else:
        print(f"❌ File not found: {excel_file1}")
        df1 = None
    
    # Read second Excel file
    if os.path.exists(excel_file2):
        try:
            df2 = pd.read_excel(excel_file2, sheet_name=0)
            
            print(f"\n{'='*60}")
            print(f"✅ Loaded {excel_file2}")
            print(f"   Shape: {df2.shape} (rows, columns)")
            print(f"   Columns: {list(df2.columns)}")
            print(f"\nFirst few rows:")
            print(df2.head())
            
            print(f"\nData types:")
            print(df2.dtypes)
            print(f"\nBasic statistics:")
            print(df2.describe())
            
            print(f"\nMissing values:")
            print(df2.isnull().sum())
            
        except Exception as e:
            print(f"❌ Error loading {excel_file2}: {e}")
            import traceback
            traceback.print_exc()
            df2 = None
    else:
        print(f"\n❌ File not found: {excel_file2}")
        df2 = None
    
    # Return dataframes for further analysis
    return df1, df2


def create_age_distribution_chart(df1, output_file='age_distribution_by_region.png'):
    """Create chart showing age distribution % for each region."""
    
    if df1 is None:
        print("Error: No data available for chart")
        return
    
    # Calculate age distribution percentages by region
    age_cols = ['Age_0_19', 'Age_20_44', 'Age_45_64', 'Age_65_plus']
    age_labels = ['0-19', '20-44', '45-64', '65+']
    
    # Prepare data
    age_data = []
    for _, row in df1.iterrows():
        region = row['Region']
        total_pop = row['Total_Population']
        
        for age_col, age_label in zip(age_cols, age_labels):
            age_pop = row[age_col]
            age_pct = (age_pop / total_pop * 100) if total_pop > 0 else 0
            age_data.append({
                'Region': region,
                'Age Group': age_label,
                'Percentage': age_pct,
                'Population': age_pop
            })
    
    age_df = pd.DataFrame(age_data)
    age_pivot = age_df.pivot(index='Region', columns='Age Group', values='Percentage')
    age_pivot = age_pivot.reindex(columns=['0-19', '20-44', '45-64', '65+'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Age Distribution by Region', fontsize=16, fontweight='bold', y=1.02)
    
    # Chart 1: Stacked Bar Chart
    ax1 = axes[0]
    age_pivot.plot(kind='bar', stacked=True, ax=ax1,
                   color=['#3498db', '#5dade2', '#85c1e2', '#aed6f1'],
                   alpha=0.85, edgecolor='white', linewidth=1)
    ax1.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax1.set_ylabel('% of Total Population', fontsize=11, fontweight='bold')
    ax1.set_title('Age Distribution by Region (Stacked)', fontsize=12, fontweight='bold')
    ax1.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels on stacked bars
    for container in ax1.containers:
        labels = [f'{v:.1f}%' if v > 3 else '' for v in container.datavalues]
        ax1.bar_label(container, labels=labels, label_type='center', fontsize=8)
    
    # Chart 2: Grouped Bar Chart
    ax2 = axes[1]
    age_pivot.plot(kind='bar', ax=ax2,
                   color=['#3498db', '#5dade2', '#85c1e2', '#aed6f1'],
                   alpha=0.85, edgecolor='white', linewidth=1)
    ax2.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax2.set_ylabel('% of Total Population', fontsize=11, fontweight='bold')
    ax2.set_title('Age Distribution by Region (Grouped)', fontsize=12, fontweight='bold')
    ax2.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved as '{output_file}'")
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("Age Distribution Summary by Region:")
    print("="*60)
    print(age_pivot.to_string())
    
    return age_df


def create_mode_by_region_chart(df1, output_file='mode_by_region.png'):
    """Create chart showing mode usage by region."""
    
    if df1 is None:
        print("Error: No data available for chart")
        return
    
    # Prepare mode data
    mode_cols = ['Users_BlueBikes_only', 'Users_LocalTransit_only', 'Users_Both_BlueBikes_and_LocalTransit']
    mode_labels = ['BlueBikes Only', 'Local Transit Only', 'Both']
    
    mode_data = []
    for _, row in df1.iterrows():
        region = row['Region']
        total_pop = row['Total_Population']
        
        for mode_col, mode_label in zip(mode_cols, mode_labels):
            mode_users = row[mode_col]
            mode_pct = (mode_users / total_pop * 100) if total_pop > 0 else 0
            mode_data.append({
                'Region': region,
                'Mode': mode_label,
                'Percentage': mode_pct,
                'Users': mode_users
            })
    
    mode_df = pd.DataFrame(mode_data)
    mode_pivot = mode_df.pivot(index='Region', columns='Mode', values='Percentage')
    mode_pivot = mode_pivot.reindex(columns=['BlueBikes Only', 'Local Transit Only', 'Both'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Mode Usage by Region', fontsize=16, fontweight='bold', y=1.02)
    
    # Chart 1: Stacked Bar Chart
    ax1 = axes[0]
    mode_pivot.plot(kind='bar', stacked=True, ax=ax1,
                    color=['#3498db', '#e67e22', '#2ecc71'],
                    alpha=0.85, edgecolor='white', linewidth=1)
    ax1.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax1.set_ylabel('% of Total Population', fontsize=11, fontweight='bold')
    ax1.set_title('Mode Usage by Region (Stacked)', fontsize=12, fontweight='bold')
    ax1.legend(title='Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for container in ax1.containers:
        labels = [f'{v:.1f}%' if v > 2 else '' for v in container.datavalues]
        ax1.bar_label(container, labels=labels, label_type='center', fontsize=8)
    
    # Chart 2: Grouped Bar Chart
    ax2 = axes[1]
    mode_pivot.plot(kind='bar', ax=ax2,
                    color=['#3498db', '#e67e22', '#2ecc71'],
                    alpha=0.85, edgecolor='white', linewidth=1)
    ax2.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax2.set_ylabel('% of Total Population', fontsize=11, fontweight='bold')
    ax2.set_title('Mode Usage by Region (Grouped)', fontsize=12, fontweight='bold')
    ax2.legend(title='Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved as '{output_file}'")
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("Mode Usage Summary by Region:")
    print("="*60)
    print(mode_pivot.to_string())
    
    return mode_df


if __name__ == '__main__':
    print("="*60)
    print("Excel File Analysis")
    print("="*60)
    df1, df2 = main()
    
    # Create charts
    if df1 is not None:
        print("\n" + "="*60)
        print("Creating Age Distribution Chart")
        print("="*60)
        age_df = create_age_distribution_chart(df1)
        
        print("\n" + "="*60)
        print("Creating Mode by Region Chart")
        print("="*60)
        mode_df = create_mode_by_region_chart(df1)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nDataFrames available:")
    print(f"  df1: {type(df1)}")
    print(f"  df2: {type(df2)}")
    
    if df1 is not None:
        print(f"\ndf1 shape: {df1.shape}")
    if df2 is not None:
        print(f"df2 shape: {df2.shape}")

