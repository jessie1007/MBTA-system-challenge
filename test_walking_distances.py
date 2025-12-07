#!/usr/bin/env python3
"""
Test script for walking distance calculation.
"""

import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from calculate_walking_distances import calculate_walking_distances_to_stations

def test_walking_distances():
    """Test walking distance calculation with a small sample."""
    print("="*70)
    print("Testing Walking Distance Calculation")
    print("="*70)
    
    # Create a small test dataset - 5 block groups around Boston
    test_block_groups = gpd.GeoDataFrame({
        'GEOID': ['250250001001', '250250001002', '250250001003', '250250001004', '250250001005'],
        'geometry': [
            Point(-71.0589, 42.3601),  # Downtown Boston
            Point(-71.0950, 42.3736),  # Cambridge area
            Point(-71.0636, 42.3581),  # South Boston
            Point(-71.0776, 42.3396),  # Dorchester area
            Point(-71.1200, 42.3500),  # Somerville area
        ]
    }, crs='EPSG:4326')
    
    # Create test train stations
    test_stations = gpd.GeoDataFrame({
        'stop_id': ['70061', '70083', '70043'],
        'stop_name': ['Alewife', 'Andrew', 'Aquarium'],
        'geometry': [
            Point(-71.1424, 42.3954),  # Alewife
            Point(-71.0552, 42.3306),  # Andrew
            Point(-71.0527, 42.3594),  # Aquarium
        ]
    }, crs='EPSG:4326')
    
    print(f"\nTest data:")
    print(f"  Block groups: {len(test_block_groups)}")
    print(f"  Train stations: {len(test_stations)}")
    
    # Calculate walking distances
    print("\nCalculating walking distances...")
    try:
        result = calculate_walking_distances_to_stations(
            test_block_groups,
            test_stations,
            place_name="Boston, Massachusetts, USA",
            cache_file='data/boston_street_network.graphml',
            max_distance_meters=10000  # 10km max for test
        )
        
        print("\n" + "="*70)
        print("Results:")
        print("="*70)
        print(result[['GEOID', 'walking_distance_miles', 'nearest_station_name']].to_string(index=False))
        
        print("\n✅ Test completed successfully!")
        print(f"   Average walking distance: {result['walking_distance_miles'].mean():.2f} miles")
        print(f"   Range: {result['walking_distance_miles'].min():.2f} - {result['walking_distance_miles'].max():.2f} miles")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_walking_distances()
    if success:
        print("\n✅ Walking distance calculation is working!")
    else:
        print("\n❌ Walking distance calculation failed. Check errors above.")

