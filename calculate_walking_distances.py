#!/usr/bin/env python3
"""
Calculate actual walking distances using OpenStreetMap street network.

This module uses OSMnx to download street network data and calculate
shortest path distances along actual roads, rather than straight-line distances.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("⚠️  Warning: osmnx not installed. Install with: pip install osmnx")
    print("   Falling back to straight-line distance calculation.")

# Massachusetts State Plane CRS (meters) for accurate distance calculations
MA_CRS = 'EPSG:26986'

def download_street_network(bbox=None, place_name="Boston, Massachusetts, USA", 
                            network_type='walk', cache_file=None):
    """
    Download street network from OpenStreetMap.
    
    Parameters:
    -----------
    bbox : tuple, optional
        (north, south, east, west) bounding box in lat/lon
    place_name : str
        Place name for OSMnx to download (default: Boston)
    network_type : str
        'walk' for pedestrian network, 'drive' for driving network
    cache_file : str, optional
        Path to cache file to save/load network
        
    Returns:
    --------
    networkx.MultiDiGraph
        Street network graph
    """
    if not OSMNX_AVAILABLE:
        raise ImportError("osmnx is required for network-based distance calculation")
    
    # Try to load from cache first
    if cache_file and Path(cache_file).exists():
        print(f"Loading street network from cache: {cache_file}")
        try:
            G = ox.load_graphml(cache_file)
            print(f"✅ Loaded network with {len(G.nodes)} nodes and {len(G.edges)} edges")
            return G
        except Exception as e:
            print(f"Error loading cache: {e}. Will download fresh network.")
    
    print("="*70)
    print("Downloading Street Network from OpenStreetMap")
    print("="*70)
    print(f"Place: {place_name}")
    print(f"Network type: {network_type}")
    
    if bbox:
        print(f"Bounding box: {bbox}")
        print("Downloading network for bounding box...")
        G = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], 
                               network_type=network_type, simplify=True)
    else:
        print("Downloading network for place name...")
        G = ox.graph_from_place(place_name, network_type=network_type, simplify=True)
    
    print(f"✅ Downloaded network: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Project to a metric CRS for accurate distance calculations
    print("Projecting network to Massachusetts State Plane (EPSG:26986)...")
    G_projected = ox.project_graph(G, to_crs=MA_CRS)
    print("✅ Network projected")
    
    # Save to cache if requested
    if cache_file:
        print(f"Saving network to cache: {cache_file}")
        try:
            ox.save_graphml(G_projected, cache_file)
            print("✅ Network cached")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    return G_projected

def get_nearest_node(G, point, method='euclidean'):
    """
    Get nearest node in network to a given point.
    
    Parameters:
    -----------
    G : networkx.MultiDiGraph
        Street network graph (should be projected to metric CRS)
    point : shapely.geometry.Point or tuple
        Point (x, y) in projected CRS or Point geometry
    method : str
        'euclidean' for projected distance (default, recommended for projected graphs)
        'haversine' for great-circle distance (only for lat/lon coordinates)
        
    Returns:
    --------
    int
        Node ID in graph
    """
    if isinstance(point, Point):
        x, y = point.x, point.y
    else:
        x, y = point
    
    # Get node coordinates from graph
    nodes = list(G.nodes(data=True))
    
    # Check if graph is projected (has 'x' and 'y' in projected coordinates)
    # or in lat/lon (has 'lat' and 'lon' or 'y' and 'x' as lat/lon)
    sample_node = nodes[0][1]
    
    if method == 'haversine' and 'lat' in sample_node and 'lon' in sample_node:
        # Use haversine distance (great circle) - only for lat/lon graphs
        from geopy.distance import geodesic
        point_coords = (y, x)  # Assuming point is (lon, lat)
        distances = [
            geodesic(point_coords, (G.nodes[node]['lat'], G.nodes[node]['lon'])).meters
            for node, _ in nodes
        ]
    else:
        # Use Euclidean distance in projected CRS (default)
        # Graph nodes have 'x' and 'y' in projected coordinates (meters)
        distances = [
            np.sqrt((G.nodes[node]['x'] - x)**2 + (G.nodes[node]['y'] - y)**2)
            for node, _ in nodes
        ]
    
    nearest_idx = np.argmin(distances)
    nearest_node = nodes[nearest_idx][0]
    
    return nearest_node

def calculate_walking_distance(G, origin_point, dest_point, max_distance_meters=50000):
    """
    Calculate shortest walking distance between two points along street network.
    
    Parameters:
    -----------
    G : networkx.MultiDiGraph
        Street network graph (projected to metric CRS)
    origin_point : shapely.geometry.Point or tuple
        Origin point (x, y) in projected CRS or Point geometry
    dest_point : shapely.geometry.Point or tuple
        Destination point (x, y) in projected CRS or Point geometry
    max_distance_meters : float
        Maximum distance to search (for performance)
        
    Returns:
    --------
    float
        Distance in meters, or np.inf if no path found
    """
    try:
        # Get nearest nodes (use Euclidean for projected graphs)
        origin_node = get_nearest_node(G, origin_point, method='euclidean')
        dest_node = get_nearest_node(G, dest_point, method='euclidean')
        
        # If same node, return 0
        if origin_node == dest_node:
            return 0.0
        
        # Calculate shortest path
        try:
            path = nx.shortest_path(G, origin_node, dest_node, weight='length')
            # Calculate total path length
            total_length = 0
            for i in range(len(path) - 1):
                edge_data = G[path[i]][path[i+1]]
                # Get length from first edge (MultiDiGraph may have multiple edges)
                if isinstance(edge_data, dict):
                    length = edge_data.get('length', 0)
                else:
                    # Multiple edges between nodes
                    length = min([e.get('length', 0) for e in edge_data.values()])
                total_length += length
            
            return total_length
        except nx.NetworkXNoPath:
            return np.inf
    except Exception as e:
        print(f"Warning: Error calculating path: {e}")
        return np.inf

def calculate_walking_distances_to_stations(block_groups_gdf, train_stations_gdf, 
                                           bbox=None, place_name="Boston, Massachusetts, USA",
                                           cache_file='data/boston_street_network.graphml',
                                           max_distance_meters=50000):
    """
    Calculate walking distances from block group centroids to nearest train stations
    using actual street network.
    
    Parameters:
    -----------
    block_groups_gdf : gpd.GeoDataFrame
        Block groups with geometry
    train_stations_gdf : gpd.GeoDataFrame
        Train stations with geometry
    bbox : tuple, optional
        (north, south, east, west) bounding box
    place_name : str
        Place name for OSMnx
    cache_file : str
        Path to cache file for street network
    max_distance_meters : float
        Maximum distance to calculate (for performance)
        
    Returns:
    --------
    gpd.GeoDataFrame
        Block groups with walking distance to nearest station
    """
    if not OSMNX_AVAILABLE:
        print("⚠️  osmnx not available. Using straight-line distance instead.")
        return calculate_straight_line_distances(block_groups_gdf, train_stations_gdf)
    
    print("="*70)
    print("Calculating Walking Distances Using Street Network")
    print("="*70)
    
    # Download/load street network
    G = download_street_network(bbox=bbox, place_name=place_name, 
                               network_type='walk', cache_file=cache_file)
    
    # Convert block groups to projected CRS
    print("\nConverting block groups to projected CRS...")
    block_groups_proj = block_groups_gdf.to_crs(MA_CRS)
    block_groups_proj['centroid'] = block_groups_proj.geometry.centroid
    print("✅ Converted block groups")
    
    # Convert stations to projected CRS
    print("Converting train stations to projected CRS...")
    train_stations_proj = train_stations_gdf.to_crs(MA_CRS)
    print("✅ Converted train stations")
    
    # Calculate distances
    print(f"\nCalculating walking distances for {len(block_groups_proj)} block groups...")
    print("This may take a while...")
    
    walking_distances_meters = []
    nearest_station_ids = []
    nearest_station_names = []
    
    # First, find nearest station by straight-line distance (for efficiency)
    # Then calculate walking distance only to nearby stations
    from scipy.spatial import cKDTree
    
    # Create spatial index for stations
    station_points = np.array([[s.geometry.x, s.geometry.y] for _, s in train_stations_proj.iterrows()])
    station_tree = cKDTree(station_points)
    
    for idx, row in block_groups_proj.iterrows():
        centroid = row['centroid']
        centroid_coords = (centroid.x, centroid.y)
        
        # Find nearest 5 stations by straight-line distance
        distances_straight, indices = station_tree.query([centroid_coords], k=min(5, len(station_points)))
        
        # Calculate walking distance to each of these nearby stations
        min_walking_dist = np.inf
        nearest_station_idx = None
        
        for i, station_idx in enumerate(indices[0]):
            station = train_stations_proj.iloc[station_idx]
            station_point = station.geometry
            
            # Skip if straight-line distance is already too far
            straight_dist = centroid.distance(station_point)
            if straight_dist > max_distance_meters:
                continue
            
            # Calculate walking distance
            # Both points are in projected CRS, so use Euclidean method
            walking_dist = calculate_walking_distance(G, centroid, station_point, max_distance_meters)
            
            if walking_dist < min_walking_dist:
                min_walking_dist = walking_dist
                nearest_station_idx = station_idx
        
        if nearest_station_idx is not None:
            nearest_station = train_stations_proj.iloc[nearest_station_idx]
            walking_distances_meters.append(min_walking_dist)
            nearest_station_ids.append(nearest_station['stop_id'])
            nearest_station_names.append(nearest_station.get('stop_name', 'Unknown'))
        else:
            # Fallback: use straight-line distance if no path found
            straight_dists = [centroid.distance(s.geometry) for _, s in train_stations_proj.iterrows()]
            min_idx = np.argmin(straight_dists)
            walking_distances_meters.append(straight_dists[min_idx])
            nearest_station = train_stations_proj.iloc[min_idx]
            nearest_station_ids.append(nearest_station['stop_id'])
            nearest_station_names.append(nearest_station.get('stop_name', 'Unknown'))
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(block_groups_proj)} block groups...")
    
    # Convert to miles
    walking_distances_miles = [d / 1609.34 for d in walking_distances_meters]
    
    block_groups_proj['walking_distance_meters'] = walking_distances_meters
    block_groups_proj['walking_distance_miles'] = walking_distances_miles
    block_groups_proj['distance_to_nearest_station_miles'] = walking_distances_miles  # For compatibility
    block_groups_proj['nearest_station_id'] = nearest_station_ids
    block_groups_proj['nearest_station_name'] = nearest_station_names
    
    valid_distances = [d for d in walking_distances_miles if not np.isinf(d) and not pd.isna(d)]
    if valid_distances:
        print(f"\n✅ Walking distance calculation complete!")
        print(f"   Range: {min(valid_distances):.2f} - {max(valid_distances):.2f} miles")
        print(f"   {len(valid_distances)}/{len(block_groups_proj)} block groups have valid distances")
        print(f"   Average: {np.mean(valid_distances):.2f} miles")
    
    # Convert back to WGS84
    block_groups_wgs84 = block_groups_proj.to_crs('EPSG:4326')
    
    return block_groups_wgs84

def calculate_straight_line_distances(block_groups_gdf, train_stations_gdf):
    """Fallback: Calculate straight-line distances if OSMnx is not available."""
    print("Using straight-line distance calculation (fallback)")
    
    block_groups_proj = block_groups_gdf.to_crs(MA_CRS)
    train_stations_proj = train_stations_gdf.to_crs(MA_CRS)
    block_groups_proj['centroid'] = block_groups_proj.geometry.centroid
    
    distances_meters = []
    nearest_station_ids = []
    
    for idx, row in block_groups_proj.iterrows():
        centroid = row['centroid']
        dists = [centroid.distance(station.geometry) for _, station in train_stations_proj.iterrows()]
        min_idx = np.argmin(dists)
        distances_meters.append(dists[min_idx])
        nearest_station_ids.append(train_stations_proj.iloc[min_idx]['stop_id'])
    
    distances_miles = [d / 1609.34 for d in distances_meters]
    
    block_groups_proj['walking_distance_meters'] = distances_meters
    block_groups_proj['walking_distance_miles'] = distances_miles
    block_groups_proj['distance_to_nearest_station_miles'] = distances_miles
    block_groups_proj['nearest_station_id'] = nearest_station_ids
    
    return block_groups_proj.to_crs('EPSG:4326')

if __name__ == '__main__':
    # Test the module
    print("Walking Distance Calculation Module")
    print("="*70)
    if OSMNX_AVAILABLE:
        print("✅ osmnx is available")
    else:
        print("❌ osmnx is not available. Install with: pip install osmnx")

