# MBTA System Challenge

## Project Overview

Analysis of MBTA system accessibility and equity across neighborhoods in the Greater Boston area.

## Geographic Scope

This project focuses on three Massachusetts counties:
- **Suffolk County** (FIPS: 25025) - Includes Boston
- **Middlesex County** (FIPS: 25017) - Includes Cambridge, Somerville
- **Norfolk County** (FIPS: 25021) - Includes Quincy, Brookline

## Problem Statement

[Describe the challenge/problem you're solving]

## Data Sources

- **ACS Income Data (B19013)**: Median Household Income from American Community Survey
  - Source: [Census Bureau](https://data.census.gov/cedsci/)
- **ACS Population Data (B01003)**: Total Population (optional)
  - Source: [Census Bureau](https://data.census.gov/cedsci/)
- **ACS Vehicle Availability (B08201)**: Households without vehicles (optional)
  - Source: [Census Bureau](https://data.census.gov/cedsci/)
- **Block Group Shapefiles**: Geographic boundaries
  - Source: [Census TIGER/Line Files](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
- **MBTA GTFS Data**: Transit stops, routes, and schedules
  - Source: [MBTA GTFS](https://www.mbta.com/developers/gtfs)
  - Files needed: `stops.txt`, `routes.txt` (optional)
- **Bluebikes Station Data**: Bike share station locations (optional)
  - Source: [Boston Open Data Portal](https://data.boston.gov/)

## Methodology

### Step 1: Merge ACS Income with Block-Group Shapefiles
- Merge ACS income data (B19013) with block-group shapefiles using GEOID
- Filter for Suffolk, Middlesex, and Norfolk counties
- Create neighborhood income map
- **Script**: `step1_merge_acs_income.py`
- **Output**: `data/neighborhood_income_merged.geojson`

### Step 2: Transit Station Accessibility Analysis
- Load MBTA GTFS stops.txt and filter for train stations (route_type: 0, 1, 2)
- Compute distance from each block group centroid to nearest train station
- Categorize into accessibility zones:
  - **walk-zone**: 0-0.5 miles
  - **bike-zone**: 0.5-2 miles
  - **far-zone**: 2+ miles
- Merge with income data from Step 1
- Add population (B01003) and vehicle availability (B08201) data
- Analyze: "X% of low-income vs Y% of high-income residents live in bike-but-not-walk zone"
- **Script**: `step2_transit_accessibility.py`
- **Output**: `data/transit_accessibility_analysis.geojson`

### Step 3: Equity Visualization
- Create income distribution heatmap (top 20% vs bottom 20% wealth)
- Create transit stop density map
- Create combined equity map showing income and transit access
- Generate bar charts showing accessibility by income level
- Analyze equity gap to support hypothesis: less wealthy people have less access
- **Script**: `step3_equity_visualization.py`
- **Output**: Multiple PNG visualization files

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/jessie1007/MBTA-system-challenge.git
cd MBTA-system-challenge

# Install dependencies
pip install -r requirements.txt
```

## Usage

[Add usage instructions]

## Results

[Add your results and findings]

## Technologies Used

- **Python** - Primary programming language
- **Pandas** - Data manipulation
- **GeoPandas** - Geospatial data analysis
- **Shapely** - Geometric operations
- **Geopy** - Distance calculations
- **Matplotlib & Seaborn** - Data visualization
- **Contextily** - Basemap tiles for maps

## License

[Add license information]

