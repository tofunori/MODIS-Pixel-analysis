#!/usr/bin/env python3
"""
Create spatial coverage heterogeneity map for MODIS pixels
Shows temporal data availability patterns across the study area
"""

import duckdb
import pandas as pd
import folium
from folium import plugins
import numpy as np
import os
from datetime import datetime

def extract_coverage_data():
    """Extract pixel coverage statistics from database"""
    print("Extracting coverage statistics from database...")
    
    conn = duckdb.connect('/home/tofunori/duckdb-data/modis_analysis.db', read_only=True)
    
    # Get comprehensive coverage statistics for each pixel location
    coverage_query = """
    SELECT 
        pixel_id,
        AVG(latitude) as latitude,
        AVG(longitude) as longitude,
        COUNT(*) as total_observations,
        COUNT(DISTINCT date) as observation_days,
        COUNT(DISTINCT year) as years_covered,
        MIN(date) as first_observation,
        MAX(date) as last_observation,
        COUNT(DISTINCT SUBSTR(date, 1, 7)) as months_covered,
        -- Summer coverage (Jun-Aug)
        COUNT(CASE WHEN CAST(SUBSTR(date, 6, 2) AS INTEGER) BETWEEN 6 AND 8 THEN 1 END) as summer_observations,
        -- Data density (obs per year)
        COUNT(*) * 1.0 / COUNT(DISTINCT year) as obs_per_year,
        -- Temporal span in days
        DATEDIFF('day', MIN(CAST(date AS DATE)), MAX(CAST(date AS DATE))) as temporal_span_days
    FROM modis_data 
    GROUP BY pixel_id
    ORDER BY total_observations DESC
    """
    
    df = conn.execute(coverage_query).df()
    conn.close()
    
    print(f"Extracted coverage data for {len(df)} pixels")
    print(f"Observation range: {df['total_observations'].min()} - {df['total_observations'].max()}")
    
    return df

def create_coverage_heatmap(df):
    """Create interactive coverage heatmap"""
    print("Creating spatial coverage heatmap...")
    
    # Calculate map center
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add satellite imagery
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Normalize coverage values for color mapping
    min_obs = df['total_observations'].min()
    max_obs = df['total_observations'].max()
    
    def get_color(obs_count):
        """Get color based on observation count (red=low, green=high)"""
        normalized = (obs_count - min_obs) / (max_obs - min_obs)
        
        if normalized < 0.2:
            return '#d73027'  # Dark red (very low)
        elif normalized < 0.4:
            return '#f46d43'  # Red-orange (low)
        elif normalized < 0.6:
            return '#fdae61'  # Orange (medium-low)
        elif normalized < 0.8:
            return '#fee08b'  # Yellow (medium-high)
        else:
            return '#66bd63'  # Green (high)
    
    # Add pixels as circle markers
    for idx, row in df.iterrows():
        # Calculate coverage percentage relative to maximum
        coverage_pct = (row['total_observations'] / max_obs) * 100
        
        # Create detailed popup
        popup_html = f"""
        <div style="width: 280px;">
            <h4>Pixel {row['pixel_id']}</h4>
            <b>Location:</b> {row['latitude']:.6f}, {row['longitude']:.6f}<br>
            <b>Total Observations:</b> {row['total_observations']}<br>
            <b>Observation Days:</b> {row['observation_days']}<br>
            <b>Coverage:</b> {coverage_pct:.1f}% of maximum<br>
            <b>Years Covered:</b> {row['years_covered']}<br>
            <b>Months Covered:</b> {row['months_covered']}<br>
            <b>Summer Observations:</b> {row['summer_observations']}<br>
            <b>Observations/Year:</b> {row['obs_per_year']:.1f}<br>
            <b>First Observation:</b> {row['first_observation']}<br>
            <b>Last Observation:</b> {row['last_observation']}<br>
            <b>Temporal Span:</b> {row['temporal_span_days']:.0f} days
        </div>
        """
        
        # Add circle marker
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=12,
            popup=folium.Popup(popup_html, max_width=300),
            color='black',
            weight=1,
            fillColor=get_color(row['total_observations']),
            fillOpacity=0.8,
            tooltip=f"Pixel {row['pixel_id']}: {row['total_observations']} observations"
        ).add_to(m)
    
    return m

def add_coverage_legend(m, df):
    """Add coverage legend to map"""
    min_obs = df['total_observations'].min()
    max_obs = df['total_observations'].max()
    
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <h4>Temporal Coverage</h4>
    <p><b>Total Pixels:</b> {len(df)}</p>
    <p><b>Observation Range:</b><br>{min_obs} - {max_obs}</p>
    
    <div style="margin-top: 10px;">
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 15px; height: 15px; background-color: #d73027; margin-right: 5px;"></div>
            <span>Very Low (&lt;20%)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 15px; height: 15px; background-color: #f46d43; margin-right: 5px;"></div>
            <span>Low (20-40%)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 15px; height: 15px; background-color: #fdae61; margin-right: 5px;"></div>
            <span>Medium (40-60%)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 15px; height: 15px; background-color: #fee08b; margin-right: 5px;"></div>
            <span>High (60-80%)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 15px; height: 15px; background-color: #66bd63; margin-right: 5px;"></div>
            <span>Very High (&gt;80%)</span>
        </div>
    </div>
    
    <p style="margin-top: 10px; font-style: italic;">
        Click pixels for detailed statistics
    </p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_seasonal_coverage_layer(df):
    """Create layer showing seasonal coverage patterns"""
    feature_group = folium.FeatureGroup(name="Summer Coverage %")
    
    for idx, row in df.iterrows():
        # Calculate summer coverage percentage
        summer_pct = (row['summer_observations'] / row['total_observations']) * 100
        
        # Color based on summer percentage
        if summer_pct < 20:
            color = '#2166ac'  # Blue (low summer coverage)
        elif summer_pct < 40:
            color = '#67a9cf'
        elif summer_pct < 60:
            color = '#d1e5f0'
        elif summer_pct < 80:
            color = '#fddbc7'
        else:
            color = '#d6604d'  # Red (high summer coverage)
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=10,
            popup=f"Summer coverage: {summer_pct:.1f}%",
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.7,
            tooltip=f"Summer: {summer_pct:.1f}%"
        ).add_to(feature_group)
    
    return feature_group

def main():
    """Main function"""
    # Extract coverage data
    df = extract_coverage_data()
    
    # Create main coverage map
    coverage_map = create_coverage_heatmap(df)
    
    # Add legend
    coverage_map = add_coverage_legend(coverage_map, df)
    
    # Add seasonal coverage layer
    seasonal_layer = create_seasonal_coverage_layer(df)
    seasonal_layer.add_to(coverage_map)
    
    # Add layer control
    folium.LayerControl().add_to(coverage_map)
    
    # Add fullscreen plugin
    plugins.Fullscreen().add_to(coverage_map)
    
    # Save map
    output_path = '/home/tofunori/Projects/MODIS Pixel analysis/results/spatial_coverage_map.html'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    coverage_map.save(output_path)
    print(f"Spatial coverage map saved to: {output_path}")
    
    # Save coverage statistics as CSV
    csv_path = '/home/tofunori/Projects/MODIS Pixel analysis/data/processed/pixel_coverage_stats.csv'
    df.to_csv(csv_path, index=False)
    print(f"Coverage statistics saved to: {csv_path}")
    
    # Print summary statistics
    print(f"\n=== COVERAGE SUMMARY ===")
    print(f"Pixels analyzed: {len(df)}")
    print(f"Observation range: {df['total_observations'].min()} - {df['total_observations'].max()}")
    print(f"Average observations per pixel: {df['total_observations'].mean():.1f}")
    print(f"Standard deviation: {df['total_observations'].std():.1f}")
    print(f"Coverage coefficient of variation: {(df['total_observations'].std() / df['total_observations'].mean() * 100):.1f}%")
    
    print("\nSpatial coverage map creation complete!")
    print(f"- Open {output_path} in a web browser")
    print(f"- Toggle between coverage layers using the layer control")
    print(f"- Click pixels for detailed temporal statistics")

if __name__ == "__main__":
    main()