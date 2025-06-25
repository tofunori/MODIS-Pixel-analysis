#!/usr/bin/env python3
"""
Create interactive map showing MODIS pixel centers and IDs
"""

import duckdb
import pandas as pd
import folium
from folium import plugins
import os

def extract_pixel_data():
    """Extract unique pixel data from database"""
    print("Extracting pixel data from database...")
    
    conn = duckdb.connect('/home/tofunori/duckdb-data/modis_analysis.db', read_only=True)
    
    # Get unique pixels with their coordinates and metadata
    query = """
    SELECT DISTINCT 
        pixel_id,
        latitude,
        longitude,
        pixel_col,
        pixel_row,
        COUNT(*) as observation_count
    FROM modis_data 
    GROUP BY pixel_id, latitude, longitude, pixel_col, pixel_row
    ORDER BY pixel_row, pixel_col
    """
    
    df = conn.execute(query).df()
    conn.close()
    
    print(f"Extracted {len(df)} unique pixels")
    return df

def create_pixel_map(df):
    """Create interactive map with pixel locations"""
    print("Creating interactive map...")
    
    # Calculate map center
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add satellite imagery as an option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Color pixels by row for better visualization
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    unique_rows = sorted(df['pixel_row'].unique())
    row_colors = {row: colors[i % len(colors)] for i, row in enumerate(unique_rows)}
    
    # Add pixel markers
    for idx, row in df.iterrows():
        # Create popup with pixel information
        popup_text = f"""
        <b>Pixel ID:</b> {row['pixel_id']}<br>
        <b>Coordinates:</b> {row['latitude']:.6f}, {row['longitude']:.6f}<br>
        <b>Grid Position:</b> Row {row['pixel_row']}, Col {row['pixel_col']}<br>
        <b>Observations:</b> {row['observation_count']}
        """
        
        # Add marker
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(popup_text, max_width=250),
            color='black',
            weight=1,
            fillColor=row_colors[row['pixel_row']],
            fillOpacity=0.7,
            tooltip=f"{row['pixel_id']}"
        ).add_to(m)
        
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>MODIS Pixels</h4>
    <p><b>Total Pixels:</b> {}</p>
    <p><b>Colors:</b> By pixel row</p>
    <p><b>Hover for pixel ID</b></p>
    <p><b>Click markers for details</b></p>
    </div>
    '''.format(len(df))
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen plugin
    plugins.Fullscreen().add_to(m)
    
    return m

def main():
    """Main function"""
    # Extract data
    df = extract_pixel_data()
    
    # Create map
    map_obj = create_pixel_map(df)
    
    # Save map
    output_path = '/home/tofunori/Projects/MODIS Pixel analysis/results/modis_pixel_map.html'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    map_obj.save(output_path)
    print(f"Interactive map saved to: {output_path}")
    
    # Also save pixel data as CSV for reference
    csv_path = '/home/tofunori/Projects/MODIS Pixel analysis/data/processed/pixel_locations.csv'
    df.to_csv(csv_path, index=False)
    print(f"Pixel data saved to: {csv_path}")
    
    print("\nMap creation complete!")
    print(f"- Open {output_path} in a web browser to view the interactive map")
    print(f"- Pixel data exported to {csv_path}")

if __name__ == "__main__":
    main()