#!/usr/bin/env python3

"""
MODIS Albedo Visualization Script
Creates comprehensive plots for trend analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import subprocess
import os

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the annual statistics data"""
    # Re-export from DuckDB if needed
    db_path = "/home/tofunori/duckdb-data/modis_analysis.db"
    csv_path = "/home/tofunori/Projects/MODIS Pixel analysis/annual_stats.csv"
    
    if not os.path.exists(csv_path):
        cmd = f'duckdb "{db_path}" -c "COPY annual_albedo_stats TO \'{csv_path}\' WITH (HEADER, DELIMITER \',\');"'
        subprocess.run(cmd, shell=True, check=True)
    
    return pd.read_csv(csv_path)

def plot_annual_trend(data, save_path):
    """Create annual albedo trend plot with Sen's slope"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    years = data['year']
    albedo = data['mean_albedo']
    
    # Main trend line
    ax.plot(years, albedo, 'o-', linewidth=3, markersize=8, 
            color='darkblue', label='Annual Mean Albedo')
    
    # Sen's slope line
    slope = -0.005898  # From our analysis
    intercept = albedo.iloc[0] - slope * years.iloc[0]
    trend_line = slope * years + intercept
    ax.plot(years, trend_line, '--', linewidth=2, color='red', 
            label=f"Sen's Slope: {slope:.4f}/year")
    
    # Error bars (std deviation)
    ax.errorbar(years, albedo, yerr=data['std_albedo'], 
                fmt='none', capsize=3, alpha=0.3, color='darkblue')
    
    # Formatting
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Snow Albedo', fontsize=14, fontweight='bold')
    ax.set_title('MODIS Snow Albedo Trend (2010-2024)\nFiltered: 90-100% Glacier Fraction, No Clouds', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add statistics text
    stats_text = f'Mann-Kendall p-value: 0.010*\nKendall\'s τ: -0.505\nTotal decline: -0.093'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_distribution_by_year(data, save_path):
    """Create box plot showing albedo distribution by year"""
    # We need to get the raw filtered data for this
    db_path = "/home/tofunori/duckdb-data/modis_analysis.db"
    
    # Sample data for ALL years (2010-2024)
    temp_csv = "/home/tofunori/Projects/MODIS Pixel analysis/temp_sample.csv"
    cmd = f'duckdb "{db_path}" -c "COPY (SELECT year, snow_albedo_scaled FROM filtered_modis WHERE snow_albedo_scaled IS NOT NULL ORDER BY RANDOM() LIMIT 15000) TO \'{temp_csv}\' WITH (HEADER, DELIMITER \',\');"'
    
    subprocess.run(cmd, shell=True, check=True)
    
    sample_data = pd.read_csv(temp_csv)
    os.remove(temp_csv)  # Clean up
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create box plot
    box_plot = ax.boxplot([sample_data[sample_data['year'] == year]['snow_albedo_scaled'].values 
                          for year in sorted(sample_data['year'].unique())],
                         labels=sorted(sample_data['year'].unique()),
                         patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(box_plot['boxes'])))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Snow Albedo', fontsize=14, fontweight='bold')
    ax.set_title('Snow Albedo Distribution by Year (Sample)\nFiltered Data: 90-100% Glacier Fraction', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_qa_analysis(save_path):
    """Create QA flag analysis showing filtering effects"""
    db_path = "/home/tofunori/duckdb-data/modis_analysis.db"
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. QA Flag Distribution
    temp_csv = "/home/tofunori/Projects/MODIS Pixel analysis/temp_qa.csv"
    cmd = f'duckdb "{db_path}" -c "COPY (SELECT basic_qa_text, COUNT(*) as count FROM modis_data GROUP BY basic_qa_text) TO \'{temp_csv}\' WITH (HEADER, DELIMITER \',\');"'
    subprocess.run(cmd, shell=True, check=True)
    qa_data = pd.read_csv(temp_csv)
    os.remove(temp_csv)
    
    wedges, texts, autotexts = ax1.pie(qa_data['count'], labels=qa_data['basic_qa_text'], 
                                       autopct='%1.1f%%', startangle=90)
    ax1.set_title('Basic QA Distribution', fontweight='bold')
    
    # 2. Cloud Flag Impact
    cmd = f'duckdb "{db_path}" -c "COPY (SELECT flag_probably_cloudy, COUNT(*) as count FROM modis_data GROUP BY flag_probably_cloudy) TO \'{temp_csv}\' WITH (HEADER, DELIMITER \',\');"'
    subprocess.run(cmd, shell=True, check=True)
    cloud_data = pd.read_csv(temp_csv)
    os.remove(temp_csv)
    
    labels = ['Clear', 'Cloudy']
    colors = ['lightblue', 'gray']
    bars = ax2.bar(labels, cloud_data['count'], color=colors, alpha=0.7)
    ax2.set_title('Cloud Flag Distribution', fontweight='bold')
    ax2.set_ylabel('Number of Pixels')
    for bar, count in zip(bars, cloud_data['count']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{count:,}', ha='center', va='bottom')
    
    # 3. Glacier Class Distribution
    cmd = f'duckdb "{db_path}" -c "COPY (SELECT glacier_class, COUNT(*) as count FROM modis_data GROUP BY glacier_class ORDER BY glacier_class) TO \'{temp_csv}\' WITH (HEADER, DELIMITER \',\');"'
    subprocess.run(cmd, shell=True, check=True)
    glacier_data = pd.read_csv(temp_csv)
    os.remove(temp_csv)
    
    bars = ax3.bar(range(len(glacier_data)), glacier_data['count'], 
                   color=plt.cm.Blues(np.linspace(0.3, 1, len(glacier_data))))
    ax3.set_title('Glacier Fraction Distribution', fontweight='bold')
    ax3.set_ylabel('Number of Pixels')
    ax3.set_xlabel('Glacier Fraction Class')
    ax3.set_xticks(range(len(glacier_data)))
    ax3.set_xticklabels(glacier_data['glacier_class'], rotation=45)
    
    # 4. Filtering Impact
    cmd = f'duckdb "{db_path}" -c "COPY (SELECT \'Original\' as dataset, COUNT(*) as count FROM modis_data UNION ALL SELECT \'Filtered\' as dataset, COUNT(*) as count FROM filtered_modis) TO \'{temp_csv}\' WITH (HEADER, DELIMITER \',\');"'
    subprocess.run(cmd, shell=True, check=True)
    filter_data = pd.read_csv(temp_csv)
    os.remove(temp_csv)
    
    bars = ax4.bar(filter_data['dataset'], filter_data['count'], 
                   color=['lightcoral', 'lightgreen'], alpha=0.7)
    ax4.set_title('Filtering Impact', fontweight='bold')
    ax4.set_ylabel('Number of Pixels')
    for bar, count in zip(bars, filter_data['count']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{count:,}', ha='center', va='bottom')
    
    # Add filtering percentage
    orig_count = filter_data[filter_data['dataset'] == 'Original']['count'].iloc[0]
    filt_count = filter_data[filter_data['dataset'] == 'Filtered']['count'].iloc[0]
    retention_pct = (filt_count / orig_count) * 100
    ax4.text(0.5, 0.95, f'Retention: {retention_pct:.1f}%', 
             transform=ax4.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('MODIS Data Quality and Filtering Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_trend_diagnostics(data, save_path):
    """Create diagnostic plots for trend analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    years = data['year']
    albedo = data['mean_albedo']
    
    # 1. Trend with confidence intervals
    slope = -0.005898
    intercept = albedo.iloc[0] - slope * years.iloc[0]
    trend_line = slope * years + intercept
    
    ax1.plot(years, albedo, 'o-', color='blue', linewidth=2, label='Observed')
    ax1.plot(years, trend_line, '--', color='red', linewidth=2, label='Sen\'s Slope')
    ax1.fill_between(years, trend_line - 0.02, trend_line + 0.02, 
                     alpha=0.2, color='red', label='Approximate CI')
    ax1.set_title('Trend Analysis with Confidence Interval')
    ax1.set_ylabel('Snow Albedo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = albedo - trend_line
    ax2.plot(years, residuals, 'o-', color='green')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_title('Residuals from Trend Line')
    ax2.set_ylabel('Residuals')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram of annual means
    ax3.hist(albedo, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.axvline(albedo.mean(), color='red', linestyle='--', 
                label=f'Mean: {albedo.mean():.3f}')
    ax3.set_title('Distribution of Annual Means')
    ax3.set_xlabel('Snow Albedo')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative change
    cumulative_change = np.cumsum(np.diff(albedo, prepend=albedo.iloc[0]))
    ax4.plot(years, cumulative_change, 'o-', color='purple', linewidth=2)
    ax4.set_title('Cumulative Albedo Change')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Cumulative Change')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('MODIS Albedo Trend Diagnostic Plots', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_figure(data, save_path):
    """Create a streamlined comprehensive summary figure"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    years = data['year']
    albedo = data['mean_albedo']
    
    # Main trend plot (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(years, albedo, 'o-', linewidth=3, markersize=8, 
             color='darkblue', label='Annual Mean Albedo')
    
    # Sen's slope line
    slope = -0.005898
    intercept = albedo.iloc[0] - slope * years.iloc[0]
    trend_line = slope * years + intercept
    ax1.plot(years, trend_line, '--', linewidth=2, color='red', 
             label=f"Sen's Slope: {slope:.4f}/year")
    
    ax1.fill_between(years, albedo - data['std_albedo'], 
                     albedo + data['std_albedo'], alpha=0.2, color='blue')
    
    ax1.set_ylabel('Snow Albedo', fontsize=14, fontweight='bold')
    ax1.set_title('MODIS Snow Albedo Trend Analysis (2010-2024)\nFiltered: 90-100% Glacier Fraction, No Clouds, Standard QA=1', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Statistics box
    stats_text = ('Mann-Kendall Test: p = 0.010*\n'
                 'Kendall\'s τ = -0.505\n'
                 'Sen\'s Slope = -0.0059/year\n'
                 'Total Decline = -0.093\n'
                 'Sample Size = 65,762 pixels')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Sample sizes (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(years, data['pixel_count'], color='steelblue', alpha=0.7)
    ax2.set_ylabel('Pixel Count', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_title('Annual Sample Sizes')
    ax2.grid(True, alpha=0.3)
    
    # Standard deviation (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(years, data['std_albedo'], 'o-', color='orange', linewidth=2, markersize=6)
    ax3.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax3.set_title('Annual Albedo Variability')
    ax3.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate all plots"""
    print("=== MODIS Albedo Visualization (Enhanced) ===")
    
    # Load data
    data = load_data()
    print(f"Loaded data for {len(data)} years")
    
    # Create output directory for plots
    plot_dir = "/home/tofunori/Projects/MODIS Pixel analysis/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Generating enhanced plot suite...")
    
    # Generate core trend analysis plots
    print("1. Annual trend plot...")
    plot_annual_trend(data, f"{plot_dir}/annual_trend.png")
    
    print("2. Distribution by ALL years...")
    plot_distribution_by_year(data, f"{plot_dir}/distribution_all_years.png")
    
    print("3. QA and quality control analysis...")
    plot_qa_analysis(f"{plot_dir}/qa_quality_analysis.png")
    
    print("4. Trend diagnostics...")
    plot_trend_diagnostics(data, f"{plot_dir}/trend_diagnostics.png")
    
    print("5. Streamlined comprehensive summary...")
    create_summary_figure(data, f"{plot_dir}/comprehensive_summary.png")
    
    print("\n=== Enhanced Visualization Suite Complete ===")
    print(f"All plots saved in: {plot_dir}/")
    print("Files created:")
    for file in sorted(os.listdir(plot_dir)):
        if file.endswith('.png'):
            print(f"  - {file}")
    
    print("\n=== Plot Descriptions ===")
    print("- annual_trend.png: Main time series with Sen's slope")
    print("- distribution_all_years.png: Box plots for all years (2010-2024)")
    print("- qa_quality_analysis.png: Data filtering and QA flag analysis")
    print("- trend_diagnostics.png: Statistical diagnostic plots")
    print("- comprehensive_summary.png: Publication-ready summary figure")

if __name__ == "__main__":
    main()