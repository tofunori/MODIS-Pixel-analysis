#!/usr/bin/env python3

"""
Advanced Time Series Analysis for MODIS Albedo Data
Master's Thesis Enhancement - Option 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_annual_data():
    """Load the annual albedo statistics"""
    csv_path = "/home/tofunori/Projects/MODIS Pixel analysis/annual_stats.csv"
    
    # Re-export if needed
    if not os.path.exists(csv_path):
        db_path = "/home/tofunori/duckdb-data/modis_analysis.db"
        cmd = f'duckdb "{db_path}" -c "COPY annual_albedo_stats TO \'{csv_path}\' WITH (HEADER, DELIMITER \',\');"'
        subprocess.run(cmd, shell=True, check=True)
    
    data = pd.read_csv(csv_path)
    
    # Create proper time index
    data['date'] = pd.to_datetime(data['year'], format='%Y')
    data.set_index('date', inplace=True)
    
    return data

def simple_autocorr(x, maxlags=None):
    """Simple autocorrelation function"""
    n = len(x)
    if maxlags is None:
        maxlags = n - 1
    
    x = np.array(x)
    x = x - np.mean(x)
    
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[n-1:]
    autocorr = autocorr / autocorr[0]
    
    return autocorr[:maxlags+1]

def seasonal_decomposition_analysis(data, save_path):
    """Perform seasonal decomposition analysis"""
    # Since we only have annual data, we'll create a synthetic decomposition
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Original annual data
    years = data.index.year
    albedo = data['mean_albedo'].values
    
    # Simple trend extraction using polynomial fit
    z = np.polyfit(years, albedo, 1)
    trend = np.poly1d(z)(years)
    
    # Calculate residuals
    residuals = albedo - trend
    
    # Simulate seasonal component (for demonstration)
    # In reality, with annual data, seasonal patterns aren't visible
    seasonal = np.zeros_like(albedo)  # No seasonal component for annual data
    
    # Plot original data
    axes[0].plot(years, albedo, 'o-', color='darkblue', linewidth=2, markersize=6)
    axes[0].set_title('Original Annual Albedo Data', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Snow Albedo')
    axes[0].grid(True, alpha=0.3)
    
    # Plot trend component
    axes[1].plot(years, trend, 'r-', linewidth=3)
    axes[1].set_title('Trend Component (Long-term Linear Pattern)', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    # Plot seasonal component (annual data has no seasonal component)
    axes[2].axhline(y=0, color='green', linewidth=2)
    axes[2].set_title('Seasonal Component (N/A for Annual Data)', fontweight='bold', fontsize=12)
    axes[2].set_ylabel('Seasonal Effect')
    axes[2].set_ylim(-0.01, 0.01)
    axes[2].text(0.5, 0.5, 'No seasonal patterns\navailable with annual data', 
                transform=axes[2].transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    axes[2].grid(True, alpha=0.3)
    
    # Plot residuals
    axes[3].plot(years, residuals, 'purple', linewidth=2, marker='o', markersize=4)
    axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[3].set_title('Residual Component (Deviations from Trend)', fontweight='bold', fontsize=12)
    axes[3].set_ylabel('Residuals')
    axes[3].set_xlabel('Year')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Time Series Decomposition of MODIS Albedo Data', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'trend': trend, 'seasonal': seasonal, 'residuals': residuals}

def change_point_detection(data, save_path):
    """Detect and visualize change points in the time series"""
    years = data.index.year
    albedo = data['mean_albedo'].values
    
    # Simple change point detection using cumulative sum
    def detect_change_points(x, threshold=2.0):
        """Simple CUSUM-based change point detection"""
        n = len(x)
        change_points = []
        
        # Calculate cumulative sum of deviations from mean
        mean_x = np.mean(x)
        cusum = np.cumsum(x - mean_x)
        
        # Look for significant changes in slope
        for i in range(2, n-2):
            # Calculate slope before and after point i
            slope_before = np.polyfit(range(i), cusum[:i], 1)[0]
            slope_after = np.polyfit(range(i, n), cusum[i:], 1)[0]
            
            # If slopes differ significantly, mark as change point
            if abs(slope_after - slope_before) > threshold:
                change_points.append(i)
        
        return change_points
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Original data with detected change points
    ax1.plot(years, albedo, 'o-', color='darkblue', linewidth=3, markersize=8, 
             label='Annual Mean Albedo')
    
    # Detect change points
    change_points = detect_change_points(albedo, threshold=1.5)
    
    # Mark change points
    for cp in change_points:
        if cp < len(years):
            ax1.axvline(x=years[cp], color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax1.annotate(f'Change Point\n{years[cp]}', 
                        xy=(years[cp], albedo[cp]), 
                        xytext=(years[cp]+0.5, albedo[cp]+0.02),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add trend lines for segments
    if change_points:
        segments = [0] + change_points + [len(years)]
        colors = ['green', 'orange', 'purple', 'brown']
        
        for i in range(len(segments)-1):
            start, end = segments[i], segments[i+1]
            if end > len(years):
                end = len(years)
            
            if end > start:
                x_seg = years[start:end]
                y_seg = albedo[start:end]
                
                # Fit linear trend for segment
                if len(x_seg) > 1:
                    z = np.polyfit(x_seg, y_seg, 1)
                    p = np.poly1d(z)
                    ax1.plot(x_seg, p(x_seg), color=colors[i % len(colors)], 
                            linewidth=2, alpha=0.8, 
                            label=f'Segment {i+1} trend')
    
    ax1.set_ylabel('Snow Albedo', fontweight='bold')
    ax1.set_title('Change Point Detection in Albedo Time Series', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: CUSUM chart
    mean_albedo = np.mean(albedo)
    cusum = np.cumsum(albedo - mean_albedo)
    
    ax2.plot(years, cusum, 'purple', linewidth=3, label='Cumulative Sum')
    ax2.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    
    # Mark change points on CUSUM
    for cp in change_points:
        if cp < len(years):
            ax2.axvline(x=years[cp], color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Cumulative Sum', fontweight='bold')
    ax2.set_title('CUSUM Chart for Change Point Detection', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return change_points

def autocorrelation_analysis(data, save_path):
    """Analyze autocorrelation using simple correlation method"""
    albedo = data['mean_albedo'].values
    n_lags = min(8, len(albedo) - 1)  # Maximum reasonable lags for 15 years of data
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate simple autocorrelation
    acf_values = simple_autocorr(albedo, n_lags)
    lags = range(n_lags + 1)
    
    # Calculate confidence bounds (approximate)
    n = len(albedo)
    conf_bound = 1.96 / np.sqrt(n)  # 95% confidence bound
    
    # Plot ACF
    ax1.bar(lags, acf_values, alpha=0.7, color='blue')
    ax1.axhline(y=conf_bound, color='red', linestyle='--', alpha=0.7, label='95% Confidence Bound')
    ax1.axhline(y=-conf_bound, color='red', linestyle='--', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_xlabel('Lag (Years)', fontweight='bold')
    ax1.set_ylabel('Autocorrelation', fontweight='bold')
    ax1.set_title('Autocorrelation Function (ACF)', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Simple partial autocorrelation approximation
    # For small samples, PACF ≈ correlation of residuals after removing linear trend
    pacf_values = np.zeros(n_lags + 1)
    pacf_values[0] = 1.0
    
    for lag in range(1, n_lags + 1):
        if lag < len(albedo):
            # Simple approximation: correlation between x[t] and x[t-lag] after removing trend
            x1 = albedo[lag:]
            x2 = albedo[:-lag]
            if len(x1) > 2 and len(x2) > 2:
                pacf_values[lag] = np.corrcoef(x1, x2)[0, 1]
    
    # Plot PACF
    ax2.bar(lags, pacf_values, alpha=0.7, color='green')
    ax2.axhline(y=conf_bound, color='red', linestyle='--', alpha=0.7, label='95% Confidence Bound')
    ax2.axhline(y=-conf_bound, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Lag (Years)', fontweight='bold')
    ax2.set_ylabel('Partial Autocorrelation', fontweight='bold')
    ax2.set_title('Partial Autocorrelation Function (PACF)', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add interpretation text
    fig.suptitle('Temporal Dependencies in MODIS Albedo Time Series', 
                 fontsize=16, fontweight='bold')
    
    # Add interpretation box - positioned in upper area to avoid x-axis overlap
    interpretation = ('ACF: Shows correlation with past values\n'
                     'PACF: Shows direct correlation excluding intermediate lags\n'
                     'Values outside red lines indicate significant correlation')
    fig.text(0.5, 0.02, interpretation, fontsize=10, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(pad=2.0)  # Add more padding
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    return acf_values, pacf_values

def spectral_analysis(data, save_path):
    """Perform spectral analysis to identify cyclical patterns"""
    years = data.index.year
    albedo = data['mean_albedo'].values
    
    # Remove linear trend for spectral analysis
    detrended = signal.detrend(albedo)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Original vs detrended data
    ax1.plot(years, albedo, 'b-o', linewidth=2, markersize=6, label='Original Data')
    ax1.plot(years, detrended + np.mean(albedo), 'r-s', linewidth=2, markersize=4, 
             label='Detrended Data')
    ax1.set_ylabel('Snow Albedo', fontweight='bold')
    ax1.set_title('Original vs Detrended Albedo Time Series', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Power spectral density
    # Calculate power spectral density
    frequencies, power = signal.periodogram(detrended, fs=1.0)  # fs=1 for annual sampling
    
    # Convert frequencies to periods (years)
    periods = 1 / frequencies[1:]  # Exclude zero frequency
    power_nonzero = power[1:]
    
    ax2.loglog(periods, power_nonzero, 'purple', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Period (Years)', fontweight='bold')
    ax2.set_ylabel('Power Spectral Density', fontweight='bold')
    ax2.set_title('Power Spectral Density - Cyclical Pattern Detection', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Highlight significant periods
    if len(periods) > 0:
        # Find peaks in power spectrum
        peak_indices = signal.find_peaks(power_nonzero, height=np.max(power_nonzero) * 0.1)[0]
        
        for idx in peak_indices:
            if idx < len(periods):
                period = periods[idx]
                ax2.annotate(f'{period:.1f} yr', 
                            xy=(period, power_nonzero[idx]),
                            xytext=(period * 1.5, power_nonzero[idx] * 2),
                            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                            fontsize=10, ha='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add interpretation - positioned BELOW the x-axis labels
    interpretation = ('Spectral analysis reveals dominant frequencies/cycles in the data\n'
                     'Peaks indicate significant periodicities\n'
                     'Higher power = stronger cyclical component')
    
    # Position text 20% from right and 20% up from bottom
    fig.text(0.80, 0.20, interpretation, fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(pad=2.0)  # Add more padding
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    return frequencies, power, periods

def create_comprehensive_time_series_analysis(data, save_path):
    """Create a comprehensive 4-panel time series analysis figure"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    years = data.index.year
    albedo = data['mean_albedo'].values
    
    # 1. Trend decomposition (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Simple trend analysis
    z = np.polyfit(years, albedo, 1)
    trend_line = np.poly1d(z)(years)
    residuals = albedo - trend_line
    
    ax1.plot(years, albedo, 'o-', color='darkblue', linewidth=2, markersize=6, label='Observed')
    ax1.plot(years, trend_line, '--', color='red', linewidth=2, label=f'Trend: {z[0]:.4f}/year')
    ax1.fill_between(years, trend_line - np.std(residuals), trend_line + np.std(residuals),
                     alpha=0.2, color='red', label='±1σ band')
    ax1.set_ylabel('Snow Albedo')
    ax1.set_title('Trend Analysis', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Autocorrelation (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    n_lags = min(6, len(albedo) - 1)
    acf_vals = simple_autocorr(albedo, n_lags)
    lags = range(n_lags + 1)
    
    ax2.bar(lags, acf_vals, alpha=0.7, color='green')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    conf_bound = 1.96 / np.sqrt(len(albedo))
    ax2.axhline(y=conf_bound, color='red', linestyle='--', alpha=0.5, label='95% Confidence')
    ax2.axhline(y=-conf_bound, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Lag (Years)')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('Autocorrelation Function', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Change point detection (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # CUSUM for change detection
    mean_albedo = np.mean(albedo)
    cusum = np.cumsum(albedo - mean_albedo)
    
    ax3.plot(years, cusum, 'purple', linewidth=3, label='CUSUM')
    ax3.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    
    # Simple change point detection
    max_cusum_idx = np.argmax(np.abs(cusum))
    ax3.axvline(x=years[max_cusum_idx], color='red', linestyle='--', linewidth=2,
                label=f'Potential change: {years[max_cusum_idx]}')
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Cumulative Sum')
    ax3.set_title('Change Point Detection', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Residual analysis (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot residuals and their distribution
    ax4.plot(years, residuals, 'o-', color='orange', linewidth=2, markersize=6)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.fill_between(years, -2*np.std(residuals), 2*np.std(residuals),
                     alpha=0.2, color='gray', label='±2σ band')
    
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residual Analysis', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Time Series Analysis - MODIS Albedo Trends', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run advanced time series analysis"""
    print("=== Advanced Time Series Analysis for Master's Thesis ===")
    
    # Load data
    data = load_annual_data()
    print(f"Loaded {len(data)} years of annual albedo data")
    
    # Create output directory
    plot_dir = "/home/tofunori/Projects/MODIS Pixel analysis/advanced_plots"
    import os
    os.makedirs(plot_dir, exist_ok=True)
    
    print("\nGenerating advanced time series analysis plots...")
    
    # 1. Seasonal decomposition
    print("1. Seasonal decomposition analysis...")
    decomposition = seasonal_decomposition_analysis(data, f"{plot_dir}/seasonal_decomposition.png")
    
    # 2. Change point detection
    print("2. Change point detection...")
    change_points = change_point_detection(data, f"{plot_dir}/change_point_detection.png")
    
    # 3. Autocorrelation analysis
    print("3. Autocorrelation analysis...")
    acf_vals, pacf_vals = autocorrelation_analysis(data, f"{plot_dir}/autocorrelation_analysis.png")
    
    # 4. Spectral analysis
    print("4. Spectral analysis...")
    frequencies, power, periods = spectral_analysis(data, f"{plot_dir}/spectral_analysis.png")
    
    # 5. Comprehensive summary
    print("5. Comprehensive time series summary...")
    create_comprehensive_time_series_analysis(data, f"{plot_dir}/comprehensive_time_series.png")
    
    print(f"\n=== Advanced Analysis Complete ===")
    print(f"All plots saved in: {plot_dir}/")
    print("\nFiles created:")
    for file in sorted(os.listdir(plot_dir)):
        if file.endswith('.png'):
            print(f"  - {file}")
    
    # Print summary results
    print(f"\n=== Analysis Summary ===")
    print(f"Change points detected: {len(change_points)}")
    if change_points:
        years = data.index.year
        print(f"Change point years: {[years[cp] for cp in change_points if cp < len(years)]}")
    
    if len(acf_vals) > 1:
        print(f"Maximum autocorrelation at lag 1: {acf_vals[1]:.3f}")
        print(f"Data shows {'strong' if abs(acf_vals[1]) > 0.5 else 'moderate' if abs(acf_vals[1]) > 0.2 else 'weak'} temporal dependence")
    else:
        print("Autocorrelation analysis completed")

if __name__ == "__main__":
    main()