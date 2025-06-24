#!/usr/bin/env python3

"""
MODIS Albedo Trend Analysis
Sen's Slope and Mann-Kendall Test
"""

import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
import subprocess
import os

def mann_kendall_test(data):
    """
    Perform Mann-Kendall test for trend detection
    Returns: tau, p_value, trend
    """
    n = len(data)
    
    # Calculate S statistic
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data[j] - data[i])
    
    # Calculate variance
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    # Calculate standardized test statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Calculate Kendall's tau
    tau = s / (0.5 * n * (n - 1))
    
    # Determine trend
    if p_value < 0.05:
        if tau > 0:
            trend = "Increasing"
        else:
            trend = "Decreasing"
    else:
        trend = "No significant trend"
    
    return tau, p_value, trend, s, z

def sens_slope(data):
    """
    Calculate Sen's slope estimator
    Returns: slope, confidence_interval
    """
    n = len(data)
    slopes = []
    
    # Calculate all pairwise slopes
    for i in range(n-1):
        for j in range(i+1, n):
            if j != i:
                slope = (data[j] - data[i]) / (j - i)
                slopes.append(slope)
    
    # Sen's slope is the median of all slopes
    sens_slope_val = np.median(slopes)
    
    # Calculate confidence interval (approximate)
    slopes_sorted = np.sort(slopes)
    n_slopes = len(slopes)
    
    # 95% confidence interval indices
    alpha = 0.05
    z_alpha = stats.norm.ppf(1 - alpha/2)
    var_s = n * (n - 1) * (2 * n + 5) / 18
    c_alpha = z_alpha * np.sqrt(var_s)
    
    lower_idx = max(0, int((n_slopes - c_alpha) / 2))
    upper_idx = min(n_slopes - 1, int((n_slopes + c_alpha) / 2))
    
    ci_lower = slopes_sorted[lower_idx] if lower_idx < len(slopes_sorted) else slopes_sorted[0]
    ci_upper = slopes_sorted[upper_idx] if upper_idx < len(slopes_sorted) else slopes_sorted[-1]
    
    return sens_slope_val, (ci_lower, ci_upper)

def main():
    print("=== MODIS Albedo Trend Analysis (2010-2024) ===")
    print("Filtered Data: 90-100% glacier fraction, no clouds, standard QA=1")
    
    # Export data from DuckDB
    db_path = "/home/tofunori/duckdb-data/modis_analysis.db"
    csv_path = "/home/tofunori/Projects/MODIS Pixel analysis/annual_stats.csv"
    
    # Export annual statistics from DuckDB
    cmd = f'duckdb "{db_path}" -c "COPY annual_albedo_stats TO \'{csv_path}\' WITH (HEADER, DELIMITER \',\');"'
    subprocess.run(cmd, shell=True, check=True)
    
    # Read the data
    data = pd.read_csv(csv_path)
    
    print(f"Total filtered records: {data['pixel_count'].sum()}")
    print()
    
    # Prepare time series data
    years = data['year'].values
    mean_albedo = data['mean_albedo'].values
    
    # Perform Sen's Slope analysis
    print("=== Sen's Slope Analysis ===")
    slope, (ci_lower, ci_upper) = sens_slope(mean_albedo)
    print(f"Sen's Slope (albedo change per year): {slope:.6f}")
    print(f"95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    # Perform Mann-Kendall test
    print("\n=== Mann-Kendall Test ===")
    tau, p_value, trend, s_stat, z_stat = mann_kendall_test(mean_albedo)
    print(f"Kendall's Tau: {tau:.4f}")
    print(f"S statistic: {s_stat}")
    print(f"Z statistic: {z_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    # Significance levels
    if p_value < 0.001:
        sig_level = "*** (p < 0.001)"
    elif p_value < 0.01:
        sig_level = "** (p < 0.01)"
    elif p_value < 0.05:
        sig_level = "* (p < 0.05)"
    elif p_value < 0.1:
        sig_level = ". (p < 0.1)"
    else:
        sig_level = "not significant (p >= 0.1)"
    
    print(f"Trend significance: {sig_level}")
    print(f"Trend direction: {trend}")
    
    # Summary statistics
    print("\n=== Annual Statistics Summary ===")
    print(f"Year Range: {years.min()}-{years.max()}")
    print(f"Mean albedo range: {mean_albedo.min():.3f}-{mean_albedo.max():.3f}")
    print(f"Overall mean albedo: {mean_albedo.mean():.3f}")
    print(f"Total decline: {mean_albedo[-1] - mean_albedo[0]:.3f}")
    
    # Create results summary
    results_summary = pd.DataFrame({
        'Metric': [
            'Sen\'s Slope', 'Sen\'s Slope CI Lower', 'Sen\'s Slope CI Upper',
            'Kendall\'s Tau', 'Mann-Kendall P-value', 'S Statistic', 'Z Statistic',
            'Overall Mean Albedo', 'First Year Mean', 'Last Year Mean', 'Total Change',
            'Trend Direction', 'Significance Level'
        ],
        'Value': [
            slope, ci_lower, ci_upper, tau, p_value, s_stat, z_stat,
            mean_albedo.mean(), mean_albedo[0], mean_albedo[-1], 
            mean_albedo[-1] - mean_albedo[0], trend, sig_level
        ]
    })
    
    # Export results
    results_path = "/home/tofunori/Projects/MODIS Pixel analysis/modis_trend_results.csv"
    annual_path = "/home/tofunori/Projects/MODIS Pixel analysis/modis_annual_stats_analysis.csv"
    
    results_summary.to_csv(results_path, index=False)
    data.to_csv(annual_path, index=False)
    
    print("\n=== Output Files Created ===")
    print(f"- {results_path}: Summary of trend analysis")
    print(f"- {annual_path}: Annual statistics")
    print(f"- {csv_path}: Raw data export from DuckDB")
    
    print("\n=== Analysis Complete ===")
    
    # Display annual data table
    print("\n=== Annual Albedo Statistics ===")
    print(data.to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    main()