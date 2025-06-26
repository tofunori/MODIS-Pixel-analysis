#!/usr/bin/env python3

"""
Autocorrelation-Corrected Statistical Analysis for MODIS Albedo Data
Addresses reviewer feedback on serial correlation and independence assumptions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.optimize import minimize
from scipy.linalg import toeplitz
import subprocess
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class AutocorrelationCorrectedAnalyzer:
    """
    Enhanced statistical analyzer with autocorrelation correction methods
    Implements Yue-Pilon, Hamed & Rao, and GLS approaches
    """
    
    def __init__(self, data):
        self.data = data
        self.years = data.index.year.values
        self.albedo = data['mean_albedo'].values
        self.n = len(self.albedo)
        self.results = {}
        
    def calculate_autocorrelation_structure(self):
        """Calculate autocorrelation structure for correction methods"""
        results = {}
        
        # Lag-1 autocorrelation (critical for correction methods)
        lag1_corr = np.corrcoef(self.albedo[:-1], self.albedo[1:])[0, 1]
        results['lag1_autocorr'] = lag1_corr
        
        # Full autocorrelation function up to n/3 lags
        max_lags = min(self.n // 3, 10)
        autocorr_full = []
        
        for lag in range(max_lags + 1):
            if lag == 0:
                autocorr_full.append(1.0)
            else:
                if self.n > lag:
                    corr = np.corrcoef(self.albedo[:-lag], self.albedo[lag:])[0, 1]
                    autocorr_full.append(corr)
                else:
                    autocorr_full.append(0.0)
        
        results['autocorr_function'] = autocorr_full
        results['max_lags'] = max_lags
        
        # Ljung-Box test for serial correlation
        ljung_box_stat, ljung_box_p = self.ljung_box_test()
        results['ljung_box'] = {
            'statistic': ljung_box_stat,
            'p_value': ljung_box_p,
            'significant_autocorr': ljung_box_p < 0.05
        }
        
        self.results['autocorr_structure'] = results
        return results
    
    def ljung_box_test(self, lags=5):
        """Ljung-Box test for serial correlation"""
        # Calculate residuals from linear trend
        slope, intercept = np.polyfit(self.years, self.albedo, 1)
        residuals = self.albedo - (slope * self.years + intercept)
        
        n = len(residuals)
        autocorrs = []
        
        # Calculate autocorrelations for specified lags
        for lag in range(1, lags + 1):
            if n > lag:
                corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                autocorrs.append(corr)
            else:
                autocorrs.append(0.0)
        
        # Ljung-Box statistic
        lb_stat = n * (n + 2) * sum([(corr**2) / (n - k) for k, corr in enumerate(autocorrs, 1)])
        
        # p-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)
        
        return lb_stat, p_value
    
    def yue_pilon_prewhitening(self):
        """
        Yue-Pilon pre-whitening approach for Mann-Kendall test
        Removes lag-1 autocorrelation before trend testing
        """
        results = {}
        
        # Step 1: Calculate lag-1 autocorrelation coefficient
        r1 = np.corrcoef(self.albedo[:-1], self.albedo[1:])[0, 1]
        results['lag1_autocorr'] = r1
        
        # Step 2: Pre-whiten the series if |r1| is significant
        # Significance test for autocorrelation
        r1_critical = 1.96 / np.sqrt(self.n - 1)  # 95% confidence limit
        results['r1_critical'] = r1_critical
        results['r1_significant'] = abs(r1) > r1_critical
        
        if abs(r1) > r1_critical:
            # Apply pre-whitening transformation
            x_pw = np.zeros(self.n - 1)
            for i in range(1, self.n):
                x_pw[i-1] = self.albedo[i] - r1 * self.albedo[i-1]
            
            years_pw = self.years[1:]  # Corresponding years
            results['prewhitened_series'] = x_pw
            results['prewhitened_years'] = years_pw
            results['prewhitening_applied'] = True
            
            # Mann-Kendall test on pre-whitened series
            mk_stat_pw, mk_p_pw = self.mann_kendall_test(x_pw)
            results['mann_kendall_prewhitened'] = {
                'statistic': mk_stat_pw,
                'p_value': mk_p_pw,
                'significant': mk_p_pw < 0.05
            }
            
        else:
            # No pre-whitening needed
            results['prewhitening_applied'] = False
            mk_stat, mk_p = self.mann_kendall_test(self.albedo)
            results['mann_kendall_original'] = {
                'statistic': mk_stat,
                'p_value': mk_p,
                'significant': mk_p < 0.05
            }
        
        # Original Mann-Kendall for comparison
        mk_stat_orig, mk_p_orig = self.mann_kendall_test(self.albedo)
        results['mann_kendall_uncorrected'] = {
            'statistic': mk_stat_orig,
            'p_value': mk_p_orig,
            'significant': mk_p_orig < 0.05
        }
        
        self.results['yue_pilon'] = results
        return results
    
    def hamed_rao_correction(self):
        """
        Hamed & Rao variance correction for Mann-Kendall test
        Adjusts variance for autocorrelated data
        """
        results = {}
        
        # Calculate Mann-Kendall statistic (unchanged)
        S, var_original = self.mann_kendall_components()
        results['mk_statistic'] = S
        results['original_variance'] = var_original
        
        # Calculate autocorrelation coefficients
        max_lags = min(self.n // 3, 10)
        autocorrs = []
        
        for lag in range(1, max_lags + 1):
            if self.n > lag:
                corr = np.corrcoef(self.albedo[:-lag], self.albedo[lag:])[0, 1]
                autocorrs.append(corr)
            else:
                autocorrs.append(0.0)
        
        results['autocorr_coefficients'] = autocorrs
        
        # Calculate Hamed & Rao variance correction
        n = self.n
        correction_sum = 0
        
        for i in range(1, max_lags + 1):
            if i < len(autocorrs) and not np.isnan(autocorrs[i-1]):
                ri = autocorrs[i-1]
                correction_sum += (n - i) * (n - i - 1) * (n - i - 2) * ri
        
        # Corrected variance
        var_corrected = var_original * (1 + (2 / (n * (n-1) * (n-2))) * correction_sum)
        results['corrected_variance'] = var_corrected
        results['variance_correction_factor'] = var_corrected / var_original
        
        # Calculate corrected Z-statistic and p-value
        if var_corrected > 0:
            if S > 0:
                z_corrected = (S - 1) / np.sqrt(var_corrected)
            elif S < 0:
                z_corrected = (S + 1) / np.sqrt(var_corrected)
            else:
                z_corrected = 0
            
            p_corrected = 2 * (1 - stats.norm.cdf(abs(z_corrected)))
        else:
            z_corrected = np.nan
            p_corrected = np.nan
        
        results['z_corrected'] = z_corrected
        results['p_corrected'] = p_corrected
        results['significant_corrected'] = p_corrected < 0.05 if not np.isnan(p_corrected) else False
        
        # Original test for comparison
        z_original = S / np.sqrt(var_original) if var_original > 0 else np.nan
        p_original = 2 * (1 - stats.norm.cdf(abs(z_original))) if not np.isnan(z_original) else np.nan
        
        results['z_original'] = z_original
        results['p_original'] = p_original
        results['significant_original'] = p_original < 0.05 if not np.isnan(p_original) else False
        
        self.results['hamed_rao'] = results
        return results
    
    def gls_trend_analysis(self):
        """
        Generalized Least Squares with AR(1) error structure
        Provides autocorrelation-corrected parametric trend estimation
        """
        results = {}
        
        # Estimate AR(1) parameter using OLS residuals
        slope_ols, intercept_ols = np.polyfit(self.years, self.albedo, 1)
        residuals_ols = self.albedo - (slope_ols * self.years + intercept_ols)
        
        # Lag-1 autocorrelation of residuals
        if len(residuals_ols) > 1:
            rho = np.corrcoef(residuals_ols[:-1], residuals_ols[1:])[0, 1]
        else:
            rho = 0
        
        results['ar1_parameter'] = rho
        results['ols_slope'] = slope_ols
        results['ols_intercept'] = intercept_ols
        
        # Construct AR(1) covariance matrix
        sigma2 = np.var(residuals_ols)
        
        # Create correlation matrix for AR(1) process
        correlation_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                correlation_matrix[i, j] = rho ** abs(i - j)
        
        # Covariance matrix
        V = sigma2 * correlation_matrix / (1 - rho**2) if abs(rho) < 1 else sigma2 * np.eye(self.n)
        
        try:
            # GLS estimation: beta = (X'V^-1 X)^-1 X'V^-1 y
            X = np.column_stack([np.ones(self.n), self.years])
            
            # Cholesky decomposition for numerical stability
            try:
                L = np.linalg.cholesky(V)
                L_inv = np.linalg.inv(L)
                
                # Transform data
                X_transformed = L_inv @ X
                y_transformed = L_inv @ self.albedo
                
                # OLS on transformed data
                XtX_inv = np.linalg.inv(X_transformed.T @ X_transformed)
                beta_gls = XtX_inv @ X_transformed.T @ y_transformed
                
                # Calculate standard errors
                residuals_gls = y_transformed - X_transformed @ beta_gls
                mse_gls = np.sum(residuals_gls**2) / (self.n - 2)
                var_beta = mse_gls * XtX_inv
                se_gls = np.sqrt(np.diag(var_beta))
                
            except np.linalg.LinAlgError:
                # Fallback to regularized approach
                V_reg = V + 1e-6 * np.eye(self.n)
                V_inv = np.linalg.inv(V_reg)
                
                XtVinvX = X.T @ V_inv @ X
                XtVinvX_inv = np.linalg.inv(XtVinvX)
                beta_gls = XtVinvX_inv @ X.T @ V_inv @ self.albedo
                
                # Standard errors
                var_beta = XtVinvX_inv
                se_gls = np.sqrt(np.diag(var_beta))
            
            results['gls_intercept'] = beta_gls[0]
            results['gls_slope'] = beta_gls[1]
            results['gls_slope_se'] = se_gls[1]
            results['gls_intercept_se'] = se_gls[0]
            
            # t-statistics and p-values
            t_slope = beta_gls[1] / se_gls[1]
            t_intercept = beta_gls[0] / se_gls[0]
            
            df = self.n - 2
            p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), df))
            p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df))
            
            results['gls_slope_t'] = t_slope
            results['gls_slope_p'] = p_slope
            results['gls_intercept_t'] = t_intercept
            results['gls_intercept_p'] = p_intercept
            
            # Confidence intervals
            t_critical = stats.t.ppf(0.975, df)
            slope_ci = [beta_gls[1] - t_critical * se_gls[1], 
                       beta_gls[1] + t_critical * se_gls[1]]
            
            results['gls_slope_ci'] = slope_ci
            results['significant_trend'] = p_slope < 0.05
            
            # Model fit statistics
            y_pred_gls = X @ beta_gls
            residuals_final = self.albedo - y_pred_gls
            
            results['gls_fitted_values'] = y_pred_gls
            results['gls_residuals'] = residuals_final
            results['gls_mse'] = np.mean(residuals_final**2)
            
            # Compare with OLS
            results['improvement_over_ols'] = {
                'ols_slope_se': np.sqrt(np.var(residuals_ols) / np.sum((self.years - np.mean(self.years))**2)),
                'gls_slope_se': se_gls[1],
                'se_ratio': se_gls[1] / np.sqrt(np.var(residuals_ols) / np.sum((self.years - np.mean(self.years))**2))
            }
            
        except Exception as e:
            results['error'] = str(e)
            results['gls_successful'] = False
            
        self.results['gls_analysis'] = results
        return results
    
    def mann_kendall_test(self, data):
        """Calculate Mann-Kendall test statistic and p-value"""
        n = len(data)
        S = 0
        
        for i in range(n-1):
            for j in range(i+1, n):
                if data[j] > data[i]:
                    S += 1
                elif data[j] < data[i]:
                    S -= 1
        
        # Variance calculation
        var_S = n * (n - 1) * (2 * n + 5) / 18
        
        # Z statistic
        if S > 0:
            Z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            Z = (S + 1) / np.sqrt(var_S)
        else:
            Z = 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        
        return S, p_value
    
    def mann_kendall_components(self):
        """Get Mann-Kendall components for variance correction"""
        n = len(self.albedo)
        S = 0
        
        for i in range(n-1):
            for j in range(i+1, n):
                if self.albedo[j] > self.albedo[i]:
                    S += 1
                elif self.albedo[j] < self.albedo[i]:
                    S -= 1
        
        # Original variance (without autocorrelation correction)
        var_S = n * (n - 1) * (2 * n + 5) / 18
        
        return S, var_S
    
    def generate_autocorr_corrected_summary(self):
        """Generate summary of autocorrelation-corrected analyses"""
        summary = {
            'dataset_info': {
                'n_observations': self.n,
                'years_range': f"{self.years[0]}-{self.years[-1]}",
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'autocorrelation_diagnostics': {},
            'corrected_trend_tests': {},
            'method_comparisons': {}
        }
        
        # Autocorrelation structure
        if 'autocorr_structure' in self.results:
            ac_struct = self.results['autocorr_structure']
            summary['autocorrelation_diagnostics'] = {
                'lag1_autocorr': round(ac_struct['lag1_autocorr'], 4),
                'ljung_box_p': round(ac_struct['ljung_box']['p_value'], 6),
                'serial_correlation_detected': ac_struct['ljung_box']['significant_autocorr']
            }
        
        # Yue-Pilon results
        if 'yue_pilon' in self.results:
            yp = self.results['yue_pilon']
            summary['corrected_trend_tests']['yue_pilon'] = {
                'prewhitening_needed': yp['r1_significant'],
                'prewhitening_applied': yp['prewhitening_applied']
            }
            
            if yp['prewhitening_applied']:
                summary['corrected_trend_tests']['yue_pilon']['mann_kendall_p'] = round(
                    yp['mann_kendall_prewhitened']['p_value'], 6)
                summary['corrected_trend_tests']['yue_pilon']['significant'] = yp['mann_kendall_prewhitened']['significant']
            else:
                summary['corrected_trend_tests']['yue_pilon']['mann_kendall_p'] = round(
                    yp['mann_kendall_original']['p_value'], 6)
                summary['corrected_trend_tests']['yue_pilon']['significant'] = yp['mann_kendall_original']['significant']
        
        # Hamed & Rao results
        if 'hamed_rao' in self.results:
            hr = self.results['hamed_rao']
            summary['corrected_trend_tests']['hamed_rao'] = {
                'original_p': round(hr['p_original'], 6),
                'corrected_p': round(hr['p_corrected'], 6),
                'variance_correction_factor': round(hr['variance_correction_factor'], 4),
                'significant_corrected': hr['significant_corrected'],
                'significant_original': hr['significant_original']
            }
        
        # GLS results
        if 'gls_analysis' in self.results:
            gls = self.results['gls_analysis']
            if 'gls_slope' in gls:
                summary['corrected_trend_tests']['gls'] = {
                    'slope': round(gls['gls_slope'], 6),
                    'slope_se': round(gls['gls_slope_se'], 6),
                    'slope_p': round(gls['gls_slope_p'], 6),
                    'slope_ci': [round(x, 6) for x in gls['gls_slope_ci']],
                    'significant': gls['significant_trend'],
                    'ar1_parameter': round(gls['ar1_parameter'], 4)
                }
        
        return summary

def load_annual_data():
    """Load the annual albedo statistics"""
    csv_path = "/home/tofunori/Projects/MODIS Pixel analysis/annual_stats.csv"
    
    if not os.path.exists(csv_path):
        db_path = "/home/tofunori/duckdb-data/modis_analysis.db"
        cmd = f'duckdb "{db_path}" -c "COPY annual_albedo_stats TO \'{csv_path}\' WITH (HEADER, DELIMITER \',\');"'
        subprocess.run(cmd, shell=True, check=True)
    
    data = pd.read_csv(csv_path)
    data['date'] = pd.to_datetime(data['year'], format='%Y')
    data.set_index('date', inplace=True)
    
    return data

def main():
    """Main function to run autocorrelation-corrected analysis"""
    print("=== Autocorrelation-Corrected Statistical Analysis ===")
    
    # Load data
    data = load_annual_data()
    print(f"Loaded {len(data)} years of data")
    
    # Initialize analyzer
    analyzer = AutocorrelationCorrectedAnalyzer(data)
    
    print("\nRunning autocorrelation diagnostics...")
    analyzer.calculate_autocorrelation_structure()
    
    print("Applying Yue-Pilon pre-whitening...")
    analyzer.yue_pilon_prewhitening()
    
    print("Applying Hamed & Rao variance correction...")
    analyzer.hamed_rao_correction()
    
    print("Running GLS trend analysis...")
    analyzer.gls_trend_analysis()
    
    # Generate summary
    summary = analyzer.generate_autocorr_corrected_summary()
    
    # Save results
    output_dir = "/home/tofunori/Projects/MODIS Pixel analysis/results/statistical_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    
    # Save detailed results
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)
    
    with open(f"{output_dir}/autocorr_corrected_results.json", 'w') as f:
        json.dump(convert_numpy(analyzer.results), f, indent=2)
    
    with open(f"{output_dir}/autocorr_corrected_summary.json", 'w') as f:
        json.dump(convert_numpy(summary), f, indent=2)
    
    print(f"\n=== Autocorrelation-Corrected Analysis Complete ===")
    print(f"Results saved in: {output_dir}/")
    print("\nKey Findings:")
    
    # Print key results
    if 'autocorr_structure' in analyzer.results:
        ac = analyzer.results['autocorr_structure']
        print(f"- Lag-1 autocorrelation: {ac['lag1_autocorr']:.4f}")
        print(f"- Ljung-Box test p-value: {ac['ljung_box']['p_value']:.6f}")
        print(f"- Serial correlation detected: {ac['ljung_box']['significant_autocorr']}")
    
    if 'yue_pilon' in analyzer.results:
        yp = analyzer.results['yue_pilon']
        print(f"- Yue-Pilon pre-whitening needed: {yp['r1_significant']}")
        if yp['prewhitening_applied']:
            print(f"- Mann-Kendall (pre-whitened) p-value: {yp['mann_kendall_prewhitened']['p_value']:.6f}")
    
    if 'hamed_rao' in analyzer.results:
        hr = analyzer.results['hamed_rao']
        print(f"- Hamed & Rao correction factor: {hr['variance_correction_factor']:.4f}")
        print(f"- Mann-Kendall (corrected) p-value: {hr['p_corrected']:.6f}")
    
    if 'gls_analysis' in analyzer.results and 'gls_slope' in analyzer.results['gls_analysis']:
        gls = analyzer.results['gls_analysis']
        print(f"- GLS slope: {gls['gls_slope']:.6f} ± {gls['gls_slope_se']:.6f}")
        print(f"- GLS slope p-value: {gls['gls_slope_p']:.6f}")

if __name__ == "__main__":
    main()