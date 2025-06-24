#!/usr/bin/env python3

"""
Comprehensive Statistical Analysis for MODIS Albedo Dataset
Advanced statistical methods for master's thesis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.optimize import minimize
import subprocess
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

class StatisticalAnalyzer:
    """Comprehensive statistical analysis class"""
    
    def __init__(self, data):
        self.data = data
        self.years = data.index.year.values
        self.albedo = data['mean_albedo'].values
        self.results = {}
        
    def distributional_analysis(self):
        """Analyze distribution properties and normality"""
        results = {}
        
        # Basic descriptive statistics
        results['descriptive'] = {
            'mean': np.mean(self.albedo),
            'median': np.median(self.albedo),
            'std': np.std(self.albedo),
            'variance': np.var(self.albedo),
            'range': np.ptp(self.albedo),
            'iqr': np.percentile(self.albedo, 75) - np.percentile(self.albedo, 25),
            'cv': np.std(self.albedo) / np.mean(self.albedo),  # Coefficient of variation
            'skewness': stats.skew(self.albedo),
            'kurtosis': stats.kurtosis(self.albedo)
        }
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(self.albedo)
        anderson_stat, anderson_crit, anderson_sig = stats.anderson(self.albedo, dist='norm')
        jb_stat, jb_p = stats.jarque_bera(self.albedo)
        
        results['normality_tests'] = {
            'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
            'anderson_darling': {'statistic': anderson_stat, 'critical_values': anderson_crit},
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p}
        }
        
        # Distribution fitting
        distributions = ['norm', 'gamma', 'beta', 'lognorm', 'weibull_min']
        results['distribution_fitting'] = {}
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(self.albedo)
                ks_stat, ks_p = stats.kstest(self.albedo, lambda x: dist.cdf(x, *params))
                aic = 2 * len(params) - 2 * np.sum(dist.logpdf(self.albedo, *params))
                
                results['distribution_fitting'][dist_name] = {
                    'parameters': params,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'aic': aic
                }
            except:
                continue
        
        # Outlier detection
        z_scores = np.abs(stats.zscore(self.albedo))
        iqr_outliers = self.detect_iqr_outliers()
        
        results['outliers'] = {
            'z_score_outliers': np.where(z_scores > 2.5)[0].tolist(),
            'iqr_outliers': iqr_outliers,
            'extreme_values': {
                'minimum': {'year': self.years[np.argmin(self.albedo)], 'value': np.min(self.albedo)},
                'maximum': {'year': self.years[np.argmax(self.albedo)], 'value': np.max(self.albedo)}
            }
        }
        
        self.results['distributional'] = results
        return results
    
    def detect_iqr_outliers(self):
        """Detect outliers using IQR method"""
        Q1 = np.percentile(self.albedo, 25)
        Q3 = np.percentile(self.albedo, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = []
        for i, value in enumerate(self.albedo):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)
        
        return outliers
    
    def robust_statistics(self):
        """Compute robust statistical measures"""
        results = {}
        
        # Robust location and scale measures
        results['robust_measures'] = {
            'median': np.median(self.albedo),
            'mad': stats.median_abs_deviation(self.albedo),  # Median Absolute Deviation
            'trimmed_mean_10': stats.trim_mean(self.albedo, 0.1),  # 10% trimmed mean
            'trimmed_mean_20': stats.trim_mean(self.albedo, 0.2),  # 20% trimmed mean
            'winsorized_mean': stats.mstats.winsorize(self.albedo, limits=0.1).mean(),
            'huber_location': self.huber_location_estimate(),
            'percentile_range': {
                '5th': np.percentile(self.albedo, 5),
                '10th': np.percentile(self.albedo, 10),
                '90th': np.percentile(self.albedo, 90),
                '95th': np.percentile(self.albedo, 95)
            }
        }
        
        # Robust trend estimation (Theil-Sen)
        theil_sen_slope, theil_sen_intercept = self.theil_sen_estimator()
        results['robust_trend'] = {
            'theil_sen_slope': theil_sen_slope,
            'theil_sen_intercept': theil_sen_intercept,
            'theil_sen_trend_total': theil_sen_slope * (self.years[-1] - self.years[0])
        }
        
        # Bootstrap confidence intervals
        bootstrap_results = self.bootstrap_analysis()
        results['bootstrap'] = bootstrap_results
        
        self.results['robust'] = results
        return results
    
    def huber_location_estimate(self, k=1.345):
        """Huber M-estimator for robust location"""
        def huber_loss(mu, x, k):
            residuals = x - mu
            mask = np.abs(residuals) <= k
            loss = np.where(mask, 0.5 * residuals**2, k * np.abs(residuals) - 0.5 * k**2)
            return np.sum(loss)
        
        result = minimize(lambda mu: huber_loss(mu, self.albedo, k), np.median(self.albedo))
        return result.x[0]
    
    def theil_sen_estimator(self):
        """Theil-Sen robust slope estimator"""
        slopes = []
        n = len(self.years)
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.years[j] != self.years[i]:
                    slope = (self.albedo[j] - self.albedo[i]) / (self.years[j] - self.years[i])
                    slopes.append(slope)
        
        median_slope = np.median(slopes)
        
        # Calculate intercept using median of residuals
        intercepts = []
        for i in range(n):
            intercept = self.albedo[i] - median_slope * self.years[i]
            intercepts.append(intercept)
        
        median_intercept = np.median(intercepts)
        
        return median_slope, median_intercept
    
    def bootstrap_analysis(self, n_bootstrap=10000):
        """Bootstrap confidence intervals for key statistics"""
        np.random.seed(42)  # For reproducibility
        
        bootstrap_means = []
        bootstrap_slopes = []
        bootstrap_stds = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(self.albedo), size=len(self.albedo), replace=True)
            boot_years = self.years[indices]
            boot_albedo = self.albedo[indices]
            
            # Calculate statistics
            bootstrap_means.append(np.mean(boot_albedo))
            bootstrap_stds.append(np.std(boot_albedo))
            
            # Calculate slope
            if len(np.unique(boot_years)) > 1:
                slope, _ = np.polyfit(boot_years, boot_albedo, 1)
                bootstrap_slopes.append(slope)
        
        results = {
            'mean_ci': np.percentile(bootstrap_means, [2.5, 97.5]),
            'std_ci': np.percentile(bootstrap_stds, [2.5, 97.5]),
            'slope_ci': np.percentile(bootstrap_slopes, [2.5, 97.5]) if bootstrap_slopes else [np.nan, np.nan],
            'mean_se': np.std(bootstrap_means),
            'slope_se': np.std(bootstrap_slopes) if bootstrap_slopes else np.nan
        }
        
        return results
    
    def advanced_trend_analysis(self):
        """Advanced trend detection and change point analysis"""
        results = {}
        
        # Pettitt test for change point
        pettitt_result = self.pettitt_test()
        results['pettitt_test'] = pettitt_result
        
        # CUSUM analysis
        cusum_result = self.cusum_analysis()
        results['cusum'] = cusum_result
        
        # Multiple change point detection
        change_points = self.multiple_change_points()
        results['change_points'] = change_points
        
        # Trend significance tests
        trend_tests = self.trend_significance_tests()
        results['trend_tests'] = trend_tests
        
        self.results['advanced_trend'] = results
        return results
    
    def pettitt_test(self):
        """Pettitt test for change point detection"""
        n = len(self.albedo)
        U = np.zeros(n)
        
        for t in range(n):
            U[t] = np.sum([np.sign(self.albedo[t] - self.albedo[j]) for j in range(n)])
        
        K = np.max(np.abs(U))
        change_point = np.argmax(np.abs(U))
        
        # Approximate p-value
        p_value = 2 * np.exp(-6 * K**2 / (n**3 + n**2))
        
        return {
            'statistic': K,
            'change_point_index': change_point,
            'change_point_year': self.years[change_point],
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def cusum_analysis(self):
        """CUSUM (Cumulative Sum) control chart analysis"""
        mean_albedo = np.mean(self.albedo)
        cusum_pos = np.zeros(len(self.albedo))
        cusum_neg = np.zeros(len(self.albedo))
        
        # Standard CUSUM with h=5 and k=0.5*std
        h = 5 * np.std(self.albedo)
        k = 0.5 * np.std(self.albedo)
        
        for i in range(1, len(self.albedo)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (self.albedo[i] - mean_albedo) - k)
            cusum_neg[i] = max(0, cusum_neg[i-1] - (self.albedo[i] - mean_albedo) - k)
        
        # Detect signals
        pos_signals = np.where(cusum_pos > h)[0]
        neg_signals = np.where(cusum_neg > h)[0]
        
        return {
            'cusum_positive': cusum_pos,
            'cusum_negative': cusum_neg,
            'threshold': h,
            'positive_signals': pos_signals.tolist(),
            'negative_signals': neg_signals.tolist(),
            'first_signal': min(pos_signals[0] if len(pos_signals) > 0 else len(self.albedo),
                              neg_signals[0] if len(neg_signals) > 0 else len(self.albedo))
        }
    
    def multiple_change_points(self):
        """Detect multiple change points using binary segmentation"""
        def find_single_change_point(data):
            n = len(data)
            if n < 4:  # Need minimum data points
                return None
            
            best_point = None
            best_score = -np.inf
            
            for i in range(2, n-2):
                left_mean = np.mean(data[:i])
                right_mean = np.mean(data[i:])
                
                # Calculate likelihood ratio test statistic
                total_var = np.var(data)
                left_var = np.var(data[:i]) if len(data[:i]) > 1 else total_var
                right_var = np.var(data[i:]) if len(data[i:]) > 1 else total_var
                
                if total_var > 0:
                    score = -n * np.log(total_var) + i * np.log(left_var) + (n-i) * np.log(right_var)
                    if score > best_score:
                        best_score = score
                        best_point = i
            
            return best_point
        
        change_points = []
        segments = [(0, len(self.albedo))]
        
        while segments:
            start, end = segments.pop(0)
            segment_data = self.albedo[start:end]
            
            if len(segment_data) < 6:  # Minimum segment size
                continue
                
            cp = find_single_change_point(segment_data)
            
            if cp is not None:
                actual_cp = start + cp
                change_points.append(actual_cp)
                
                # Add new segments for further analysis
                if cp > 3:
                    segments.append((start, start + cp))
                if end - start - cp > 3:
                    segments.append((start + cp, end))
        
        change_points.sort()
        
        return {
            'change_points': change_points,
            'change_point_years': [self.years[cp] for cp in change_points],
            'n_change_points': len(change_points)
        }
    
    def trend_significance_tests(self):
        """Multiple tests for trend significance"""
        results = {}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.years, self.albedo)
        results['linear_regression'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'significant': p_value < 0.05
        }
        
        # Spearman correlation (non-parametric)
        spearman_corr, spearman_p = stats.spearmanr(self.years, self.albedo)
        results['spearman'] = {
            'correlation': spearman_corr,
            'p_value': spearman_p,
            'significant': spearman_p < 0.05
        }
        
        # Kendall's tau
        kendall_tau, kendall_p = stats.kendalltau(self.years, self.albedo)
        results['kendall'] = {
            'tau': kendall_tau,
            'p_value': kendall_p,
            'significant': kendall_p < 0.05
        }
        
        return results
    
    def temporal_pattern_analysis(self):
        """Analyze temporal patterns and persistence"""
        results = {}
        
        # Hurst exponent (persistence analysis)
        hurst_exp = self.calculate_hurst_exponent()
        results['hurst_exponent'] = hurst_exp
        
        # Detrended fluctuation analysis
        dfa_exp = self.detrended_fluctuation_analysis()
        results['dfa_exponent'] = dfa_exp
        
        # Run test for randomness
        runs_test = self.runs_test()
        results['runs_test'] = runs_test
        
        # Lag-1 autocorrelation
        lag1_autocorr = np.corrcoef(self.albedo[:-1], self.albedo[1:])[0, 1]
        results['lag1_autocorrelation'] = lag1_autocorr
        
        self.results['temporal_patterns'] = results
        return results
    
    def calculate_hurst_exponent(self):
        """Calculate Hurst exponent using R/S analysis"""
        n = len(self.albedo)
        if n < 10:
            return {'exponent': np.nan, 'interpretation': 'Insufficient data'}
        
        # R/S analysis
        lags = range(2, min(n//2, 20))
        rs_values = []
        
        for lag in lags:
            # Divide series into non-overlapping periods of length lag
            n_periods = n // lag
            if n_periods == 0:
                continue
                
            rs_period = []
            for i in range(n_periods):
                period_data = self.albedo[i*lag:(i+1)*lag]
                if len(period_data) == lag:
                    # Calculate cumulative deviations from mean
                    mean_period = np.mean(period_data)
                    cumdev = np.cumsum(period_data - mean_period)
                    
                    # Calculate range and standard deviation
                    R = np.max(cumdev) - np.min(cumdev)
                    S = np.std(period_data)
                    
                    if S > 0:
                        rs_period.append(R / S)
            
            if rs_period:
                rs_values.append(np.mean(rs_period))
            else:
                rs_values.append(np.nan)
        
        # Fit line to log(lag) vs log(R/S)
        valid_indices = ~np.isnan(rs_values)
        if np.sum(valid_indices) > 3:
            log_lags = np.log(np.array(lags)[valid_indices])
            log_rs = np.log(np.array(rs_values)[valid_indices])
            
            hurst, _ = np.polyfit(log_lags, log_rs, 1)
            
            if hurst < 0.5:
                interpretation = "Anti-persistent (mean-reverting)"
            elif hurst > 0.5:
                interpretation = "Persistent (trending)"
            else:
                interpretation = "Random walk"
        else:
            hurst = np.nan
            interpretation = "Cannot calculate"
        
        return {
            'exponent': hurst,
            'interpretation': interpretation,
            'rs_values': rs_values,
            'lags': list(lags)
        }
    
    def detrended_fluctuation_analysis(self):
        """Detrended Fluctuation Analysis (DFA)"""
        n = len(self.albedo)
        if n < 10:
            return {'exponent': np.nan}
        
        # Integrate the series
        y = np.cumsum(self.albedo - np.mean(self.albedo))
        
        # Define box sizes
        box_sizes = [int(n//(4*2**i)) for i in range(int(np.log2(n/4)))]
        box_sizes = [s for s in box_sizes if s >= 4]
        
        fluctuations = []
        
        for box_size in box_sizes:
            # Divide into non-overlapping boxes
            n_boxes = n // box_size
            
            detrended_fluct = []
            for i in range(n_boxes):
                start = i * box_size
                end = start + box_size
                box_data = y[start:end]
                
                # Fit linear trend
                x = np.arange(len(box_data))
                trend = np.polyfit(x, box_data, 1)
                detrended = box_data - np.polyval(trend, x)
                
                # Calculate fluctuation
                detrended_fluct.append(np.mean(detrended**2))
            
            fluctuations.append(np.sqrt(np.mean(detrended_fluct)))
        
        # Fit power law
        if len(fluctuations) > 3:
            log_sizes = np.log(box_sizes)
            log_flucts = np.log(fluctuations)
            dfa_exponent, _ = np.polyfit(log_sizes, log_flucts, 1)
        else:
            dfa_exponent = np.nan
        
        return {
            'exponent': dfa_exponent,
            'box_sizes': box_sizes,
            'fluctuations': fluctuations
        }
    
    def runs_test(self):
        """Wald-Wolfowitz runs test for randomness"""
        median = np.median(self.albedo)
        runs = []
        current_run = None
        
        for value in self.albedo:
            if value >= median:
                symbol = '+'
            else:
                symbol = '-'
            
            if symbol != current_run:
                runs.append(symbol)
                current_run = symbol
        
        n_runs = len(runs)
        n_plus = np.sum(self.albedo >= median)
        n_minus = len(self.albedo) - n_plus
        
        # Expected runs and variance under null hypothesis
        expected_runs = (2 * n_plus * n_minus) / (n_plus + n_minus) + 1
        variance_runs = (2 * n_plus * n_minus * (2 * n_plus * n_minus - n_plus - n_minus)) / \
                       ((n_plus + n_minus)**2 * (n_plus + n_minus - 1))
        
        # Z-score
        z_score = (n_runs - expected_runs) / np.sqrt(variance_runs) if variance_runs > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'n_runs': n_runs,
            'expected_runs': expected_runs,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Non-random' if p_value < 0.05 else 'Random'
        }
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        report = {
            'dataset_info': {
                'n_years': len(self.years),
                'start_year': int(self.years[0]),
                'end_year': int(self.years[-1]),
                'total_pixels': int(self.data['pixel_count'].sum()),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'key_findings': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        # Extract key findings from all analyses
        if 'distributional' in self.results:
            dist = self.results['distributional']['descriptive']
            report['key_findings']['descriptive'] = {
                'mean_albedo': round(dist['mean'], 4),
                'trend_magnitude': round(dist['mean'] - self.albedo[0], 4),
                'variability_cv': round(dist['cv'], 4),
                'distribution_shape': 'Skewed' if abs(dist['skewness']) > 0.5 else 'Normal-like'
            }
        
        if 'robust' in self.results:
            rob = self.results['robust']
            report['key_findings']['trend'] = {
                'theil_sen_slope': round(rob['robust_trend']['theil_sen_slope'], 6),
                'total_change': round(rob['robust_trend']['theil_sen_trend_total'], 4),
                'bootstrap_slope_ci': [round(x, 6) for x in rob['bootstrap']['slope_ci']]
            }
        
        if 'advanced_trend' in self.results:
            trend = self.results['advanced_trend']
            report['statistical_tests']['trend_significance'] = {
                'linear_regression_p': round(trend['trend_tests']['linear_regression']['p_value'], 6),
                'spearman_p': round(trend['trend_tests']['spearman']['p_value'], 6),
                'kendall_p': round(trend['trend_tests']['kendall']['p_value'], 6),
                'change_points_detected': trend['change_points']['n_change_points']
            }
        
        if 'temporal_patterns' in self.results:
            temp = self.results['temporal_patterns']
            report['key_findings']['persistence'] = {
                'hurst_exponent': round(temp['hurst_exponent']['exponent'], 3) if not np.isnan(temp['hurst_exponent']['exponent']) else 'N/A',
                'persistence_type': temp['hurst_exponent']['interpretation'],
                'lag1_autocorr': round(temp['lag1_autocorrelation'], 3)
            }
        
        # Add recommendations
        if 'distributional' in self.results:
            normality_p = self.results['distributional']['normality_tests']['shapiro_wilk']['p_value']
            if normality_p < 0.05:
                report['recommendations'].append("Use non-parametric statistical tests (data not normally distributed)")
            else:
                report['recommendations'].append("Parametric tests appropriate (data approximately normal)")
        
        if 'robust' in self.results:
            slope_ci = self.results['robust']['bootstrap']['slope_ci']
            if slope_ci[0] < 0 and slope_ci[1] < 0:
                report['recommendations'].append("Significant decreasing trend confirmed by robust methods")
            elif slope_ci[0] > 0 and slope_ci[1] > 0:
                report['recommendations'].append("Significant increasing trend confirmed by robust methods")
            else:
                report['recommendations'].append("Trend direction uncertain - confidence interval includes zero")
        
        return report

def create_statistical_plots(analyzer, save_dir):
    """Create comprehensive statistical visualization plots"""
    
    # 1. Distribution Analysis Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram with fitted distribution
    ax1.hist(analyzer.albedo, bins=8, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(analyzer.albedo)
    x = np.linspace(analyzer.albedo.min(), analyzer.albedo.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')
    ax1.set_title('Distribution Analysis', fontweight='bold')
    ax1.set_xlabel('Snow Albedo')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(analyzer.albedo, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Box plot with outlier detection
    box_plot = ax3.boxplot(analyzer.albedo, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightgreen')
    ax3.set_title('Box Plot with Outliers', fontweight='bold')
    ax3.set_ylabel('Snow Albedo')
    ax3.grid(True, alpha=0.3)
    
    # Time series with robust trend
    ax4.plot(analyzer.years, analyzer.albedo, 'o-', color='blue', linewidth=2, markersize=6, label='Observed')
    
    if 'robust' in analyzer.results:
        robust_trend = analyzer.results['robust']['robust_trend']
        slope = robust_trend['theil_sen_slope']
        intercept = robust_trend['theil_sen_intercept']
        trend_line = slope * analyzer.years + intercept
        ax4.plot(analyzer.years, trend_line, '--', color='red', linewidth=2, label=f'Theil-Sen trend: {slope:.6f}/year')
    
    ax4.set_title('Robust Trend Analysis', fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Snow Albedo')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Statistical Analysis - MODIS Albedo', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comprehensive_statistical_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Change Point Detection Plot
    if 'advanced_trend' in analyzer.results:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Pettitt test visualization
        pettitt = analyzer.results['advanced_trend']['pettitt_test']
        ax1.plot(analyzer.years, analyzer.albedo, 'o-', color='blue', linewidth=2, markersize=6)
        
        if pettitt['significant']:
            cp_year = pettitt['change_point_year']
            ax1.axvline(x=cp_year, color='red', linestyle='--', linewidth=2, 
                       label=f'Change point: {cp_year} (p={pettitt["p_value"]:.4f})')
        
        ax1.set_title('Pettitt Change Point Test', fontweight='bold')
        ax1.set_ylabel('Snow Albedo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CUSUM chart
        cusum = analyzer.results['advanced_trend']['cusum']
        ax2.plot(analyzer.years, cusum['cusum_positive'], 'g-', linewidth=2, label='CUSUM+')
        ax2.plot(analyzer.years, cusum['cusum_negative'], 'r-', linewidth=2, label='CUSUM-')
        ax2.axhline(y=cusum['threshold'], color='black', linestyle='--', alpha=0.7, label='Control limit')
        ax2.axhline(y=-cusum['threshold'], color='black', linestyle='--', alpha=0.7)
        
        ax2.set_title('CUSUM Control Chart', fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Sum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/change_point_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run comprehensive statistical analysis"""
    print("=== Comprehensive Statistical Analysis for MODIS Dataset ===")
    
    # Load data
    data = load_annual_data()
    print(f"Loaded {len(data)} years of data")
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(data)
    
    # Create output directory
    plot_dir = "/home/tofunori/Projects/MODIS Pixel analysis/statistical_analysis"
    os.makedirs(plot_dir, exist_ok=True)
    
    print("\nRunning comprehensive statistical analyses...")
    
    # Run all analyses
    print("1. Distributional analysis...")
    analyzer.distributional_analysis()
    
    print("2. Robust statistics...")
    analyzer.robust_statistics()
    
    print("3. Advanced trend analysis...")
    analyzer.advanced_trend_analysis()
    
    print("4. Temporal pattern analysis...")
    analyzer.temporal_pattern_analysis()
    
    print("5. Generating plots...")
    create_statistical_plots(analyzer, plot_dir)
    
    # Generate summary report
    print("6. Generating summary report...")
    report = analyzer.generate_summary_report()
    
    # Save detailed results
    import json
    with open(f"{plot_dir}/detailed_results.json", 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
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
                return str(obj)  # Convert any remaining objects to string
        
        json.dump(convert_numpy(analyzer.results), f, indent=2)
    
    # Save summary report
    with open(f"{plot_dir}/summary_report.json", 'w') as f:
        json.dump(convert_numpy(report), f, indent=2)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved in: {plot_dir}/")
    print("\nKey Findings:")
    
    # Print key results
    if 'distributional' in analyzer.results:
        desc = analyzer.results['distributional']['descriptive']
        print(f"- Mean albedo: {desc['mean']:.4f} ± {desc['std']:.4f}")
        print(f"- Coefficient of variation: {desc['cv']:.3f}")
        print(f"- Skewness: {desc['skewness']:.3f}")
    
    if 'robust' in analyzer.results:
        trend = analyzer.results['robust']['robust_trend']
        bootstrap = analyzer.results['robust']['bootstrap']
        print(f"- Theil-Sen slope: {trend['theil_sen_slope']:.6f} ± {bootstrap['slope_se']:.6f}")
        print(f"- Total change: {trend['theil_sen_trend_total']:.4f}")
    
    if 'temporal_patterns' in analyzer.results:
        hurst = analyzer.results['temporal_patterns']['hurst_exponent']
        print(f"- Hurst exponent: {hurst['exponent']:.3f} ({hurst['interpretation']})")
    
    print(f"\nDetailed results and plots available in: {plot_dir}/")

if __name__ == "__main__":
    main()