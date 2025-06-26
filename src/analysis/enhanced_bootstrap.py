#!/usr/bin/env python3

"""
Enhanced Bootstrap Methods for Autocorrelated Time Series
Implements block bootstrap methods to preserve serial correlation structure
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedBootstrap:
    """
    Enhanced bootstrap methods for time series with serial correlation
    Implements moving block, circular block, and stationary bootstrap
    """
    
    def __init__(self, data, block_length=None, n_bootstrap=10000):
        """
        Initialize bootstrap analyzer
        
        Parameters:
        -----------
        data : array-like
            Time series data
        block_length : int, optional
            Block length for block bootstrap (default: √n ≈ 4-5 for 15 years)
        n_bootstrap : int
            Number of bootstrap iterations
        """
        self.data = np.array(data)
        self.n = len(self.data)
        self.n_bootstrap = n_bootstrap
        
        # Optimal block length (reviewer recommendation: √N ≈ 4-5 years)
        if block_length is None:
            self.block_length = max(3, min(int(np.sqrt(self.n)), 5))
        else:
            self.block_length = block_length
            
        self.results = {}
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def moving_block_bootstrap(self, statistic_func, years=None):
        """
        Moving Block Bootstrap (MBB)
        
        Parameters:
        -----------
        statistic_func : callable
            Function to calculate statistic (e.g., slope, mean)
        years : array-like, optional
            Year values for trend calculation
        
        Returns:
        --------
        dict : Bootstrap results including confidence intervals
        """
        if years is None:
            years = np.arange(self.n)
        
        bootstrap_stats = []
        
        # Create overlapping blocks
        n_blocks_available = self.n - self.block_length + 1
        n_blocks_needed = int(np.ceil(self.n / self.block_length))
        
        for bootstrap_iter in range(self.n_bootstrap):
            # Sample blocks with replacement
            block_starts = np.random.choice(n_blocks_available, size=n_blocks_needed, replace=True)
            
            # Construct bootstrap sample
            bootstrap_sample = []
            bootstrap_years = []
            
            for i, block_start in enumerate(block_starts):
                block_end = min(block_start + self.block_length, self.n)
                block_data = self.data[block_start:block_end]
                block_years_data = years[block_start:block_end]
                
                # Truncate last block if necessary
                if len(bootstrap_sample) + len(block_data) > self.n:
                    remaining = self.n - len(bootstrap_sample)
                    block_data = block_data[:remaining]
                    block_years_data = block_years_data[:remaining]
                
                bootstrap_sample.extend(block_data)
                bootstrap_years.extend(block_years_data)
                
                if len(bootstrap_sample) >= self.n:
                    break
            
            # Ensure exact length
            bootstrap_sample = np.array(bootstrap_sample[:self.n])
            bootstrap_years = np.array(bootstrap_years[:self.n])
            
            # Calculate statistic
            try:
                if len(np.unique(bootstrap_years)) > 1:
                    stat = statistic_func(bootstrap_sample, bootstrap_years)
                    bootstrap_stats.append(stat)
                else:
                    # Handle case where all years are the same
                    bootstrap_stats.append(np.nan)
            except:
                bootstrap_stats.append(np.nan)
        
        # Remove NaN values
        bootstrap_stats = np.array(bootstrap_stats)
        bootstrap_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]
        
        if len(bootstrap_stats) == 0:
            return {
                'method': 'moving_block_bootstrap',
                'n_valid_samples': 0,
                'error': 'No valid bootstrap samples generated'
            }
        
        # Calculate confidence intervals and statistics
        results = self._calculate_bootstrap_statistics(bootstrap_stats, 'moving_block_bootstrap')
        results['block_length'] = self.block_length
        results['n_valid_samples'] = len(bootstrap_stats)
        
        return results
    
    def circular_block_bootstrap(self, statistic_func, years=None):
        """
        Circular Block Bootstrap (CBB)
        Treats the time series as circular to avoid edge effects
        """
        if years is None:
            years = np.arange(self.n)
        
        bootstrap_stats = []
        
        # Extend data circularly
        extended_data = np.concatenate([self.data, self.data[:self.block_length-1]])
        extended_years = np.concatenate([years, years[:self.block_length-1]])
        
        n_blocks_needed = int(np.ceil(self.n / self.block_length))
        
        for bootstrap_iter in range(self.n_bootstrap):
            # Sample blocks with replacement (can start anywhere in extended series)
            block_starts = np.random.choice(self.n, size=n_blocks_needed, replace=True)
            
            # Construct bootstrap sample
            bootstrap_sample = []
            bootstrap_years = []
            
            for block_start in block_starts:
                block_end = block_start + self.block_length
                block_data = extended_data[block_start:block_end]
                block_years_data = extended_years[block_start:block_end]
                
                # Truncate if necessary
                if len(bootstrap_sample) + len(block_data) > self.n:
                    remaining = self.n - len(bootstrap_sample)
                    block_data = block_data[:remaining]
                    block_years_data = block_years_data[:remaining]
                
                bootstrap_sample.extend(block_data)
                bootstrap_years.extend(block_years_data)
                
                if len(bootstrap_sample) >= self.n:
                    break
            
            # Ensure exact length
            bootstrap_sample = np.array(bootstrap_sample[:self.n])
            bootstrap_years = np.array(bootstrap_years[:self.n])
            
            # Calculate statistic
            try:
                if len(np.unique(bootstrap_years)) > 1:
                    stat = statistic_func(bootstrap_sample, bootstrap_years)
                    bootstrap_stats.append(stat)
                else:
                    bootstrap_stats.append(np.nan)
            except:
                bootstrap_stats.append(np.nan)
        
        # Remove NaN values
        bootstrap_stats = np.array(bootstrap_stats)
        bootstrap_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]
        
        if len(bootstrap_stats) == 0:
            return {
                'method': 'circular_block_bootstrap',
                'n_valid_samples': 0,
                'error': 'No valid bootstrap samples generated'
            }
        
        results = self._calculate_bootstrap_statistics(bootstrap_stats, 'circular_block_bootstrap')
        results['block_length'] = self.block_length
        results['n_valid_samples'] = len(bootstrap_stats)
        
        return results
    
    def stationary_bootstrap(self, statistic_func, years=None, p=None):
        """
        Stationary Bootstrap with geometric block lengths
        
        Parameters:
        -----------
        p : float, optional
            Probability parameter for geometric distribution (default: 1/block_length)
        """
        if years is None:
            years = np.arange(self.n)
        
        if p is None:
            p = 1.0 / self.block_length
        
        bootstrap_stats = []
        
        for bootstrap_iter in range(self.n_bootstrap):
            bootstrap_sample = []
            bootstrap_years = []
            
            while len(bootstrap_sample) < self.n:
                # Random starting point
                start_idx = np.random.randint(0, self.n)
                
                # Generate block length from geometric distribution
                block_len = np.random.geometric(p)
                
                # Extract block (with wrapping if necessary)
                for i in range(block_len):
                    if len(bootstrap_sample) >= self.n:
                        break
                    
                    idx = (start_idx + i) % self.n
                    bootstrap_sample.append(self.data[idx])
                    bootstrap_years.append(years[idx])
            
            # Ensure exact length
            bootstrap_sample = np.array(bootstrap_sample[:self.n])
            bootstrap_years = np.array(bootstrap_years[:self.n])
            
            # Calculate statistic
            try:
                if len(np.unique(bootstrap_years)) > 1:
                    stat = statistic_func(bootstrap_sample, bootstrap_years)
                    bootstrap_stats.append(stat)
                else:
                    bootstrap_stats.append(np.nan)
            except:
                bootstrap_stats.append(np.nan)
        
        # Remove NaN values
        bootstrap_stats = np.array(bootstrap_stats)
        bootstrap_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]
        
        if len(bootstrap_stats) == 0:
            return {
                'method': 'stationary_bootstrap',
                'n_valid_samples': 0,
                'error': 'No valid bootstrap samples generated'
            }
        
        results = self._calculate_bootstrap_statistics(bootstrap_stats, 'stationary_bootstrap')
        results['geometric_p'] = p
        results['expected_block_length'] = 1.0 / p
        results['n_valid_samples'] = len(bootstrap_stats)
        
        return results
    
    def bias_corrected_accelerated(self, bootstrap_stats, original_stat, 
                                    statistic_func, years=None, alpha=0.05):
        """
        Bias-Corrected and Accelerated (BCa) confidence intervals
        More accurate than percentile method for small samples
        """
        if years is None:
            years = np.arange(self.n)
        
        n_boot = len(bootstrap_stats)
        
        # Bias correction
        z0 = stats.norm.ppf((bootstrap_stats < original_stat).mean())
        
        # Acceleration parameter using jackknife
        jackknife_stats = []
        for i in range(self.n):
            # Leave-one-out sample
            jackknife_data = np.concatenate([self.data[:i], self.data[i+1:]])
            jackknife_years = np.concatenate([years[:i], years[i+1:]])
            
            try:
                if len(np.unique(jackknife_years)) > 1:
                    jack_stat = statistic_func(jackknife_data, jackknife_years)
                    jackknife_stats.append(jack_stat)
                else:
                    jackknife_stats.append(np.nan)
            except:
                jackknife_stats.append(np.nan)
        
        jackknife_stats = np.array(jackknife_stats)
        valid_jack = jackknife_stats[~np.isnan(jackknife_stats)]
        
        if len(valid_jack) > 3:
            jack_mean = np.mean(valid_jack)
            numerator = np.sum((jack_mean - valid_jack)**3)
            denominator = 6 * (np.sum((jack_mean - valid_jack)**2))**1.5
            
            if denominator != 0:
                acceleration = numerator / denominator
            else:
                acceleration = 0
        else:
            acceleration = 0
        
        # BCa confidence intervals
        z_alpha_2 = stats.norm.ppf(alpha/2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha/2)
        
        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha_2)/(1 - acceleration*(z0 + z_alpha_2)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2)/(1 - acceleration*(z0 + z_1_alpha_2)))
        
        # Ensure valid quantiles
        alpha1 = max(0, min(1, alpha1))
        alpha2 = max(0, min(1, alpha2))
        
        if alpha1 < alpha2:
            bca_lower = np.percentile(bootstrap_stats, 100 * alpha1)
            bca_upper = np.percentile(bootstrap_stats, 100 * alpha2)
        else:
            # Fallback to percentile method
            bca_lower = np.percentile(bootstrap_stats, 100 * alpha/2)
            bca_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
        
        return {
            'bca_ci': [bca_lower, bca_upper],
            'bias_correction': z0,
            'acceleration': acceleration,
            'alpha1': alpha1,
            'alpha2': alpha2
        }
    
    def _calculate_bootstrap_statistics(self, bootstrap_stats, method_name):
        """Calculate comprehensive bootstrap statistics"""
        results = {
            'method': method_name,
            'n_bootstrap': len(bootstrap_stats),
            'mean': np.mean(bootstrap_stats),
            'std': np.std(bootstrap_stats),
            'bias': np.mean(bootstrap_stats) - np.mean(self.data) if method_name == 'mean' else 0,
            'percentile_ci_95': [np.percentile(bootstrap_stats, 2.5), 
                                np.percentile(bootstrap_stats, 97.5)],
            'percentile_ci_90': [np.percentile(bootstrap_stats, 5), 
                                np.percentile(bootstrap_stats, 95)],
            'percentile_ci_99': [np.percentile(bootstrap_stats, 0.5), 
                                np.percentile(bootstrap_stats, 99.5)],
            'bootstrap_distribution': bootstrap_stats.tolist()
        }
        
        return results
    
    def slope_statistic(self, data, years):
        """Calculate slope for bootstrap"""
        if len(np.unique(years)) <= 1:
            return np.nan
        try:
            slope, _ = np.polyfit(years, data, 1)
            return slope
        except:
            return np.nan
    
    def mean_statistic(self, data, years):
        """Calculate mean for bootstrap"""
        return np.mean(data)
    
    def std_statistic(self, data, years):
        """Calculate standard deviation for bootstrap"""
        return np.std(data)
    
    def theil_sen_statistic(self, data, years):
        """Calculate Theil-Sen slope for bootstrap"""
        if len(np.unique(years)) <= 1:
            return np.nan
        
        slopes = []
        n = len(data)
        
        for i in range(n):
            for j in range(i + 1, n):
                if years[j] != years[i]:
                    slope = (data[j] - data[i]) / (years[j] - years[i])
                    slopes.append(slope)
        
        return np.median(slopes) if slopes else np.nan
    
    def comprehensive_bootstrap_analysis(self, years=None, statistics=['slope', 'mean', 'std']):
        """
        Run comprehensive bootstrap analysis with all methods
        
        Parameters:
        -----------
        years : array-like, optional
            Year values for trend calculation
        statistics : list
            List of statistics to bootstrap ['slope', 'mean', 'std', 'theil_sen']
        
        Returns:
        --------
        dict : Comprehensive bootstrap results
        """
        if years is None:
            years = np.arange(self.n)
        
        results = {
            'block_length_used': self.block_length,
            'n_bootstrap_iterations': self.n_bootstrap,
            'methods': {}
        }
        
        # Define statistic functions
        stat_functions = {
            'slope': self.slope_statistic,
            'mean': self.mean_statistic,
            'std': self.std_statistic,
            'theil_sen': self.theil_sen_statistic
        }
        
        # Bootstrap methods to apply
        methods = {
            'moving_block': self.moving_block_bootstrap,
            'circular_block': self.circular_block_bootstrap,
            'stationary': self.stationary_bootstrap
        }
        
        for stat_name in statistics:
            if stat_name not in stat_functions:
                continue
                
            stat_func = stat_functions[stat_name]
            results['methods'][stat_name] = {}
            
            # Calculate original statistic
            original_stat = stat_func(self.data, years)
            results['methods'][stat_name]['original_value'] = original_stat
            
            # Apply each bootstrap method
            for method_name, method_func in methods.items():
                print(f"Running {method_name} bootstrap for {stat_name}...")
                
                try:
                    if method_name == 'stationary':
                        method_results = method_func(stat_func, years)
                    else:
                        method_results = method_func(stat_func, years)
                    
                    results['methods'][stat_name][method_name] = method_results
                    
                    # Add BCa confidence intervals for slope
                    if stat_name == 'slope' and 'bootstrap_distribution' in method_results:
                        bootstrap_dist = np.array(method_results['bootstrap_distribution'])
                        bca_results = self.bias_corrected_accelerated(
                            bootstrap_dist, original_stat, stat_func, years)
                        results['methods'][stat_name][method_name]['bca'] = bca_results
                    
                except Exception as e:
                    results['methods'][stat_name][method_name] = {
                        'error': str(e),
                        'method': method_name
                    }
        
        self.results = results
        return results
    
    def generate_bootstrap_summary(self):
        """Generate summary of bootstrap results for reporting"""
        if not hasattr(self, 'results') or not self.results:
            return {'error': 'No bootstrap results available'}
        
        summary = {
            'configuration': {
                'block_length': self.block_length,
                'n_bootstrap': self.n_bootstrap,
                'series_length': self.n
            },
            'recommendations': [],
            'method_comparison': {}
        }
        
        # Compare methods for slope estimation
        if 'slope' in self.results['methods']:
            slope_results = self.results['methods']['slope']
            original_slope = slope_results.get('original_value', np.nan)
            
            summary['method_comparison']['slope'] = {
                'original_value': round(original_slope, 6) if not np.isnan(original_slope) else 'N/A'
            }
            
            for method in ['moving_block', 'circular_block', 'stationary']:
                if method in slope_results and 'percentile_ci_95' in slope_results[method]:
                    ci = slope_results[method]['percentile_ci_95']
                    mean_est = slope_results[method]['mean']
                    
                    summary['method_comparison']['slope'][method] = {
                        'mean_estimate': round(mean_est, 6),
                        'ci_95': [round(ci[0], 6), round(ci[1], 6)],
                        'ci_width': round(ci[1] - ci[0], 6)
                    }
                    
                    # BCa results if available
                    if 'bca' in slope_results[method]:
                        bca_ci = slope_results[method]['bca']['bca_ci']
                        summary['method_comparison']['slope'][method]['bca_ci_95'] = [
                            round(bca_ci[0], 6), round(bca_ci[1], 6)]
        
        # Generate recommendations
        if self.block_length >= 4:
            summary['recommendations'].append("Block length appropriately sized for serial correlation preservation")
        else:
            summary['recommendations'].append("Consider increasing block length for better serial correlation preservation")
        
        if self.n_bootstrap >= 10000:
            summary['recommendations'].append("Sufficient bootstrap iterations for stable results")
        else:
            summary['recommendations'].append("Consider increasing bootstrap iterations for more stable results")
        
        return summary

def main():
    """Example usage of enhanced bootstrap methods"""
    # This would typically be called from the main analysis script
    pass

if __name__ == "__main__":
    main()