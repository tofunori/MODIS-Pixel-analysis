#!/usr/bin/env python3

"""
Seasonal Mann-Kendall Test Implementation
Addresses reviewer feedback on seasonal analysis capabilities
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SeasonalMannKendall:
    """
    Seasonal Mann-Kendall test for trend detection in seasonal time series
    Improves power over annual aggregation by considering seasonal patterns
    """
    
    def __init__(self, data, time_index=None, seasonal_periods=None):
        """
        Initialize Seasonal Mann-Kendall analyzer
        
        Parameters:
        -----------
        data : array-like or pandas Series
            Time series data
        time_index : array-like or pandas DatetimeIndex, optional
            Time index for the data
        seasonal_periods : int or list, optional
            Number of seasons per year (e.g., 12 for monthly, 4 for quarterly)
        """
        if isinstance(data, pd.Series):
            self.data = data.values
            self.time_index = data.index if time_index is None else pd.to_datetime(time_index)
        else:
            self.data = np.array(data)
            self.time_index = pd.to_datetime(time_index) if time_index is not None else pd.date_range('2010', periods=len(data), freq='AS')
        
        self.n = len(self.data)
        
        # Default to monthly analysis if not specified
        if seasonal_periods is None:
            seasonal_periods = 12
        
        self.seasonal_periods = seasonal_periods
        self.results = {}
        
    def seasonal_mann_kendall_test(self, remove_trend_within_seasons=True):
        """
        Perform Seasonal Mann-Kendall test
        
        Parameters:
        -----------
        remove_trend_within_seasons : bool
            Whether to remove trend within each season before testing
        
        Returns:
        --------
        dict : Seasonal Mann-Kendall test results
        """
        results = {
            'method': 'seasonal_mann_kendall',
            'seasonal_periods': self.seasonal_periods,
            'remove_trend_within_seasons': remove_trend_within_seasons
        }
        
        # Create seasonal decomposition
        seasonal_data = self._create_seasonal_structure()
        results['seasonal_structure'] = {
            'seasons_detected': len(seasonal_data),
            'observations_per_season': {season: len(data) for season, data in seasonal_data.items()}
        }
        
        # Apply trend removal if requested
        if remove_trend_within_seasons:
            seasonal_data = self._remove_within_season_trends(seasonal_data)
            results['trend_removal_applied'] = True
        else:
            results['trend_removal_applied'] = False
        
        # Calculate seasonal S statistics
        seasonal_S = {}
        seasonal_var_S = {}
        
        for season, season_data in seasonal_data.items():
            if len(season_data) > 1:
                S, var_S = self._calculate_mann_kendall_components(season_data)
                seasonal_S[season] = S
                seasonal_var_S[season] = var_S
            else:
                seasonal_S[season] = 0
                seasonal_var_S[season] = 0
        
        results['seasonal_statistics'] = {
            'seasonal_S': seasonal_S,
            'seasonal_var_S': seasonal_var_S
        }
        
        # Combined test statistic
        S_total = sum(seasonal_S.values())
        var_S_total = sum(seasonal_var_S.values())
        
        results['combined_statistic'] = S_total
        results['combined_variance'] = var_S_total
        
        # Calculate Z-statistic and p-value
        if var_S_total > 0:
            if S_total > 0:
                Z = (S_total - 1) / np.sqrt(var_S_total)
            elif S_total < 0:
                Z = (S_total + 1) / np.sqrt(var_S_total)
            else:
                Z = 0
            
            p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        else:
            Z = np.nan
            p_value = np.nan
        
        results['z_statistic'] = Z
        results['p_value'] = p_value
        results['significant'] = p_value < 0.05 if not np.isnan(p_value) else False
        
        # Calculate seasonal trend direction
        results['trend_direction'] = 'increasing' if S_total > 0 else 'decreasing' if S_total < 0 else 'no trend'
        
        # Individual season significance
        season_tests = {}
        for season in seasonal_S.keys():
            if seasonal_var_S[season] > 0:
                if seasonal_S[season] > 0:
                    z_season = (seasonal_S[season] - 1) / np.sqrt(seasonal_var_S[season])
                elif seasonal_S[season] < 0:
                    z_season = (seasonal_S[season] + 1) / np.sqrt(seasonal_var_S[season])
                else:
                    z_season = 0
                
                p_season = 2 * (1 - stats.norm.cdf(abs(z_season)))
                
                season_tests[season] = {
                    'S': seasonal_S[season],
                    'Z': z_season,
                    'p_value': p_season,
                    'significant': p_season < 0.05
                }
            else:
                season_tests[season] = {
                    'S': seasonal_S[season],
                    'Z': np.nan,
                    'p_value': np.nan,
                    'significant': False
                }
        
        results['seasonal_tests'] = season_tests
        
        self.results['seasonal_mann_kendall'] = results
        return results
    
    def seasonal_sens_slope(self):
        """
        Calculate seasonal Sen's slope estimates
        
        Returns:
        --------
        dict : Seasonal slope estimates
        """
        results = {
            'method': 'seasonal_sens_slope',
            'seasonal_slopes': {},
            'overall_slope': None
        }
        
        seasonal_data = self._create_seasonal_structure()
        all_slopes = []
        
        for season, season_data in seasonal_data.items():
            if len(season_data) >= 2:
                # Calculate Sen's slope for this season
                slopes = []
                n = len(season_data)
                
                for i in range(n):
                    for j in range(i + 1, n):
                        # Use time difference in years
                        time_diff = (season_data.index[j] - season_data.index[i]).days / 365.25
                        if time_diff > 0:
                            slope = (season_data.iloc[j] - season_data.iloc[i]) / time_diff
                            slopes.append(slope)
                
                if slopes:
                    seasonal_slope = np.median(slopes)
                    results['seasonal_slopes'][season] = {
                        'slope': seasonal_slope,
                        'n_pairs': len(slopes)
                    }
                    all_slopes.extend(slopes)
                else:
                    results['seasonal_slopes'][season] = {
                        'slope': np.nan,
                        'n_pairs': 0
                    }
            else:
                results['seasonal_slopes'][season] = {
                    'slope': np.nan,
                    'n_pairs': 0
                }
        
        # Overall slope estimate
        if all_slopes:
            results['overall_slope'] = np.median(all_slopes)
        else:
            results['overall_slope'] = np.nan
        
        self.results['seasonal_sens_slope'] = results
        return results
    
    def _create_seasonal_structure(self):
        """
        Create seasonal data structure based on the time index
        
        Returns:
        --------
        dict : Dictionary with seasonal data
        """
        # Convert to pandas Series for easier manipulation
        ts = pd.Series(self.data, index=self.time_index)
        
        # Group by season based on seasonal_periods
        if self.seasonal_periods == 12:
            # Monthly seasons
            seasonal_data = {}
            for month in range(1, 13):
                month_data = ts[ts.index.month == month]
                if len(month_data) > 0:
                    seasonal_data[f'month_{month:02d}'] = month_data
        elif self.seasonal_periods == 4:
            # Quarterly seasons
            seasonal_data = {}
            quarter_months = {1: [12, 1, 2], 2: [3, 4, 5], 3: [6, 7, 8], 4: [9, 10, 11]}
            for quarter, months in quarter_months.items():
                quarter_data = ts[ts.index.month.isin(months)]
                if len(quarter_data) > 0:
                    seasonal_data[f'quarter_{quarter}'] = quarter_data
        else:
            # Custom seasonal periods - use modulo approach
            seasonal_data = {}
            day_of_year = ts.index.dayofyear
            season_length = 365 // self.seasonal_periods
            
            for season in range(self.seasonal_periods):
                start_day = season * season_length + 1
                end_day = (season + 1) * season_length
                
                if season == self.seasonal_periods - 1:
                    # Last season includes remaining days
                    season_mask = day_of_year >= start_day
                else:
                    season_mask = (day_of_year >= start_day) & (day_of_year <= end_day)
                
                season_data = ts[season_mask]
                if len(season_data) > 0:
                    seasonal_data[f'season_{season+1:02d}'] = season_data
        
        return seasonal_data
    
    def _remove_within_season_trends(self, seasonal_data):
        """
        Remove linear trends within each season
        
        Parameters:
        -----------
        seasonal_data : dict
            Dictionary of seasonal time series
        
        Returns:
        --------
        dict : Detrended seasonal data
        """
        detrended_data = {}
        
        for season, season_ts in seasonal_data.items():
            if len(season_ts) > 2:
                # Convert dates to numeric values for regression
                numeric_time = np.array([(date - season_ts.index[0]).days for date in season_ts.index])
                
                try:
                    # Fit linear trend
                    slope, intercept = np.polyfit(numeric_time, season_ts.values, 1)
                    
                    # Remove trend
                    trend = slope * numeric_time + intercept
                    detrended_values = season_ts.values - trend + np.mean(season_ts.values)
                    
                    # Create detrended series
                    detrended_series = pd.Series(detrended_values, index=season_ts.index)
                    detrended_data[season] = detrended_series
                    
                except:
                    # If detrending fails, use original data
                    detrended_data[season] = season_ts
            else:
                # Insufficient data for detrending
                detrended_data[season] = season_ts
        
        return detrended_data
    
    def _calculate_mann_kendall_components(self, series):
        """
        Calculate Mann-Kendall S statistic and variance for a single series
        
        Parameters:
        -----------
        series : pandas Series
            Time series data
        
        Returns:
        --------
        tuple : (S statistic, variance of S)
        """
        data = series.values
        n = len(data)
        
        if n < 2:
            return 0, 0
        
        # Calculate S statistic
        S = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    S += 1
                elif data[j] < data[i]:
                    S -= 1
        
        # Calculate variance of S
        # Account for ties
        unique_values, counts = np.unique(data, return_counts=True)
        
        # Tie correction
        tie_correction = 0
        for count in counts:
            if count > 1:
                tie_correction += count * (count - 1) * (2 * count + 5)
        
        var_S = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18
        
        return S, var_S
    
    def monthly_mann_kendall_test(self):
        """
        Convenience method for monthly Mann-Kendall test
        """
        original_periods = self.seasonal_periods
        self.seasonal_periods = 12
        
        try:
            result = self.seasonal_mann_kendall_test()
            result['method'] = 'monthly_mann_kendall'
            return result
        finally:
            self.seasonal_periods = original_periods
    
    def quarterly_mann_kendall_test(self):
        """
        Convenience method for quarterly Mann-Kendall test
        """
        original_periods = self.seasonal_periods
        self.seasonal_periods = 4
        
        try:
            result = self.seasonal_mann_kendall_test()
            result['method'] = 'quarterly_mann_kendall'
            return result
        finally:
            self.seasonal_periods = original_periods
    
    def generate_seasonal_summary(self):
        """
        Generate comprehensive summary of seasonal analysis
        """
        summary = {
            'analysis_type': f'{self.seasonal_periods}_period_seasonal',
            'data_characteristics': {
                'total_observations': self.n,
                'time_span': {
                    'start': str(self.time_index[0].date()),
                    'end': str(self.time_index[-1].date())
                }
            }
        }
        
        # Seasonal Mann-Kendall results
        if 'seasonal_mann_kendall' in self.results:
            smk = self.results['seasonal_mann_kendall']
            summary['trend_detection'] = {
                'overall_significant': smk['significant'],
                'p_value': smk['p_value'],
                'trend_direction': smk['trend_direction'],
                'z_statistic': smk['z_statistic']
            }
            
            # Season-specific results
            significant_seasons = []
            for season, test in smk['seasonal_tests'].items():
                if test['significant']:
                    significant_seasons.append({
                        'season': season,
                        'p_value': test['p_value'],
                        'trend': 'increasing' if test['S'] > 0 else 'decreasing'
                    })
            
            summary['seasonal_patterns'] = {
                'seasons_with_significant_trends': len(significant_seasons),
                'significant_seasons': significant_seasons
            }
        
        # Seasonal slope results
        if 'seasonal_sens_slope' in self.results:
            sss = self.results['seasonal_sens_slope']
            summary['slope_estimates'] = {
                'overall_seasonal_slope': sss['overall_slope'],
                'seasonal_slopes': sss['seasonal_slopes']
            }
        
        return summary
    
    def compare_with_annual_test(self, annual_data, annual_years):
        """
        Compare seasonal results with annual Mann-Kendall test
        
        Parameters:
        -----------
        annual_data : array-like
            Annual aggregated data
        annual_years : array-like
            Years for annual data
        
        Returns:
        --------
        dict : Comparison results
        """
        # Calculate annual Mann-Kendall
        annual_S, annual_var_S = self._calculate_mann_kendall_components(
            pd.Series(annual_data, index=annual_years))
        
        if annual_var_S > 0:
            if annual_S > 0:
                annual_Z = (annual_S - 1) / np.sqrt(annual_var_S)
            elif annual_S < 0:
                annual_Z = (annual_S + 1) / np.sqrt(annual_var_S)
            else:
                annual_Z = 0
            
            annual_p = 2 * (1 - stats.norm.cdf(abs(annual_Z)))
        else:
            annual_Z = np.nan
            annual_p = np.nan
        
        comparison = {
            'annual_test': {
                'S': annual_S,
                'Z': annual_Z,
                'p_value': annual_p,
                'significant': annual_p < 0.05 if not np.isnan(annual_p) else False
            }
        }
        
        # Add seasonal results if available
        if 'seasonal_mann_kendall' in self.results:
            smk = self.results['seasonal_mann_kendall']
            comparison['seasonal_test'] = {
                'S': smk['combined_statistic'],
                'Z': smk['z_statistic'],
                'p_value': smk['p_value'],
                'significant': smk['significant']
            }
            
            # Power comparison
            comparison['power_comparison'] = {
                'seasonal_more_significant': (smk['p_value'] < annual_p) if not np.isnan(annual_p) and not np.isnan(smk['p_value']) else False,
                'p_value_improvement': (annual_p - smk['p_value']) if not np.isnan(annual_p) and not np.isnan(smk['p_value']) else np.nan
            }
        
        return comparison

def main():
    """Example usage - typically called from main analysis"""
    pass

if __name__ == "__main__":
    main()