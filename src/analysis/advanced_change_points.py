#!/usr/bin/env python3

"""
Advanced Change Point Detection Methods
Implements Bai-Perron multiple break testing and segmented regression
Addresses reviewer feedback on change point analysis improvements
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedChangePointDetector:
    """
    Advanced change point detection using multiple methodologies
    Implements Bai-Perron test, segmented regression, and confidence intervals
    """
    
    def __init__(self, data, years=None, max_breaks=3):
        """
        Initialize change point detector
        
        Parameters:
        -----------
        data : array-like
            Time series data
        years : array-like, optional
            Time points (default: sequential integers)
        max_breaks : int
            Maximum number of breaks to consider
        """
        self.data = np.array(data)
        self.years = np.array(years) if years is not None else np.arange(len(data))
        self.n = len(self.data)
        self.max_breaks = max_breaks
        self.results = {}
        
        # Minimum segment size (typically 15% of series or 3 observations)
        self.min_segment_size = max(3, int(0.15 * self.n))
    
    def bai_perron_test(self, trimming=0.15, max_breaks=None):
        """
        Bai-Perron multiple structural break test
        
        Parameters:
        -----------
        trimming : float
            Fraction of sample to trim from each end (default 0.15)
        max_breaks : int, optional
            Maximum breaks to test (default: self.max_breaks)
        
        Returns:
        --------
        dict : Bai-Perron test results
        """
        if max_breaks is None:
            max_breaks = self.max_breaks
        
        results = {
            'method': 'bai_perron',
            'trimming': trimming,
            'max_breaks_tested': max_breaks,
            'models': {},
            'break_dates': {},
            'information_criteria': {},
            'sequential_test': {}
        }
        
        # Determine break search range
        start_idx = int(trimming * self.n)
        end_idx = self.n - int(trimming * self.n)
        potential_breaks = list(range(start_idx, end_idx))
        
        # Test models with 0 to max_breaks breaks
        for m in range(max_breaks + 1):
            if m == 0:
                # No break model
                model_result = self._fit_linear_model(self.data, self.years)
                results['models'][m] = model_result
                results['break_dates'][m] = []
                
            else:
                # Find optimal break points for m breaks
                best_breaks, best_ssr = self._find_optimal_breaks(m, potential_breaks)
                
                if best_breaks is not None:
                    # Fit segmented model
                    segments = self._create_segments(best_breaks)
                    model_result = self._fit_segmented_model(segments)
                    
                    results['models'][m] = model_result
                    results['break_dates'][m] = [self.years[bp] for bp in best_breaks]
                    
                else:
                    results['models'][m] = {'error': 'Could not find valid breaks'}
                    results['break_dates'][m] = []
        
        # Calculate information criteria
        results['information_criteria'] = self._calculate_information_criteria(results['models'])
        
        # Select optimal number of breaks
        optimal_breaks = self._select_optimal_breaks(results['information_criteria'])
        results['optimal_breaks'] = optimal_breaks
        
        # Sequential F-test
        results['sequential_test'] = self._sequential_f_test(results['models'])
        
        self.results['bai_perron'] = results
        return results
    
    def segmented_regression(self, initial_breaks=None, confidence_level=0.95):
        """
        Segmented regression with simultaneous break detection and slope estimation
        
        Parameters:
        -----------
        initial_breaks : list, optional
            Initial guess for break points
        confidence_level : float
            Confidence level for break point intervals
        
        Returns:
        --------
        dict : Segmented regression results with confidence intervals
        """
        results = {
            'method': 'segmented_regression',
            'confidence_level': confidence_level
        }
        
        if initial_breaks is None:
            # Use Bai-Perron results if available, otherwise use simple detection
            if 'bai_perron' in self.results and 'optimal_breaks' in self.results['bai_perron']:
                optimal_m = self.results['bai_perron']['optimal_breaks']
                if optimal_m > 0 and optimal_m in self.results['bai_perron']['break_dates']:
                    break_years = self.results['bai_perron']['break_dates'][optimal_m]
                    initial_breaks = [np.where(self.years == year)[0][0] for year in break_years if len(np.where(self.years == year)[0]) > 0]
                else:
                    initial_breaks = []
            else:
                initial_breaks = self._simple_break_detection()
        
        results['initial_breaks'] = initial_breaks
        
        if not initial_breaks:
            # No breaks - fit simple linear model
            model = self._fit_linear_model(self.data, self.years)
            results['segments'] = [model]
            results['break_points'] = []
            results['break_confidence_intervals'] = []
            
        else:
            # Optimize break points and fit segmented model
            optimized_breaks = self._optimize_break_points(initial_breaks)
            results['optimized_breaks'] = optimized_breaks
            
            # Create segments and fit model
            segments = self._create_segments(optimized_breaks)
            segmented_model = self._fit_segmented_model(segments)
            
            results['segments'] = segmented_model['segments']
            results['break_points'] = [self.years[bp] for bp in optimized_breaks]
            results['global_ssr'] = segmented_model['global_ssr']
            results['global_r_squared'] = segmented_model['global_r_squared']
            
            # Calculate confidence intervals for break points
            break_cis = self._break_point_confidence_intervals(optimized_breaks, confidence_level)
            results['break_confidence_intervals'] = break_cis
            
            # Structural stability tests
            stability_tests = self._structural_stability_tests(segments)
            results['stability_tests'] = stability_tests
        
        self.results['segmented_regression'] = results
        return results
    
    def _find_optimal_breaks(self, m, potential_breaks):
        """Find optimal placement of m breaks"""
        if m == 0:
            return [], None
        
        min_ssr = np.inf
        best_breaks = None
        
        # For small m, use exhaustive search; for larger m, use greedy approach
        if m <= 2 and len(potential_breaks) <= 20:
            # Exhaustive search for small problems
            from itertools import combinations
            
            for break_combination in combinations(potential_breaks, m):
                if self._valid_break_combination(break_combination):
                    ssr = self._calculate_ssr_for_breaks(break_combination)
                    if ssr < min_ssr:
                        min_ssr = ssr
                        best_breaks = list(break_combination)
        else:
            # Greedy approach for larger problems
            best_breaks = self._greedy_break_selection(m, potential_breaks)
            if best_breaks:
                min_ssr = self._calculate_ssr_for_breaks(best_breaks)
        
        return best_breaks, min_ssr
    
    def _greedy_break_selection(self, m, potential_breaks):
        """Greedy algorithm for break selection"""
        selected_breaks = []
        
        for _ in range(m):
            best_ssr = np.inf
            best_new_break = None
            
            for candidate_break in potential_breaks:
                if candidate_break not in selected_breaks:
                    test_breaks = sorted(selected_breaks + [candidate_break])
                    
                    if self._valid_break_combination(test_breaks):
                        ssr = self._calculate_ssr_for_breaks(test_breaks)
                        if ssr < best_ssr:
                            best_ssr = ssr
                            best_new_break = candidate_break
            
            if best_new_break is not None:
                selected_breaks.append(best_new_break)
            else:
                break
        
        return sorted(selected_breaks) if selected_breaks else None
    
    def _valid_break_combination(self, breaks):
        """Check if break combination produces valid segments"""
        if not breaks:
            return True
        
        # Check minimum segment sizes
        segment_starts = [0] + list(breaks)
        segment_ends = list(breaks) + [self.n]
        
        for start, end in zip(segment_starts, segment_ends):
            if end - start < self.min_segment_size:
                return False
        
        return True
    
    def _calculate_ssr_for_breaks(self, breaks):
        """Calculate sum of squared residuals for given break configuration"""
        segments = self._create_segments(breaks)
        total_ssr = 0
        
        for segment in segments:
            start_idx, end_idx = segment
            y_seg = self.data[start_idx:end_idx]
            x_seg = self.years[start_idx:end_idx]
            
            if len(y_seg) > 1 and len(np.unique(x_seg)) > 1:
                # Fit linear model to segment
                slope, intercept = np.polyfit(x_seg, y_seg, 1)
                y_pred = slope * x_seg + intercept
                ssr = np.sum((y_seg - y_pred)**2)
                total_ssr += ssr
            else:
                total_ssr += np.sum((y_seg - np.mean(y_seg))**2)
        
        return total_ssr
    
    def _create_segments(self, breaks):
        """Create segment indices from break points"""
        if not breaks:
            return [(0, self.n)]
        
        segments = []
        starts = [0] + list(breaks)
        ends = list(breaks) + [self.n]
        
        for start, end in zip(starts, ends):
            segments.append((start, end))
        
        return segments
    
    def _fit_linear_model(self, y, x):
        """Fit simple linear model"""
        if len(np.unique(x)) <= 1:
            return {
                'slope': 0,
                'intercept': np.mean(y),
                'r_squared': 0,
                'ssr': np.sum((y - np.mean(y))**2),
                'n_obs': len(y),
                'error': 'Insufficient x variation'
            }
        
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        ssr = np.sum((y - y_pred)**2)
        tss = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ssr/tss if tss > 0 else 0
        
        # Standard errors
        df = len(y) - 2
        mse = ssr / df if df > 0 else np.inf
        
        x_centered = x - np.mean(x)
        slope_se = np.sqrt(mse / np.sum(x_centered**2)) if np.sum(x_centered**2) > 0 else np.inf
        
        # t-statistic and p-value for slope
        t_stat = slope / slope_se if slope_se > 0 and slope_se < np.inf else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if df > 0 else 1
        
        return {
            'slope': slope,
            'intercept': intercept,
            'slope_se': slope_se,
            'slope_t': t_stat,
            'slope_p': p_value,
            'r_squared': r_squared,
            'ssr': ssr,
            'mse': mse,
            'n_obs': len(y),
            'fitted_values': y_pred,
            'residuals': y - y_pred
        }
    
    def _fit_segmented_model(self, segments):
        """Fit segmented linear model"""
        segment_models = []
        total_ssr = 0
        total_tss = 0
        total_n = 0
        
        for i, (start_idx, end_idx) in enumerate(segments):
            y_seg = self.data[start_idx:end_idx]
            x_seg = self.years[start_idx:end_idx]
            
            # Fit model to segment
            model = self._fit_linear_model(y_seg, x_seg)
            model['segment_id'] = i
            model['start_year'] = x_seg[0]
            model['end_year'] = x_seg[-1]
            model['start_idx'] = start_idx
            model['end_idx'] = end_idx
            
            segment_models.append(model)
            total_ssr += model['ssr']
            total_tss += np.sum((y_seg - np.mean(self.data))**2)
            total_n += model['n_obs']
        
        # Global R-squared
        global_r_squared = 1 - total_ssr/total_tss if total_tss > 0 else 0
        
        return {
            'segments': segment_models,
            'global_ssr': total_ssr,
            'global_r_squared': global_r_squared,
            'total_observations': total_n,
            'n_segments': len(segment_models)
        }
    
    def _optimize_break_points(self, initial_breaks):
        """Refine break point locations using optimization"""
        if not initial_breaks:
            return []
        
        def objective(break_positions):
            # Convert continuous positions to integer indices
            break_indices = [max(self.min_segment_size, 
                               min(self.n - self.min_segment_size, int(pos))) 
                           for pos in break_positions]
            
            # Ensure breaks are sorted and valid
            break_indices = sorted(list(set(break_indices)))
            
            if not self._valid_break_combination(break_indices):
                return np.inf
            
            return self._calculate_ssr_for_breaks(break_indices)
        
        # Set up optimization bounds
        bounds = []
        for break_idx in initial_breaks:
            lower = max(self.min_segment_size, break_idx - 2)
            upper = min(self.n - self.min_segment_size, break_idx + 2)
            bounds.append((lower, upper))
        
        try:
            from scipy.optimize import differential_evolution
            
            result = differential_evolution(objective, bounds, seed=42, maxiter=100)
            
            if result.success:
                optimized_breaks = [max(self.min_segment_size, 
                                      min(self.n - self.min_segment_size, int(pos))) 
                                  for pos in result.x]
                return sorted(list(set(optimized_breaks)))
            else:
                return initial_breaks
        except:
            # Fallback to initial breaks if optimization fails
            return initial_breaks
    
    def _break_point_confidence_intervals(self, breaks, confidence_level):
        """Calculate confidence intervals for break points using bootstrap"""
        if not breaks:
            return []
        
        n_bootstrap = 1000
        break_estimates = [[] for _ in breaks]
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(self.n, size=self.n, replace=True)
            boot_data = self.data[indices]
            boot_years = self.years[indices]
            
            # Sort by years (important for break detection)
            sort_idx = np.argsort(boot_years)
            boot_data = boot_data[sort_idx]
            boot_years = boot_years[sort_idx]
            
            # Detect breaks in bootstrap sample
            try:
                # Simple approach: find closest years to original breaks
                for i, original_break_year in enumerate([self.years[bp] for bp in breaks]):
                    closest_idx = np.argmin(np.abs(boot_years - original_break_year))
                    break_estimates[i].append(boot_years[closest_idx])
            except:
                continue
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        confidence_intervals = []
        
        for i, estimates in enumerate(break_estimates):
            if len(estimates) > 10:  # Sufficient estimates
                lower = np.percentile(estimates, 100 * alpha/2)
                upper = np.percentile(estimates, 100 * (1 - alpha/2))
                confidence_intervals.append({
                    'break_year': self.years[breaks[i]],
                    'confidence_interval': [lower, upper],
                    'n_bootstrap_estimates': len(estimates)
                })
            else:
                confidence_intervals.append({
                    'break_year': self.years[breaks[i]],
                    'confidence_interval': [np.nan, np.nan],
                    'n_bootstrap_estimates': len(estimates)
                })
        
        return confidence_intervals
    
    def _simple_break_detection(self):
        """Simple break detection as fallback"""
        # Use existing Pettitt test result if available
        if hasattr(self, 'pettitt_result'):
            if self.pettitt_result.get('significant', False):
                return [self.pettitt_result['change_point_index']]
        
        # Otherwise, look for largest change in slope
        if self.n < 6:
            return []
        
        max_f_stat = 0
        best_break = None
        
        for potential_break in range(self.min_segment_size, self.n - self.min_segment_size):
            # Split into two segments
            y1, x1 = self.data[:potential_break], self.years[:potential_break]
            y2, x2 = self.data[potential_break:], self.years[potential_break:]
            
            if len(y1) >= 3 and len(y2) >= 3:
                # Fit models to each segment
                model1 = self._fit_linear_model(y1, x1)
                model2 = self._fit_linear_model(y2, x2)
                
                # F-test for structural break
                ssr_restricted = self._fit_linear_model(self.data, self.years)['ssr']
                ssr_unrestricted = model1['ssr'] + model2['ssr']
                
                if ssr_restricted > ssr_unrestricted:
                    f_stat = ((ssr_restricted - ssr_unrestricted) / 2) / (ssr_unrestricted / (self.n - 4))
                    
                    if f_stat > max_f_stat:
                        max_f_stat = f_stat
                        best_break = potential_break
        
        # Check significance (approximate)
        if max_f_stat > 3.0:  # Rough threshold
            return [best_break]
        else:
            return []
    
    def _calculate_information_criteria(self, models):
        """Calculate AIC and BIC for model selection"""
        criteria = {}
        
        for m, model in models.items():
            if 'error' in model:
                criteria[m] = {'aic': np.inf, 'bic': np.inf}
                continue
            
            # Extract SSR and calculate log-likelihood
            if m == 0:
                ssr = model['ssr']
                k = 3  # intercept, slope, sigma
            else:
                ssr = model.get('global_ssr', np.inf)
                k = 3 + 2 * m  # original params + 2 params per break
            
            # Log-likelihood (assuming normal errors)
            if ssr > 0:
                log_likelihood = -0.5 * self.n * (np.log(2 * np.pi) + np.log(ssr / self.n) + 1)
            else:
                log_likelihood = 0
            
            # Information criteria
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(self.n) - 2 * log_likelihood
            
            criteria[m] = {'aic': aic, 'bic': bic, 'log_likelihood': log_likelihood}
        
        return criteria
    
    def _select_optimal_breaks(self, criteria):
        """Select optimal number of breaks using information criteria"""
        # Use BIC (more conservative) as primary criterion
        best_bic = np.inf
        optimal_m = 0
        
        for m, crit in criteria.items():
            if crit['bic'] < best_bic:
                best_bic = crit['bic']
                optimal_m = m
        
        return optimal_m
    
    def _sequential_f_test(self, models):
        """Sequential F-test for break significance"""
        test_results = {}
        
        for m in range(1, len(models)):
            if 'error' in models[m] or 'error' in models[m-1]:
                test_results[f'{m-1}_vs_{m}'] = {'f_stat': np.nan, 'p_value': np.nan}
                continue
            
            # F-test: H0: m-1 breaks vs H1: m breaks
            ssr_restricted = models[m-1].get('ssr', models[m-1].get('global_ssr', np.inf))
            ssr_unrestricted = models[m].get('global_ssr', np.inf)
            
            if ssr_restricted > ssr_unrestricted and ssr_unrestricted > 0:
                f_stat = ((ssr_restricted - ssr_unrestricted) / 2) / (ssr_unrestricted / (self.n - 2 * m - 2))
                
                # p-value from F-distribution
                df1 = 2  # 2 additional parameters per break
                df2 = self.n - 2 * m - 2
                
                if df2 > 0:
                    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                else:
                    p_value = np.nan
            else:
                f_stat = np.nan
                p_value = np.nan
            
            test_results[f'{m-1}_vs_{m}'] = {
                'f_stat': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05 if not np.isnan(p_value) else False
            }
        
        return test_results
    
    def _structural_stability_tests(self, segments):
        """Additional structural stability tests"""
        if len(segments) < 2:
            return {}
        
        stability_tests = {}
        
        # Test for equal slopes across segments
        slopes = [seg['slope'] for seg in segments if 'slope' in seg]
        slope_vars = [seg['slope_se']**2 for seg in segments if 'slope_se' in seg and seg['slope_se'] < np.inf]
        
        if len(slopes) >= 2 and len(slope_vars) >= 2:
            # Weighted average slope
            weights = [1/var if var > 0 else 0 for var in slope_vars]
            if sum(weights) > 0:
                pooled_slope = sum(w*s for w, s in zip(weights, slopes)) / sum(weights)
                
                # Chi-square test for slope equality
                chi2_stat = sum(w * (s - pooled_slope)**2 for w, s in zip(weights, slopes))
                p_value = 1 - stats.chi2.cdf(chi2_stat, len(slopes) - 1)
                
                stability_tests['slope_equality'] = {
                    'chi2_statistic': chi2_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'pooled_slope': pooled_slope
                }
        
        return stability_tests
    
    def generate_change_point_summary(self):
        """Generate comprehensive summary of change point analysis"""
        summary = {
            'methods_applied': [],
            'detected_breaks': {},
            'model_selection': {},
            'confidence_intervals': {}
        }
        
        # Bai-Perron results
        if 'bai_perron' in self.results:
            bp = self.results['bai_perron']
            summary['methods_applied'].append('Bai-Perron')
            
            optimal_m = bp.get('optimal_breaks', 0)
            summary['detected_breaks']['bai_perron'] = {
                'optimal_number': optimal_m,
                'break_years': bp['break_dates'].get(optimal_m, []),
                'information_criteria': bp['information_criteria']
            }
        
        # Segmented regression results
        if 'segmented_regression' in self.results:
            sr = self.results['segmented_regression']
            summary['methods_applied'].append('Segmented Regression')
            
            summary['detected_breaks']['segmented_regression'] = {
                'break_years': sr.get('break_points', []),
                'confidence_intervals': sr.get('break_confidence_intervals', [])
            }
            
            if 'segments' in sr:
                summary['model_selection']['segmented_regression'] = {
                    'n_segments': len(sr['segments']),
                    'global_r_squared': sr.get('global_r_squared', np.nan)
                }
        
        return summary

def main():
    """Example usage - typically called from main analysis"""
    pass

if __name__ == "__main__":
    main()