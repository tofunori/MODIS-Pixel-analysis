#!/usr/bin/env python3

"""
Enhanced Residual Diagnostics and Homoscedasticity Testing
Addresses reviewer feedback on residual analysis and assumption validation
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EnhancedDiagnostics:
    """
    Enhanced diagnostic tests for regression models
    Focuses on residual analysis, homoscedasticity, and normality
    """
    
    def __init__(self, data, years=None, fitted_values=None, residuals=None):
        """
        Initialize diagnostics analyzer
        
        Parameters:
        -----------
        data : array-like
            Original time series data
        years : array-like, optional
            Time points
        fitted_values : array-like, optional
            Model fitted values
        residuals : array-like, optional
            Model residuals
        """
        self.data = np.array(data)
        self.years = np.array(years) if years is not None else np.arange(len(data))
        self.n = len(self.data)
        
        # If residuals not provided, calculate from simple linear model
        if residuals is None or fitted_values is None:
            slope, intercept = np.polyfit(self.years, self.data, 1)
            self.fitted_values = slope * self.years + intercept
            self.residuals = self.data - self.fitted_values
        else:
            self.fitted_values = np.array(fitted_values)
            self.residuals = np.array(residuals)
        
        self.results = {}
    
    def comprehensive_residual_analysis(self):
        """
        Comprehensive residual analysis including all diagnostic tests
        """
        results = {
            'normality_tests': self.residual_normality_tests(),
            'homoscedasticity_tests': self.homoscedasticity_tests(),
            'autocorrelation_tests': self.residual_autocorrelation_tests(),
            'outlier_detection': self.residual_outlier_detection(),
            'influence_diagnostics': self.influence_diagnostics(),
            'linearity_tests': self.linearity_tests()
        }
        
        # Overall assessment
        results['overall_assessment'] = self._generate_overall_assessment(results)
        
        self.results = results
        return results
    
    def residual_normality_tests(self):
        """
        Test normality of residuals using multiple methods
        Addresses reviewer feedback on testing residuals rather than raw data
        """
        results = {}
        
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(self.residuals)
        results['shapiro_wilk'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'normal': shapiro_p > 0.05
        }
        
        # Anderson-Darling test
        anderson_result = stats.anderson(self.residuals, dist='norm')
        results['anderson_darling'] = {
            'statistic': anderson_result.statistic,
            'critical_values': anderson_result.critical_values.tolist(),
            'significance_levels': anderson_result.significance_level.tolist(),
            'normal': anderson_result.statistic < anderson_result.critical_values[2]  # 5% level
        }
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(self.residuals)
        results['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'normal': jb_p > 0.05
        }
        
        # Lilliefors test (Kolmogorov-Smirnov with estimated parameters)
        try:
            # Standardize residuals
            standardized_residuals = (self.residuals - np.mean(self.residuals)) / np.std(self.residuals)
            ks_stat, ks_p = stats.kstest(standardized_residuals, 'norm')
            results['lilliefors'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'normal': ks_p > 0.05
            }
        except:
            results['lilliefors'] = {'error': 'Could not compute Lilliefors test'}
        
        # D'Agostino-Pearson test
        try:
            dp_stat, dp_p = stats.normaltest(self.residuals)
            results['dagostino_pearson'] = {
                'statistic': dp_stat,
                'p_value': dp_p,
                'normal': dp_p > 0.05
            }
        except:
            results['dagostino_pearson'] = {'error': 'Could not compute D\'Agostino-Pearson test'}
        
        # Summary
        normal_tests = [test for test in ['shapiro_wilk', 'anderson_darling', 'jarque_bera', 'lilliefors', 'dagostino_pearson'] 
                       if test in results and 'normal' in results[test]]
        
        if normal_tests:
            normal_count = sum(results[test]['normal'] for test in normal_tests)
            results['summary'] = {
                'tests_indicating_normality': normal_count,
                'total_tests': len(normal_tests),
                'proportion_normal': normal_count / len(normal_tests),
                'consensus': 'normal' if normal_count >= len(normal_tests) / 2 else 'non_normal'
            }
        
        return results
    
    def homoscedasticity_tests(self):
        """
        Test for homoscedasticity (constant variance) of residuals
        Implements Breusch-Pagan and White tests as requested by reviewer
        """
        results = {}
        
        # Breusch-Pagan test
        bp_results = self._breusch_pagan_test()
        results['breusch_pagan'] = bp_results
        
        # White test
        white_results = self._white_test()
        results['white'] = white_results
        
        # Goldfeld-Quandt test
        gq_results = self._goldfeld_quandt_test()
        results['goldfeld_quandt'] = gq_results
        
        # Modified Breusch-Pagan (Koenker test)
        koenker_results = self._koenker_test()
        results['koenker'] = koenker_results
        
        # Summary
        homo_tests = [test for test in results.keys() if 'homoscedastic' in results[test]]
        
        if homo_tests:
            homo_count = sum(results[test]['homoscedastic'] for test in homo_tests)
            results['summary'] = {
                'tests_indicating_homoscedasticity': homo_count,
                'total_tests': len(homo_tests),
                'proportion_homoscedastic': homo_count / len(homo_tests),
                'consensus': 'homoscedastic' if homo_count >= len(homo_tests) / 2 else 'heteroscedastic'
            }
        
        return results
    
    def _breusch_pagan_test(self):
        """
        Breusch-Pagan test for heteroscedasticity
        Tests if residual variance depends on fitted values
        """
        try:
            # Regress squared residuals on fitted values
            squared_residuals = self.residuals ** 2
            
            # Design matrix: [1, fitted_values]
            X = np.column_stack([np.ones(self.n), self.fitted_values])
            
            # OLS regression: squared_residuals = X * beta + error
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ squared_residuals
            
            # Predicted values and residuals
            predicted_sq_res = X @ beta
            aux_residuals = squared_residuals - predicted_sq_res
            
            # R-squared from auxiliary regression
            ss_reg = np.sum((predicted_sq_res - np.mean(squared_residuals))**2)
            ss_tot = np.sum((squared_residuals - np.mean(squared_residuals))**2)
            r_squared = ss_reg / ss_tot if ss_tot > 0 else 0
            
            # Test statistic: n * R-squared ~ chi2(1)
            lm_statistic = self.n * r_squared
            p_value = 1 - stats.chi2.cdf(lm_statistic, df=1)
            
            return {
                'statistic': lm_statistic,
                'p_value': p_value,
                'homoscedastic': p_value > 0.05,
                'r_squared_auxiliary': r_squared
            }
            
        except Exception as e:
            return {'error': f'Breusch-Pagan test failed: {str(e)}'}
    
    def _white_test(self):
        """
        White test for heteroscedasticity
        Tests if residual variance depends on fitted values and their squares
        """
        try:
            # Regress squared residuals on fitted values and fitted values squared
            squared_residuals = self.residuals ** 2
            fitted_sq = self.fitted_values ** 2
            
            # Design matrix: [1, fitted_values, fitted_values^2]
            X = np.column_stack([np.ones(self.n), self.fitted_values, fitted_sq])
            
            # Check for multicollinearity
            if np.linalg.cond(X.T @ X) > 1e12:
                # Use simpler model if multicollinearity detected
                X = np.column_stack([np.ones(self.n), self.fitted_values])
            
            # OLS regression
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ squared_residuals
            
            # Predicted values
            predicted_sq_res = X @ beta
            
            # R-squared from auxiliary regression
            ss_reg = np.sum((predicted_sq_res - np.mean(squared_residuals))**2)
            ss_tot = np.sum((squared_residuals - np.mean(squared_residuals))**2)
            r_squared = ss_reg / ss_tot if ss_tot > 0 else 0
            
            # Test statistic: n * R-squared ~ chi2(k-1), where k is number of regressors
            df = X.shape[1] - 1
            lm_statistic = self.n * r_squared
            p_value = 1 - stats.chi2.cdf(lm_statistic, df=df)
            
            return {
                'statistic': lm_statistic,
                'p_value': p_value,
                'homoscedastic': p_value > 0.05,
                'degrees_of_freedom': df,
                'r_squared_auxiliary': r_squared
            }
            
        except Exception as e:
            return {'error': f'White test failed: {str(e)}'}
    
    def _goldfeld_quandt_test(self):
        """
        Goldfeld-Quandt test for heteroscedasticity
        Compares variance in first and last thirds of data
        """
        try:
            # Sort data by fitted values
            sort_idx = np.argsort(self.fitted_values)
            sorted_residuals = self.residuals[sort_idx]
            
            # Split into thirds, use first and last third
            n_third = self.n // 3
            first_third = sorted_residuals[:n_third]
            last_third = sorted_residuals[-n_third:]
            
            # Calculate variances
            var1 = np.var(first_third, ddof=1)
            var2 = np.var(last_third, ddof=1)
            
            # F-statistic (larger variance in numerator)
            if var2 > var1:
                f_stat = var2 / var1
                df1, df2 = len(last_third) - 1, len(first_third) - 1
            else:
                f_stat = var1 / var2
                df1, df2 = len(first_third) - 1, len(last_third) - 1
            
            # Two-tailed p-value
            p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
            
            return {
                'statistic': f_stat,
                'p_value': p_value,
                'homoscedastic': p_value > 0.05,
                'variance_ratio': var2 / var1 if var1 > 0 else np.inf,
                'df1': df1,
                'df2': df2
            }
            
        except Exception as e:
            return {'error': f'Goldfeld-Quandt test failed: {str(e)}'}
    
    def _koenker_test(self):
        """
        Koenker test (studentized Breusch-Pagan test)
        Robust version of Breusch-Pagan test
        """
        try:
            # Standardized squared residuals
            mean_sq_res = np.mean(self.residuals ** 2)
            standardized_sq_res = (self.residuals ** 2) / mean_sq_res - 1
            
            # Regress standardized squared residuals on fitted values
            X = np.column_stack([np.ones(self.n), self.fitted_values])
            
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ standardized_sq_res
            
            # Predicted values
            predicted = X @ beta
            
            # Explained sum of squares
            ss_reg = np.sum((predicted - np.mean(standardized_sq_res))**2)
            
            # Test statistic
            lm_statistic = 0.5 * ss_reg
            p_value = 1 - stats.chi2.cdf(lm_statistic, df=1)
            
            return {
                'statistic': lm_statistic,
                'p_value': p_value,
                'homoscedastic': p_value > 0.05
            }
            
        except Exception as e:
            return {'error': f'Koenker test failed: {str(e)}'}
    
    def residual_autocorrelation_tests(self):
        """
        Test for autocorrelation in residuals
        """
        results = {}
        
        # Durbin-Watson test
        dw_stat = self._durbin_watson_test()
        results['durbin_watson'] = dw_stat
        
        # Ljung-Box test on residuals
        lb_results = self._ljung_box_residuals()
        results['ljung_box'] = lb_results
        
        # Runs test on residuals
        runs_results = self._runs_test_residuals()
        results['runs_test'] = runs_results
        
        return results
    
    def _durbin_watson_test(self):
        """
        Durbin-Watson test for first-order autocorrelation in residuals
        """
        # Calculate DW statistic
        diff_residuals = np.diff(self.residuals)
        dw_stat = np.sum(diff_residuals**2) / np.sum(self.residuals**2)
        
        # Rough interpretation (exact critical values depend on X matrix)
        if dw_stat < 1.5:
            interpretation = 'positive_autocorrelation'
        elif dw_stat > 2.5:
            interpretation = 'negative_autocorrelation'
        else:
            interpretation = 'no_autocorrelation'
        
        return {
            'statistic': dw_stat,
            'interpretation': interpretation,
            'no_autocorrelation': interpretation == 'no_autocorrelation'
        }
    
    def _ljung_box_residuals(self, lags=5):
        """
        Ljung-Box test for serial correlation in residuals
        """
        n = len(self.residuals)
        autocorrs = []
        
        # Calculate autocorrelations
        for lag in range(1, lags + 1):
            if n > lag:
                corr = np.corrcoef(self.residuals[:-lag], self.residuals[lag:])[0, 1]
                autocorrs.append(corr if not np.isnan(corr) else 0)
            else:
                autocorrs.append(0)
        
        # Ljung-Box statistic
        lb_stat = n * (n + 2) * sum([(corr**2) / (n - k) for k, corr in enumerate(autocorrs, 1)])
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)
        
        return {
            'statistic': lb_stat,
            'p_value': p_value,
            'lags': lags,
            'no_autocorrelation': p_value > 0.05,
            'autocorrelations': autocorrs
        }
    
    def _runs_test_residuals(self):
        """
        Runs test for randomness of residual signs
        """
        # Convert residuals to signs
        signs = np.where(self.residuals >= np.median(self.residuals), 1, -1)
        
        # Count runs
        runs = 1
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1]:
                runs += 1
        
        # Expected runs and variance under null hypothesis
        n_pos = np.sum(signs == 1)
        n_neg = np.sum(signs == -1)
        
        expected_runs = (2 * n_pos * n_neg) / (n_pos + n_neg) + 1
        variance_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / \
                       ((n_pos + n_neg)**2 * (n_pos + n_neg - 1))
        
        # Z-score
        if variance_runs > 0:
            z_score = (runs - expected_runs) / np.sqrt(variance_runs)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0
            p_value = 1
        
        return {
            'runs_observed': runs,
            'runs_expected': expected_runs,
            'z_score': z_score,
            'p_value': p_value,
            'random': p_value > 0.05
        }
    
    def residual_outlier_detection(self):
        """
        Detect outliers in residuals using multiple methods
        """
        results = {}
        
        # Standardized residuals
        standardized_residuals = self.residuals / np.std(self.residuals)
        results['standardized_residuals'] = standardized_residuals.tolist()
        
        # Outliers based on standardized residuals
        outlier_threshold = 2.5
        outlier_indices = np.where(np.abs(standardized_residuals) > outlier_threshold)[0]
        
        results['standardized_outliers'] = {
            'threshold': outlier_threshold,
            'outlier_indices': outlier_indices.tolist(),
            'n_outliers': len(outlier_indices),
            'outlier_years': self.years[outlier_indices].tolist() if len(outlier_indices) > 0 else []
        }
        
        # Studentized residuals (if possible to calculate)
        try:
            studentized_residuals = self._calculate_studentized_residuals()
            results['studentized_residuals'] = studentized_residuals.tolist()
            
            # Outliers based on studentized residuals
            student_outliers = np.where(np.abs(studentized_residuals) > 2.5)[0]
            results['studentized_outliers'] = {
                'outlier_indices': student_outliers.tolist(),
                'n_outliers': len(student_outliers),
                'outlier_years': self.years[student_outliers].tolist() if len(student_outliers) > 0 else []
            }
        except:
            results['studentized_residuals'] = 'Could not calculate'
        
        return results
    
    def _calculate_studentized_residuals(self):
        """
        Calculate studentized residuals (externally studentized)
        """
        # This requires leave-one-out residual calculation
        # Simplified approximation for now
        mse = np.sum(self.residuals**2) / (self.n - 2)
        leverage = self._calculate_leverage()
        
        # Studentized residuals approximation
        studentized = self.residuals / np.sqrt(mse * (1 - leverage))
        
        return studentized
    
    def _calculate_leverage(self):
        """
        Calculate leverage values (hat matrix diagonal)
        """
        # Design matrix for simple linear regression
        X = np.column_stack([np.ones(self.n), self.years])
        
        try:
            # Hat matrix diagonal: h_ii = x_i^T (X^T X)^-1 x_i
            XtX_inv = np.linalg.inv(X.T @ X)
            leverage = np.array([X[i] @ XtX_inv @ X[i] for i in range(self.n)])
            return leverage
        except:
            return np.ones(self.n) / self.n  # Fallback
    
    def influence_diagnostics(self):
        """
        Calculate influence diagnostics (Cook's distance, DFBETAS, etc.)
        """
        results = {}
        
        try:
            leverage = self._calculate_leverage()
            results['leverage'] = leverage.tolist()
            
            # Cook's distance
            mse = np.sum(self.residuals**2) / (self.n - 2)
            cooks_d = (self.residuals**2 / (2 * mse)) * (leverage / (1 - leverage)**2)
            results['cooks_distance'] = cooks_d.tolist()
            
            # Influential points (Cook's D > 4/n)
            influential_threshold = 4 / self.n
            influential_indices = np.where(cooks_d > influential_threshold)[0]
            
            results['influential_points'] = {
                'threshold': influential_threshold,
                'influential_indices': influential_indices.tolist(),
                'n_influential': len(influential_indices),
                'influential_years': self.years[influential_indices].tolist() if len(influential_indices) > 0 else []
            }
            
        except Exception as e:
            results['error'] = f'Could not calculate influence diagnostics: {str(e)}'
        
        return results
    
    def linearity_tests(self):
        """
        Test for linearity assumptions
        """
        results = {}
        
        # Rainbow test for linearity
        rainbow_results = self._rainbow_test()
        results['rainbow_test'] = rainbow_results
        
        # Reset test (RESET)
        reset_results = self._reset_test()
        results['reset_test'] = reset_results
        
        return results
    
    def _rainbow_test(self):
        """
        Rainbow test for linearity
        Tests whether linear model is adequate
        """
        try:
            # Sort data by years
            sort_idx = np.argsort(self.years)
            sorted_data = self.data[sort_idx]
            sorted_years = self.years[sort_idx]
            
            # Fit to middle portion and compare with full model
            n_middle = int(0.5 * self.n)
            start_idx = (self.n - n_middle) // 2
            end_idx = start_idx + n_middle
            
            middle_years = sorted_years[start_idx:end_idx]
            middle_data = sorted_data[start_idx:end_idx]
            
            # Fit models
            slope_full, intercept_full = np.polyfit(sorted_years, sorted_data, 1)
            slope_middle, intercept_middle = np.polyfit(middle_years, middle_data, 1)
            
            # Calculate SSR for each model
            pred_full = slope_full * sorted_years + intercept_full
            ssr_full = np.sum((sorted_data - pred_full)**2)
            
            pred_middle = slope_middle * middle_years + intercept_middle
            ssr_middle = np.sum((middle_data - pred_middle)**2)
            
            # F-test for linearity
            f_stat = ((ssr_full - ssr_middle) / (self.n - n_middle)) / (ssr_middle / (n_middle - 2))
            df1 = self.n - n_middle
            df2 = n_middle - 2
            
            if df2 > 0:
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)
            else:
                p_value = np.nan
            
            return {
                'statistic': f_stat,
                'p_value': p_value,
                'linear': p_value > 0.05 if not np.isnan(p_value) else True,
                'df1': df1,
                'df2': df2
            }
            
        except Exception as e:
            return {'error': f'Rainbow test failed: {str(e)}'}
    
    def _reset_test(self):
        """
        RESET test for functional form
        Tests if higher-order terms of fitted values are significant
        """
        try:
            # Auxiliary regression: y = X*beta + gamma*fitted^2 + gamma*fitted^3 + error
            fitted_sq = self.fitted_values ** 2
            fitted_cube = self.fitted_values ** 3
            
            # Design matrices
            X_original = np.column_stack([np.ones(self.n), self.years])
            X_extended = np.column_stack([X_original, fitted_sq, fitted_cube])
            
            # Fit both models
            beta_original = np.linalg.inv(X_original.T @ X_original) @ X_original.T @ self.data
            beta_extended = np.linalg.inv(X_extended.T @ X_extended) @ X_extended.T @ self.data
            
            # Calculate SSR for both models
            pred_original = X_original @ beta_original
            pred_extended = X_extended @ beta_extended
            
            ssr_original = np.sum((self.data - pred_original)**2)
            ssr_extended = np.sum((self.data - pred_extended)**2)
            
            # F-test
            f_stat = ((ssr_original - ssr_extended) / 2) / (ssr_extended / (self.n - 4))
            p_value = 1 - stats.f.cdf(f_stat, 2, self.n - 4)
            
            return {
                'statistic': f_stat,
                'p_value': p_value,
                'linear': p_value > 0.05,
                'df1': 2,
                'df2': self.n - 4
            }
            
        except Exception as e:
            return {'error': f'RESET test failed: {str(e)}'}
    
    def _generate_overall_assessment(self, results):
        """
        Generate overall assessment of model assumptions
        """
        assessment = {
            'normality': 'unknown',
            'homoscedasticity': 'unknown',
            'independence': 'unknown',
            'linearity': 'unknown',
            'overall_adequacy': 'unknown',
            'recommendations': []
        }
        
        # Normality assessment
        if 'normality_tests' in results and 'summary' in results['normality_tests']:
            assessment['normality'] = results['normality_tests']['summary']['consensus']
            
            if assessment['normality'] == 'non_normal':
                assessment['recommendations'].append('Consider robust or non-parametric methods due to non-normal residuals')
        
        # Homoscedasticity assessment
        if 'homoscedasticity_tests' in results and 'summary' in results['homoscedasticity_tests']:
            assessment['homoscedasticity'] = results['homoscedasticity_tests']['summary']['consensus']
            
            if assessment['homoscedasticity'] == 'heteroscedastic':
                assessment['recommendations'].append('Consider robust standard errors or weighted least squares due to heteroscedasticity')
        
        # Independence assessment
        if 'autocorrelation_tests' in results:
            if 'durbin_watson' in results['autocorrelation_tests']:
                dw_ok = results['autocorrelation_tests']['durbin_watson']['no_autocorrelation']
            else:
                dw_ok = True
                
            if 'ljung_box' in results['autocorrelation_tests']:
                lb_ok = results['autocorrelation_tests']['ljung_box']['no_autocorrelation']
            else:
                lb_ok = True
            
            assessment['independence'] = 'adequate' if (dw_ok and lb_ok) else 'violated'
            
            if assessment['independence'] == 'violated':
                assessment['recommendations'].append('Consider autocorrelation correction methods (GLS, pre-whitening)')
        
        # Linearity assessment
        if 'linearity_tests' in results:
            linearity_tests = [test for test in results['linearity_tests'].values() 
                              if isinstance(test, dict) and 'linear' in test]
            
            if linearity_tests:
                linear_count = sum(test['linear'] for test in linearity_tests)
                assessment['linearity'] = 'adequate' if linear_count >= len(linearity_tests) / 2 else 'questionable'
                
                if assessment['linearity'] == 'questionable':
                    assessment['recommendations'].append('Consider non-linear models or transformation of variables')
        
        # Overall adequacy
        adequate_count = sum(1 for aspect in ['normality', 'homoscedasticity', 'independence', 'linearity']
                           if assessment[aspect] in ['normal', 'homoscedastic', 'adequate'])
        
        total_aspects = 4
        adequacy_ratio = adequate_count / total_aspects
        
        if adequacy_ratio >= 0.75:
            assessment['overall_adequacy'] = 'good'
        elif adequacy_ratio >= 0.5:
            assessment['overall_adequacy'] = 'fair'
        else:
            assessment['overall_adequacy'] = 'poor'
        
        if assessment['overall_adequacy'] == 'poor':
            assessment['recommendations'].append('Model assumptions are significantly violated - consider alternative methods')
        
        return assessment

def main():
    """Example usage - typically called from main analysis"""
    pass

if __name__ == "__main__":
    main()