import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import scipy.stats as stats
import sys
import os

class SimpleLinearRegressionAnalysis():
    def __init__(self, dataframe, response_variable, explanatory_variable, round_digits=5, alpha=0.05, alternative="two_sided"):
        
        self.response_variable = response_variable
        self.explanatory_variable = explanatory_variable
        self.alpha = alpha
        self.alternative = alternative
        
        
        self.data = dataframe
        self.response_data = list(self.data[self.response_variable].to_numpy())
        self.explanatory_data = list(self.data[self.explanatory_variable].to_numpy())
        self.round_digits = round_digits
        self.statistics = self.calculate_statistics()
        
        self.conf_pred_table = self._build_conf_pred_table()
    
    def get_ci_table(self):
        return self.conf_pred_table
    
    def get_summary_table(self):
        index = ['slope', 'intercept', 'pearsons_r', 'r_squared', 'mse', 'rmse', 'standard_error_slope', 'standard_error_intercept']
        values = [self.statistics[key] for key in index]
        summary_data = {"Statistic":values}
        temp_df = pd.DataFrame(data=summary_data, index=index)
        return temp_df
    
    def calculate_statistics(self):
        sample_stats = {}
        
        data = [self.response_data, self.explanatory_data]
        names = [self.response_variable, self.explanatory_variable]
        
        for index, (variable_data, variable_name) in enumerate(zip(data, names)):
            sample_stats[f'{variable_name}_variance'] = np.var(variable_data, ddof=1)
            sample_stats[f'{variable_name}_std'] = np.std(variable_data, ddof=1)
            sample_stats[f'{variable_name}_mean'] = np.mean(variable_data)
            sample_stats[f'{variable_name}_deviations_from_mean'] = [(value - sample_stats[f'{variable_name}_mean']) for value in variable_data]
            sample_stats[f'{variable_name}_sum_squared_dev_mean'] = np.sum([value**2 for value in sample_stats[f'{variable_name}_deviations_from_mean']])
            sample_stats[f'{variable_name}_n'] = len(variable_data)
            sample_stats[f'{variable_name}_z_scores'] = [(value - sample_stats[f'{variable_name}_mean'])/sample_stats[f'{variable_name}_std'] for value in variable_data]
            sample_stats[f'{variable_name}_min'] = np.min(variable_data)
            sample_stats[f'{variable_name}_max'] = np.max(variable_data)
            
        sample_stats[f'sum_product_xy_deviations'] = np.sum([x*y for x,y in zip(sample_stats[f'{self.response_variable}_deviations_from_mean'],
                                                                                sample_stats[f'{self.explanatory_variable}_deviations_from_mean'])])
        
        
        # Calculate slope (beta_1) and intercept (beta_0)
        sample_stats['slope'] = sample_stats[f'sum_product_xy_deviations'] / sample_stats[f'{self.explanatory_variable}_sum_squared_dev_mean'] 
        sample_stats['intercept'] = sample_stats[f'{self.response_variable}_mean'] - sample_stats['slope']*sample_stats[f'{self.explanatory_variable}_mean']
        
        # String Form of the Regression equation
        reg_eq = (f"{self.response_variable} = {round(sample_stats['slope'], self.round_digits)}*{self.explanatory_variable} + "
                  f"{round(sample_stats['intercept'], self.round_digits)}")
        sample_stats['regression_equation'] = reg_eq
        
        # Calculate Pearsons Correlation
        sample_stats['pearsons_r'] = np.sum([z_x * z_y for z_x, z_y in zip(sample_stats[f'{self.explanatory_variable}_z_scores'], 
                                                                           sample_stats[f'{self.response_variable}_z_scores'])]) / (sample_stats[f'{self.response_variable}_n'] - 1)
        
        # This is true for simple linear regression, but does not hold for multiple linear regression. 
        sample_stats['r_squared'] = sample_stats['pearsons_r']**2
        
        # Calculate Residuals
        sample_stats['residuals'] = self._calculate_residuals(sample_stats)
        
        # Calculate the estimate of the variance (mse) and standard deviation (rmse)
        # of the errors (residuals) about the regression line
        sample_stats['mse'] = self._calculate_mse(sample_stats)
        sample_stats['rmse'] = np.sqrt(sample_stats['mse'])
        
        # Calculate the standard error of slope (beta_1)
        sample_stats['standard_error_slope'] = self._calculate_standard_error_slope(sample_stats)
        
        # Calculate the standard error of intercept (beta_0)
        sample_stats['standard_error_intercept'] = self._calcuate_standard_error_int(sample_stats)
                
        return sample_stats
    
    def _calcuate_standard_error_int(self, sample_stats):
        key="standard_error_slope"
        rmse=sample_stats['rmse']
        n = sample_stats[f'{self.response_variable}_n']
        dof=sample_stats[f'{self.response_variable}_n'] - 1
        var_x = sample_stats[f'{self.explanatory_variable}_variance']
        mean_x = sample_stats[f'{self.explanatory_variable}_mean']
        
        inner_term = (1/n) + ((mean_x**2)/(dof*var_x))
        
        std_err_int = rmse * np.sqrt(inner_term)
        return std_err_int
    
    def _calculate_standard_error_slope(self, sample_stats):
        key="standard_error_slope"
        rmse=sample_stats['rmse']
        dof=sample_stats[f'{self.response_variable}_n'] - 1
        var_x = sample_stats[f'{self.explanatory_variable}_variance']
        
        std_err_slope = rmse * np.sqrt(1/(dof*var_x))
        return std_err_slope
    
    def _calculate_residuals(self, sample_stats):
        self.data['residuals'] = self.data.apply(lambda row: row[self.response_variable] - 
                                                 (row[self.explanatory_variable]*sample_stats['slope'] + sample_stats['intercept']),
                                                 axis=1)
        
        resids = list(self.data['residuals'].to_numpy())
        return resids
    
    def _calculate_mse(self, sample_stats):
        mse = np.sum([resid**2 for resid in sample_stats['residuals']]) / (sample_stats[f'{self.response_variable}_n'] - 2)
        return mse
    
    def _calculate_mean_response(self, x0):
        return self.statistics['slope']*x0 + self.statistics['intercept']
    
    # Standard error of the mean at x=x0
    def _calc_std_err_mean_response(self, x0):
        
        n = self.statistics[f'{self.response_variable}_n']
        
        x_variability = ((x0 - self.statistics[f'{self.explanatory_variable}_mean'])**2)/((n-1)*self.statistics[f'{self.explanatory_variable}_variance'])
        
        std_err = self.statistics['rmse'] * np.sqrt((1/n) + x_variability)
        
        return std_err
    
    def _calc_std_err_prediction(self, x0):
        
        uncertainty_mean_loc = self._calc_std_err_mean_response(x0)
        
        # Prediction uncertainty is the sum of 1 and 2
        # 1) uncertainty from sampling from a normal distribution 
        # 2) uncertainty from not knowing exactly where the mean of that normal distribution is
        return np.sqrt(self.statistics['mse'] + uncertainty_mean_loc**2)
    
    def _calculate_std_err_xhat(self, xhat, mean_response=True):
        
        if mean_response:
            numerator = self._calc_std_err_mean_response(x0=xhat)
        else:
            numerator = self. _calc_std_err_prediction(x0=xhat)
        
        denominator = np.abs(self.statistics['slope'])
        
        std_err_xhat = numerator / denominator
        
        return std_err_xhat
    
    def _calc_wald_calibration_ci(self, y0, multiplier_type="t", mean_response=True, alpha=None):
        
        xhat = self._calculate_xhat(y0=y0)
        
        multiplier = self._calculate_multiplier(multiplier_type=multiplier_type, alpha=alpha)
        std_error = self._calculate_std_err_xhat(xhat=xhat, mean_response=mean_response)
        
        lwr_ci = xhat - multiplier * std_error
        upr_ci = xhat + multiplier * std_error
        
        ci = {"lower_ci": lwr_ci, "upper_ci": upr_ci, "best_estimate": xhat}
        
        return ci
        
    def _calculate_xhat(self, y0):
        return (y0 - self.statistics['intercept']) / self.statistics['slope']
    
    def _build_conf_pred_table(self):
        
        confidence_data = self._calculate_confidence_band(x_values=self.explanatory_data)
        prediction_data = self._calculate_prediction_band(x_values=self.explanatory_data)
        
        schef_ci_data = self._calculate_confidence_band(x_values=self.explanatory_data, multiplier_type="scheffe")
        schef_pred_data = self._calculate_prediction_band(x_values=self.explanatory_data, multiplier_type="scheffe")
                
        confidence_table = {f'Dependent Variable ({self.response_variable})':self.response_data,
                            f'Predicted Value': confidence_data[f'estimated_{self.response_variable}'],
                            f'Std Error Mean Predict': confidence_data['std_err_mean'],
                            f'{1-self.alpha}% CL Mean Lower': confidence_data['lower'],
                            f'{1-self.alpha}% CL Mean Upper': confidence_data['upper'],
                            f'{1-self.alpha}% CL Predict Lower': prediction_data['lower'], 
                            f'{1-self.alpha}% CL Predict Upper': prediction_data['upper'],
                            'residual':self.statistics['residuals'],
                            f'SCHEFFE {1-self.alpha}% CL Mean Lower': schef_ci_data['lower'],
                            f'SCHEFFE {1-self.alpha}% CL Mean Upper': schef_ci_data['upper'],
                            f'SCHEFFE {1-self.alpha}% CL Predict Lower': schef_pred_data['lower'], 
                            f'SCHEFFE {1-self.alpha}% CL Predict Upper': schef_pred_data['upper']}
        
        conf_pred_df = pd.DataFrame(confidence_table)
        
        return conf_pred_df
    
    def _calculate_multiplier(self, multiplier_type, alpha=None):
        
        if alpha is not None:
            sig_level = alpha
        else:
            sig_level = self.alpha
        
        if multiplier_type == 't':
            multiplier = self._calculate_t_multiplier(alpha=sig_level)
        elif multiplier_type == 'scheffe':
            multiplier = self._calculate_scheffe_multiplier(alpha=sig_level)
            
        return multiplier
    
    def _calculate_confidence_band(self, x_values, alpha=None, multiplier_type='t'):
        
        multiplier = self._calculate_multiplier(multiplier_type=multiplier_type, alpha=alpha)
        
        # PROC REG USES THIS T-MULTIPLIER WHEN DOING CI'S FOR EACH DATA POINT BY DEFAULT (INSTEAD OF A BONFERRONI OR A
        # SCHEFFE MULTIPLIER)
        ###scheffe_multiplier = stats.t.ppf(q=1-(sig_level/2), df=(self.statistics[f'{self.response_variable}_n'] - 2), loc=0, scale=1)
        
        mean_responses = [self._calculate_mean_response(x) for x in x_values]
        std_err_mean_responses = [self._calc_std_err_mean_response(x) for x in x_values]
        
        conf_band = {f"{self.explanatory_variable}":x_values,
                     f"estimated_{self.response_variable}":mean_responses,
                     f"std_err_mean":std_err_mean_responses}
        
        conf_band["lower"] = [mean - multiplier*std_err_mean for mean, std_err_mean in zip(mean_responses, std_err_mean_responses)]
        conf_band["upper"] = [mean + multiplier*std_err_mean for mean, std_err_mean in zip(mean_responses, std_err_mean_responses)]
        
        return conf_band
    
    def _calculate_prediction_band(self, x_values, alpha=None, multiplier_type='t'):
        
        multiplier = self._calculate_multiplier(multiplier_type=multiplier_type, alpha=alpha)
            
        # Best estimate
        mean_responses = [self._calculate_mean_response(x) for x in x_values]
        
        # Standard error for prediction of an individual at x=x0
        std_err_predicted_responses = [self._calc_std_err_prediction(x) for x in x_values]
        
        pred_band = {f"{self.explanatory_variable}":x_values,
                     f"estimated_{self.response_variable}":mean_responses,
                     f"std_err_mean":std_err_predicted_responses}
        
        pred_band["lower"] = [mean - multiplier*std_err_pred for mean, std_err_pred in zip(mean_responses, std_err_predicted_responses)]
        pred_band["upper"] = [mean + multiplier*std_err_pred for mean, std_err_pred in zip(mean_responses, std_err_predicted_responses)]
        
        return pred_band
    
    def _calculate_t_multiplier(self, alpha):
        return stats.t.ppf(q=1-(alpha/2), df=(self.statistics[f'{self.response_variable}_n'] - 2), loc=0, scale=1)
        
    def _calculate_scheffe_multiplier(self, alpha):
        
        scheffe_multiplier = np.sqrt(2 * stats.f.ppf(q=1-alpha,
                                                     dfn=2,
                                                     dfd=(self.statistics[f'{self.response_variable}_n'] - 2),
                                                     loc=0, 
                                                     scale=1))
        
        return scheffe_multiplier
    
    def get_statistics(self):
        return self.statistics
        
    def create_seaborn_residplot(self, figsize=(12, 6), lowess=False, order=1, robust=True, dropna=False, 
                                 seaborn_style="darkgrid"):
        
        sns.set_style(seaborn_style)
        
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        
        sns.residplot(x=self.explanatory_variable, y=self.response_variable, data=self.data, lowess=lowess,
                      order=order, robust=robust, dropna=dropna, ax=axs)
        
        axs.set_title(f"Residuals for Regressing {self.response_variable} on {self.explanatory_variable}", fontsize=24, weight="bold")
        axs.set_ylabel(f"{self.response_variable}_observed - {self.response_variable}_predicted", fontsize=18, weight="bold")
        axs.set_xlabel(f"{self.explanatory_variable}", fontsize=18, weight="bold")
        axs.tick_params(labelsize="xx-large")
    
    # Confidence level for drawing confidence band on regression plot
    def _get_conf_level(self, alpha):
        
        if alpha is not None:
            return 1 - alpha
        else:
            return 1 - self.alpha
        
    def plot_regression_grid(self, figsize=(30,8), seaborn_style="darkgrid", num_points=1000, legend_size=14, legend_loc="best", 
                             alpha=None, point_color="#003F5C", reg_line_color="#ffa600", conf_line_color="#ef5675", conf_band=True,
                             pred_line_color="#7a5195", conf_band_alpha=0.15, pred_linestyle="--"):
        
        sns.set_style(seaborn_style)
        
        width, height = figsize
        
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        
        self.plot_regression(figsize=(width/2, height), seaborn_style=seaborn_style, num_points=num_points, legend_size=legend_size,
                                      legend_loc=legend_loc, conf_band=conf_band, alpha=alpha, point_color=point_color, reg_line_color=reg_line_color,
                                      conf_line_color=conf_line_color, pred_line_color=pred_line_color, conf_band_alpha=conf_band_alpha, pred_band=True,
                                      pred_linestyle=pred_linestyle, grid=True, axs=axs[0])
        
        self.plot_regression(figsize=(width/2, height), seaborn_style=seaborn_style, num_points=num_points, legend_size=legend_size,
                                      legend_loc=legend_loc, conf_band=conf_band, alpha=alpha, point_color=point_color, reg_line_color=reg_line_color,
                                      conf_line_color=conf_line_color, pred_line_color=pred_line_color, conf_band_alpha=conf_band_alpha, pred_band=False,
                                      pred_linestyle=pred_linestyle, grid=True, axs=axs[1])
        
        plt.tight_layout()
        
    def plot_vertical_normal_curve(self, axs=None):
        pass
        #if axs is None:
        #    fig, axs = plt.subplots(nrows=1, ncols=1, )
    
    def _calibration_comparison(self, estimated_mean=True, num_comparisons=100, num_points=5_000, alpha=None, multiplier_type='t', 
                                extrapolate_pct=0.1, verbose=False, save_every=10, save_final=True, save_dir="Cal_Comparisons/New_Plots"):
        
        os.makedirs(f"./{save_dir}", exist_ok=True)
        
        if alpha is None:
            alpha = self.alpha
        
        min_obs_x = self.statistics[f"{self.explanatory_variable}_min"]
        max_obs_x = self.statistics[f"{self.explanatory_variable}_max"]
        range_obs_x = self.statistics[f"{self.explanatory_variable}_max"] - self.statistics[f"{self.explanatory_variable}_min"]
        
        # X-axis values we are going to compare calibration intervals for
        min_x = math.floor(min_obs_x - range_obs_x*extrapolate_pct)
        max_x = math.ceil(max_obs_x + range_obs_x*extrapolate_pct)
        x_values = np.linspace(start=min_x, stop=max_x, num=num_comparisons)
        
        # Reponse (y-values) for every x above.
        response_values = [self._calculate_mean_response(x) for x in x_values]
        
        keys = ["wald_lower", "wald_upper", "wald_best", "graphical_lower", "graphical_upper", "graphical_best", 
                "lower_difference", "upper_difference", "best_difference", "response_value", "wald_width",
                "graphical_width", "width_difference"]
        results = {key:[] for key in keys}
        
        num_comparisons = len(response_values)
        
        if estimated_mean:
            interval_type = "Estimated Mean"
        else:
            interval_type = "Individual Prediction"
        
        results["xs_at_responses"] = x_values
        results["min_obs_x"] = [min_obs_x] * num_comparisons # Min and max x values before extrapolating
        results["max_obs_x"] = [max_obs_x] * num_comparisons
        results["mean_obs_x"] = [self.statistics[f"{self.explanatory_variable}_mean"]] * num_comparisons
        results["min_x_used"] = [min_x] * num_comparisons     # Min and max x values used (including any extrapolation)
        results["max_x_used"] = [max_x] * num_comparisons
        results["mean_x_used"] = [np.mean(x_values)] * num_comparisons
        results["min_obs_response"] = [np.min(self.response_data)] * num_comparisons
        results["max_obs_response"] = [np.max(self.response_data)] * num_comparisons
        results["mean_obs_response"] = [np.mean(self.response_data)] * num_comparisons
        results["min_response_used"] = [np.min(response_values)] * num_comparisons
        results["max_response_used"] = [np.max(response_values)] * num_comparisons
        results["mean_response_used"] = [np.mean(response_values)] * num_comparisons
        results["alpha"] = [alpha] * num_comparisons
        results["interval_type"] = [interval_type] * num_comparisons
        results["response_variable"] = [self.response_variable] * num_comparisons
        results["explanatory_variable"] = [self.explanatory_variable] * num_comparisons
        results["regression_equation"] = [self.statistics["regression_equation"]] * num_comparisons
        results["rmse"] = [self.statistics["rmse"]] * num_comparisons
        results["multiplier_type"] = [multiplier_type] * num_comparisons
        results["pearsons_r"] = [self.statistics['pearsons_r']] * num_comparisons
        results["r_squared"] = [self.statistics['r_squared']] * num_comparisons
        
        for index, y0 in enumerate(response_values):
        
            graphical_ci = self._calc_graphical_calibration_interval(y_value=y0, multiplier_type=multiplier_type, estimated_mean=estimated_mean,
                                                                     num_points=num_points, alpha=alpha)
        
            wald_ci = self._calc_wald_calibration_ci(y0=y0, multiplier_type=multiplier_type, mean_response=estimated_mean, alpha=alpha)
            
            graphical_width = graphical_ci["calibration_upper"] - graphical_ci["calibration_lower"]
            wald_width = wald_ci["upper_ci"] - wald_ci["lower_ci"]
            
            results["wald_lower"].append(wald_ci["lower_ci"])
            results["wald_upper"].append(wald_ci["upper_ci"])
            results["wald_best"].append(wald_ci["best_estimate"])
            results["wald_width"].append(wald_width)
            results["graphical_lower"].append(graphical_ci["calibration_lower"])
            results["graphical_upper"].append(graphical_ci["calibration_upper"])
            results["graphical_best"].append(graphical_ci["calibration_best"])
            results["graphical_width"].append(graphical_width)
            results["lower_difference"].append(graphical_ci["calibration_lower"] - wald_ci["lower_ci"])
            results["upper_difference"].append(graphical_ci["calibration_upper"] - wald_ci["upper_ci"])
            results["best_difference"].append(graphical_ci["calibration_best"] - wald_ci["best_estimate"])
            results["width_difference"].append(graphical_width - wald_width)
            results["response_value"].append(y0)
            
            if verbose and index !=0 and index % save_every == 0:
                print("\n===================================================================")
                print(f"Finished calculating {index + 1} of {len(response_values)} intervals")
                print("===================================================================\n")
            
            
        result_df = pd.DataFrame(results)
        
        if save_final:
            run_number = time.strftime("_%Y_%m_%d-%H_%M_%S")
            
            if estimated_mean:
                ident = f"CI_{multiplier_type}_"
            else:
                ident = f"PI_{multiplier_type}_"
            
            # CI --> Confidence intervals or PI --> prediction intervals
            # C --> number of comparisons
            # A --> alpha used
            
            try:
                name = f"./{save_dir}/Cal_{self.response_variable}_{self.explanatory_variable}_{ident}_{num_comparisons}C_{alpha}A_{run_number}.csv"
                self.cal_compare_filename = name
                result_df.to_csv(name, index=False)
            except:
                print("Save Failed! Exiting function...\n")
            
        self.calibration_comparison_results = result_df
        
        return result_df
            
    def _calc_graphical_calibration_interval(self, y_value, estimated_mean, num_points, alpha, multiplier_type):
        
        cal_data = {}
        
        top_band_intersect = False
        bottom_band_intersect = False
        start_x = self.statistics[f'{self.explanatory_variable}_min']
        end_x = self.statistics[f'{self.explanatory_variable}_max']
        
        while not top_band_intersect or not bottom_band_intersect:
        
            x_values = np.linspace(start=int(start_x),
                                   stop=int(end_x),
                                   num=int(num_points))
        
            # If making a calibration interval for an estimated mean
            if estimated_mean:
                band = self._calculate_confidence_band(x_values, alpha, multiplier_type=multiplier_type)
            else: # else, confidence interval for an individual (prediction interval).
                band = self._calculate_prediction_band(x_values, alpha, multiplier_type=multiplier_type)
            
            if np.min(band['upper']) <= y_value:
                top_band_intersect = True
            else:
                start_x = start_x - 0.5*np.abs(end_x - start_x)
                num_points *= 1.5
            
            if np.max(band['lower']) >= y_value:
                bottom_band_intersect = True
            else:
                end_x = end_x + 0.5*np.abs(end_x - start_x)
                num_points *= 1.5
        
        # Calculate Lower Calibration Interval Limit
        upper_intersection_y = np.max([value for value in band['upper'] if value <= y_value])
        upper_intersection_arg = band['upper'].index(upper_intersection_y)
        upper_intersection_x = x_values[upper_intersection_arg]
        
        # Calculate Upper Calibration Interval Limit
        lower_intersection_y = np.min([value for value in band['lower'] if value >= y_value])
        lower_intersection_arg = band['lower'].index(lower_intersection_y)
        lower_intersection_x = x_values[lower_intersection_arg]
        
        # Calculate Calibration Interval Best Estimate
        reg_line_pts = [self._calculate_mean_response(x) for x in x_values]
        reg_intersection_y = np.max([value for value in reg_line_pts if value <= y_value])
        reg_intersection_arg = reg_line_pts.index(reg_intersection_y)
        reg_intersection_x = x_values[reg_intersection_arg]
        
        # Graphical Calibration Interval Info
        calibration_lower = upper_intersection_x
        calibration_upper = lower_intersection_x
        calibration_best = reg_intersection_x
        
        cal_data['upper_intersection_y'] = upper_intersection_y
        cal_data['upper_intersection_arg'] = upper_intersection_arg
        cal_data['upper_intersection_x'] = upper_intersection_x
        cal_data['lower_intersection_y'] = lower_intersection_y
        cal_data['lower_intersection_arg'] = lower_intersection_arg
        cal_data['lower_intersection_x'] = lower_intersection_x
        cal_data['reg_line_pts'] = reg_line_pts
        cal_data['reg_intersection_y'] = reg_intersection_y
        cal_data['reg_intersection_arg'] = reg_intersection_arg
        cal_data['reg_intersection_x'] = reg_intersection_x
        cal_data['calibration_lower'] = calibration_lower
        cal_data['calibration_upper'] = calibration_upper
        cal_data['calibration_best'] = calibration_best
        cal_data['num_points'] = num_points
        cal_data['x_values'] = x_values
        
        return cal_data
        
    # GRAPHICAL METHOD OF FINDING CALIBRATION INTERVALS
    def plot_calibration_interval(self, y_value, estimated_mean=True, seaborn_style="darkgrid", figsize=(15,8), legend_size=14, legend_loc="best",
                                  conf_band=True, alpha=None, point_color="#003F5C", reg_line_color="#ffa600", conf_line_color="#ef5675", num_points=5_000,
                                  pred_line_color="#7a5195", conf_band_alpha=0.15, pred_band=True, pred_linestyle="--", grid=False, axs=None,
                                  calibration_markertype="*", marker_color="#007F80", marker_size=12, linecolor="#30D5C8", linewidth=2, round_digits=4,
                                  multiplier_type='t', wald_txt_color="#2f4b7c", annot_cal_interval_below=False):
        
        sns.set_style(seaborn_style)
        
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        
        
        cal_data = self._calc_graphical_calibration_interval(y_value=y_value, estimated_mean=estimated_mean, alpha=alpha,
                                                             multiplier_type=multiplier_type, num_points=num_points)

        
        axs = self.plot_regression(seaborn_style=seaborn_style, num_points=cal_data['num_points'], legend_size=legend_size, x_values=cal_data['x_values'],
                                   legend_loc=legend_loc, conf_band=conf_band, alpha=alpha, point_color=point_color, reg_line_color=reg_line_color,
                                   conf_line_color=conf_line_color, pred_line_color=pred_line_color, conf_band_alpha=conf_band_alpha, pred_band=pred_band,
                                   pred_linestyle=pred_linestyle, grid=False, calibration_plot=True, axs=axs, multiplier_type=multiplier_type)
        
        left_hline_xs = [cal_data['upper_intersection_x'], cal_data['reg_intersection_x']]
        left_hline_ys = [cal_data['upper_intersection_y'], cal_data['reg_intersection_y']]
        
        right_hline_xs = [cal_data['reg_intersection_x'], cal_data['lower_intersection_x']]
        right_hline_ys = [cal_data['reg_intersection_y'], cal_data['lower_intersection_y']]
        
        left_vline_xs = [cal_data['upper_intersection_x'], cal_data['upper_intersection_x']]
        left_vline_ys = [0, cal_data['upper_intersection_y']]
        
        right_vline_xs= [cal_data['lower_intersection_x'], cal_data['lower_intersection_x']]
        right_vline_ys = [0, cal_data['lower_intersection_y']]
        
        # Lower and upper refer to the bands below and above the regression line.
        # Lower intersection = right
        # Upper intersection = left
        calibration_lower = cal_data['upper_intersection_x']
        calibration_upper = cal_data['lower_intersection_x']
        calibration_best = cal_data['reg_intersection_x']
        
        cal_lwr, cal_upr, best_est = self._round_ci(lower=calibration_lower, upper=calibration_upper, best=calibration_best, round_digits=round_digits)
        
        if alpha is None:
            pct = 100*(1-self.alpha)
        else:
            pct = 100*(1-alpha)
        
        if estimated_mean:
            cal_type = "Mean"
        else:
            cal_type = "Individual Pred"
        calibration_txt = (f"Calibration Interval for {cal_type} {self.explanatory_variable} at {self.response_variable}={y_value}\n"
                           f"{pct}% Calibration Interval=({cal_lwr},{cal_upr}) "
                           f"Best Estimate={best_est}")
        
        # Vertical lines directly down to the x-axis, from the point where the horizontal lines intersect with the upper and lower bands.
        axs.vlines(x=cal_data['upper_intersection_x'], ymin=0, ymax=cal_data['upper_intersection_y'], color=linecolor, linewidth=linewidth)
        axs.vlines(x=cal_data['lower_intersection_x'], ymin=0, ymax=cal_data['lower_intersection_y'], color=linecolor, linewidth=linewidth)
        
        # Horizontal lines, from regression line to the left and to the right, until they
        # intersect with the upper (left) and lower (right) bands.
        sns.lineplot(x=left_hline_xs, y=left_hline_ys, ax=axs, color=linecolor, linewidth=linewidth, label=calibration_txt)
        sns.lineplot(x=right_hline_xs, y=right_hline_ys, ax=axs, color=linecolor, linewidth=linewidth)
        
        sns.lineplot(x=right_hline_xs, y=right_hline_ys, ax=axs, color=linecolor, linewidth=linewidth)
        axs.plot(cal_data['upper_intersection_x'], cal_data['upper_intersection_y'], marker=".", markerfacecolor=marker_color, markeredgecolor=marker_color, markersize = marker_size)
        axs.plot(cal_data['reg_intersection_x'], cal_data['reg_intersection_y'], marker=calibration_markertype, markerfacecolor=marker_color, markeredgecolor=marker_color, markersize = marker_size)
        axs.plot(cal_data['lower_intersection_x'], cal_data['lower_intersection_y'], marker=".", markerfacecolor=marker_color, markeredgecolor=marker_color, markersize = marker_size)
             
        if annot_cal_interval_below:
            y_lwr_ci_txt = 0
            y_upper_ci_txt = 0
            y_best_est_ci_txt = 0
            axs.vlines(x=cal_data['reg_intersection_x'], ymin=0, ymax=cal_data['reg_intersection_y'], color=linecolor, linewidth=linewidth)
            style = dict(size=14, color = marker_color, ha = 'center', va="top", fontweight="bold")
        else:
            style = dict(size=14, color = marker_color, ha = 'center', va="bottom", fontweight="bold")
            y_lwr_ci_txt = cal_data['upper_intersection_y']
            y_upper_ci_txt = cal_data['lower_intersection_y']
            y_best_est_ci_txt = cal_data['reg_intersection_y']
        
        axs.text(x=cal_data['upper_intersection_x'], y=y_lwr_ci_txt, s=f"{cal_lwr}", **style)
        axs.text(x=cal_data['lower_intersection_x'], y=y_upper_ci_txt, s=f"{cal_upr}", **style)
        axs.text(x=cal_data['reg_intersection_x'], y=y_best_est_ci_txt, s=f"{best_est}", **style)
        
        wald_ci = self._calc_wald_calibration_ci(y0=y_value, multiplier_type=multiplier_type, mean_response=estimated_mean, alpha=alpha)
        wald_lwr, wald_upr, wald_best_est = self._round_ci(lower=wald_ci['lower_ci'], upper=wald_ci['upper_ci'], best=wald_ci['best_estimate'], round_digits=round_digits)
        
        wald_txt = (f"{pct}% Wald Calibration Interval\n"
                    f"Interval: ({wald_lwr}, {wald_upr})\n"
                    f"Best Estimate: {wald_best_est}")
        
        style = dict(size=14, color = wald_txt_color, ha = 'right', va="bottom", fontweight="bold")
        axs.text(1,0, s=wald_txt, **style, transform=axs.transAxes)
        
        '''
        print(f"num_points: {num_points}")
        print(f"y_value: {y_value}")
        print(f" upper_intersection_y: {upper_intersection_y}" )
        print(f" upper_intersection_arg: {upper_intersection_arg}" )
        print(f" upper_intersection_x: {upper_intersection_x}" )
        print(f" lower_intersection_y: {lower_intersection_y}" )
        print(f" lower_intersection_arg: {lower_intersection_arg}" )
        print(f" lower_intersection_x: {lower_intersection_x}" )
        '''
        return cal_data

    def _round_ci(self, lower, upper, best, round_digits):
        
        lwr = round(lower, round_digits)
        upr = round(upper, round_digits)
        best_est = round(best, round_digits)
        
        return lwr, upr, best_est
    
    def plot_regression(self, figsize=(15, 8), seaborn_style="darkgrid", num_points=1000, legend_size=14, legend_loc="best",
                        conf_band=True, alpha=None, point_color="#003F5C", reg_line_color="#ffa600", conf_line_color="#ef5675",
                        pred_line_color="#7a5195", conf_band_alpha=0.15, pred_band=True, pred_linestyle="--", grid=False, axs=None,
                        calibration_plot=False, multiplier_type='t', x_values=None):
        
        if axs is None:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        
        sns.scatterplot(x=self.explanatory_variable, y=self.response_variable, data=self.data, color=point_color, ax=axs)
        
        if x_values is None:
            # Get X, and Y values to plot the regression line.
            regression_x_values = np.linspace(start=self.statistics[f'{self.explanatory_variable}_min'],
                                              stop=self.statistics[f'{self.explanatory_variable}_max'],
                                              num=num_points)
        else:
            regression_x_values = x_values
        
        regression_y_values = [self._calculate_mean_response(x) for x in regression_x_values]
        sns.lineplot(x=regression_x_values, y=regression_y_values, label=self.statistics['regression_equation'], color=reg_line_color, ax=axs)
        
        conf_level = self._get_conf_level(alpha=alpha)
        
        # Prediction band
        if pred_band:
            prediction_band = self._calculate_prediction_band(x_values=regression_x_values, alpha=alpha, multiplier_type=multiplier_type)
            self.prediction_band_plot_data = prediction_band
            
            sns.lineplot(x=regression_x_values, y=prediction_band['lower'], color=pred_line_color, linestyle=pred_linestyle,label=f"{conf_level}% Prediction Band", ax=axs)
            sns.lineplot(x=regression_x_values, y=prediction_band['upper'], color=pred_line_color, linestyle=pred_linestyle,ax=axs)

        # Confidence band
        if conf_band:
            
            confidence_band = self._calculate_confidence_band(x_values=regression_x_values, alpha=alpha, multiplier_type=multiplier_type)
            self.confidence_band_plot_data = confidence_band
            
            sns.lineplot(x=regression_x_values, y=confidence_band['lower'], color=conf_line_color, label=f"{conf_level}% Confidence Band", ax=axs)
            sns.lineplot(x=regression_x_values, y=confidence_band['upper'], color=conf_line_color, ax=axs)
            axs.fill_between(x=regression_x_values, y1=confidence_band['upper'], y2=confidence_band['lower'], color=conf_line_color, alpha=conf_band_alpha)
                        
        axs.set_title(f"{self.response_variable} vs {self.explanatory_variable} Regression", fontsize=24, weight="bold")
        axs.set_ylabel(f"{self.response_variable}", fontsize=18, weight="bold")
        axs.set_xlabel(f"{self.explanatory_variable}", fontsize=18, weight="bold")
        axs.tick_params(labelsize="xx-large")
        axs.legend(loc=legend_loc, prop={'size':legend_size})
        
        if not grid and not calibration_plot:
            plt.show()
    
        return axs
    
    def plot_histogram_of_residuals(self, figsize=(12, 8), discrete=False, stat="density", normal_curve_pts=2_000, round_digits=4, axis_fontsize=18, studentized=False):
        
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        
        if studentized:
            resids=self.statistics['studentized_resids']=[resid/self.statistics['rmse'] for resid in self.statistics['residuals']]
            resid_type = " Studentized "
        else:
            resids=self.statistics['residuals']
            resid_type =" "
        
        sns.histplot(resids, ax=axs, discrete=discrete, stat=stat)
        
        mu, sd = stats.norm.fit(resids)
        
        x_min, x_max = axs.get_xlim()
        x_range = x_max - x_min
        x_norm = np.linspace(x_min - (x_range * 0.1), x_max + (x_range * 0.1), normal_curve_pts)
        y_norm = stats.norm.pdf(x_norm, mu, sd)
        axs.plot(x_norm, y_norm, color="#FE019A", linewidth=3)
        
        title = (f"Histogram of{resid_type}Residuals, regressing {self.response_variable} onto {self.explanatory_variable}\n"
                 f"Normal Curve Overlay, mu={round(mu,round_digits)}, sigma={round(sd,round_digits)}")           
        
        axs.set_title(title, fontsize=22, weight="bold")
        axs.tick_params(labelsize="xx-large")
        axs.set_xlabel("Residual", fontsize=axis_fontsize)
        axs.set_ylabel(f"{stat}", fontsize=axis_fontsize)

        