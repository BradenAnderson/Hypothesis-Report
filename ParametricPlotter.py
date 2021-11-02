
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import statsmodels.stats as smstats
import itertools
import mdutils
import latex2markdown
import copy
import os
import scipy.stats as stats
import statsmodels.stats.power as smp

class ParametricPlotter():
    def __init__(self, sample1, first_class, sample2, second_class, test_value, 
                 alpha, test_type, alternative, format_params, project_directory,
                 relative_images_directory, df, test_statistic, relative_file_paths):

        self.sample1 = sample1
        self.sample2 = sample2
        self.test_value = test_value
        self.alpha = alpha
        self.alternative = alternative
        self.format_params = format_params
        self.df = df
        self.test_statistic = test_statistic


        self.project_directory = project_directory
        self.relative_images_directory = relative_images_directory
        self.relative_file_paths = relative_file_paths

        self.neg_t_crit, self.pos_t_crit = self.calculate_critical_values()

        # None of these are currently used inside ParameterPlotter class! Only passed incase
        # a later use for them arises... 
        self.first_class = first_class    # Name of the first sample (e.g. "placebo")
        self.second_class = second_class  # Name of the second sample
        self.test_type = test_type        # Type of test (e.g. "students_t")

    def run_test(self):
        pass

    def get_critical_values(self):
        return self.neg_t_crit, self.pos_t_crit

    def calculate_critical_values(self):

        # q is the lower tail area. 
        # so this returns the critical value where the lower tail area is alpha/2
        if self.alternative == "two_sided":
            neg_t_crit = stats.t.ppf(q=self.alpha/2, df=self.df, loc=0, scale=1)
            pos_t_crit = np.abs(neg_t_crit)

        elif self.alternative == "one_sided_right":
            neg_t_crit = None
            pos_t_crit = np.abs(stats.t.ppf(q=self.alpha, df=self.df, loc=0, scale=1))

        elif self.alternative == "one_sided_left":
            neg_t_crit = stats.t.ppf(q=self.alpha, df=self.df, loc=0, scale=1)
            pos_t_crit = None

        return neg_t_crit, pos_t_crit
    
    def update_t_dist_plot_params(self, x):
        
        # While the test statistic will be off the plot to the left
        if np.min(x) > self.test_statistic:
        
            while np.min(x) > self.test_statistic:
                self.format_params['ds_max_ppf'] = self.format_params['ds_max_ppf'] + self.format_params['ds_min_ppf'] * 0.9
                self.format_params['ds_min_ppf'] /= 10
                self.format_params['ds_num_pts'] *= 10

                x = np.linspace(stats.t.ppf(self.format_params['ds_min_ppf'], self.df, loc=0, scale=1),
                                stats.t.ppf(self.format_params['ds_max_ppf'], self.df, loc=0, scale=1),
                                1000)
                
        elif np.max(x) < self.test_statistic: 
            
            while np.max(x) < self.test_statistic:
                self.format_params['ds_max_ppf'] = self.format_params['ds_max_ppf'] + self.format_params['ds_min_ppf'] * 0.9
                self.format_params['ds_min_ppf'] /= 10
                self.format_params['ds_num_pts'] *= 10

                x = np.linspace(stats.t.ppf(self.format_params['ds_min_ppf'], self.df, loc=0, scale=1),
                                stats.t.ppf(self.format_params['ds_max_ppf'], self.df, loc=0, scale=1),
                                1000)

        return x
    
    def draw_and_shade(self):

        sns.set_style(self.format_params['ds_plot_style'])

        fig, axs = plt.subplots(nrows=1, ncols=1,  figsize=self.format_params['ds_figsize'])
        
        #### DO SOMETHING WITH THESE LATER??? 
        mean, var, skew, kurt = stats.t.stats(self.df, moments = "mvsk")

        # ppf (percent point function) is the inverse of the cdf
        # Ex: t.ppf(0.95, 10) = 1.812
        # 1.812 is the critical value for a 95% onsided interval (95% of area is left of 1.812 in 10 df t-dist)
        # Also, 1.812 would be the upper and lower critical values for 90% two sided interval.
        x = np.linspace(stats.t.ppf(self.format_params['ds_min_ppf'], self.df, loc=0, scale=1),
                        stats.t.ppf(self.format_params['ds_max_ppf'], self.df, loc=0, scale=1),
                        self.format_params['ds_num_pts'])
        
        # self.sample_center
            
        # If the test statistic is way too extreme to try to draw, just note the value in the upper left corner.
        if np.abs(self.test_statistic) > self.format_params['ds_max_drawable_tstat']:
                
            draw_tstat = False
                
            # Format parameters for test-statistic text
            style = dict(size= self.format_params['ds_tstat_fontsize'],
                         color = self.format_params['ds_tstat_fontcolor'],
                         ha = "left",
                         va = "top")
            
            
            y = [stats.t.pdf(x_value, self.df) for x_value in x]
                
            tstat_string = ("Test Statistic too extreme to draw!\n"
                            f"Test Statistic = {round(self.test_statistic, 3)}")
            
            # Put the test statistic note in the upper left corner
            axs.text(np.min(x),
                     np.max(y) + 0.01,
                     tstat_string,
                     **style)
            
            # If the test statistic is just a little more extreme that the values we are currently plotting
            # but not more extreme than the max_drawable value, then adjust the x values we are plotting
            # so the test statistic is included nicely. 
        elif ((np.min(x) > self.test_statistic or np.max(x) < self.test_statistic) and    
                np.abs(self.test_statistic) < self.format_params['ds_max_drawable_tstat']):
                
            x = self.update_t_dist_plot_params(x)
            draw_tstat = True
                
        else:
            draw_tstat = True
        
        # USE THE PDF TO GET THE Y VALUES FOR THE T-DISTRIBUTION! 
        y = [stats.t.pdf(x_value, self.df) for x_value in x]
        max_y = np.max(y)
        
        # Defaults to placing the top of the critical line 80% of the way to the top of the figure
        crit_line_top = max_y * self.format_params['ds_crit_line_top']  
        
        # Defaults to placing the text 85% of the way to the top of the figure (5% above where the line stops).
        crit_text_y = max_y * self.format_params['ds_crit_text_y'] 
        
        tstat_text_y = max_y * self.format_params['ds_tstat_text_y']
        tstat_line_top = max_y * self.format_params['ds_tstat_line_top']
        
        # Draw the t-distribution
        sns.lineplot(x=x, y=y, ax=axs, color=self.format_params['ds_dist_line_color'])  # Default linecolor to black
        
        if draw_tstat:
            
            # Format parameters for test-statistic text
            style = dict(size= self.format_params['ds_tstat_fontsize'],
                            color = self.format_params['ds_tstat_fontcolor'],
                            ha = self.format_params['ds_tstat_ha'])
            
            # Add a vertical line for the test-statistic
            axs.vlines(x=self.test_statistic,
                        ymin=0,
                        ymax=tstat_line_top,
                        colors=self.format_params['ds_tstat_line_color'])


            # Write the test statistic just above the line indicating its location.
            axs.text(self.test_statistic,
                        tstat_text_y,
                        f"t_stat = {round(self.test_statistic, 3)}", **style)

        # Overwrite style to use the format parameters for the critical values now.
        style = dict(size= self.format_params['ds_crit_fontsize'],
                        color = self.format_params['ds_crit_fontcolor'],
                        ha = self.format_params['ds_crit_ha'])

        if self.neg_t_crit is not None:
            
            axs.vlines(x=self.neg_t_crit,
                        ymin=0,
                        ymax=crit_line_top,
                        colors=self.format_params['ds_crit_line_color'])
            
            lower_tail_x = [x_value for x_value in x if x_value <= self.neg_t_crit]
            
            plt.fill_between(lower_tail_x,
                                stats.t.pdf(lower_tail_x, self.df),
                                color=self.format_params['ds_crit_fill_color'])
            
            axs.text(self.neg_t_crit,
                        crit_text_y,
                        f"-t_crit = {round(self.neg_t_crit, 3)}", **style)

        if self.pos_t_crit is not None:
            
            axs.vlines(x=self.pos_t_crit,
                        ymin=0,
                        ymax=crit_line_top,
                        colors=self.format_params['ds_crit_line_color'])
            
            upper_tail_x = [x_value for x_value in x if x_value >= self.pos_t_crit]
            
            plt.fill_between(upper_tail_x,
                                stats.t.pdf(upper_tail_x, self.df),
                                color=self.format_params['ds_crit_fill_color'])
            
            axs.text(self.pos_t_crit,
                        crit_text_y,
                        f"t_crit = {round(self.pos_t_crit, 3)}", **style)
        
        title_string = (
        f"T-Distribution with {self.df} df"
        )

        axs.set_title(title_string,
                        weight=self.format_params['ds_fig_title_weight'],
                        fontsize=self.format_params['ds_fig_title_fontsize'])
                                                                    
        axs.tick_params(axis="both",
                        labelsize=self.format_params["ds_fig_tick_labelsize"])
        
        # Filename for saving this image
        img_filename = f"{self.format_params['ds_img_save_name']}.{self.format_params['ds_img_save_format']}"
        
        img_relative_path = os.path.join(self.relative_images_directory, img_filename)
        
        # Full path from root of project to save location for the image
        full_save_path = os.path.join(self.project_directory, os.path.normpath(img_relative_path))
        
        # Save the relative image page
        self.relative_file_paths['step_2_ds_img'] = img_relative_path
        
        # Save the figure using the full path
        plt.savefig(full_save_path,
                    transparent=self.format_params['ds_img_trnsprnt_bkgrd'],
                    format=self.format_params['ds_img_save_format'])
        
        # Return the relative path that needs to be added to the markdown file
        return img_relative_path
