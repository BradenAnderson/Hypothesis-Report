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


class EquationImageGenerator():
    def __init__(self, first_class, second_class, test_value, sample_statistics,
                 alternative, format_params, project_directory, test_statistic, p_value,
                 test_type, samples_to_test_map, relative_images_directory, relative_file_paths,
                 test_statistic_type):
        
        self.first_class = first_class
        self.second_class = second_class
        self.test_value = test_value
        self.alternative = alternative
        self.format_params = format_params
        self.sample_statistics = sample_statistics
        self.project_directory = project_directory
        self.test_statistic = test_statistic
        self.p_value = p_value
        self.test_type = test_type
        self.samples_to_test_map = samples_to_test_map
        self.relative_images_directory = relative_images_directory
        self.relative_file_paths = relative_file_paths
        self.test_statistic_type = test_statistic_type
        
    def create_hypothesis_image(self):

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.format_params['hyp_figsize'])

        if self.test_type in self.samples_to_test_map["one_sample"] and self.alternative == "two_sided":

            # Null hypothesis, mu_first_class equals test_value
            null_string = f"$\mathbf{{H_0:}}$ $\mu_{{{self.first_class}}}$ = ${{{self.test_value}}}$"

            # Alt hypothesis is mu_first_class does not equal test value
            alt_string = f"$\mathbf{{H_a:}}$ $\mu_{{{self.first_class}}}$ $\\neq$ ${{{self.test_value}}}$"

        if self.test_type in self.samples_to_test_map["one_sample"] and self.alternative == "one_sided_left":

            # Null hypothesis: mu_first_class is greater than or equal to test_value
            null_string = f"$\mathbf{{H_0:}}$ $\mu_{{{self.first_class}}}$ $\\geq$ ${{{self.test_value}}}$"

            # Alt hypothesis: mu_first_class less than test value
            alt_string = f"$\mathbf{{H_a:}}$ $\mu_{{{self.first_class}}}$ < ${{{self.test_value}}}$"

        if self.test_type in self.samples_to_test_map["one_sample"] and self.alternative == "one_sided_right":

            # Null hypothesis: mu_first_class is less than or equal to test_value
            null_string = f"$\mathbf{{H_0:}}$ $\mu_{{{self.first_class}}}$ $\\leq$ ${{{self.test_value}}}$"

            # Alt hypothesis is mu_first_class greater than test_value 
            alt_string = f"$\mathbf{{H_a:}}$ $\mu_{{{self.first_class}}}$ > ${{{self.test_value}}}$"

        if self.test_type in self.samples_to_test_map["two_sample"] and self.alternative == "two_sided":

            # Null hypothesis, mu_first_class minus mu_second_class equals test_value
            null_string = f"$\mathbf{{H_0:}}$ $\mu_{{{self.first_class}}}$ - $\mu_{{{self.second_class}}}$ = ${{{self.test_value}}}$"

            # Alt hypothesis is mu_first_class minus mu_second_class does not equal test value
            alt_string = f"$\mathbf{{H_a:}}$ $\mu_{{{self.first_class}}}$ - $\mu_{{{self.second_class}}}$ $\\neq$ ${{{self.test_value}}}$"

        if self.test_type in self.samples_to_test_map["two_sample"] and self.alternative == "one_sided_left":

            # Null hypothesis, mu_first_class - mu_second_class greater than or equal to test_value
            null_string = f"$\mathbf{{H_0:}}$ $\mu_{{{self.first_class}}}$ - $\mu_{{{self.second_class}}}$ $\\geq$ ${{{self.test_value}}}$"

             # Alt hypothesis is mu_first_class - mu_second_class less than test value
            alt_string = f"$\mathbf{{H_a:}}$ $\mu_{{{self.first_class}}}$ - $\mu_{{{self.second_class}}}$ < ${{{self.test_value}}}$"

        if self.test_type in self.samples_to_test_map["two_sample"] and self.alternative == "one_sided_right":

            # Null hypothesis, mu_first_class - mu_second_class less than or equal to test_value
            null_string = f"$\mathbf{{H_0:}}$ $\mu_{{{self.first_class}}}$ - $\mu_{{{self.second_class}}}$ $\\leq$ ${{{self.test_value}}}$"

             # Alt hypothesis is mu_first_class - mu_second_class greater than test_value
            alt_string = f"$\mathbf{{H_a:}}$ $\mu_{{{self.first_class}}}$ - $\mu_{{{self.second_class}}}$ > ${{{self.test_value}}}$"

        if self.test_type in self.samples_to_test_map["paired"] and self.alternative == "two_sided":

            # Null hypothesis, mu_difference equals test value
            null_string = f"$\mathbf{{H_0:}}$ $\mu_{{{self.first_class}-{self.second_class}}}$ = ${{{self.test_value}}}$"

            # Alt hypothesis is mu_difference does not equal test_value
            alt_string = f"$\mathbf{{H_a:}}$ $\mu_{{{self.first_class}-{self.second_class}}}$ $\\neq$ ${{{self.test_value}}}$"

        if self.test_type in self.samples_to_test_map["paired"] and self.alternative == "one_sided_left":

            # Null hypothesis, mu_difference greater than or equal to test value
            null_string = f"$\mathbf{{H_0:}}$ $\mu_{{{self.first_class}-{self.second_class}}}$ $\\geq$ ${{{self.test_value}}}$"

            # Alt hypothesis is mu_difference less than test_value
            alt_string = f"$\mathbf{{H_a:}}$ $\mu_{{{self.first_class}-{self.second_class}}}$ < ${{{self.test_value}}}$"

        if self.test_type in self.samples_to_test_map["paired"] and self.alternative == "one_sided_right":

            # Null hypothesis, mu_difference greater than or equal to test value
            null_string = f"$\mathbf{{H_0:}}$ $\mu_{{{self.first_class}-{self.second_class}}}$ $\\leq$ ${{{self.test_value}}}$"

            # Alt hypothesis is mu_difference less than test_value
            alt_string = f"$\mathbf{{H_a:}}$ $\mu_{{{self.first_class}-{self.second_class}}}$ > ${{{self.test_value}}}$"

        if self.test_type == "permutation" and self.test_statistic_type in ['mean', 'median', 'rank_sum']:

            if self.test_statistic_type == "mean":
                symbol = "\mu"
            elif self.test_statistic_type in ['median', 'rank_sum']:
                symbol = "median"
            
            if self.alternative == "two_sided":

                # Null hypothesis, mu_first_class minus mu_second_class equals test_value
                null_string = f"$\mathbf{{H_0:}}$ ${{{{{symbol}}}_{{{self.first_class}}}}}$ - ${{{{{symbol}}}_{{{self.second_class}}}}}$ = ${{{self.test_value}}}$"

                # Alt hypothesis is mu_first_class minus mu_second_class does not equal test value
                alt_string = f"$\mathbf{{H_a:}}$ ${{{{{symbol}}}_{{{self.first_class}}}}}$ - ${{{{{symbol}}}_{{{self.second_class}}}}}$ $\\neq$ ${{{self.test_value}}}$"

            elif self.alternative == "one_sided_left":
            
                # Null hypothesis, mu_first_class - mu_second_class greater than or equal to test_value
                null_string = f"$\mathbf{{H_0:}}$ ${{{{{symbol}}}_{{{self.first_class}}}}}$ - ${{{{{symbol}}}_{{{self.second_class}}}}}$ $\\geq$ ${{{self.test_value}}}$"

                # Alt hypothesis is mu_first_class - mu_second_class less than test value
                alt_string = f"$\mathbf{{H_a:}}$ ${{{{{symbol}}}_{{{self.first_class}}}}}$ - ${{{{{symbol}}}_{{{self.second_class}}}}}$ < ${{{self.test_value}}}$"

            elif self.alternative == "one_sided_right":

                # Null hypothesis, mu_first_class - mu_second_class less than or equal to test_value
                null_string = f"$\mathbf{{H_0:}}$ ${{{{{symbol}}}_{{{self.first_class}}}}}$ - ${{{{{symbol}}}_{{{self.second_class}}}}}$ $\\leq$ ${{{self.test_value}}}$"

                # Alt hypothesis is mu_first_class - mu_second_class greater than test_value
                alt_string = f"$\mathbf{{H_a:}}$ ${{{{{symbol}}}_{{{self.first_class}}}}}$ - ${{{{{symbol}}}_{{{self.second_class}}}}}$ > ${{{self.test_value}}}$"

        if self.test_type == "permutation" and self.test_statistic_type == "signed_rank":

            symbol = "median"

            if self.alternative == "two_sided":

                # Null hypothesis, mu_difference equals test value
                null_string = f"$\mathbf{{H_0:}}$ ${{{{{symbol}}}_{{{self.first_class}-{self.second_class}}}}}$ = ${{{self.test_value}}}$"

                # Alt hypothesis is mu_difference does not equal test_value
                alt_string = f"$\mathbf{{H_a:}}$ ${{{{{symbol}}}_{{{self.first_class}-{self.second_class}}}}}$ $\\neq$ ${{{self.test_value}}}$"

            elif self.alternative == "one_sided_left":
                
                # Null hypothesis, mu_difference greater than or equal to test value
                null_string = f"$\mathbf{{H_0:}}$ ${{{{{symbol}}}_{{{self.first_class}-{self.second_class}}}}}$ $\\geq$ ${{{self.test_value}}}$"

                # Alt hypothesis is mu_difference less than test_value
                alt_string = f"$\mathbf{{H_a:}}$ ${{{{{symbol}}}_{{{self.first_class}-{self.second_class}}}}}$ < ${{{self.test_value}}}$"

            elif self.alternative == "one_sided_right":

                # Null hypothesis, mu_difference greater than or equal to test value
                null_string = f"$\mathbf{{H_0:}}$ ${{{{{symbol}}}_{{{self.first_class}-{self.second_class}}}}}$ $\\leq$ ${{{self.test_value}}}$"

                # Alt hypothesis is mu_difference less than test_value
                alt_string = f"$\mathbf{{H_a:}}$ ${{{{{symbol}}}_{{{self.first_class}-{self.second_class}}}}}$ > ${{{self.test_value}}}$"


        # Create the image filename
        img_filename = f"{self.format_params['hyp_img_save_name']}.{self.format_params['hyp_img_save_format']}"
        
        # Create the relative path to this image (from the project root).
        img_relative_path = os.path.join(self.relative_images_directory, img_filename)
        
        # Create the full save path for the image
        full_save_path = os.path.join(self.project_directory, os.path.normpath(img_relative_path))
        
        # Save the relative path to the dictionary of relative paths
        self.relative_file_paths['step_1_hyp_img'] = img_relative_path
        
        ax.text(self.format_params['hyp_null_x_coord'],
                self.format_params['hyp_null_y_coord'],
                null_string,
                fontsize=self.format_params['hyp_fontsize'],
                color=self.format_params['hyp_text_color'])
         
        ax.text(self.format_params['hyp_alt_x_coord'],
                self.format_params['hyp_alt_y_coord'],
                alt_string,
                fontsize=self.format_params['hyp_fontsize'],
                color=self.format_params['hyp_text_color'])

        ax.axis("off")
        
        # Save the image
        plt.savefig(full_save_path,
                    transparent=self.format_params['hyp_img_trnsprnt_bkgrd'],
                    format=self.format_params['hyp_img_save_format'])
        plt.close()
        
        # Return the relative path
        return img_relative_path
    
    def create_test_statistic_image(self):
    
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.format_params['ts_figsize'])

        if self.test_type == "one_sample_t":
        
            t_stat = (f"$\\mathbf{{t_{{{'stat'}}}}}$ = " 
                      f"$\\frac{{\\bar x_{{{self.first_class}}} - \mu_{{{'Null'}}}}}{{\\frac{{s}}{{\\sqrt{{n}}}}}}$"
                      "\n\n\t = "
                      f"$\\frac{{\\bar x_{{{self.first_class}}} - \mu_{{{'Null'}}}}}{{{{SE_\\barx}}}}$"
                      "\n\n\t = "
                      f"$\\frac{{{round(self.sample_statistics['mean'], 3)} - {self.test_value}}}{{{round(self.sample_statistics['std_error'], 3)}}}$ = " 
                      f" {round(self.test_statistic, 3)}")
            
        elif self.test_type == "students_t":
            
            t_stat = (f"$\\mathbf{{t_{{{'stat'}}}}}$ = " 
                      f"$\\frac{{(\\bar x_{{{self.first_class}}} - \\bar x_{{{self.second_class}}}) - \mu_{{diff\_means\_null}}}}"
                      f"{{s_p\\sqrt{{\\frac{{1}}{{n_{{{self.first_class}}}}} + \\frac{{1}}{{n_{{{self.second_class}}}}}}}}}$"
                      "\n\n\t = "
                      f"$\\frac{{(\\bar x_{{{self.first_class}}} - \\bar x_{{{self.second_class}}}) - \mu_{{diff\_means\_null}}}}"
                      f"{{SE_{{\\barx_{{{self.first_class}}} - \\barx_{{{self.second_class}}}}}}}$"
                      "\n\n\t"
                      f"= $\\frac{{{round(self.sample_statistics['diff_sample_means'],3)} - {{{self.test_value}}}}}" 
                      f"{{{{{round(self.sample_statistics['pooled_std_error'], 3)}}}}}$ = "
                      f" {round(self.test_statistic, 3)}")
        
        elif self.test_type == "welches_t":

            t_stat = (f"$\\mathbf{{t_{{{'stat'}}}}}$ = " 
                      f"$\\frac{{(\\bar x_{{{self.first_class}}} - \\bar x_{{{self.second_class}}}) - \mu_{{diff\_means\_null}}}}"
                      f"{{\\sqrt{{\\frac{{s^2_{{{self.first_class}}}}}{{n_{{{self.first_class}}}}} + \\frac{{s^2_{{{self.first_class}}}}}{{n_{{{self.second_class}}}}}}}}}$"
                      "\n\n\t = "
                      f"$\\frac{{(\\bar x_{{{self.first_class}}} - \\bar x_{{{self.second_class}}}) - \mu_{{diff\_means\_null}}}}"
                      f"{{SE_{{\\barx_{{{self.first_class}}} - \\barx_{{{self.second_class}}}}}}}$"
                      "\n\n\t"
                      f"= $\\frac{{{round(self.sample_statistics['diff_sample_means'],3)} - {{{self.test_value}}}}}" 
                      f"{{{{{round(self.sample_statistics['pooled_std_error'], 3)}}}}}$ = "
                      f" {round(self.test_statistic, 3)}")
        
        elif self.test_type == "paired_t":
            
            t_stat = (f"$\\mathbf{{t_{{{'stat'}}}}}$ = " 
                      f"$\\frac{{\\bar x_{{{{{self.first_class}}} - {{{self.second_class}}}}} - \mu_{{{'Null'}}}}}"
                      f"{{\\frac{{s_{{{{{self.first_class}}} - {{{self.second_class}}}}}}}{{\\sqrt{{n}}}}}}$"
                      "\n\n\t = "
                      f"$\\frac{{\\bar x_{{{{{self.first_class}}} - {{{self.second_class}}}}} - \mu_{{{'Null'}}}}}"
                      f"{{SE_{{\\barx_{{{{{self.first_class}}} - {{{self.second_class}}}}}}}}}$"
                      "\n\n\t = "
                      f"$\\frac{{{round(self.sample_statistics['mean'], 3)} - {self.test_value}}}{{{round(self.sample_statistics['std_error'], 3)}}}$ = " 
                      f" {round(self.test_statistic, 3)}")
            
        
        # Image file name for saving.
        img_filename = f"{self.format_params['ts_img_save_name']}.{self.format_params['ts_img_save_format']}"
        
        # Relative path to the image
        img_relative_path = os.path.join(self.relative_images_directory, img_filename)
        
        # Full path where we will save the image
        full_save_path = os.path.join(self.project_directory, os.path.normpath(img_relative_path))
        
        # Add the relative path to the dictionary of relative paths
        self.relative_file_paths['step_3_tstat_img'] = img_relative_path
    
        ax.text(x=self.format_params['ts_x_coord'],
                y=self.format_params['ts_y_coord'],
                s=t_stat,
                fontsize=self.format_params['ts_fontsize'],
                color=self.format_params['ts_text_color'])
         
        ax.axis("off")
        
        plt.savefig(full_save_path,
                    transparent=self.format_params['ts_img_trnsprnt_bkgrd'],
                    format=self.format_params['ts_img_save_format'])
        plt.close()
        
        return img_relative_path

    def create_p_value_image(self):

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=self.format_params['pv_figsize'])
        
        p_value_text = (f"$\\mathbf{{{'P-value'}}} = {{{round(self.p_value, self.format_params['pv_round_digits'])}}}$")

        # Image file name for saving.
        img_filename = f"{self.format_params['pv_img_save_name']}.{self.format_params['pv_img_save_format']}"
        
        # Relative path to the image
        img_relative_path = os.path.join(self.relative_images_directory, img_filename)
        
        # Full path where we will save the image
        full_save_path = os.path.join(self.project_directory, os.path.normpath(img_relative_path))
        
        # Add the relative path to the dictionary of relative paths
        self.relative_file_paths['step_4_pvalue_img'] = img_relative_path

        ax.text(x=self.format_params['pv_x_coord'],
                y=self.format_params['pv_y_coord'],
                s=p_value_text,
                fontsize=self.format_params['pv_fontsize'],
                color=self.format_params['pv_text_color'])

        ax.axis("off")

        plt.savefig(full_save_path,
                    transparent=self.format_params['pv_img_trnsprnt_bkgrd'],
                    format=self.format_params['pv_img_save_format'])

        plt. close()

        return img_relative_path