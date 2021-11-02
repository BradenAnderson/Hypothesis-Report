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
import shutil
import scipy.stats as stats
import statsmodels.stats.power as smp
from SampleAnalyzer import SampleAnalyzer
from EquationImageGenerator import EquationImageGenerator
from ParametricPlotter import ParametricPlotter
from PermutationTest import PermutationTest
from PowerPlotter import PowerPlotter

class HypothesisReport():
    def __init__(self, sample1, first_class, sample2=None, second_class=None, test_value=0,
                 alpha=0.05, test_type="one_sample_t", num_permutations=None, test_statistic_type="mean", power_plot_types=None,
                 alternative="two_sided", project_directory=None, report_filename=None, report_title = "Hypothesis Test Report:",
                 confidence_intervals=None, format_args={}, duplicate_ci_tables=False, power_given_effectsize=None,
                 sample_size_given_power_args=None):
        
        
        ##### Need input validation and error checking! 
        self.valid_test_types = ["one_sample_t", "students_t", "welches_t", "paired_t", "permutation"]
        self.valid_alternatives = ["one_sided_right", "one_sided_left", "two_sided"]
        
        ## TROUBLESHOOTING PARAMS ONLY
        self.duplicate_ci_tables = duplicate_ci_tables
        
        ### PARAMS FOR INTERFACING WITH OTHER LIBRARIES
        
        # Map the terms I use for the alternative hypothesis choices
        # To the term Scipy uses (lets me easily convert to their convention without adopting it).
        self.scipy_alt_hyp_map = {"two_sided":"two-sided",
                                  "one_sided_right":"greater",
                                  "one_sided_left":"less"}
        
        # Maps the terms I use for alternative hypothesis choices to the terms statsmodels uses.
        self.sm_alt_hyp_map = {"two_sided": "two-sided",
                               "one_sided_right": "larger",
                               "one_sided_left": "smaller"}
        
        self.samples_to_test_map = {"one_sample" : ["one_sample_t"],
                                    "two_sample" : ["students_t", "welches_t", "paired_t", "permutation"],
                                    "paired" : ["paired_t"]}
        
        
        # Dictionary containing parameters that specify all report formatting choices.
        # This function updates the default formatting options based on any the user 
        # decided to specify in the format_args parameter
        self.format_params = self.set_formatting_params(format_args)
        
        # These need to be python lists containing sample1 and sample2
        # If doing a 1 sample test, then sample2 can be None.
        self.sample1=sample1
        self.sample2=sample2
        self.num_permutations = num_permutations  # number of permutations to use (if doing a permutation test)
        self.first_class = first_class    # String - what we are calling the first class (e.g. Placebo)
        self.second_class = second_class  # String - what we are calling the second class (e.g. Ritalin)
        self.test_value = test_value      # Value assumed under the null (usually zero)
        self.test_type = test_type        # "permutation", "one_sample_t", "students_t", "welches_t", "paired_t"
        self.alternative = alternative    # "one_sided_right", "one_sided_left", "two_sided"
        self.alpha = alpha
        self.test_statistic_type = test_statistic_type

        self.valid_power_calcs = ["paired_t", "students_t", "one_sample_t"]   # which test types have power calc functions implemented (hopefully this can be all eventually)...
        self.pwr_plots_created = False  # Flag that monitors if power plots have been created to we don't recreate (maybe not necessary)

        # Relative location for image storage, relative to the project directory ".".
        self.relative_images_directory = "./images/"
        self.relative_file_paths = {}                  # Dictionary mapping each image to its relative path
                                                       # Need relative path for embedding in markdown. Otherwise path breaks whenever folder moves.        
        
        # Set up the project directory. This is the root folder, all images and the
        # output report will be stored here or a folder below it.
        self.project_directory = self.create_project_directory(project_directory)
        self.report_filename = self.create_report_filename(report_filename)
        
        # Markdown file which will be created when the build_report method is called.
        self.report = mdutils.MdUtils(file_name=os.path.join(self.project_directory, self.report_filename),
                                      title=report_title)
        
        # Instantiate the SampleAnalyzer class. This class takes care of generating all the necessary sample statistics.
        self.sample_analyzer = SampleAnalyzer(sample1=sample1, sample2=sample2, test_value=test_value,
                                              alpha=alpha, test_type=test_type, alternative=alternative, valid_power_calcs=self.valid_power_calcs,
                                              confidence_intervals=confidence_intervals, valid_alternatives=self.valid_alternatives, first_class=self.first_class,
                                              second_class=self.second_class)
        

        self.permutation_tester = self._create_permutation_tester()


        # Get statistics from the sample_analyzer
        self.df = self.sample_analyzer.get_degrees_of_freedom()
        self.standard_error = self.sample_analyzer.get_standard_error()
        self.sample_statistics = self.sample_analyzer.get_sample_statistics()
        
        # Get the confidence intervals from the sample analyzer (redundant calculations, for checking implementation only, both are not needed).
        self.manual_conf_intervals = self.sample_analyzer.get_manual_confidence_intervals()
        self.conf_intervals = self.sample_analyzer.get_scipy_confidence_intervals()

        ##### START POWER SECTION

        self.power_plotter = PowerPlotter(test_type=self.test_type, test_value=self.test_value, relative_imgs_dir=self.relative_images_directory,
                                        project_dir=self.project_directory, alpha=self.alpha, alternative=self.alternative, sample_statistics=self.sample_statistics)

        ##self.power_plot_types = self.get_power_plot_types(desired_plot_types=power_plot_types, test_type=test_type)

        ### Power tables

        self.power_given_effectsize = power_given_effectsize
        self.samples_at_pwr_threshs_s1_fixed = None
        self.samples_at_pwr_threshs_ratio_fixed = None

        if self.power_given_effectsize is not None:
            self.pwr_effectsize_dict = self.sample_analyzer.calculate_power_given_effect_table(effect_sizes=self.power_given_effectsize)
        # Put an else here with a default value based on the observed effect size?

        # Add a function that lets you just pass the alternate effect size by checking data types...
        if sample_size_given_power_args is None:
            self.sample_size_pwr_args = {"power": [0.75, 0.80, 0.85, 0.90, 0.925]}
        else:
            self.sample_size_pwr_args = sample_size_given_power_args

        return_value = self.sample_analyzer.calculate_sample_size_given_power_table(power_args=self.sample_size_pwr_args)

        if return_value is not None:
            self.samples_at_pwr_threshs_s1_fixed, self.samples_at_pwr_threshs_ratio_fixed = return_value


        ##### END POWER SECTION

        # Run the test to get the test statistic and p_value
        self.test_statistic, self.p_value = self.run_test()

        # Equation generator for creating the visualizations for the 6 step calculations...
        self.equation_generator = EquationImageGenerator(first_class=first_class, second_class=second_class, test_value=test_value,
                                                         sample_statistics=self.sample_statistics, alternative=alternative,
                                                         format_params=self.format_params, project_directory=self.project_directory,
                                                         test_statistic=self.test_statistic, p_value=self.p_value, test_type=test_type,
                                                         samples_to_test_map=self.samples_to_test_map, relative_images_directory = self.relative_images_directory,
                                                         relative_file_paths=self.relative_file_paths, test_statistic_type=self.test_statistic_type)

        self.hypothesis_img_filename = self.equation_generator.create_hypothesis_image()

        # Instantiate the ParametricPlotter for the draw_and_shade step. This is only applicable to t-tests though (not permutation tests).
        if self.test_type in ['one_sample_t', "students_t", 'welches_t', 'paired_t']:

            self.parametric_plotter = ParametricPlotter(sample1=self.sample1, first_class=first_class, sample2=sample2, second_class=second_class,
                                                        test_value=test_value, alpha=alpha, test_type=test_type, format_params=self.format_params,
                                                        project_directory=self.project_directory, relative_images_directory=self.relative_images_directory,
                                                        test_statistic=self.test_statistic, relative_file_paths=self.relative_file_paths, df=self.df,
                                                        alternative=alternative)

            self.neg_t_crit, self.pos_t_crit = self.parametric_plotter.get_critical_values()
    

    def _create_permutation_tester(self):

        if self.test_type == "permutation":

            return PermutationTest(num_permutations=self.num_permutations, sample1=self.sample1, sample2=self.sample2, test_type=self.test_type,
                                                    alternative=self.alternative, alpha=self.alpha, test_statistic_type=self.test_statistic_type,
                                                    format_params=self.format_params, relative_file_paths=self.relative_file_paths,
                                                    relative_images_directory=self.relative_images_directory, test_value=self.test_value,
                                                    project_directory=self.project_directory)
        else:
            return None

    def run_test(self):
        
        if self.test_type == "permutation":
            test_statistic, p_value = self.permutation_tester.run_test()
        
        elif self.test_type == "students_t":
            test_statistic, p_value = stats.ttest_ind(a=self.sample1,
                                                      b=self.sample2,
                                                      equal_var=True,
                                                      alternative=self.scipy_alt_hyp_map[self.alternative])
        
        elif  self.test_type == "welches_t":
            test_statistic, p_value = stats.ttest_ind(a=self.sample1,
                                                      b=self.sample2,
                                                      equal_var=False,
                                                      alternative=self.scipy_alt_hyp_map[self.alternative])
            
        elif self.test_type == "paired_t":
            test_statistic, p_value = stats.ttest_rel(a=self.sample1,
                                                      b=self.sample2,
                                                      alternative=self.scipy_alt_hyp_map[self.alternative])
            
        elif self.test_type == "one_sample_t":
            test_statistic, p_value = stats.ttest_1samp(a=self.sample1,
                                                        popmean=self.test_value,
                                                        alternative=self.scipy_alt_hyp_map[self.alternative])
            
        
        return test_statistic, p_value
    
    def build_output_file(self):

        if self.test_type in ['one_sample_t', 'students_t', 'paired_t', 'welches_t']:
            self.build_ttest_output_file()
        
        elif self.test_type == "permutation":
            self.build_permutation_test_output_file()

        # Always last
        self.report.create_md_file()

    def build_permutation_test_output_file(self):

        self.add_opening_summary_to_report()

        # Step 1: Identify the null and alternative hypothesis
        self.add_null_and_alt_hypothesis_to_report()

        # Step 2: Draw distribution
        self.report.new_line()
        self.add_permutation_distribution_to_report()

        self.report.new_header(level=1, title='Step 3 - Calculate the Test Statistic')   # Step 5

        # Step 4: Calculate the p-value
        self.add_p_value_to_report()

        self.report.new_header(level=1, title='Step 5 - Reject or fail to reject:')   # Step 5
        self.report.new_header(level=1, title='Step 6 - Conclusion:')                 # Step 6

        self.add_permutation_summary_table_to_report()
        
    def build_ttest_output_file(self):
        
        self.add_opening_summary_to_report()

        # Step 1: Identify the null and alternative hypothesis
        self.add_null_and_alt_hypothesis_to_report()
        
        # Step 2: Draw and shade, and find the critical value
        self.add_draw_and_shade_to_report()
        
        # Step 3: Calculate test statistic
        self.add_test_statistic_to_report()
        
        # Step 4: Calculate the p-value
        self.add_p_value_to_report()

        self.report.new_header(level=1, title='Step 5 - Reject or fail to reject:')   # Step 5
        self.report.new_header(level=1, title='Step 6 - Conclusion:')                 # Step 6
        
        # Add confidence intervals table 
        self.add_scipy_confidence_intervals_to_report()

        # If we want to display the confidence interval table that comes from the
        # manual calculations (for compairson only, duplicate of above).
        if self.duplicate_ci_tables:      
            self.add_manual_confidence_intervals_to_report()
        
        # Add table of summary statistics
        self.add_summary_statistics_table()

        # Add power plots
        if self.test_type in ["paired_t", "one_sample_t", "students_t"]:
            self.add_power_plots_to_report()

        # Add a small table that shows the power of the test that was run for each
        # of the three possible choices of alternative hypothesis.
        if self.test_type in self.valid_power_calcs and False:
            self.add_test_power_to_report()

            if self.power_given_effectsize is not None:
                self.add_power_vs_effectsize_table_to_report()

            self.add_power_sample_size_tables_to_report()

        # Add the power plots to the report
        # self.add_power_plots_to_report()

    def create_power_sample_size_tables(self):
        
        # This dictionary will hold two other dictionaries, one for each table.
        table_dicts = dict.fromkeys(["table_s1_fixed", "table_ratio_fixed"])

        # Table #1: Power calcs where number of samples in class 1 is fixed.
        table_s1_fixed_rows = [f"samples_{self.first_class}", f"samples_{self.second_class}", "ratio", "power"]

        samples_s1 = self.samples_at_pwr_threshs_s1_fixed[f"samples_{self.first_class}"]
        samples_s2 = self.samples_at_pwr_threshs_s1_fixed[f"samples_{self.second_class}"]
        r = self.samples_at_pwr_threshs_s1_fixed[f"ratio"]
        pwr = self.samples_at_pwr_threshs_s1_fixed[f"power"]

        num_rows = len(table_s1_fixed_rows)
        num_columns = len(pwr) + 1

        for s1, s2, ratio, power in zip(samples_s1, samples_s2, r, pwr):
            table_s1_fixed_rows.extend([round(s1, 4), round(s2, 4), round(ratio, 4), round(power, 4)])
        
        table_dicts["table_s1_fixed"] = {"table_data": table_s1_fixed_rows, "num_rows":num_rows, "num_columns":num_columns}
        
        # Starting table #2: ratio fixed
        table_ratio_fixed_rows = [f"samples_{self.first_class}", f"samples_{self.second_class}", "ratio", "power"]

        samples_s1 = self.samples_at_pwr_threshs_ratio_fixed[f"samples_{self.first_class}"]
        samples_s2 = self.samples_at_pwr_threshs_ratio_fixed[f"samples_{self.second_class}"]
        r = self.samples_at_pwr_threshs_ratio_fixed[f"ratio"]
        pwr = self.samples_at_pwr_threshs_ratio_fixed[f"power"]

        num_rows = len(table_ratio_fixed_rows)
        num_columns = len(pwr) + 1

        for s1, s2, ratio, power in zip(samples_s1, samples_s2, r, pwr):
            table_ratio_fixed_rows.extend([round(s1, 4), round(s2, 4), round(ratio, 4), round(power, 4)])
        
        table_dicts["table_ratio_fixed"] = {"table_data": table_ratio_fixed_rows, "num_rows":num_rows, "num_columns":num_columns}

        return table_dicts

    def add_power_sample_size_tables_to_report(self):

        tables = self.create_power_sample_size_tables()

        s1_fixed_table = tables['table_s1_fixed']
        ratio_fixed_table = tables['table_ratio_fixed']

        self.report.new_line()
        self.report.new_line(f"Calculating Sample Size at Power Levels when size of \"{self.first_class}\" group is fixed")
        self.report.new_table(columns=s1_fixed_table['num_columns'],
                              rows=s1_fixed_table['num_rows'],
                              text=s1_fixed_table['table_data'],
                              text_align='center')
        
        self.report.new_line()
        self.report.new_line(f"Calculating Sample Size at Power Levels when ratio \"{self.second_class}/{self.first_class}\" (s2/s1) group is fixed")
        self.report.new_table(columns=ratio_fixed_table['num_columns'],
                              rows=ratio_fixed_table['num_rows'],
                              text=ratio_fixed_table['table_data'],
                              text_align='center')

        

    def add_power_vs_effectsize_table_to_report(self):

        pwr_table_info = self.create_power_effectsize_table()
        self.report.new_line()
        self.report.new_table(columns=pwr_table_info['num_columns'],
                              rows=pwr_table_info['num_rows'],
                              text=pwr_table_info['table_data'],
                              text_align='center')

    def add_permutation_summary_table_to_report(self):

        table_info = self.permutation_tester.create_permutation_summary_table()

        self.report.new_line()
        self.report.new_header(level=1, title='Permutation Summary Data')
        self.report.new_line("\n\n")
        self.report.new_table(columns=table_info['num_columns'],
                              rows=table_info['num_rows'],
                              text=table_info['table_data'],
                              text_align='left')
        self.report.new_line()

    def create_power_effectsize_table(self):

        table_rows = ["Effect Size", "Cohens d", "Power"]
        num_columns = len(table_rows)
        num_rows = len(self.pwr_effectsize_dict['power']) + 1

        for effect, cohen, pwr in zip(self.pwr_effectsize_dict['effect_size'], self.pwr_effectsize_dict['cohens_d'], self.pwr_effectsize_dict['power']):
            table_rows.extend([effect, round(cohen, 4), round(pwr, 5)])

        return {'table_data': table_rows, 'num_rows':num_rows, 'num_columns': num_columns}

    def conf_dict_to_table(self, conf_dict):
        
        # The confidence level for the confidence interval that will agree with 
        # the hypothesis test we ran. (depends on if we ran a 1 or two sided test).
        agreeing_conf_lvl = self.sample_analyzer.get_agreeing_conf_level()
        
        table_rows = ["Confidence Level ", "Lower Bound ", "Upper Bound "]
        
        num_columns = len(table_rows)
        num_rows = len(conf_dict.keys()) + 1 # One row for each CI, plus header row
        
        for key, value in conf_dict.items():
            lower_bound, upper_bound = value
            rounded_lower_bound = round(lower_bound, self.format_params['ci_round_digits'])
            rounded_upper_bound = round(upper_bound, self.format_params['ci_round_digits'])
            
            if key == agreeing_conf_lvl:
                table_rows.extend([f" <font color={self.format_params['ci_agreeing_lvl_color']}>**{key * 100}%**</font> ", 
                                   f" <font color={self.format_params['ci_agreeing_lvl_color']}>**{rounded_lower_bound}**</font> ", 
                                   f" <font color={self.format_params['ci_agreeing_lvl_color']}>**{rounded_upper_bound}</font>** "])
            else:
                table_rows.extend([f"{key * 100}%", rounded_lower_bound, rounded_upper_bound])
            
        table_info = {"table_data" : table_rows,
                      "num_rows": num_rows,
                      "num_columns": num_columns}
        
        return table_info
    
    # Get the markdown for a table that shows the power values for each alterntive hypothesis option
    def create_power_table(self):
        
        table_rows = ["Test Alternative ", "Power "]
        num_columns = len(table_rows)
        pwr_dict = {key:value for key,value in self.sample_statistics.items() if key.startswith("pwr")}
        num_rows = len(pwr_dict.keys()) + 1
        
        for key, value in pwr_dict.items():
            power = round(value, self.format_params["pwr_round_digits"])
            alternative = key.strip("pwr_")
            
            # If this is the power for the test we actually ran, highlight it.
            if alternative == self.alternative:
                
                table_rows.extend([f" <font color={self.format_params['pwr_agreeing_test_color']}>**{alternative}**</font> ",
                                   f" <font color={self.format_params['pwr_agreeing_test_color']}>**{power}**</font> " ])
            else:
                 table_rows.extend([f"{alternative}", power])
        
        table_info = {"table_data" : table_rows,
                      "num_rows": num_rows,
                      "num_columns": num_columns}
        
        return table_info
    
    
    def add_opening_summary_to_report(self):

        if self.test_type == "one_sample_t":

            table_data = ["Test Type: ", f"{self.test_type}",
                          "Alternative: ", f"{self.alternative}",
                          "Sample1 Name: ", f"{self.first_class}"]

            ncols = 2
            nrows = 3

        else:
            table_data = ["Test Type: ", f"{self.test_type}",
                          "Alternative: ", f"{self.alternative}",
                          "Sample1 Name: ", f"{self.first_class}",
                          "Sample2 Name: ", f"{self.second_class}",
                          "Test Statistic", f"{self.test_statistic_type}"]

            ncols = 2
            nrows = 5

        self.report.new_line()
        self.report.new_table(columns=ncols,
                              rows=nrows,
                              text=table_data,
                              text_align='center')
    
    def add_null_and_alt_hypothesis_to_report(self):
        
        self.report.new_header(level=1, title='Step 1 - Identify the null and alternative hypothesis:')

        hyp_img = self.report.new_inline_image(text="Step 1 - Identify the null and alternative hypothesis",
                                               path=self.relative_file_paths['step_1_hyp_img'])
        
        self.report.new_paragraph(f"{hyp_img}")

    def add_power_plots_to_report(self):

        self.power_plotter.build_power_plots()

        plot_filepaths = self.power_plotter.get_plot_relative_paths()

        self.report.new_header(level=1, title="Power Plots")

        self.report.new_line()
        self.report.new_line("Note: Regardless of what alternative hypothesis was chosen, power plots are shown for the associated two sided test.")
        self.report.new_line()

        for filename, filepath in plot_filepaths.items():

            img = self.report.new_inline_image(text=f"{filename}", path=filepath)

            self.report.new_paragraph(f"{img}")

            self.report.new_line()


    def add_draw_and_shade_to_report(self):

        self.report.new_header(level=1, title='Step 2 - Find the critical value, draw and shade the distribution:')
        
        ds_img_filename = self.parametric_plotter.draw_and_shade()
        
        ds_img = self.report.new_inline_image(text="Step 2 - Find t_crit, draw and shade",
                                              path=ds_img_filename)
        
        self.report.new_paragraph(f"{ds_img}")

    def add_permutation_distribution_to_report(self):
        
        self.report.new_header(level=1, title='Step 2 - Find the critical value, draw and shade the distribution:')

        perm_dist_img_filename = self.permutation_tester.create_permutation_distribution_plot()

        perm_dist_img = self.report.new_inline_image(text="Step 2 - Find critical vals, draw and shade",
                                                     path=perm_dist_img_filename)

        self.report.new_paragraph(f"{perm_dist_img}")

    def add_p_value_to_report(self):

        self.report.new_header(level=1, title='Step 4 - Find the p-value:')

        pv_img_filename = self.equation_generator.create_p_value_image()

        pv_img = self.report.new_inline_image(text="Step 4 - Calculate P-Value",
                                              path=pv_img_filename)

        self.report.new_paragraph(f"{pv_img}")

    def add_test_statistic_to_report(self):

        self.report.new_header(level=1, title='Step 3 - Calculate the test statistic:')
        
        ts_img_filename = self.equation_generator.create_test_statistic_image()
        
        ts_img = self.report.new_inline_image(text="Step 3 - Calculate the test statistic",
                                              path=ts_img_filename)
        
        self.report.new_paragraph(f"{ts_img}")
        
        note_string = ("Note: The final test statistic (RHS above) was calculated using SciPy, however the intermediate values shown "
                       "above (mean, standard error, etc.) were calculated separately and included in the equations above for "
                       "completeness. Any differences between the LHS and RHS above should be minimal and simply the result of "
                       "different rounding errors between the manual calculates and the calculations SciPy performed")

        self.report.new_paragraph(note_string)

    def add_scipy_confidence_intervals_to_report(self):

        # Add header for confidence Intervals Tables
        self.report.new_header(level=1, title="Confidence Intervals")
        
        # Turn the dictionary of confidence intervals into a markdown table.
        scipy_ci = self.conf_dict_to_table(self.conf_intervals)
        
        self.report.new_line("\n\nConfidence Intervals Table:\n",
                             bold_italics_code="",
                             color=self.format_params['ci_text_color'])
        
        # Add the markdown table.
        self.report.new_table(columns=scipy_ci['num_columns'],
                              rows=scipy_ci['num_rows'],
                              text=scipy_ci['table_data'],
                              text_align='center')

    def add_manual_confidence_intervals_to_report(self):

        manual_ci = self.conf_dict_to_table(self.manual_conf_intervals)

        self.report.new_line("\n\nConfidence Intervals Manual Calculations:\n",
                             bold_italics_code="",
                             color=self.format_params['ci_text_color'])

        self.report.new_table(columns=manual_ci['num_columns'],
                              rows=manual_ci['num_rows'],
                              text=manual_ci['table_data'],
                              text_align='center')
    
    def add_summary_statistics_table(self):

        self.report.new_header(level=1, title='Summary Statistics')

        if self.test_type == "one_sample_t":

            table_data = ["Statistic", "Value",
                          "Sample Mean: ", f"{round(self.sample_statistics['mean'], self.format_params['summary_stats_round_digits'])}",
                          "Sample SD: ", f"{round(self.sample_statistics['sample_sd'], self.format_params['summary_stats_round_digits'])}",
                          "Sample Variance: ", f"{round(self.sample_statistics['variance'], self.format_params['summary_stats_round_digits'])}",
                          "Sample Size: ", f"{round(self.sample_statistics['n'], self.format_params['summary_stats_round_digits'])}",
                          "Standard Error: ", f"{round(self.sample_statistics['std_error'], self.format_params['summary_stats_round_digits'])}",
                          "Max: ", f"{round(self.sample_statistics['max'], self.format_params['summary_stats_round_digits'])}",
                          "Min: ", f"{round(self.sample_statistics['min'], self.format_params['summary_stats_round_digits'])}",
                          "Median: ", f"{round(self.sample_statistics['median'], self.format_params['summary_stats_round_digits'])}",
                          "Cohens d: ", f"{round(self.sample_statistics['cohens_d'], self.format_params['summary_stats_round_digits'])}"]

            self.report.new_table(columns=2,
                                  rows=10,
                                  text=table_data,
                                  text_align='center')
        
        elif self.test_type == "paired_t":

            s1_table_data = ["Sample 1 Statistics", "Value",
                             "Mean: ", f"{round(self.sample_statistics['s1_sample_mean'], self.format_params['summary_stats_round_digits'])}",
                             "SD: ", f"{round(self.sample_statistics['s1_sample_sd'], self.format_params['summary_stats_round_digits'])}",
                             "Variance: ", f"{round(self.sample_statistics['s1_var'], self.format_params['summary_stats_round_digits'])}",
                             "Sample Size: ", f"{round(self.sample_statistics['s1_n'], self.format_params['summary_stats_round_digits'])}",
                             "Standard Error: ", f"{round(self.sample_statistics['s1_std_error'], self.format_params['summary_stats_round_digits'])}",
                             "Max: ", f"{round(self.sample_statistics['s1_max'], self.format_params['summary_stats_round_digits'])}",
                             "Min: ", f"{round(self.sample_statistics['s1_min'], self.format_params['summary_stats_round_digits'])}",
                             "Median: ", f"{round(self.sample_statistics['s1_median'], self.format_params['summary_stats_round_digits'])}"]
            
            s2_table_data = ["Sample 2 Statistics", "Value",
                             "Mean: ", f"{round(self.sample_statistics['s2_sample_mean'], self.format_params['summary_stats_round_digits'])}",
                             "SD: ", f"{round(self.sample_statistics['s2_sample_sd'], self.format_params['summary_stats_round_digits'])}",
                             "Variance: ", f"{round(self.sample_statistics['s2_var'], self.format_params['summary_stats_round_digits'])}",
                             "Sample Size: ", f"{round(self.sample_statistics['s2_n'], self.format_params['summary_stats_round_digits'])}",
                             "Standard Error: ", f"{round(self.sample_statistics['s2_std_error'], self.format_params['summary_stats_round_digits'])}",
                             "Max: ", f"{round(self.sample_statistics['s2_max'], self.format_params['summary_stats_round_digits'])}",
                             "Min: ", f"{round(self.sample_statistics['s2_min'], self.format_params['summary_stats_round_digits'])}",
                             "Median: ", f"{round(self.sample_statistics['s2_median'], self.format_params['summary_stats_round_digits'])}"]

            differences_table_data = ["Sample of Differences Statistics", "Value",
                                      "Mean: ", f"{round(self.sample_statistics['mean'], self.format_params['summary_stats_round_digits'])}",
                                      "SD: ", f"{round(self.sample_statistics['sample_sd'], self.format_params['summary_stats_round_digits'])}",
                                      "Variance: ", f"{round(self.sample_statistics['var'], self.format_params['summary_stats_round_digits'])}",
                                      "Sample Size: ", f"{round(self.sample_statistics['n'], self.format_params['summary_stats_round_digits'])}",
                                      "Standard Error: ", f"{round(self.sample_statistics['std_error'], self.format_params['summary_stats_round_digits'])}",
                                      "Max: ", f"{round(self.sample_statistics['max'], self.format_params['summary_stats_round_digits'])}",
                                      "Min: ", f"{round(self.sample_statistics['min'], self.format_params['summary_stats_round_digits'])}",
                                      "Median: ", f"{round(self.sample_statistics['median'], self.format_params['summary_stats_round_digits'])}",
                                      "Cohens d: ", f"{round(self.sample_statistics['cohens_d'], self.format_params['summary_stats_round_digits'])}"]



            # If we want to stack the summary tables horizontally (otherwise they stack vertically)... 
            # I think horizontal looks better :) but go ahead and change this param to False to try out vertical!
            if self.format_params["summary_tables_oriented_horizontal"]:
                s1_table = mdutils.tools.Table.Table().create_table(columns=2, rows=9, text=s1_table_data, text_align="left")
                s2_table = mdutils.tools.Table.Table().create_table(columns=2, rows=9, text=s2_table_data, text_align="left")
                diff_table = mdutils.tools.Table.Table().create_table(columns=2, rows=10, text=differences_table_data, text_align="left")
                self.report.new_paragraph("<table><tr><td> " + s1_table + " </td><td> " + s2_table + " </td><td> " + diff_table + " </td></tr></table>")

            else: # else stack the three tables vertically

                self.report.new_table(columns=2,
                                      rows=9,
                                      text=s1_table_data,
                                      text_align='center')

                self.report.new_table(columns=2,
                                      rows=9,
                                      text=s2_table_data,
                                      text_align='center')

                self.report.new_table(columns=2,
                                      rows=10,
                                      text=differences_table_data,
                                      text_align='center')

        else: ### Else its students_t, welches_t or permutation (perm probably needs its own!)
            s1_table_data = [f"{self.first_class}_stats", "Value",
                             "Mean:", f"{round(self.sample_statistics['s1_sample_mean'], self.format_params['summary_stats_round_digits'])}",
                             "SD:", f"{round(self.sample_statistics['s1_sample_sd'], self.format_params['summary_stats_round_digits'])}",
                             "Var:", f"{round(self.sample_statistics['s1_var'], self.format_params['summary_stats_round_digits'])}",
                             "Sample_Size:", f"{round(self.sample_statistics['s1_n'], self.format_params['summary_stats_round_digits'])}",
                             "Standard_Error:", f"{round(self.sample_statistics['s1_std_error'], self.format_params['summary_stats_round_digits'])}",
                             "Max:", f"{round(self.sample_statistics['s1_max'], self.format_params['summary_stats_round_digits'])}",
                             "Min:", f"{round(self.sample_statistics['s1_min'], self.format_params['summary_stats_round_digits'])}",
                             "Median:", f"{round(self.sample_statistics['s1_median'], self.format_params['summary_stats_round_digits'])}"]

            s2_table_data = [f"{self.second_class}_stats", "Value",
                             "Mean:", f"{round(self.sample_statistics['s2_sample_mean'], self.format_params['summary_stats_round_digits'])}",
                             "SD:", f"{round(self.sample_statistics['s2_sample_sd'], self.format_params['summary_stats_round_digits'])}",
                             "Var:", f"{round(self.sample_statistics['s2_var'], self.format_params['summary_stats_round_digits'])}",
                             "Sample_Size:", f"{round(self.sample_statistics['s2_n'], self.format_params['summary_stats_round_digits'])}",
                             "Standard_Error:", f"{round(self.sample_statistics['s2_std_error'], self.format_params['summary_stats_round_digits'])}",
                             "Max:", f"{round(self.sample_statistics['s2_max'], self.format_params['summary_stats_round_digits'])}",
                             "Min:", f"{round(self.sample_statistics['s2_min'], self.format_params['summary_stats_round_digits'])}",
                             "Median:", f"{round(self.sample_statistics['s2_median'], self.format_params['summary_stats_round_digits'])}"]

            diff_means_data = ["Combined Stats", "Value",
                               "Diff Means:", f"{round(self.sample_statistics['diff_sample_means'], self.format_params['summary_stats_round_digits'])}",
                               "Pooled SD:", f"{round(self.sample_statistics['pooled_sd'], self.format_params['summary_stats_round_digits'])}",
                               "Satterthwaite df:", f"{round(self.sample_statistics['satter_df'], self.format_params['summary_stats_round_digits'])}",
                               "Students df:", f"{round(self.sample_statistics['students_df'], self.format_params['summary_stats_round_digits'])}",
                               "Pooled Standard Error:", f"{round(self.sample_statistics['pooled_std_error'], self.format_params['summary_stats_round_digits'])}",
                               "Welches Standard Error:", f"{round(self.sample_statistics['welches_std_error'], self.format_params['summary_stats_round_digits'])}",
                               "Cohens d:", f"{round(self.sample_statistics['cohens_d'], self.format_params['summary_stats_round_digits'])}"]

                               #stat_names = ["Combined Stats", ""]
                               #
                               # Try with pandas to markdown??

            # If we want to stack the summary tables horizontally (otherwise they stack vertically)... 
            # I think horizontal looks better :) but go ahead and change this param to False to try out vertical!
            if self.format_params["summary_tables_oriented_horizontal"]:
                s1_table = mdutils.tools.Table.Table().create_table(columns=2, rows=9, text=s1_table_data)
                s2_table = mdutils.tools.Table.Table().create_table(columns=2, rows=9, text=s2_table_data)
                diff_table = mdutils.tools.Table.Table().create_table(columns=2, rows=8, text=diff_means_data)
                self.report.write("\n\n<table><tr><td>")
                self.report.write(s1_table)
                self.report.write("</td><td>")
                self.report.write(s2_table)
                self.report.write("</td><td>")
                self.report.write(diff_table)
                self.report.write("</td></tr></table>\n\n")
                #self.report.new_paragraph("<table><tr><td>\n\n" + s1_table + "\n\n</td><td>\n\n" + s2_table + "\n\n</td><td>\n\n" + diff_table + "\n\n</td></tr></table>")

            else: # else stack the three tables vertically

                self.report.new_table(columns=2,
                                    rows=9,
                                    text=s1_table_data,
                                    text_align='center')

                self.report.new_table(columns=2,
                                      rows=9,
                                      text=s2_table_data,
                                      text_align='center')

                self.report.new_table(columns=2,
                                      rows=8,
                                      text=diff_means_data,
                                      text_align='center')


    # Creates a small markdown table that contains the tests power
    # for each of the three possible alternative hypothesis choices.
    def add_test_power_to_report(self):

        self.report.new_header(level=1, title="Power")  

        self.report.new_line("\n\nPower values:\n",
                             bold_italics_code="",
                             color=self.format_params['pwr_text_color'])

        power_table_info = self.create_power_table()

        self.report.new_table(columns=power_table_info['num_columns'],
                              rows=power_table_info['num_rows'],
                              text=power_table_info['table_data'],
                              text_align='center')
               
    # Get a list of filepaths, specifying the file locations for each power plot we made.
    def get_power_plot_filepaths(self):
        
        # List of all the keys in the file paths dictionary
        file_keys = list(self.relative_file_paths.keys())
        
        # List of all the power plots we created
        power_plot_types = self.power_plot_types[self.test_type] 
        
        # Keys that are associated with power plot filepaths
        pwr_plot_file_keys = [key for key in file_keys if key in power_plot_types]
        
        # For each power plot file, grab both the filename and the path to the file
        pwr_plot_info = [(key, self.relative_file_paths[key]) for key in pwr_plot_file_keys]
        
        return pwr_plot_info

    # Pass desired_plot_types = False to generate no power plots
    # Pass desired_plot_type = None (default) to use the default plot types
    # Pass a list of valid plot types to use that list
    def get_power_plot_types(self, desired_plot_types, test_type):
        
        defaults = {"students_t": ["Power_vs_Samples_S1_Fixed",
                                   "Power_vs_Samples_S2_Fixed",
                                   "Power_vs_Samples_Vary_Together"],
                    "one_sample_t": [],
                    "paired_t": [],
                    "welches_t":[]}
        
        if desired_plot_types == False:  # If we don't want any plots
            defaults[test_type] = None
        elif desired_plot_types is None: # Using the default plot types
            return defaults
        else:                            # Using custom list of plot types
            defaults[test_type] = desired_plot_types
            return defaults

    def create_project_directory(self, project_dir):
        
        if project_dir is None:
            
            # Make the project directory
            # Make the relative images directory
            # Return the project directory
            if self.second_class is not None:
                
                # If a project directory wasn't passed, set up the default
                project_directory = f"./output_files/{self.first_class}_{self.second_class}/{self.test_type}_{self.alternative}_{self.first_class}_{self.second_class}/"

                # Remove this folder if it already exists, to make sure we don't use stale information.
                shutil.rmtree(project_directory, ignore_errors=True)
                
                os.makedirs(project_directory, exist_ok=True)
                os.makedirs(os.path.join(project_directory, os.path.normpath(self.relative_images_directory)), exist_ok=True)
                return project_directory
            elif self.second_class is None:
                
                # If a project directory wasn't passed, set up the default
                project_directory = f"./output_files/{self.first_class}/{self.test_type}_{self.alternative}_{self.first_class}/"

                # Remove this folder if it already exists, to make sure we don't use stale information.
                shutil.rmtree(project_directory, ignore_errors=True)
                
                os.makedirs(project_directory, exist_ok=True)
                os.makedirs(os.path.join(project_directory, os.path.normpath(self.relative_images_directory)), exist_ok=True)

                return project_directory 
        else:
            return project_dir
    
    def create_report_filename(self, report_filename):
        
        if report_filename is None:
            return f"Hypothesis_Report_{self.test_type}"
        else:
            return report_filename

    def set_formatting_params(self, format_args): 

        ### Include error checking for invalid keys being passed with format_args
        power_plot_defaults = {"figsize": (12, 8),
                               "title_fontsize": 20,
                               "title_weight": "bold",
                               "tick_label_size": 16,
                               "img_trnsprnt_bkgrd": False,
                               "img_save_format": "png"}
        
        # Default formatting behavior
        defaults = {"hyp_null_x_coord": 0.2,
                    "hyp_null_y_coord": 0.6,
                    "hyp_alt_x_coord": 0.2,
                    "hyp_alt_y_coord": 0.3,
                    "hyp_img_trnsprnt_bkgrd": True,
                    "hyp_text_color": "#FF69B4",
                    "hyp_img_save_format": "png",
                    "hyp_img_save_name": "null_alt_hyp_img",
                    "hyp_figsize":(12,2),
                    "hyp_fontsize":26,
                    "ds_figsize":(12,6),
                    "ds_crit_line_top" : 0.8,
                    "ds_crit_text_y" : 0.85,
                    "ds_crit_fontsize":16,
                    "ds_crit_fontcolor":"red",
                    "ds_crit_ha":"center",
                    "ds_dist_line_color": "black",
                    "ds_crit_line_color": "r",
                    "ds_crit_fill_color": "red",
                    "ds_fig_title_fontsize":20,
                    "ds_fig_title_weight":"bold",
                    "ds_fig_tick_labelsize":16,
                    "ds_img_save_name" : "draw_and_shade_img",
                    "ds_img_save_format": "png",
                    "ds_img_trnsprnt_bkgrd" : False,
                    "ds_plot_style": "dark",
                    "ds_tstat_line_color": "black",
                    "ds_tstat_fontcolor": "black",
                    "ds_tstat_ha": "center",
                    "ds_tstat_fontsize":16,
                    "ds_tstat_line_top" : 0.9,
                    "ds_tstat_text_y": 0.95,
                    "ds_max_drawable_tstat": 6,
                    "ds_min_ppf":0.01,
                    "ds_max_ppf":0.99,
                    "ds_num_pts" : 1000,
                    "ts_x_coord": 0.05,
                    "ts_y_coord": 0.25,
                    "ts_fontsize": 32,
                    "ts_figsize": (12, 8),
                    "ts_text_color": "#FF69B4",
                    "ts_img_trnsprnt_bkgrd" : True,
                    "ts_img_save_format":"png",
                    "ts_img_save_name" : "test_statistic_img",
                    "pv_figsize":(12, 1),
                    "pv_x_coord" : 0.05,
                    "pv_y_coord": 0.25,
                    "pv_fontsize" : 32,
                    "pv_text_color": "#FF69B4",
                    "pv_img_save_name": "p_value_image",
                    "pv_img_save_format": "png",
                    "pv_img_trnsprnt_bkgrd": True,
                    "pv_round_digits": 20,
                    "ci_round_digits" : 3, 
                    "ci_text_color": "#FF69B4",
                    "ci_agreeing_lvl_color":"#FFFF00",
                    "pwr_text_color": "#FF69B4",
                    "pwr_agreeing_test_color": "#FFFF00",
                    "pwr_round_digits": 5,
                    "pwr_plt_PvS_samples_scaler" : 3,
                    "Power_vs_Samples_S1_Fixed": power_plot_defaults,
                    "Power_vs_Samples_S2_Fixed": power_plot_defaults,
                    "Power_vs_Samples_Vary_Together": power_plot_defaults,
                    "summary_stats_round_digits": 4,
                    "summary_tables_oriented_horizontal": False, # If false, stack tables veritically (horizontally currently doesn't work...)
                    "perm_figsize" : (10, 8),
                    "perm_obs_line_scaler" : 0.9,
                    "perm_annot_fontsize" : 12,
                    "perm_annot_ha" : "center",
                    "perm_annot_va":"top",
                    "perm_annot_fontweight":"bold",
                    "perm_annot_round_digits": 4,
                    "perm_crit_linecolor": "red",
                    "perm_axis_fontsize":16,
                    "perm_title_fontsize":16,
                    "perm_axis_fontweight" : "bold",
                    "perm_title_fontweight": "bold",
                    "perm_tick_labelsize": "xx-large",
                    "perm_img_save_name" : "permutation_distribution",
                    "perm_img_save_format" : "png",
                    "perm_img_trnsprnt_bkgrd": False}

        # Update defaults based on any params the user chose to specify
        for key, value in format_args.items():
            defaults[key] = value

        return defaults
    
    ### Getter functions for external use
    def get_sample_statistics(self):
        return self.sample_statistics

    def get_format_params(self):
        return self.format_params
    
    def get_project_directory(self):
        return self.project_directory
    
    def get_relative_images_directory(self):
        return self.relative_images_directory

    def get_degrees_of_freedom(self):
        return self.df

    def get_test_statistic(self):
        return self.test_statistic

    def get_relative_file_paths(self):
        return self.relative_file_paths

    def print_summary(self):
        
        pwr_key = f"pwr_{self.alternative}"
        
        print("\n==================================== Test Summary ====================================")
        print(f"Test type: {self.test_type}")
        print(f"Alt Hyp: {self.alternative}")
        print(f"P-value: {self.p_value}")
        print(f"Test Statistic: {self.test_statistic}")
        print(f"Number of permutations: {self.num_permutations}")
        print("\n")
        print(f"power: {self.sample_statistics[pwr_key]}")
        print(f"cohens_d: {self.sample_statistics['cohens_d']}")
        print("=============================================================================================\n")