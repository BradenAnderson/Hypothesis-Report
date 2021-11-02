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
import math
import scipy.stats as stats
import statsmodels.stats.power as smp

class PermutationTest():
    def __init__(self, num_permutations, sample1, sample2, alternative="two_sided", alpha=0.05, test_type="two_sample", test_statistic_type="mean",
                 format_params=None, project_directory="./", relative_images_directory="./images/", relative_file_paths={}, fast=True,
                 test_value=0, overlay_normal_curve=True):
         

        self.fast = fast

        self.num_permutations = num_permutations
        self.sample1 = sample1
        self.sample2 = sample2
        self.test_statistic_type=test_statistic_type


        self.all_samples = self._create_all_samples_list()

        self.p_value = None
        self.num_extreme_samples = None
        self.num_extreme_low = None
        self.num_extreme_high = None

        self.all_permutations = None
        self.num_permutations_performed = 0
        self.alternative = alternative
        self.alpha = alpha
        self.test_value = 0

        # For overlaying a normal curve onto the permutation dist histogram
        self.overlay_normal_curve = overlay_normal_curve

        ### Make sure we use the correct "n" for the signed rank test
        if self.test_statistic_type == "signed_rank":
            self.differences = [s1 - s2 for s1, s2 in zip(sample1, sample2)]
            self.num_differences = len(self.differences)
            self.non_zero_differences = [s1 - s2 for s1, s2 in zip(sample1, sample2) if s1 - s2 != 0]
            self.num_nonzero_differences = len(self.non_zero_differences)
        else:
            self.differences = None
            self.num_differences = None
            self.non_zero_differences = None
            self.num_nonzero_differences = None

        # What the center of the distribution should be under the null
        self.null_expectation = self._calculate_null_expectation()
        self.observed_test_statistic = None
        self.observed_deviation_from_null_expected = None

        self.project_directory = project_directory
        self.relative_images_directory = relative_images_directory
        self.relative_file_paths = relative_file_paths

        if format_params is not None:
            self.format_params = format_params
        else:
            self.format_params = self.get_format_defaults()

        self.generated_permutations = set()

        # Will be a list to hold the test statistics calculated for each permuted sample.
        self.permuted_test_stats = None
    
    def _create_all_samples_list(self):
        if self.sample2 is not None:
            return self.sample1 + self.sample2
        else:
            return self.sample1

     # https://stackoverflow.com/questions/54050322/random-sample-of-n-distinct-permutations-of-a-list
    def permutation_generator(self, sequence):

        permutation_length = len(sequence)

        while True:
            permutation = tuple(random.sample(sequence, permutation_length))
            if permutation not in self.generated_permutations:
                self.generated_permutations.add(permutation)
                yield permutation

    def calculate_test_statistic(self, sample1, sample2, perm_assignments="0"):

        if self.test_statistic_type == "mean":
            test_statistic = np.mean(sample1) - np.mean(sample2)

        elif self.test_statistic_type == "median":
            test_statistic = np.median(sample1) - np.median(sample2)
        
        elif self.test_statistic_type == "rank_sum" and self.fast:
            test_statistic = self.calculate_rank_sum_statistic_fast(sample1, sample2)

        elif self.test_statistic_type == "rank_sum" and not self.fast:
            test_statistic = self._calculate_rank_sum_statistic(sample1, sample2)

        elif self.test_statistic_type == "signed_rank":
            test_statistic =  self._calculate_signed_rank_test_statistic(perm_assignments=perm_assignments)

        return test_statistic

    def permutation_test(self):
        
        if self.num_permutations is None:
            self.num_permutations = 10_000

        # Single list that holds the samples for both classes
        all_samples = self.sample1 + self.sample2

        # Get the difference in means (and absolute value) for the sample we actually observed.
        observed_test_statistic = self.calculate_test_statistic(sample1=self.sample1, sample2=self.sample2)
        self.observed_test_statistic = observed_test_statistic

        # How far the test statistic we observed deviations from what the value should be under the null
        # A lot of the time the value under the null (e.g. difference in means) is simply zero, however
        # that is not always the case. For example in a rank_sum test, the expected value under the null
        # is the sum of ranks for the smaller group assuming the class labels do not influence the ranks at all.
        self.observed_deviation_from_null_expected = self.observed_test_statistic - self.null_expectation

        # Take the absolute value of the observed deviation, we need this quantity for calculating p_values correctly.
        self.absolute_observed_deviation = np.abs(self.observed_deviation_from_null_expected)

        # Generate num_permutations number of random permutations of the data
        random_permutations = self.permutation_generator(all_samples)
        all_permutations = [next(random_permutations) for _ in range(self.num_permutations)]

        # List to store the test statistic for each of the permuted versions of the data.
        self.permuted_test_stats = []

        # Seperate list to store how far each test stat deviations from the expected value under the null.
        # In many cases these lists will be identical, but not always (see rank_sum comments).
        self.permuted_test_stat_deviations = []
        
        # Save the actual permutations that were used, in case we want to inspect them later.
        self.all_permutations = all_permutations
        
        # Iterate over the permutations
        for permutation in all_permutations:
            
            self.num_permutations_performed += 1
            
            # Create the versions of sample sample1 and sample 2
            # that exist under this random permutation
            s1 = permutation[:len(self.sample1)]
            s2 = permutation[len(self.sample1):]

            # Calculate the test statistic on this permuted version of the data.
            permuted_test_stat = self.calculate_test_statistic(sample1=s1, sample2=s2)

            # Append the test statistic for this permuted version of the data to our list.
            self.permuted_test_stats.append(permuted_test_stat)

            # Append the difference between this test statistics value and what we expected under the null.
            self.permuted_test_stat_deviations.append(permuted_test_stat - self.null_expectation)
        
        p_value = self._calculate_permutation_statistics()
        
        # Return the test statistic (which is the value we observed in the sample we actually took)
        # and the p-value (which is the number of permuted versions of the data where the test statistic was as or more extreme than the
        # test statistic for the sample we actually took)
        return observed_test_statistic, p_value

    def _calculate_permutation_statistics(self):

        if self.alternative == "two_sided":

            # Count the number of values in permuted test statistics that were as or more extreme
            # than the difference we observed in the sample we actually took. 
            self.num_extreme_values = sum([1 if np.abs(value) >= self.absolute_observed_deviation else 0 for value in self.permuted_test_stat_deviations])

            # Saving these numbers for potential analysis at a later time...
            self.num_extreme_low = sum([1 if value <= -self.absolute_observed_deviation else 0 for value in self.permuted_test_stat_deviations])
            self.num_extreme_high = sum([1 if value >= self.absolute_observed_deviation else 0 for value in self.permuted_test_stat_deviations])

        elif self.alternative == "one_sided_right": 
            
            self.num_extreme_values = sum([1 if value >=  self.observed_deviation_from_null_expected else 0 for value in self.permuted_test_stat_deviations])
            self.num_extreme_high = self.num_extreme_values
            self.num_extreme_low = None

        elif self.alternative == "one_sided_left":

            self.num_extreme_values = sum([1 if value <=  self.observed_deviation_from_null_expected else 0 for value in self.permuted_test_stat_deviations])
            self.num_extreme_low = self.num_extreme_values
            self.num_extreme_high = None

        p_value = self.num_extreme_values / self.num_permutations
        self.p_value = p_value

        # Average value of the test statistic from the permuted samples
        self.mean_test_statistic = np.mean(self.permuted_test_stats)
        self.min_test_statistic = np.min(self.permuted_test_stats)
        self.max_test_statistic = np.max(self.permuted_test_stats)
        self.median_test_statistic = np.median(self.permuted_test_stats)

        # Average value of the differences between (permuted test statistic) - (test statistic expectation under the null)
        # I think this will always be a pretty small value
        self.mean_test_statistic_deviation = np.mean(self.permuted_test_stat_deviations)
        self.min_test_statistic_deviation = np.min(self.permuted_test_stat_deviations)
        self.max_test_statistic_deviation = np.max(self.permuted_test_stat_deviations)
        self.median_test_statistic_deviation = np.median(self.permuted_test_stat_deviations)

        self.permutation_statistics = {'observed_test_statistic': self.observed_test_statistic,
                                       'mean_tstat' : self.mean_test_statistic,
                                       "min_tstat": self.min_test_statistic,
                                       "max_tstat":self.max_test_statistic,
                                       "median_tstat": self.median_test_statistic,
                                       'mean_tstat_dev' : self.mean_test_statistic_deviation,
                                       "min_tstat_dev": self.min_test_statistic_deviation,
                                       "max_tstat_dev":self.max_test_statistic_deviation,
                                       "median_tstat_dev": self.median_test_statistic_deviation}

        return p_value

    def _calculate_null_expected_rank_statistic(self):

        # Put the samples into a pandas dataframe for convenient sorting. 
        # This function only needs to run once, so speed isn't too important. 
        data = {"samples": self.all_samples}
        df = pd.DataFrame(data)
    
        # Step 2: Sort the data smallest to largest
        df.sort_values(by="samples", inplace=True)
    
        # Step 3: Assign ranks
        df['ranks'] = [num for num in range(1, len(df) + 1)]

        # Get a list of unique values, for checking to see if we have any tied ranks.
        unique_values = df['samples'].unique()
    
        #  If there are any ties, update the rank for ties to the average rank.
        if len(unique_values) != len(df):
        
            # Update the rank for any tied values to be the average rank 
            # across all elements that share this value. 
            for value in unique_values:
            
                # Take the mean of the ranks for all samples with this value.
                avg_rank = df.loc[df['samples'] == value, "ranks"].mean()
            
                # Reassign the ranks for these samples to the average rank
                df.loc[df['samples'] == value, "ranks"] = avg_rank

        # Get the size of the smaller sample
        smaller_sample_size = min(len(self.sample1), len(self.sample2))

        # Under the null hypothesis, the ranks in each class group should simply be a random
        # sample of the n1 + n2 total ranks. By convention, we calculate the rank sum statistic
        # as the sum of ranks for the class with the smaller sample size. This means the expected
        # value of the rank sum statistic under the null hypothesis is just...
        # (number of samples in smaller class) * (average rank, regardless of class label)

        # Get the mean rank, regardless of class
        mean_rank = df['ranks'].mean()

        null_expected_rank_sum_stat = smaller_sample_size * mean_rank

        return null_expected_rank_sum_stat

    def _calculate_null_expectation(self):

        if self.test_statistic_type == "mean":
            return self.test_value

        elif self.test_statistic_type == "median":
            return self.test_value
        
        elif self.test_statistic_type == "rank_sum":
            expected_rank_statistic = self._calculate_null_expected_rank_statistic()
            return expected_rank_statistic
        
        # Sum of all ranks will always be n*(n+1)/2 ... so the expected value under the null is the
        # middle of this. (if the median difference between the distributions is zero, then the
        # sum of positive ranks should always be half the total rank).
        elif self.test_statistic_type == "signed_rank":
            expected_sign_rank_stat = self.num_nonzero_differences * (self.num_nonzero_differences + 1)/4
            return expected_sign_rank_stat

    def _calculate_critical_values(self):

        if self.alternative == "two_sided":
            lower_critical, upper_critical = np.percentile(a=self.permuted_test_stats, q=[(self.alpha/2) * 100, (1 - self.alpha/2) * 100])

            lower_crit_info = (lower_critical, f"{(self.alpha/2) * 100}")
            upper_crit_info = (upper_critical, f"{(1 - self.alpha/2) * 100}")

        elif self.alternative == "one_sided_left":
            lower_critical = np.percentile(a=self.permuted_test_stats, q=[self.alpha * 100])[0]

            lower_crit_info = (lower_critical, f"{self.alpha * 100}")
            upper_crit_info = (None, None)

        elif self.alternative == "one_sided_right":
            
            upper_critical = np.percentile(self.permuted_test_stats, q=[(1 - self.alpha) * 100])[0]

            lower_crit_info = (None, None)
            upper_crit_info = (upper_critical, f"{(1 - self.alpha) * 100}")

        return lower_crit_info, upper_crit_info 

    def get_format_defaults(self):

        defaults = {"perm_figsize" : (10, 8),
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

        return defaults

    def create_permutation_distribution_plot(self):

        sns.set_style("darkgrid")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.format_params['perm_figsize'])

        if self.test_statistic_type == "rank_sum" or self.test_statistic_type == "signed_rank":
            discrete = True
        else:
            discrete = None

        if self.overlay_normal_curve:
            stat = "probability"
        else:
            stat = "percent"

        # Plot the histogram
        sns.histplot(x=self.permuted_test_stats, ax=ax, stat=stat, discrete=discrete)

        # Create transformations
        # https://stackoverflow.com/questions/29107800/python-matplotlib-convert-axis-data-coordinates-systems
        axis_to_data = ax.transAxes + ax.transData.inverted()
        data_to_axis = axis_to_data.inverted()

        lower_crit_info, upper_crit_info = self._calculate_critical_values()
        self.lower_critical, lower_percentile = lower_crit_info
        self.upper_critical, upper_percentile = upper_crit_info

        # Get the x and y max values in data coordinates
        y_min, y_max = ax.get_ylim()

        ## ===== Test statistic vertical line section =====

        # Get the top of the observed test statistic line, in the data coordinate frame.
        observed_tstat_vert_line_top = y_max * self.format_params['perm_obs_line_scaler']

        # Convert the y coordinate for the top of the line to the axes coordinate frame.
        _, observed_tstat_vert_line_top = data_to_axis.transform((0, observed_tstat_vert_line_top))

        # Plot the vertical line for the observed test statistic
        # x is in the data coordinate from
        # y_min and y_max are in the axis coordinate frame (0=bottom, 1=top)
        ax.axvline(x=self.observed_test_statistic, ymin=0, ymax=observed_tstat_vert_line_top, color='black')

        ## ===== Test statistic text section =====

        # Get the text for the observed test statistic, in the axis coordinate frame
        # It should always be slightly above the top of the line.
        observed_text_scaler = self.format_params['perm_obs_line_scaler'] + 0.06
        observed_text_y_coord = y_max * observed_text_scaler

        # Convert the text location y coordinate to the data coordinate frame.
        # We need the data coordinate frame because the x-value depends on the data.
        #_ , converted_y = axis_to_data.transform((0, observed_text_y_coord))

        # Text for the observed test statistic
        style = dict(size=self.format_params["perm_annot_fontsize"],
                     ha = self.format_params["perm_annot_ha"],
                     va=self.format_params["perm_annot_va"],
                     fontweight=self.format_params["perm_annot_fontweight"])

        obs_txt = f"observed test statistic\n= {round(self.observed_test_statistic, self.format_params['perm_annot_round_digits'])}"

        # Add the text (x, y both in data coordinates)
        ax.text(x=self.observed_test_statistic, y=observed_text_y_coord, s=obs_txt, **style)

        # Location of the top of the red lines showing critical percentiles, in the data coordinate frame.
        crit_value_vert_line_scaler = self.format_params['perm_obs_line_scaler'] - 0.2
        crit_value_line_top = y_max * crit_value_vert_line_scaler

        # Convert the y coordinate for the top of the line to the axes coordinate frame.
        _, crit_value_line_top = data_to_axis.transform((0, crit_value_line_top))

        # y coordinates for the critical line text.
        critical_text_scaler = crit_value_line_top + 0.06
        critical_text_y_coord = y_max * critical_text_scaler

        if self.lower_critical is not None:
    
            lower_crit_txt = f"{lower_percentile} percentile\n= {round(self.lower_critical, self.format_params['perm_annot_round_digits'])}"
    
            ax.axvline(x=self.lower_critical, ymin=0, ymax=crit_value_line_top, color=self.format_params["perm_crit_linecolor"])
            ax.text(x=self.lower_critical, y=critical_text_y_coord, s=lower_crit_txt, **style)
    
        if self.upper_critical is not None:
    
            upper_crit_txt = f"{upper_percentile} percentile\n= {round(self.upper_critical, self.format_params['perm_annot_round_digits'])}"
    
            ax.axvline(x=self.upper_critical, ymin=0, ymax=crit_value_line_top, color=self.format_params["perm_crit_linecolor"])
            ax.text(x=self.upper_critical, y=critical_text_y_coord, s=upper_crit_txt, **style)

        
        if self.overlay_normal_curve:

            # Get the mean and standard deviation for the best fitting normal curve
            mu, sd = stats.norm.fit(self.permuted_test_stats)

            x_min, x_max = ax.get_xlim()
            x_norm = np.linspace(x_min, x_max, 2_000)
            y_norm = stats.norm.pdf(x_norm, mu, sd)

            ax.plot(x_norm, y_norm, color="#FE019A", linewidth=3)

            self.mu_normal = mu
            self.sd_normal = sd

    
        plot_info = self._create_plot_info()
    
        ax.set_title(plot_info['title'], fontsize=self.format_params["perm_title_fontsize"], weight=self.format_params["perm_title_fontweight"])
        ax.set_xlabel(plot_info['xlab'], fontsize=self.format_params["perm_axis_fontsize"], weight=self.format_params["perm_axis_fontweight"])
        ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=self.format_params["perm_axis_fontsize"], weight=self.format_params["perm_axis_fontweight"])
        ax.tick_params(labelsize=self.format_params["perm_tick_labelsize"])

        plt.tight_layout()

        # Filename for saving this image
        img_filename = f"{self.format_params['perm_img_save_name']}.{self.format_params['perm_img_save_format']}"
        
        # Path to image location, relative to project directory
        img_relative_path = os.path.join(self.relative_images_directory, img_filename)

        # Full path to image save location (including project directory)
        full_save_path = os.path.join(self.project_directory, os.path.normpath(img_relative_path))

        self.relative_file_paths['step_2_ds_img_perm'] = img_relative_path

        # Make the save directory if it does not already exist
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

        plt.savefig(full_save_path,
                    transparent=self.format_params["perm_img_trnsprnt_bkgrd"],
                    format=self.format_params["perm_img_save_format"])

        return img_relative_path

    def _create_plot_info(self):

        if self.test_statistic_type == "mean":

            plot_info = {"title": "Permutaiton Distribution for Difference in Means",
                         "xlab": "Difference in means for permuted samples"}

            return plot_info

        elif self.test_statistic_type == "median":

            plot_info = {"title": "Permutation Distribution for Difference in Medians",
                         "xlab": "Difference in medians for permuted samples"}

            return plot_info

        elif self.test_statistic_type == "rank_sum":

            plot_info = {"title": "Permutation Distribution for the Rank Sum Statistic",
                         "xlab": "Rank Sum Statistic Values for Permuted Samples"}

            return plot_info

        elif self.test_statistic_type == "signed_rank":

            plot_info = {"title": "Permutation Distribution for the Signed Rank Statistic",
                         "xlab": "Signed Rank Statistic Values for Permuted Samples"}

            return plot_info

    def print_summary(self):

        print("\n============================================================================")
        print(f"P-Value: {self.p_value}\n")
        print(f"Observed Test_Statistic... {self.test_statistic_type}: {self.observed_test_statistic}")
        print(f"Expected Test Statistic Under the Null: {self.null_expectation}")
        print(f"Observed test stat deviation from null expected: {self.observed_deviation_from_null_expected}\n")
        print(f"Average Test Statistic {self.test_statistic_type}s from PERMUTED samples: {self.mean_test_statistic}")
        print(f"Average Test Statistic Deviation from Null Expected from PERMUTED samples: {self.mean_test_statistic_deviation}\n")
        print(f"Num Permutations Performed: {self.num_permutations_performed}")
        print(f"Num values as or more extreme than observed: {self.num_extreme_values}")
        print(f"Num extreme values low: {self.num_extreme_low}")
        print(f"Num extreme high: {self.num_extreme_high}")
        print("============================================================================\n")

    # For signed rank test
    def get_num_permutations_paired(self):

        # 2^n possible assignments with n-pairs
        max_paired_assignments = 2 ** len(self.sample1)

        # If we didn't specify a number of permutations, use all of them as long as that's below 50k.
        if self.num_permutations is None and max_paired_assignments <= 50_000:
            num_permutations = max_paired_assignments

        # If we specified more permutations than is possible, use the max possible.
        elif self.num_permutations > max_paired_assignments:

            print("\n========================================================================================")
            print(f"Num Permutations Requested {self.num_permutations} is larger than 2**n={max_paired_assignments}")
            print(f"which is the maximum possible number of paired assignments.")
            print(f"Using {max_paired_assignments} instead of the requested number of permuations.")
            print("========================================================================================\n")

            num_permutations = max_paired_assignments
        
        # If they didn't specify None, or too many, use what they specified.
        else:
            num_permutations = self.num_permutations

        return num_permutations

    # For signed rank test
    def paired_assignment_generator(self, num_pairs):
        
        # Keep track of the random numbers we have generated
        randomly_assigned_numbers = set()
        
        while True:
            
            # Generate a random number between 0 and the number of pairs we have in the dataset
            # This is because for n pairs we have 2^n possible assignments
            rand_num = random.randint(0, 2**num_pairs)
            
            if rand_num not in randomly_assigned_numbers:
                
                randomly_assigned_numbers.add(rand_num)
                
                permutation_assignments = tuple(format(rand_num, f"0{num_pairs}b"))
                
                yield permutation_assignments

    # For signed rank test
    def get_paired_permutations(self):
        
        num_pairs = len(self.sample1)

        if num_pairs < 60: 
            
            random_assignments = [format(rand_num, f"0{num_pairs}b") for rand_num in random.sample(range(2**num_pairs), self.num_permutations)]

            return random_assignments
            
        else:
            
            assignment_generator = self.paired_assignment_generator(num_pairs)
            
            random_assignments = [next(assignment_generator) for _ in range(self.num_permutations)]
            
            return random_assignments

    def paired_permutation_test(self):

        self.num_permutations = self.get_num_permutations_paired()

        self.all_permutations = self.get_paired_permutations()

        initial_assignment = "0"*len(self.sample1)

        # Don't need to pass sample1 and sample2 for a paired test, because self.sample1 and self.sample2 are always used.
        # The only thing that can change for a paired test is the subtraction order. The mixing of subtraction orders is 
        # handled via a binary string passed to perm_assignments. One bit for each sample. A "0" indicates s1-s1, "1" for s2-s1.
        observed_test_statistic = self.calculate_test_statistic(sample1=None, sample2=None, perm_assignments=initial_assignment)

        self.observed_test_statistic = observed_test_statistic
        
        # How far the test statistic we observed was from the expected test statistic if the null is True.
        self.observed_deviation_from_null_expected = self.observed_test_statistic - self.null_expectation

        # Take the absolute value of the observed deviation, we need this quantity for calculating p_values correctly.
        self.absolute_observed_deviation = np.abs(self.observed_deviation_from_null_expected)

        self.permuted_test_stats = [self.calculate_test_statistic(None, None, perm_assign) 
                                    for perm_assign in self.all_permutations]

        self.permuted_test_stat_deviations = [t_stat - self.null_expectation for t_stat in self.permuted_test_stats]

        self.num_permutations_performed = self.num_permutations

        p_value = self._calculate_permutation_statistics()

        return observed_test_statistic, p_value

    def run_test(self, verbose=False):
        
        # If this is not a paired permutation test
        if self.test_statistic_type != "signed_rank":
            test_statistic, p_value = self.permutation_test()
        else:
            test_statistic, p_value = self.paired_permutation_test()

        if verbose:
            self.print_summary()

        return test_statistic, p_value

    def get_permuted_test_stats(self):
        return self.permuted_test_stats

    def _calculate_signed_rank_test_statistic(self, perm_assignments):
        
        # Step 1: Compute all the differences
        # Step 2: Remove zeros from the list
        all_differences = [(s1 - s2, np.abs(s1 -s2)) if perm_assignments[index] == '0' else (s2 - s1, np.abs(s2-s1)) 
                            for index, (s1, s2) in enumerate(zip(self.sample1, self.sample2)) if s1 - s2 != 0]
        
        # Step 3: Sort the absolute value of the differences from smallest to largest
        sorted_differences = sorted(all_differences, key = lambda sublist: sublist[1])
        
        # Step 4: Assign ranks
        ranked_diffs = [(difference, abs_difference, rank) for rank, (difference, abs_difference) in enumerate(sorted_differences, start=1)]
        
        # List of only the absolute differences, used to check for ties.
        abs_differences = [abs_diff for diff, abs_diff in all_differences]
        
        # List of unique absolute differences, used to check for ties.
        unique_abs_differences = list(np.unique(abs_differences))
        
        # If the length of the unique absolute differences is less than the 
        # length of the absolute differences, then we have some ties.
        if len(unique_abs_differences) < len(abs_differences):
            
            for value in unique_abs_differences:
                
                # Calculate the average rank of all samples where the absolute difference is equal to this unique absolute difference (value).
                avg_rank = np.mean([rank for diff, abs_diff, rank in ranked_diffs if abs_diff == value])
                
                # Update the ranks for the absolute differences tied at "value" to be avg_rank.
                ranked_diffs = [(diff, abs_diff, avg_rank) if abs_diff == value else (diff, abs_diff, rank) for diff, abs_diff, rank in ranked_diffs]
                
        # Signed rank statistic is the sum of the ranks for the pairs for which the difference is positive.
        signed_rank_stat = np.sum([rank for diff, abs_diff, rank in ranked_diffs if diff > 0])
        
        return signed_rank_stat

    def calculate_rank_sum_statistic_fast(self, sample1, sample2):
        
        all_samples = sample1 + sample2
        
        # Step 1: Combine the data from both classes into a single list. 
        data = [("class_1", value) for value in sample1]
        data.extend([("class_2", value) for value in sample2])
        
        data_sorted = sorted(data, key = lambda sublist: sublist[1])
        
        ranked_data = [(category, value, rank) for rank, (category, value) in enumerate(data_sorted, start=1)]
        
        unique_values = list(np.unique(all_samples))
        
        if len(unique_values) < len(all_samples):
            
            #counts = [(all_samples.count(value), value) for value in unique_values]
            #ties = [value for count, value in counts if count > 1]
            
            for value in unique_values:
                
                avg_rank = np.mean([rank for category, sample_value, rank in ranked_data if sample_value == value])
                
                ranked_data = [(category, sample_value, avg_rank) if sample_value == value else (category, sample_value, rank) for category, sample_value, rank in ranked_data]
        
                
        if len(sample1) > len(sample2):
            rank_sum_statistic = np.sum([rank for category, value, rank in ranked_data if category == "class_2"])
        
        else:
            
            rank_sum_statistic = np.sum([rank for category, value, rank in ranked_data if category == "class_1"])
            
        return rank_sum_statistic

    def _calculate_rank_sum_statistic(self, sample1, sample2):
    
        # Step 1: Combine the data from both classes into a single list. 
        all_samples = sample1 + sample2
    
        class1 = ["class_1"] * len(sample1)
        class2 = ["class_2"] * len(sample2)
    
        all_classes = class1 + class2
    
        data = {"class":all_classes, "samples": all_samples}
    
        df = pd.DataFrame(data)
    
        # Step 2: Sort the data smallest to largest
        df.sort_values(by="samples", inplace=True)
    
        # Step 3: Assign ranks
        df['ranks'] = [num for num in range(1, len(df) + 1)]
    
        unique_values = df['samples'].unique()
    
        #  If there are any ties
        if len(unique_values) != len(df):
        
            # Update the rank for any tied values to be the average rank 
            # across all elements that share this value. 
            for value in unique_values:
            
                # Take the mean of the ranks for all samples with this value.
                avg_rank = df.loc[df['samples'] == value, "ranks"].mean()
            
                # Reassign the ranks for these samples to the average rank
                df.loc[df['samples'] == value, "ranks"] = avg_rank
    
        # The rank sum statistic is the sum of the ranks for the class with
        # fewer observations in it!
    
        # If there are more samples in class 1 than in class 2, use the sum of the ranks
        # for class_2 as the rank sum statistic, else, use sum of ranks for class_1 as the statistic
        if len(df.loc[df['class'] == 'class_1', :]) > len(df.loc[df['class'] == 'class_2', :]):
            rank_sum_statistic = df.loc[df['class'] == "class_2", 'ranks'].sum()
        else:
            rank_sum_statistic = df.loc[df['class'] == "class_1", 'ranks'].sum()
        
        del df

        return rank_sum_statistic

    def get_test_string(self):

        if self.test_statistic_type == "mean":
            return "difference in means"
        elif self.test_statistic_type == "median":
            return "difference in medians"
        elif self.test_statistic_type == "rank_sum":
            return "rank sum"
        elif self.test_statistic_type == "signed_rank":
            return "signed rank"

    def create_permutation_summary_table(self):

        table_data = ["Test: ", f"{self.get_test_string()}", "Number of Permutations :", f"{self.num_permutations_performed}",
                      "P-value: ", f"{self.p_value}", "Alpha: ", f"{self.alpha}",
                      "Num Extreme Values: ", f"{self.num_extreme_values}", "Extreme values low, high: ", f"{self.num_extreme_low}, {self.num_extreme_high}",
                      "Test Stat (observed): ", f"{self.observed_test_statistic}", "Expected Test Stat (Null): ", f"{self.null_expectation}",
                      "Test Stat Deviation (observed): ", f"{self.observed_deviation_from_null_expected}", "Avg Test Stat Deviation (Perms): ", f"{self.mean_test_statistic_deviation}",
                      "Min Test Stat (Perms): ", f"{self.min_test_statistic}", "Min Test Stat Deviation (Perms): ", f"{self.min_test_statistic_deviation}",
                      "Max Test Stat (Perms): ", f"{self.max_test_statistic}", "Max Test Stat Deviation (Perms): ", f"{self.max_test_statistic_deviation}",
                      "Median Test Stat (Perms): ", f"{self.median_test_statistic}", "Median Test Stat Deviation (Perms): ", f"{self.median_test_statistic_deviation}"
                      ]

        nrows = 8
        ncols = 4

        if self.test_statistic_type == "signed_rank":

            nrows += 1

            table_data.extend(["Num Differences: ", f"{self.num_differences}", "Num Non-Zero Differences: ", f"{self.num_nonzero_differences}"])

        if self.overlay_normal_curve:

            nrows += 1
            table_data.extend(["Normal Curve Mean: ", f"{self.mu_normal}", "Normal Curve Std: ", f"{self.sd_normal}"])

        table_info = {"table_data": table_data, "num_rows" : nrows, "num_columns": ncols}

        return table_info