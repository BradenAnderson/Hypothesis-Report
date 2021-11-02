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


# ===============================================================================================
# This class is used to calculate a set of summary statistics for a given sample or samples, where
# the summary statistics are also dependent on the type of hypothesis test that is being
# performed using the sample(s). 
# ===============================================================================================
class SampleAnalyzer():
    def __init__(self, sample1, sample2=None, test_value=0, alpha=0.05, test_type="one_sample_t",
                 alternative="two_sided", valid_power_calcs=None, valid_alternatives=None, confidence_intervals=None,
                 first_class=None, second_class=None):

        self.sm_alt_hyp_map = {"two_sided": "two-sided",      # Maps the terms I use for the alternative hypothesis choices to the
                               "one_sided_right": "larger",   # terms statsmodels uses for the alternative hypothesis. This allows
                               "one_sided_left": "smaller"}   # me to easily interface with their functions without adopting their convention.

        
        self.sample1 = sample1           # The actual samples in sample1
        self.sample2 = sample2           # The actual samples in sample2 (none for one sample tests).
        self.test_value = test_value     # The value under the null hypothesis
        self.alpha = alpha               # The significance level for the test being run
        self.test_type = test_type       # Type of hypothesis test
                                         # Valid test_types --> one_sample_t, paired_t, students_t, welches_t, permutation

        self.first_class = first_class
        self.second_class = second_class       
        self.alternative = alternative   # The alternative hypothesis. One of --> one_sided_left, one_sided_right, two_sided

        # List of test_types where we have a valid method for calculating power
        # Statsmodels currently doesn't give you an option to calculate power when the
        # equal variance assumption isn't used (welches_t)... need a work around for this! 
        if valid_power_calcs is None:
            self.valid_power_calcs = ['paired_t', 'students_t', 'one_sample_t']
        else:
            self.valid_power_calcs = valid_power_calcs
        
        if valid_alternatives is None:
            self.valid_alternatives = ["one_sided_left", "one_sided_right", "two_sided"]
        else:
            self.valid_alternatives = valid_alternatives

        # Calculate the correct sample_statistics for this particular test_type.
        if sample2 is not None and test_type != "paired_t":
            self.sample_statistics = self.get_two_sample_stats(sample1, sample2)
            self.sample_center = self.sample_statistics['diff_sample_means']
        elif sample2 is not None and test_type == "paired_t":
            self.sample_statistics = self.get_paired_sample_stats(sample1, sample2)
            self.sample_center = self.sample_statistics['mean']
        else: 
            self.sample_statistics = self.get_one_sample_stats(sample1)
            self.sample_center = self.sample_statistics['mean']
            
        self.df = self.calculate_degrees_of_freedom()
        self.standard_error = self.calculate_standard_error()

        if self.test_type in ["one_sample_t", "students_t", "paired_t", "welches_t"]:
            self.manual_conf_intervals, self.conf_intervals = self.calculate_confidence_intervals(confidence_intervals)
        
        # Cannot calculate confidence intervals via the above methods for permutation tests.
        # Other methods exist for doing this and will require their own special functions. TODO!! 
        else:
            self.manual_conf_intervals = None
            self.conf_intervals = None 

    # ========================================================
    # Function to calculate statistics on a single sample
    # ========================================================
    def get_one_sample_stats(self, sample):

        variance = np.var(sample, ddof=1)
        sample_standard_deviation = np.sqrt(variance)

        sample_size = len(sample)
        
        # Note, the standard error could be calculated in a single line using scipy.stats.sem()
        # and will yeild the exact same result as this more verbose approach. 
        standard_error = sample_standard_deviation / np.sqrt(sample_size)

        min_value = np.min(sample)
        max_value = np.max(sample)
        median = np.median(sample)
        mean = np.mean(sample)

        sample_statistics = {"variance":variance,
                             "sample_sd": sample_standard_deviation,
                             "n":sample_size,
                             "std_error":standard_error,
                             "min":min_value,
                             "max":max_value,
                             "median":median,
                             "mean":mean}
        
        # Add power info if this is just a one_sample t-test, power info is added
        # later for all other test types.
        if self.test_type == "one_sample_t":
            
            # Cohens d for the one_sample_t test ...
            # This value is not used (and the proper cohens_d calculated)
            # for all other test types.. (paired_t, students_t)
            cohens_d = (mean - self.test_value) / sample_standard_deviation
            
            sample_statistics['cohens_d'] = cohens_d
            sample_statistics['ratio'] = None
            
            power = self.get_power(effect_size = cohens_d,
                                   num_observations= sample_size,
                                   alpha=self.alpha)
            
            for key, value in power.items():
                sample_statistics[f"pwr_{key}"] = value

        return sample_statistics

    # ========================================================
    # Calculate the pooled estimate of the standard deviation. 
    # Only used under the equal variance assumption (students test)
    # ========================================================
    def get_pooled_standard_deviation(self, sample1_stats, sample2_stats):

        # Weight the sample one variance by its degrees of freedom
        numerator_sample1 = (sample1_stats['n'] - 1)*sample1_stats['variance']

        # Weight the sample two variance by its degrees of freedom
        numerator_sample2 = (sample2_stats['n'] - 1)*sample2_stats['variance']

        # Calculate the total degrees of freedom
        denominator = (sample1_stats['n'] + sample2_stats['n'] - 2)

        # Calcualte the pooled standard deviation as the weighted averages
        # of the two sample variances (weighted by their degrees of freedom).
        pooled_sd = np.sqrt((numerator_sample1 + numerator_sample2)/denominator)

        return pooled_sd
    
    # ========================================================
    # Degrees of freedom used in welches t-test
    # ========================================================
    def get_satterthwaite_df(self, s1_stats, s2_stats):

        numerator = ((s1_stats['variance']/s1_stats['n']) + (s2_stats['variance']/s2_stats['n']))**2

        denom1 = (s1_stats['variance'] / s1_stats['n'])**2 / (s1_stats['n'] - 1)
        denom2 = (s2_stats['variance'] / s2_stats['n'])**2 / (s2_stats['n'] - 1)

        satterthwaite_df = numerator / (denom1 + denom2)

        return satterthwaite_df

    # ========================================================
    # Statistics for a paired sample
    # ========================================================
    def get_paired_sample_stats(self, sample1, sample2):
        
        sample1_stats = self.get_one_sample_stats(sample1)
        sample2_stats = self.get_one_sample_stats(sample2)
        
        differences = [x1 - x2 for (x1, x2) in zip(sample1, sample2)]
        
        differences_stats = self.get_one_sample_stats(differences)
        
        # Cohens d for the paired test case
        cohens_d = np.abs(differences_stats['mean'] - self.test_value) / differences_stats['sample_sd']
        
        paired_sample_stats = {"s1_var":sample1_stats['variance'],
                               "s1_sample_sd": sample1_stats['sample_sd'],
                               "s1_n":sample1_stats['n'],
                               "s1_std_error":sample1_stats['std_error'],
                               "s1_min":sample1_stats['min'], 
                               "s1_max":sample1_stats['max'],
                               "s1_median":sample1_stats['median'],
                               "s1_sample_mean":sample1_stats['mean'],
                               "s2_var":sample2_stats['variance'],
                               "s2_sample_sd": sample2_stats['sample_sd'],
                               "s2_n":sample2_stats['n'],
                               "s2_std_error":sample2_stats['std_error'],
                               "s2_min":sample2_stats['min'], 
                               "s2_max":sample2_stats['max'],
                               "s2_median":sample2_stats['median'],
                               "s2_sample_mean":sample2_stats['mean'],                                
                               "var":differences_stats['variance'],            # The statistics from this line and below
                               "sample_sd": differences_stats['sample_sd'],    # were all calculated on the sample of DIFFERENCES
                               "n":differences_stats['n'],                     # created by subtracting each pair, sample1 - sample2
                               "std_error":differences_stats['std_error'],
                               "min":differences_stats['min'], 
                               "max":differences_stats['max'],
                               "median":differences_stats['median'],
                               "mean":differences_stats['mean'],
                               "cohens_d": cohens_d,
                               "ratio": None}
        
        power = self.get_power(effect_size = cohens_d,
                               num_observations=sample1_stats['n'],
                               alpha=self.alpha)        
        
        for key, value in power.items():
            paired_sample_stats[f"pwr_{key}"] = value
        
        return paired_sample_stats

    # ========================================================
    # Two sample statistics
    # ========================================================
    def get_two_sample_stats(self, sample1, sample2):

        sample1_stats = self.get_one_sample_stats(sample1)
        sample2_stats = self.get_one_sample_stats(sample2)

        pooled_sd = self.get_pooled_standard_deviation(sample1_stats, sample2_stats)

        # Standard error for the sampling distribution of the difference 
        # in sample means when we use the "equal variance" assumption (students t-test)
        pooled_std_error = pooled_sd * np.sqrt((1/sample1_stats['n']) + (1/sample2_stats['n']))

        # Standard error for the sampling distribution of the difference 
        # in sample means when we do NOT use the "equal variance" assumption
        welches_std_error = np.sqrt((sample1_stats['variance']/sample1_stats['n']) + (sample2_stats['variance']/sample2_stats['n']))

        satterthwaite_df = self.get_satterthwaite_df(sample1_stats, sample2_stats)
        students_df = sample1_stats['n'] + sample2_stats['n'] - 2
        
        # Cohens d using the pooled standard deviation. 
        # Can we calculate cohens d when we aren't using an equal variance assumption ?? 
        cohens_d = np.abs(((sample1_stats['mean'] - sample2_stats['mean']) - self.test_value)) / pooled_sd

        two_sample_stats = {"s1_var":sample1_stats['variance'],
                            "s1_sample_sd": sample1_stats['sample_sd'],
                            "s1_n":sample1_stats['n'],
                            "s1_std_error":sample1_stats['std_error'],
                            "s1_min":sample1_stats['min'], 
                            "s1_max":sample1_stats['max'],
                            "s1_median":sample1_stats['median'],
                            "s1_sample_mean":sample1_stats['mean'],
                            "s2_var":sample2_stats['variance'],
                            "s2_sample_sd": sample2_stats['sample_sd'],
                            "s2_n":sample2_stats['n'],
                            "s2_std_error":sample2_stats['std_error'],
                            "s2_min":sample2_stats['min'], 
                            "s2_max":sample2_stats['max'],
                            "s2_median":sample2_stats['median'],
                            "s2_sample_mean":sample2_stats['mean'],
                            "diff_sample_means": sample1_stats['mean'] - sample2_stats['mean'],
                            "pooled_sd":pooled_sd,
                            "satter_df":satterthwaite_df,
                            "students_df":students_df,
                            "pooled_std_error": pooled_std_error,
                            "welches_std_error": welches_std_error,
                            "cohens_d": cohens_d,
                            "ratio": sample2_stats['n'] / sample1_stats['n']}
        
        power = self.get_power(effect_size = cohens_d,
                               num_observations=sample1_stats['n'],
                               ratio=sample2_stats['n'] / sample1_stats['n'],
                               alpha=self.alpha)
        
        if self.test_type in self.valid_power_calcs:
            for key, value in power.items():
                two_sample_stats[f"pwr_{key}"] = value
        else:
            for alternative in self.valid_alternatives:
                two_sample_stats[f"pwr_{alternative}"] = "N/A"

        return two_sample_stats

    def get_power(self, effect_size=None, num_observations=None, ratio=None, all_alternatives=True, power=None, alpha=None):
        
        alternatives = ["one_sided_left", "one_sided_right", "two_sided"]

        if self.test_type == "students_t":

            powers = {}   # Dictionary mapping each alternative hypothesis type to the tests power.
            
            # Caculate power for this alternative hypothesis
            for alternative in alternatives:
                
                pwr = smp.TTestIndPower()
                power_param = pwr.solve_power(effect_size=effect_size,
                                              nobs1=num_observations,
                                              alpha=alpha,
                                              ratio=ratio,
                                              alternative=self.sm_alt_hyp_map[alternative],
                                              power=power)
                
                powers[alternative] = power_param
            
        elif self.test_type == "one_sample_t" or self.test_type == "paired_t":

            #alternatives = ["two-sided", "one-sided"]
            
            powers = {} # Dictionary mapping each alternative hypothesis type to the tests power.
            
            for alternative in alternatives:
            
                pwr = smp.TTestPower()
                power_param = pwr.solve_power(effect_size=effect_size,
                                              nobs=num_observations,
                                              alpha=alpha,
                                              power=power,
                                              alternative=self.sm_alt_hyp_map[alternative])
                
                powers[alternative] = power_param
                
        else: # Currently power test is only implemented for the above test types! (not welch or permutation)
            return "N/A"
        
        # Return either a dictionary of powers for all alternatives (if called from the sample statistics code)
        # Or just a single power value (if called from the plotting code).
        return powers
    
    def calculate_sample_size_given_power_table(self, power_args):

        if self.test_type == "students_t":

            powers = power_args['power']

            if 'effect' not in power_args.keys():
                effects = [self.sample_statistics['diff_sample_means']] * len(powers)
            else:
                effects = [power_args['effect']] * len(powers)
            
            samples1 = []
            ratios = []

            for effect, power in zip(effects, powers):

                pwr = smp.TTestIndPower()

                power_param = pwr.solve_power(effect_size=effect,
                                              nobs1=None,
                                              alpha=self.alpha,
                                              ratio=1.0,
                                              alternative=self.sm_alt_hyp_map[self.alternative],
                                              power=power)

                #power_param = power_param[0]  # Have to do this because statsmodels is returning an array for nobs1
                # print(f"samples 1: {power_param}")

                samples1.append(power_param)
                ratios.append(1.0)

            samples2 = [ratio * s1 for ratio, s1 in zip(ratios, samples1)]

            samples_ratio_fixed = {"power": powers, f"samples_{self.first_class}":samples1,
                                   f"samples_{self.second_class}": samples2, "ratio": ratios,
                                   "effects": effects}

            samples1 = []
            ratios = []
            samples2 = []

            for effect, power in zip(effects, powers):

                pwr = smp.TTestIndPower()

                power_param = pwr.solve_power(effect_size=effect,
                                              nobs1=self.sample_statistics['s1_n'],
                                              alpha=self.alpha,
                                              ratio=None,
                                              alternative=self.sm_alt_hyp_map[self.alternative],
                                              power=power)

                #print(f"ratio: {power_param}")

                samples1.append(self.sample_statistics['s1_n'])
                ratios.append(power_param)
            
            # Same calculation as above, only this time the ratio changes each time, and all values in the samples1 
            # list are the same. 
            samples2 = [ratio * s1 for ratio, s1 in zip(ratios, samples1)]

            samples_s1_fixed = {"power": powers, f"samples_{self.first_class}":samples1,
                                f"samples_{self.second_class}":samples2, "ratio":ratios,
                                "effects": effects}
            
            #print(f"samples_s1_fixed:{samples_s1_fixed}\n\n")
            #print(f"samples_ratio_fixed:{samples_ratio_fixed}\n\n")
            
            return samples_s1_fixed, samples_ratio_fixed


    def calculate_power_given_effect_table(self, effect_sizes):

        if self.test_type == "students_t":
            
            # Convert the passed list of effect sizes to cohens_d values
            cohens_ds = [(effect - self.test_value) / self.sample_statistics['pooled_sd'] for effect in effect_sizes]
            powers = []

            for cohens_d in cohens_ds: 

                pwr = smp.TTestIndPower()

                power_param = pwr.solve_power(effect_size=cohens_d,
                                              nobs1=self.sample_statistics['s1_n'],
                                              alpha=self.alpha,
                                              ratio=self.sample_statistics['ratio'],
                                              alternative=self.sm_alt_hyp_map[self.alternative],
                                              power=None)

                powers.append(power_param)
            
            power_dict = {"effect_size" : effect_sizes, "cohens_d": cohens_ds, "power": powers}
            return power_dict
        

    def calculate_standard_error(self):
        if self.test_type == "one_sample_t" or self.test_type == "paired_t":
            return self.sample_statistics['std_error']
        elif self.test_type == "students_t":
            return self.sample_statistics['pooled_std_error']
        elif self.test_type == "welches_t":
            return self.sample_statistics['welches_std_error']
    
    def get_standard_error(self):
        return self.standard_error
    
    def calculate_degrees_of_freedom(self): 
        if self.test_type == "one_sample_t" or self.test_type == "paired_t":
            return self.sample_statistics['n'] - 1
        elif self.test_type == "students_t":
            return self.sample_statistics['students_df']
        elif self.test_type == "welches_t":
            return self.sample_statistics['satter_df']
    
    def get_degrees_of_freedom(self):
        return self.df

    # ==========================================================
    # Get the confidence interval that will always agree with 
    # the hypothesis test we ran.
    # ==========================================================
    def get_agreeing_conf_level(self):
        
        if self.alternative == "two_sided":
            return 1 - self.alpha
        else:
            return 1 - (self.alpha * 2)

    def calculate_confidence_intervals(self, confidence_intervals=None):
        
        # If the user passed a set of confidence intervals to use
        if confidence_intervals is not None:
            conf_intervals = dict.fromkeys(confidence_intervals)  # Initialize the dictionary of confidence_intervals
                                                                  # Keys are the CI's we want to calculate, values initialized to None. 
        
        # If the user did not pass a set of confidence intervals to use, then use the defaults below.
        else:
            default_intervals = [.80, .85, .90, .95, .98, .99]
            conf_intervals = dict.fromkeys(default_intervals)
        
        # Two sided confidence interval that will agree with a one sided
        # hypothesis test. Doing a quick check here to make sure this always gets includd.
        agreeing_interval = self.get_agreeing_conf_level()
        if agreeing_interval not in conf_intervals.keys():   # If the agreeing interval isn't in the set we are going to calculate
            conf_intervals[agreeing_interval] = None         # Add the key so this agreeing interval gets calculated too.
        
        # Doing all the confidence intervals twice, once with only
        # scipy and again more manually (for compairson / learning).
        manual_conf_intervals = copy.deepcopy(conf_intervals)
        
        for conf_level in conf_intervals.keys():
            
            # Automated confidence interval calculation from scipy.
            conf_intervals[conf_level] = stats.t.interval(conf_level,
                                                          df=self.df,
                                                          loc=self.sample_center,
                                                          scale=self.standard_error)
            
            # Getting left tail area for manual CI calculated. This is used as input to the ppf function. 
            left_tail_area = (1 - conf_level) / 2
            
            # More manual confidence interval calculation, we don't need both but nice to compare while building.
            neg_t_crit = stats.t.ppf(q=left_tail_area, df=self.df, loc=0, scale=1)
            pos_t_crit = np.abs(neg_t_crit)
            
            lower_ci = self.sample_center + self.standard_error * neg_t_crit
            upper_ci = self.sample_center + self.standard_error * pos_t_crit
        
            manual_conf_intervals[conf_level] = (lower_ci, upper_ci)
        
        return (manual_conf_intervals, conf_intervals)
    
    def get_scipy_confidence_intervals(self):
        return self.conf_intervals
    
    def get_manual_confidence_intervals(self):
        return self.manual_conf_intervals

    def get_sample_statistics(self):
        return self.sample_statistics