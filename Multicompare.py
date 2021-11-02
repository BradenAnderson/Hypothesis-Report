import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats as sms
import itertools
from statsmodels.stats.weightstats import _tstat_generic
from statsmodels.graphics.gofplots import qqplot
from tabulate import tabulate
import math



class MultipleComparisons():
    def __init__(self, dataframe, test_type, numeric_variable, grouping_variable, ride_alongs=None, reduced_combined=None, lc_coefs=None, alpha=0.05,
                 test_value=0, alternative="two_sided", round_digits=10, lc_type="sums_of_group_means", mc_correction="Bonferroni", pairwise_comparisons="all",
                 pandas_display_precision=10):
        
        pd.set_option('precision', pandas_display_precision)
        
        # Let pairwise_comparisons parameter be a list of tuples that specify which groups the user wants to compare
        # mc_correction is the adjustment factor for multiple comparisons...
        ### bf = "Bonferroni Adjustment"
        
        self.mc_correction = mc_correction
        self.pairwise_comparisons = pairwise_comparisons
        self.mc_results = None
        
        self.data = dataframe
        self.test_type = test_type
        self.test_value = test_value
        self.numeric_variable = numeric_variable
        self.grouping_variable = grouping_variable
        self.round_digits = round_digits
        self.group_statistics = self._calculate_group_statistics()
        
        # For extra sum of squares test
        self.ride_alongs = ride_alongs
        self.reduced_combined = reduced_combined
        self.anova_table = None
        
        # For linear_contrast test
        self.lc_coefs = self._assemble_linear_contrast_coefficients(lc_coefs, lc_type)
        self.alpha=alpha
        self.alternative=alternative
        
        
        self.intermediate_values = {}
        
        self.test_statistic, self.p_value = self.run_test()
        
    def run_test(self):
        if self.test_type == "basic_anova":
            test_statistic, p_value = self.basic_anova()
        elif self.test_type == "extra_sum_of_squares":
            test_statistic, p_value = self.extra_sum_of_squares_test()
        elif self.test_type == "linear_contrast":
            test_statistic, p_value = self.linear_contrast()
        elif self.test_type == "pairwise_comparisons":
            self.mc_results = self.multiple_pairwise_comparisons()
            test_statistic = None
            p_value = None
            
        return test_statistic, p_value 
    def _calculate_group_statistics(self):
        
        group_means = self.data.groupby(by=self.grouping_variable)[[self.numeric_variable]].mean()
        
        # Get the mean of each group.
        group_stats = {f"mean_{key}":value[0] for key, value in group_means.iterrows()}
        
        group_sizes =  self.data.groupby(by=self.grouping_variable)[[self.numeric_variable]].count()
        
        for group_name, count_data, in group_sizes.iterrows():
            group_stats[f"n_{group_name}"] = count_data[0]
        
        group_stats["group_names"] = list(self.data[self.grouping_variable].unique())
        
        # Variance and SD for each group
        for group_name in group_stats["group_names"]:
            group_stats[f"variance_{group_name}"] = np.var(self.data.loc[self.data[self.grouping_variable]== group_name, self.numeric_variable].to_numpy(),
                                                           ddof=1)
            
            group_stats[f"sample_sd_{group_name}"] = np.sqrt(group_stats[f"variance_{group_name}"])
        
        # Grand mean across all groups
        group_stats["grand_mean"] = self.data[self.numeric_variable].mean()
        
        group_stats = self._add_pooled_stdev_info(group_statistics=group_stats)
        
        return group_stats
    
    def _add_pooled_stdev_info(self, group_statistics):
        
        numerator = np.sum([(group_statistics[f"n_{group_name}"]-1)*group_statistics[f"variance_{group_name}"] for group_name in group_statistics["group_names"]])
        denominator = np.sum([group_statistics[f"n_{group_name}"] for group_name in group_statistics["group_names"]])
        denominator -= len(group_statistics["group_names"])
        
        group_statistics["pooled_df"] = denominator
        
        pooled_stdev = np.sqrt(numerator/denominator)
        group_statistics["pooled_sd"] = pooled_stdev
        
        group_statistics["pooled_standard_error"] = pooled_stdev * np.sqrt(np.sum([1/group_statistics[f"n_{group_name}"] 
                                                                                   for group_name in group_statistics["group_names"]]))
        
        return group_statistics
    
    def _assemble_linear_contrast_coefficients(self, lc_coefs, lc_type):
        
        if type(lc_coefs) == dict: 
            groups = lc_coefs.keys()
            coefs = lc_coefs.values()
            
            # Assumes the linear contrast is looking at the difference in sums of group means
            linear_contrast_coefs = {key:(lc_coefs[key] if key in groups else 0) for key in self.group_statistics["group_names"]}
            
            # Same as the "Divisor" argument in estimate statement used inside SAS proc glm. 
            if lc_type == "avgs_of_group_means":
                
                positive_coef_divisor = np.sum([1 for coef in linear_contrast_coefs.values() if coef > 0])
                negative_coef_divisor = np.sum([1 for coef in linear_contrast_coefs.values() if coef < 0])

                linear_contrast_coefs = {key:(value/positive_coef_divisor if value >= 0 else value/negative_coef_divisor)
                                         for key,value in linear_contrast_coefs.items()}
            
            return linear_contrast_coefs
            
    def basic_anova(self):
        
        # Sum of squared residuals under the reduced model.
        # Reduced model has all means being equal.
        reduced_sos_resids = np.sum([(value - self.group_statistics['grand_mean'])**2 for 
                                     value in self.data[self.numeric_variable]])

        self.intermediate_values["reduced_sos_resids"] = reduced_sos_resids
        
        # Full model SOS resids is the same as the "Within Group" Sum of squared residuals.
        # This is because it measures the residuals "within" each group 
        # by subtracting out the groups sample mean.
        full_model_sos_resids = np.sum([(value - self.group_statistics[f"mean_{sport}"])**2 for value, sport 
                                            in zip(self.data[self.numeric_variable], self.data[self.grouping_variable])])
        
        self.intermediate_values["full_sos_resids"] = full_model_sos_resids

        # The "Extra Sum of Squares"  is the total variability
        # we started with when we used a single
        # shared mean to describe the groups, minus the variability that 
        # we had when each group got its own mean.
        # This could also be called the "Between groups sum of squares"
        extra_sos = reduced_sos_resids - full_model_sos_resids
        
        self.intermediate_values["extra_sos_resids"] = extra_sos

        # Reduced model degrees of freedom is number of samples - 1.
        # Because under the reduced model, there is only one (shared) mean being estimated.
        df_reduced = len(self.data[self.grouping_variable]) - 1
        
        self.intermediate_values["df_reduced"] = df_reduced

        # Degrees of freedom for the full model is number of samples - number of groups
        # Because under the full model, we estimate a separate mean for each group
        df_full = len(self.data[self.grouping_variable]) - self.data[self.grouping_variable].nunique()
        
        self.intermediate_values["df_full"] = df_full

        # "Extra" degrees of freedom is the difference between reduced and full. 
        df_extra = df_reduced - df_full
        
        self.intermediate_values["df_extra"] = df_extra
        
        # A mean square is the ratio of a sum of squares to its degrees of freedom

        # Extra (between) group mean square
        # This could also be called the "Between groups mean square"
        mean_square_between = extra_sos / df_extra
        
        self.intermediate_values["mean_square_between"] = mean_square_between

        # Within (full model) group mean square
        mean_square_within = full_model_sos_resids / df_full
        
        self.intermediate_values["mean_square_within"] = mean_square_within

        # F-statistic is the ratio of the between MS to the within MS
        # If the null is correct, the numerator and denominator are both
        # estimates of the shared variance. So if the null is true, this 
        # F-statistic will be close to one.
        f_stat = mean_square_between / mean_square_within

        # The f_statistic takes on an f_distribution, where the 
        # numerator degrees of freedom (dfn) = df_extra (df between), and,
        # denominator degrees of freedom (dfd) = df_full (full model df aka "within" df)
        p_value = 1 - stats.f.cdf(f_stat, dfn=df_extra, dfd=df_full)
        
        # Createa dataframe set up like an ANOVA table.
        self.anova_table = self.build_anova_table(reduced_df=df_reduced,
                                        full_df=df_full,
                                        extra_df=df_extra,
                                        reduced_sos_resids=reduced_sos_resids,
                                        within_sos_resids=full_model_sos_resids,
                                        between_sos_resids=extra_sos,
                                        within_mean_square=mean_square_within,
                                        between_mean_square=mean_square_between,
                                        f_stat=f_stat,
                                        p_value=p_value)

        return f_stat, p_value
    
    def build_anova_table(self, reduced_df, full_df, extra_df, reduced_sos_resids, within_sos_resids, between_sos_resids,
                          within_mean_square, between_mean_square, f_stat, p_value):

        data = {"Source of Variation" : ["Model\n(Between Groups)", "Error\n(Within Groups)", "Total"],
                "DF":[extra_df, full_df, reduced_df],
                "Sum of Squares": [between_sos_resids, within_sos_resids, reduced_sos_resids],
                "Mean Square": [str(round(between_mean_square, self.round_digits)),
                                str(round(within_mean_square, self.round_digits)), "--"],
                "F Value": [str(round(f_stat, self.round_digits)), "--", "--"],
                "Probability > F Value\n(p_value)" : [str(round(p_value, self.round_digits)), "--", "--"]}

        anova_df = pd.DataFrame(data=data)
        
        return anova_df
    
    def calculate_squared_resids(self, row, numeric_variable, grouping_variable, group_means):

        group = row[grouping_variable]
        value = row[numeric_variable]
        squared_resid = (value - group_means[group])**2

        return squared_resid

    def extra_sum_of_squares_test(self):

        unique_groups = self.data[self.grouping_variable].unique()

        if self.ride_alongs is not None:
            reduced_combined = [variable for variable in unique_groups if variable not in self.ride_alongs]
        else:
            reduced_combined = self.reduced_combined
            
        self.intermediate_values["reduced_combined"] = reduced_combined

        # Shared mean for all groups that are being combined under the reduced model.
        reduced_combined_mean = self.data.loc[self.data[self.grouping_variable].isin(reduced_combined), self.numeric_variable].mean()
        
        self.intermediate_values["reduced_combined_mean"] = reduced_combined_mean

        # Dictionary mapping each group to the proper mean (or shared mean) under the reduced (null hypothesis) model. 
        group_means = {variable:(self.data.loc[self.data[self.grouping_variable]==variable, self.numeric_variable].mean() 
                                 if variable not in self.reduced_combined else reduced_combined_mean)
                        for variable in unique_groups}
        
        
        self.intermediate_values["group_means_under_null"] = group_means

        # Apply the "calculate_squared_resids" function to each row of the dataframe to get the squared
        # residuals as they exist under the reduced model for each sample, them sum the result to get the sum
        # of squared residuals under the null hypothesis.
        sos_resids_reduced = np.sum(self.data.apply(func=self.calculate_squared_resids, 
                                             args=(self.numeric_variable, self.grouping_variable, group_means), 
                                             axis=1))

        
        self.intermediate_values["reduced_sos_resids"] = sos_resids_reduced
        
        # Degrees of freedom for the reduced model is:
        # Number of samples - (Number_of_groups - Number_of_groups_with_combined_means + 1)
        deg_freedom_reduced = len(self.data) - (self.data[self.grouping_variable].nunique() - len(self.reduced_combined) + 1)
        
        self.intermediate_values["df_reduced"] = deg_freedom_reduced
        
        # Full model, each group gets its own mean. 
        deg_freedom_full = len(self.data) - self.data[self.grouping_variable].nunique()
        
        self.intermediate_values["df_full"] = deg_freedom_full

        # Means for each group under the alternative (full) model
        group_means_full = {variable:self.data.loc[self.data[self.grouping_variable]==variable, self.numeric_variable].mean() for variable in unique_groups}
        
        
        self.intermediate_values["group_means_under_full"] = group_means_full
        
        # Sum of squared residuals under the full model
        sos_resids_full = np.sum(self.data.apply(func=self.calculate_squared_resids, 
                                             args=(self.numeric_variable, self.grouping_variable, group_means_full), 
                                             axis=1))
        
        self.intermediate_values["full_sos_resids"] = sos_resids_full

        # Extra SOS resids and extra degrees of freedom found through subtraction.
        sos_resids_between = sos_resids_reduced - sos_resids_full
        deg_freedom_extra = deg_freedom_reduced - deg_freedom_full
        
        self.intermediate_values["extra_sos_resids"] = sos_resids_between

        # Mean square is the sum of squared residuals per degree of freedom. 
        mean_square_within = sos_resids_full / deg_freedom_full
        mean_square_between = sos_resids_between / deg_freedom_extra
        
        self.intermediate_values["mean_square_within"] = mean_square_within
        self.intermediate_values["mean_square_between"] = mean_square_between
        
        f_statistic = mean_square_between / mean_square_within

        p_value = 1 - stats.f.cdf(f_statistic, dfn=deg_freedom_extra, dfd=deg_freedom_full)

        self.anova_table = self.build_anova_table(reduced_df=deg_freedom_reduced,
                                        full_df=deg_freedom_full,
                                        extra_df=deg_freedom_extra,
                                        reduced_sos_resids=sos_resids_reduced,
                                        within_sos_resids=sos_resids_full,
                                        between_sos_resids=sos_resids_between,
                                        within_mean_square=mean_square_within,
                                        between_mean_square=mean_square_between,
                                        f_stat=f_statistic,
                                        p_value=p_value)

        return f_statistic, p_value
    
    def _calculate_t_critical_value(self, alpha):
        if self.alternative == "two_sided":
            t_critical = stats.t.ppf(q=1 - alpha/2, df=self.group_statistics["pooled_df"], loc=0, scale=1)
        elif self.alternative == "one_sided_right":
            t_critical = stats.t.ppf(q=1-alpha, df=self.group_statistics["pooled_df"], loc=0, scale=1)
        elif self.alternative == "one_sided_left":
            t_critical = stats.t.ppf(q=alpha, df=self.group_statistics["pooled_df"], loc=0, scale=1)
        
        return t_critical
    
    def _calculate_lc_p_value(self, test_statistic):
        
        # Note: stats.t.sf is the "survival function", which is the
        # same thing as 1 - stats.t.cdf, except sometimes using the
        # survival function can be more accurate.
        
        if self.alternative == "two_sided":
            
            # Multiplied by two to get both tail areas.
            p_value = stats.t.sf(x=np.abs(test_statistic), df=self.group_statistics["pooled_df"], loc=0, scale=1) * 2
            
        elif self.alternative == "one_sided_right":
            p_value = stats.t.sf(x=test_statistic, df=self.group_statistics["pooled_df"], loc=0, scale=1)
            
        elif self.alternative == "one_sided_left":
            
            # Left sided test, t-stat will already be negative, so we want that left tail area from cdf not 1-cdf from survival function.s
            p_value = stats.t.cdf(x=test_statistic, df=self.group_statistics["pooled_df"], loc=0, scale=1)
            
        return p_value
    
    
    def _calculate_sum_of_squares_contrast(self):
        
        group_sizes = [self.group_statistics[f"n_{group_name}"] for group_name in self.group_statistics["group_names"]]
        
        all_sizes_equal = group_sizes.count(group_sizes[0]) == len(group_sizes)
        
        if True:
            
            n = group_sizes[0]
            
            num = np.sum([(self.lc_coefs[group_name]*self.group_statistics[f"mean_{group_name}"])**2 for group_name in self.group_statistics["group_names"]])
            denom = np.sum([self.lc_coefs[group_name]**2 for group_name in self.group_statistics["group_names"]])
            
            ss_contrast = num / denom
            
            return ss_contrast
            
    def linear_contrast(self):
        
        g = np.sum([self.lc_coefs[name]*self.group_statistics[f"mean_{name}"] for name in self.group_statistics["group_names"]])
        
        self.intermediate_values["linear_contrast_g"] = g
        self.intermediate_values["sos_contrast"] = self._calculate_sum_of_squares_contrast()
        
        standard_error_g = self.group_statistics["pooled_sd"]*np.sqrt(np.sum([self.lc_coefs[name]**2/self.group_statistics[f"n_{name}"] 
                                                                              for name in self.group_statistics["group_names"]]))
        
        self.intermediate_values["standard_error_g"] = standard_error_g
        
        test_statistic = (g - self.test_value) / standard_error_g
        
        self.intermediate_values["test_statistic"] = test_statistic
        
        #t_critical = self._calculate_t_critical_value(alpha=self.alpha)
        
        #self.intermediate_values["t_critical"] = t_critical
        
        p_value = self._calculate_lc_p_value(test_statistic=test_statistic)
        
        return test_statistic, p_value
    
    def _add_bonferroni_adjustment(self, results, num_comparisons):
        
        results["adjusted_p_value"] = [min(p_val * num_comparisons, 1.0) for p_val in results["p_value"]]
        results["significant"] = [True if p_val<self.alpha else False for p_val in results["p_value"]]
            
        adjusted_alpha = self.alpha / num_comparisons
        
        t_multiplier = self._calculate_t_critical_value(alpha=adjusted_alpha)
        
        self.intermediate_values["t_critical"] = t_multiplier
        
        results["lower_ci"] = [diff_mean - t_multiplier*std_err for diff_mean, std_err in zip(results['difference between means'], results["standard_error"])]
        results["upper_ci"] = [diff_mean + t_multiplier*std_err for diff_mean, std_err in zip(results['difference between means'], results["standard_error"])]
        
        result_df = pd.DataFrame(results)
        
        result_df.sort_values(by="adjusted_p_value", inplace=True)
        
        return result_df
    
    def multiple_pairwise_comparisons(self):
        
        results = {"group_1":[], "group_2":[], "difference between means": [],
                   "test_statistic":[], "p_value":[], "standard_error":[]}

        if self.pairwise_comparisons == "all":
            pairs = [combo for combo in itertools.combinations(self.group_statistics["group_names"], 2)]
            
            num_comparisons = len(pairs)
            
            reverse_pairs = [(group2, group1) for group1, group2 in pairs]
            pairs.extend(reverse_pairs)
        
        # Else, we passed a list of tuples specifying exactly which comparisons to make.
        else:
            pairs = self.pairwise_comparisons
            num_comparisons = len(pairs)

        for group1, group2 in pairs:

            group1_data = df.loc[df[self.grouping_variable] == group1, self.numeric_variable].to_numpy()
            group2_data = df.loc[df[self.grouping_variable] == group2, self.numeric_variable].to_numpy()
            
            std_err = self.group_statistics["pooled_sd"] * np.sqrt((1/self.group_statistics[f"n_{group1}"]) + (1/self.group_statistics[f"n_{group2}"]))
            results["standard_error"].append(std_err)
            
            # Needed to use https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats._tstat_generic.html#statsmodels.stats.weightstats._tstat_generic
            # because Scipy.stats.ttest_ind doesn't let you specify a degrees of freedom. 
            t_stat, p_val = _tstat_generic(value1=self.group_statistics[f"mean_{group1}"],
                                                           value2=self.group_statistics[f"mean_{group2}"],
                                                           std_diff=std_err,
                                                           dof=self.group_statistics["pooled_df"],
                                                           alternative="two-sided",
                                                           diff=self.test_value # value of the difference under the null, usually 0. 
                                                          )
            
            results["group_1"].append(group1)
            results["group_2"].append(group2)
            results["test_statistic"].append(t_stat)
            results["p_value"].append(p_val)
            results["difference between means"].append(self.group_statistics[f"mean_{group1}"] - self.group_statistics[f"mean_{group2}"])
        
        
        if self.mc_correction == "Bonferroni":
            result_df = self._add_bonferroni_adjustment(results=results, num_comparisons=num_comparisons)
            
        else:
            result_df = pd.DataFrame(results)
            
        return result_df
        
    def get_anova_table(self, verbose=False):
        
        if verbose:
            print(tabulate(self.anova_table, headers="keys", showindex=False))
        
        return self.anova_table
    
    def get_test_results(self):
        return self.test_statistic, self.p_value
    
    def get_group_statistics(self):
        return self.group_statistics
    
    def get_intermediate_values(self):
        return self.intermediate_values
    
    def get_mc_results(self):
        return self.mc_results
    
    def print_full_summary(self):
        
        if self.test_type == "pairwise_comparisons":
            print(self.mc_results)
        else:
            print("=========================================================================")
            print(f"test statistic: {self.test_statistic}, p_value: {self.p_value}\n")
            print(f"intermediate values:\n {self.intermediate_values}\n")
            print(f"group stats: {self.group_statistics}")
            print("=========================================================================\n")
            
    def plot_pairwise_comparison_matrix(self, mask_location="upper_right", cmap='Blues', figsize=(12,12), rotate_xticks=45,
                                        text_color="g", na_fontsize=18, na_fontcolor="y", pval_round_digits=10):
        
        
        if self.pairwise_comparisons == "all":
            labels = self.mc_results['group_1'].unique()
            num_groups = self.mc_results['group_1'].nunique()
        else:
            used_labels = [lab for group in self.pairwise_comparisons for lab in group]
            unique_used = list(np.unique(used_labels))
            all_unique_labels = self.group_statistics["group_names"]
            needed_unique = [lab for lab in all_unique_labels if lab not in unique_used]
            labels = unique_used + needed_unique
            num_groups = len(labels)

                
        # Create a matrix that will plot each group (y-axis)
        # against every other group (x-axis).
        matrix = []
        for index in range(len(labels)):
            row = [(lab, labels[index]) for lab in labels]
            matrix.extend(row)
        
        # A list for each thing we want to be
        # able to use in an annotation on the plot.
        values = []
        diff_means = []
        tstats = []
        upperci = []
        lowerci = []
        
        skips = []
        
        # for all group1, group2 pairs in the matrix
        for g1, g2 in matrix: 
            
            # "-1" as a placeholder on the diagonals, because
            # we do not test a group against itself!
            if g1 == g2:
                values.append(0.5)
                diff_means.append(-1)
                tstats.append(-1)
                upperci.append(-1)
                lowerci.append(-1)
                skips.append(True)
            
            # If we didn't test all the pairs, and this isn't a pair we tested.
            elif self.pairwise_comparisons != "all" and (g1, g2) not in self.pairwise_comparisons:
                values.append(0.5)
                diff_means.append(-1)
                tstats.append(-1)
                upperci.append(-1)
                lowerci.append(-1)
                skips.append(True)
            
            # Collecting values for annotation
            else:
                value = result_df.loc[(result_df['group_1'] == g1) & (result_df["group_2"] == g2), "adjusted_p_value"].to_numpy()[0]
                delta_means = result_df.loc[(result_df['group_1'] == g1) & (result_df["group_2"] == g2), "difference between means"].to_numpy()[0]
                tstat = result_df.loc[(result_df['group_1'] == g1) & (result_df["group_2"] == g2), "test_statistic"].to_numpy()[0]
                uci = result_df.loc[(result_df['group_1'] == g1) & (result_df["group_2"] == g2), "upper_ci"].to_numpy()[0]
                lci = result_df.loc[(result_df['group_1'] == g1) & (result_df["group_2"] == g2), "lower_ci"].to_numpy()[0]
                values.append(value)
                diff_means.append(delta_means)
                tstats.append(tstat)
                upperci.append(uci)
                lowerci.append(lci)
                skips.append(False)

        # Each piece of data collected above needs to be converted into a 
        # numpy array, and reshaped to it is square (each side of square is the number
        # of groups we have in the dataset).
        values_arr = np.array(values).reshape(num_groups, num_groups)
        diff_means = np.array(diff_means).reshape(num_groups, num_groups)
        tstats = np.array(tstats).reshape(num_groups, num_groups)
        lowerci = np.array(lowerci).reshape(num_groups, num_groups)
        upperci = np.array(upperci).reshape(num_groups, num_groups)
        skips = np.array(skips).reshape(num_groups, num_groups)

        
        # Create a mask so we can throw away redundant values (either upper or lower diagonal)
        
        if mask_location == "upper_right":
            mask = np.tri(values_arr.shape[0], k=0).T
            values_masked = np.ma.array(values_arr, mask=mask)
        elif mask_location == "lower_left":
            mask = np.tri(values_arr.shape[0], k=0)
            values_masked = np.ma.array(values_arr, mask=mask)
        elif mask_location == "none":
            mask = np.array([False]*values_arr.shape[0]*values_arr.shape[1]).reshape(values_arr.shape[0], values_arr.shape[1])
            values_masked = np.ma.array(values_arr, mask=mask)
        
        # Create the figure and axis.
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        
        # plot the matrix of adjusted p-values on the image.
        im = axs.imshow(values_masked, cmap=cmap, vmin=0.0, vmax=1.0)

        # Adjust tick labels
        axs.set_xticks(np.arange(num_groups))
        axs.set_yticks(np.arange(num_groups))
        axs.set_xticklabels(labels)
        axs.set_yticklabels(labels)
        axs.tick_params(axis='both', labelsize=14)
        
        if rotate_xticks != 0:
            plt.setp(axs.get_xticklabels(), rotation=rotate_xticks, ha="right", rotation_mode="anchor")
        
        # Turn off the frame around the axis.
        axs.spines["top"].set_visible(False)
        axs.spines["bottom"].set_visible(False)
        axs.spines["left"].set_visible(False)
        axs.spines["right"].set_visible(False)
        
        # Add a title for the plot
        axs.set_title("All Pairwise T-tests with Bonferroni Adjusted P-Values", fontsize=22, weight='bold')
        
        # Add a colorbar on the right side of the plot.
        divider = make_axes_locatable(axs)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        fig.colorbar(mappable=im, cax=cax, orientation="vertical")
        
        # Add our annotations.
        for i in range(num_groups):
            for j in range(num_groups):

                if not(values_masked.mask[i, j]):
                    
                    if not(skips[i, j]):
                        p_val_txt = round(values_arr[i, j], pval_round_digits)
                        
                        if p_val_txt == 0.0:
                            p_val_txt = f"<2e-{pval_round_digits}"
                        
                        
                        tstat_txt = round(tstats[i, j], 3)
                        diff_means_txt = round(diff_means[i, j], 3)
                        ci_txt = f"({round(lowerci[i, j], 2)}, {round(upperci[i, j], 2)})"

                        txt1 = (f"adj pvalue = {p_val_txt}\n"
                                f"{ci_txt}\n\n")
                        txt2 = (f"tstat = {tstat_txt}\n"
                               f"diff means = {diff_means_txt}")

                        text = axs.text(j, i, txt1, ha="center", va="bottom", color=text_color, fontweight="bold")
                        text = axs.text(j, i, txt2, ha="center", va="center", color=text_color)
                        
                    else:
                        text = axs.text(j, i, "N/A", ha="center", va="bottom", color=na_fontcolor, fontweight="bold", fontsize=na_fontsize)
                        

        plt.tight_layout()
        plt.savefig(fname="./Multiple_Comparison_Matrix.png", format="png")