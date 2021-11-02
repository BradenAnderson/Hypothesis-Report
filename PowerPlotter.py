import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import scipy.stats as stats
import statsmodels.stats.power as smp

class PowerPlotter():
    def __init__(self, test_type, test_value, relative_imgs_dir, project_dir, alpha=None, alternative=None, sample_statistics=None):
        
        self.sm_alt_hyp_map = {"two_sided": "two-sided",
                               "one_sided_right": "larger",
                               "one_sided_left": "smaller"}
        
        self.test_type = test_type
        self.test_value = test_value
        self.alternative = "two_sided"   ### Restricting to only two sided alternatives for now.
        self.alpha = alpha
        self.sample_statistics = sample_statistics
        
        self.common_significance_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        
        # For saving
        self.relative_images_directory = relative_imgs_dir
        self.project_directory = project_dir
        self.relative_file_paths = {}
        
        self.plot_types = {"one_sample_t": [ "Power_vs_Sample_Size", "Power_vs_Effect_Size"], 
                           "students_t": ["Power_vs_Sample_Size", "Power_vs_Effect_Size"],
                           "paired_t":["Power_vs_Sample_Size", "Power_vs_Effect_Size"]}
    
    
    def _get_power_data(self, power_min, power_max, sig_lvls, effect_size=None, solve_using=None):
        
        pwr = smp.TTestPower()
        
        for sig_lvl, data in sig_lvls.items():
            
            
            # If solving for power based on sample size
            # Always assume samples are the same size (ratio = 1).
            if solve_using == "sample_sizes":
                sample_size = None

            # If solving for power based on effect size, use the actual values
            # for sample1 and sample2 sizes.
            elif solve_using == "effect_sizes":
                effect_size=None
                sample_size = self.sample_statistics['n']

            # Find the sample size required for us to achieve our minimum power (0.5 by default)
            min_x_value = pwr.solve_power(effect_size=effect_size,
                                              nobs=sample_size,
                                              alpha=sig_lvl,
                                              power=power_min)

            # Find the sample size required to reach the max power we want to plot (default is 0.95)
            max_x_value = pwr.solve_power(effect_size=effect_size,
                                              nobs=sample_size,
                                              alpha=sig_lvl,
                                              power=power_max)
            
            if solve_using == "sample_sizes":
                
                # Make sure our bounds always include the value we actually have. 
                min_x = math.floor(min(min_x_value, self.sample_statistics['n']))
                max_x = math.ceil(max(max_x_value, self.sample_statistics['n']))

                data[solve_using] = [value for value in range(min_x, max_x + 1)]
                data['power_values'] = []
                
            elif solve_using == "effect_sizes":
                
                # Make sure bounds always include the effect size values we actually have.
                min_x = min(min_x_value, self.sample_statistics['cohens_d'])
                max_x = max(max_x_value, self.sample_statistics['cohens_d'])
                
                data[solve_using] = list(np.linspace(start=min_x, stop=max_x, num=2_000))
                
                # Convert the cohens d effect sizes back to raw effect size (difference in sample means)
                # For a more interpretable x-axis plot option.
                data[f"{solve_using}_raw"] = [self._cohen_to_raw_effect(es) for es in data[solve_using]]
            
            data['power_values'] = []
            
            for x_value in data[solve_using]:
                
                if solve_using == "sample_sizes":
                    sample_size = x_value
                elif solve_using == "effect_sizes":
                    effect_size = x_value

                power= pwr.solve_power(effect_size=effect_size,
                                       nobs=sample_size,
                                       alpha=sig_lvl,
                                       alternative=self.sm_alt_hyp_map[self.alternative])

                data['power_values'].append(power)

        return sig_lvls
    
    def _get_power_data_ind(self, power_min, power_max, sig_lvls, effect_size=None, solve_using=None):
        
        pwr = smp.TTestIndPower()
                 
        for sig_lvl, data in sig_lvls.items():
            
            # If solving for power based on sample size
            # Always assume samples are the same size (ratio = 1).
            if solve_using == "sample_sizes":
                ratio = 1
                sample_size = None

            # If solving for power based on effect size, use the actual values
            # for sample1 and sample2 sizes.
            elif solve_using == "effect_sizes":
                effect_size=None
                sample_size = self.sample_statistics['s1_n']
                ratio = self.sample_statistics['s2_n'] / self.sample_statistics['s1_n']
            
            # Find the sample size required for us to achieve our minimum power (0.6 by default)
            min_x_value = pwr.solve_power(effect_size=effect_size,
                                          nobs1=sample_size,
                                          alpha=sig_lvl,
                                          power=power_min,
                                          ratio=ratio)

            # Find the sample size required to reach the max power we want to plot (default is 0.95)
            max_x_value = pwr.solve_power(effect_size=effect_size,
                                          nobs1=sample_size,
                                          alpha=sig_lvl,
                                          power=power_max,
                                          ratio=ratio)

            
            if solve_using == "sample_sizes":
                
                # Make sure our bounds always include the sample size values value we actually have.
                min_x = math.floor(min(min_x_value, min(self.sample_statistics['s1_n'], self.sample_statistics['s2_n'])))
                max_x = math.ceil(max(max_x_value, max(self.sample_statistics['s1_n'], self.sample_statistics['s2_n'])))
                
                data[solve_using] = [value for value in range(min_x, max_x + 1)]
            
            
            elif solve_using == "effect_sizes":
                
                # Make sure bounds always include the effect size values we actually have.
                min_x = min(min_x_value, self.sample_statistics['cohens_d'])
                max_x = max(max_x_value, self.sample_statistics['cohens_d'])
                
                data[solve_using] = list(np.linspace(start=min_x, stop=max_x, num=2_000))
                
                # Convert the cohens d effect sizes back to raw effect size (difference in sample means)
                # For a more interpretable x-axis plot option.
                data[f"{solve_using}_raw"] = [self._cohen_to_raw_effect(es) for es in data[solve_using]]
            
            data['power_values'] = []
            
            for x_value in data[solve_using]:
                
                if solve_using == "sample_sizes":
                    sample_size = x_value
                elif solve_using == "effect_sizes":
                    effect_size = x_value

                power= pwr.solve_power(effect_size=effect_size,
                                       nobs1=sample_size,
                                       alpha=sig_lvl,
                                       ratio=ratio,
                                       alternative=self.sm_alt_hyp_map[self.alternative])

                data['power_values'].append(power)

        return sig_lvls
    
    def _cohen_to_raw_effect(self, cohens_d):
        
        if self.test_type == "students_t":
            return (cohens_d * self.sample_statistics['pooled_sd']) + self.test_value
        elif self.test_type in ["one_sample_t", "paired_t"]:
            return (cohens_d * self.sample_statistics['sample_sd']) + self.test_value
    
    def _get_significance_levels(self):
        
        significance_lvls_to_plot = {}
        significance_lvls_to_plot[self.alpha] = {}
        
        # Common intervals below and above the chosen significance.
        below = [level for level in self.common_significance_levels if level < self.alpha]
        above = [level for level in self.common_significance_levels if level > self.alpha]
        
        # If there is one, grab the largest common sig. level that is
        # lower than the chosen level.
        if below != []:
            significance_lvls_to_plot[max(below)] = {}

        # If there is one, grab the smallest common sig. level that is
        # bigger than the chosen one.
        if above != []:
            significance_lvls_to_plot[min(above)] = {}
        
        return significance_lvls_to_plot
        
    def build_power_plots(self, power_min=0.6, power_max=0.95):
        
        for plot_type in self.plot_types[self.test_type]:
            
            if plot_type == "Power_vs_Sample_Size" and self.test_type in ["one_sample_t", "paired_t"]:
                
                significance_levels = self._get_significance_levels()
                
                data_by_sig_lvl = self._get_power_data(power_min=power_min,
                                                       power_max=power_max,
                                                       effect_size=self.sample_statistics['cohens_d'],
                                                       sig_lvls = significance_levels,
                                                       solve_using="sample_sizes")
                
                
                
                self.plot_power_vs_samples_or_effect(data=data_by_sig_lvl, x_val_type="sample_sizes", plot_name=plot_type)
                
            elif plot_type == "Power_vs_Sample_Size" and self.test_type == "students_t":
                
                significance_levels = self._get_significance_levels()
                
                data_by_sig_lvl = self._get_power_data_ind(power_min=power_min,
                                                           power_max=power_max,
                                                           effect_size=self.sample_statistics['cohens_d'],
                                                           sig_lvls=significance_levels,
                                                           solve_using="sample_sizes")
                
                self.plot_power_vs_samples_or_effect(data=data_by_sig_lvl, x_val_type="sample_sizes", plot_name=plot_type)
                
            
            elif plot_type == "Power_vs_Effect_Size" and self.test_type in ["one_sample_t", "paired_t"]:
                
                significance_levels = self._get_significance_levels()
                
                data_by_sig_lvl = self._get_power_data(power_min=power_min,
                                                       power_max=power_max,
                                                       effect_size=None,
                                                       sig_lvls = significance_levels,
                                                       solve_using="effect_sizes")
                
                self.plot_power_vs_samples_or_effect(data=data_by_sig_lvl, x_val_type="effect_sizes", plot_name=plot_type)
                
                
            elif plot_type == "Power_vs_Effect_Size" and self.test_type == "students_t":
                
                significance_levels = self._get_significance_levels()
                
                data_by_sig_lvl = self._get_power_data_ind(power_min=power_min,
                                                           power_max=power_max,
                                                           effect_size=None,
                                                           sig_lvls=significance_levels,
                                                           solve_using="effect_sizes")
                
                self.plot_power_vs_samples_or_effect(data=data_by_sig_lvl, x_val_type="effect_sizes", plot_name=plot_type)
                

                
    
    def _get_power_vs_sample_or_effect_plot_text(self, x_val_type, annot_coords):
        
        plot_text = {}
        
        if x_val_type == "sample_sizes":
            x_annot = "Samples ="
        elif x_val_type == "effect_sizes":
            x_annot = "Effect Size = "
            
        for threshold in annot_coords.keys():
            x, y = annot_coords[threshold]
    
            plot_text[f'pwr_annot_{threshold}']  = (f"Pwr = {round(y, 3)}\n"
                                                    f"{x_annot} = {round(x, 3)}")
            
            if x_val_type == "effect_sizes":
                plot_text[f'pwr_annot_{threshold}_raw']  = (f"Pwr = {round(y, 3)}\n"
                                                            f"{x_annot} = {round(self._cohen_to_raw_effect(y), 3)}")

        
        if self.test_type == "one_sample_t" and x_val_type == "sample_sizes":
            effect_size = round(self.sample_statistics['mean'] - self.test_value, 3)
            plot_text['title'] = ["Power vs Sample Size"]*2
            plot_text['xlabel'] = "Sample Size"
            
            # Note that shows other params that also affect power.
            note = (f"Effect Size (sample_mean - hypothesized) = {effect_size}\n"
                    f"Effect Size (cohens d) = {round(self.sample_statistics['cohens_d'], 3)}")
            
        elif self.test_type == "paired_t" and x_val_type == "sample_sizes":
            effect_size = round(self.sample_statistics['mean'] - self.test_value, 3)
            plot_text['title'] = ["Power vs Sample Size"]*2
            plot_text['xlabel'] = ["Number of paired samples"]*2
            
            # Note that shows other params that also affect power.
            note = (f"Effect Size (differences - hypothesized) = {effect_size}\n"
                    f"Effect Size (cohens d) = {round(self.sample_statistics['cohens_d'], 3)}")
            
        elif self.test_type == "students_t" and x_val_type == "sample_sizes":
            effect_size = round(self.sample_statistics['diff_sample_means'] - self.test_value, 3)
            plot_text['title'] = ["Power vs Sample Size\n(Assuming each group has equal number of samples)"]*2
            plot_text['xlabel'] = ["Sample Size (Per Group)"]*2
            
            # Note that shows other params that also affect power.
            note = (f"Effect Size (True diff means) = {effect_size}\n"
                    f"Effect Size (cohens d) = {round(self.sample_statistics['cohens_d'], 3)}")
        
        elif self.test_type == "one_sample_t" and x_val_type == "effect_sizes":
            plot_text['title'] = ["Power vs Effect Size (Cohens D)"]*2 + ["Power vs Effect Size (Raw)"]*2
            plot_text['xlabel'] = ["Effect Size (Cohens D)"]*2 + ["Effect Size (x_bar obs - x_bar hypothesized)"]*2
            
            # Note that shows other params that also affect power.
            note = (f"Sample Size = {self.sample_statistics['n']}\n")
            
        elif self.test_type == "paired_t" and x_val_type == "effect_sizes":
            plot_text['title'] = ["Power vs Effect Size (Cohens D)"]*2 + ["Power vs Effect Size (Raw)"]*2
            plot_text['xlabel'] = ["Effect Size (Cohens D)"]*2 + ["Effect Size (mean differences obs - hypothesized)"]*2
            
            # Note that shows other params that also affect power.
            note = (f"Number of pairs = {self.sample_statistics['n']}\n")
            
        elif self.test_type == "students_t" and x_val_type == "effect_sizes":
            plot_text['title'] = ["Power vs Effect Size (Cohens D)"]*2 + ["Power vs Effect Size (Raw)"]*2
            plot_text['xlabel'] = ["Effect Size (Cohens D)"]*2 + ["Effect Size (Obs difference in means - hypothesized)"]*2
            
            # Note that shows other params that also affect power.
            note = (f"Sample1 Size = {self.sample_statistics['s1_n']}\n"
                    f"Sample2 Size = {self.sample_statistics['s2_n']}\n")
        
        plot_text['ylabel'] = "Statistical Power"
        plot_text['note'] = note
        
        return plot_text
    
    def plot_power_vs_samples_or_effect(self, data, plot_name, x_val_type):

        sns.set_style("darkgrid")
        
        if x_val_type == "sample_sizes":
            nrows=1
            figsize=(24,8)
        elif x_val_type == "effect_sizes":
            nrows=2
            figsize=(24,16)

        fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
        
        if x_val_type == "effect_sizes":
            
            axs = axs.flatten()
            
            sns.lineplot(x=data[self.alpha][f'{x_val_type}_raw'], y=data[self.alpha]['power_values'], ax=axs[2], label=f"alpha {self.alpha}")
            
            y_min_orig, y_max_orig = axs[2].get_ylim()
            x_min_orig, x_max_orig = axs[2].get_xlim()
            
        
        sns.lineplot(x=data[self.alpha][x_val_type], y=data[self.alpha]['power_values'], ax=axs[0], label=f"alpha {self.alpha}")
        
        # Get axes limits for notes
        y_min, y_max = axs[0].get_ylim()
        x_min, x_max = axs[0].get_xlim()
        
        annot_coords = dict.fromkeys([0.80, 0.85, 0.90])
        
        ### ANNOTATE PLOT WITH COHENS_D ON THE X-AXIS
        
        for threshold in annot_coords.keys():
            # Get coordinates where we pass the threshold
            annot_coords[threshold] = sorted([(x,y) for x, y in zip(data[self.alpha][x_val_type], data[self.alpha]["power_values"]) 
                                              if y >= threshold], key = lambda sublist : sublist[0])[0]
            
            x_pwr, y_pwr = annot_coords[threshold]
            
            # Annotate where we hit the power threshold
            axs[0].plot(x_pwr, y_pwr, marker="o", color="#FE53BB")
            
            if x_val_type == "effect_sizes":
                x_pwr_raw = self._cohen_to_raw_effect(x_pwr)
                axs[2].plot(x_pwr_raw, y_pwr, marker="o", color="#FE53BB")
        
        # Get a dictionary filled with the text we are going to annotate this plot with.
        plot_text = self._get_power_vs_sample_or_effect_plot_text(x_val_type=x_val_type, annot_coords=annot_coords)
        
        # Add note to upper left corner
        style = dict(size=14, color = "#000000", ha = 'left', va="top", fontweight="bold")
        axs[0].text(x=x_min, y=y_max, s=plot_text['note'], **style)
        
        if x_val_type == "effect_sizes":
            style = dict(size=14, color = "#000000", ha = 'left', va="top", fontweight="bold")
            axs[2].text(x=x_min_orig, y=y_max_orig, s=plot_text['note'], **style)
            
        offset = (x_max - x_min)/100 # 1% offset consistently work?
        
        for threshold in annot_coords.keys():
            x_pwr, y_pwr = annot_coords[threshold]
            style = dict(size=12, color = '#FE53BB', ha = 'left', va="top", fontweight="bold", fontstyle="oblique")
            axs[0].text(x=x_pwr+offset, y=y_pwr, s=plot_text[f'pwr_annot_{threshold}'], **style)
            
            if x_val_type == "effect_sizes":
                x_pwr_raw = self._cohen_to_raw_effect(x_pwr)
                offset_orig = (x_max_orig - x_min_orig)/100
                axs[2].text(x=x_pwr_raw+offset_orig, y=y_pwr, s=plot_text[f'pwr_annot_{threshold}_raw'], **style)
        
        ### START RHS PLOT (MULTIPLE ALPHAS)
        
        for sig_lvl, points in data.items():
            sns.lineplot(x=points[x_val_type], y=points['power_values'], ax=axs[1], label=f"alpha {sig_lvl}")    
            
            if x_val_type == "effect_sizes":
                sns.lineplot(x=points[f'{x_val_type}_raw'], y=points['power_values'], ax=axs[3], label=f"alpha {sig_lvl}")
        
        ### END RHS PLOT (MULTIPLE ALPHAS)
        
        for index, ax in enumerate(axs):
            ax.set_title(label=plot_text['title'][index], fontsize=24, weight="bold")
            ax.tick_params(axis='both', labelsize=20)
            ax.set_xlabel(plot_text['xlabel'][index], fontsize=20, weight="bold")
            ax.set_ylabel(plot_text['ylabel'], fontsize=20, weight="bold")
            ax.legend(loc="lower right", fontsize="large")
        
        
        ### START SAVING CODE
        
        # Relative path to the image
        img_relative_path = os.path.join(self.relative_images_directory, plot_name)
        
        # Full path where we will save the image
        full_save_path = os.path.join(self.project_directory, os.path.normpath(img_relative_path))
        
        # Add the relative path to the dictionary of relative paths
        self.relative_file_paths[plot_name] = img_relative_path
        
        plt.tight_layout()

        plt.savefig(full_save_path,
                    transparent=False,
                    format="png")

    def get_plot_relative_paths(self):
        return self.relative_file_paths