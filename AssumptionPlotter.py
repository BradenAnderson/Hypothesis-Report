import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot
import math
import os
import mdutils


class AssumptionPlotter():
    def __init__(self, dataframe, numeric_variable, grouping_variable):
        
        self.data = dataframe
        self.numeric_variable = numeric_variable
        self.grouping_variable = grouping_variable
        
        self.num_groups = self.data[self.grouping_variable].nunique()
        self.group_names = list(self.data[self.grouping_variable].unique())

        self.report = mdutils.MdUtils(file_name=f"./ASSUMPTIONS_{numeric_variable}_{grouping_variable}",
                                      title=f"assumption_plots_{numeric_variable}_{grouping_variable}")

        self.relative_file_paths = {}
        self.relative_images_directory = "./images/"
    
    def build_report(self, include_transform_plots=True):

        os.makedirs(self.relative_images_directory, exist_ok=True)

        # Histograms
        self.report.new_header(level=1, title='Histograms')
        self.report.new_line()

        plot = self.create_histograms()

        img = self.report.new_inline_image(text="Histograms",
                                               path=self.relative_file_paths['Histograms'])
        
        self.report.new_paragraph(f"{img}")


        # QQs
        self.report.new_header(level=1, title='QQ-Plots')
        self.report.new_line()

        plot = self.create_qq_plots()

        img = self.report.new_inline_image(text="QQ",
                                               path=self.relative_file_paths['QQ'])
        
        self.report.new_paragraph(f"{img}")


        # Boxplots
        self.report.new_header(level=1, title='Boxplots')
        self.report.new_line()

        plot = self.create_boxplots()

        img = self.report.new_inline_image(text="boxplot",
                                               path=self.relative_file_paths['Boxplots'])
        
        self.report.new_paragraph(f"{img}")


        if include_transform_plots:
            
            # Transform comparison histograms
            self.report.new_header(level=1, title='Log Comparison Histograms')
            self.report.new_line()

            plot = self.create_transform_comparison_histograms()

            img = self.report.new_inline_image(text="transform histograms",
                                                path=self.relative_file_paths['Histogram_Transform'])
            
            self.report.new_paragraph(f"{img}")
        
            # Transform comparison QQ-plots

            self.report.new_header(level=1, title='Log Comparison QQ-Plots')
            self.report.new_line()

            plot = self.create_transform_comparison_qqplots()

            img = self.report.new_inline_image(text="transform QQs",
                                                path=self.relative_file_paths['QQ_Transform'])
            
            self.report.new_paragraph(f"{img}")

        self.report.create_md_file()


    def _plot_setup(self, plots_per_row, as_array):
        
        # List of (data, group_name) tuples. 
        # data = numpy array, group_name = string
        if as_array:
            plot_data = [(self.data.loc[self.data[self.grouping_variable] == group_name, self.numeric_variable].to_numpy(), group_name) for group_name in self.group_names]
        else:
            plot_data = [(self.data.loc[self.data[self.grouping_variable] == group_name, [self.numeric_variable]], group_name) for group_name in self.group_names]
        
        # Number of rows we need for the Q-Q plots
        num_rows = math.ceil(self.num_groups / plots_per_row)
        
        if self.num_groups < plots_per_row:
            num_rows = 1
            num_cols = self.num_groups
        else:
            num_cols = plots_per_row
            
        # If we won't have complete grid, update so we turn off the axes we don't use
        if self.num_groups < num_rows * num_cols:
            axes_off = [(None, None)] * ((num_rows * num_cols) - self.num_groups)
            plot_data.extend(axes_off)
        
        plot_info = {"plot_data":plot_data, "num_rows":num_rows, "num_cols":num_cols}
        
        return plot_info
        
    def create_qq_plots(self, plots_per_row=3, loc=0, scale=1, reference_line_type="q", fit=False, figsize_scale_factor=5, save_name="QQPlots",
                        save_format="png", seaborn_style="darkgrid", axis_fontsize=12, title_fontsize=16):
        
        plot_info = self._plot_setup(plots_per_row=plots_per_row, as_array=True)
        
        sns.set_style(seaborn_style)
            
        fig, axs = plt.subplots(nrows=plot_info["num_rows"],
                                ncols=plot_info["num_cols"],
                                figsize=(figsize_scale_factor*plot_info["num_cols"], figsize_scale_factor*plot_info["num_rows"]),
                                squeeze=False)
        
        # For each qq-plot (one per group).
        for index, plot_data in enumerate(plot_info['plot_data']): 
            
            # Unpack the data and the groups name
            data, category = plot_data
            
            # Get the axis location for this plot
            row = index // plot_info["num_cols"]
            col = index % plot_info["num_cols"]
            
            # If this isn't an empty plot
            if data is not None: 

                figure = qqplot(data=data, ax=axs[row][col], line=reference_line_type, fit=fit, loc=loc, scale=scale)
                
                axs[row][col].set_title(f"{self.grouping_variable}={category} Q-Q Plot", fontsize=title_fontsize, weight="bold")
                axs[row][col].set_xlabel("Theoretical Quantiles", fontsize=axis_fontsize, weight="bold")
                axs[row][col].set_ylabel("Sample Quantiles", fontsize=axis_fontsize, weight="bold")

            else:
                axs[row][col].axis('off')

        plt.tight_layout()

        save_path = f"./{save_name}.{save_format}"

        self.relative_file_paths["QQ"] = save_path

        plt.savefig(fname=save_path, format=save_format)
        
    def create_histograms(self, plots_per_row=3, figsize_scale_factor=5, save_name="Histograms",
                          save_format="png", seaborn_style="darkgrid", add_kde_line=True, stat="percent",
                          axis_fontsize=12, title_fontsize=16):
        
        
        plot_info = self._plot_setup(plots_per_row=plots_per_row, as_array=False)
        
        sns.set_style(seaborn_style)
            
        fig, axs = plt.subplots(nrows=plot_info["num_rows"],
                                ncols=plot_info["num_cols"],
                                figsize=(figsize_scale_factor*plot_info["num_cols"], figsize_scale_factor*plot_info["num_rows"]),
                                squeeze=False)
        
        # For each histogram (one per group).
        for index, plot_data in enumerate(plot_info['plot_data']): 
            
            # Unpack the data and the groups name
            data, category = plot_data
            
            # Get the axis location for this plot
            row = index // plot_info["num_cols"]
            col = index % plot_info["num_cols"]
            
            # If this isn't an empty plot
            if data is not None: 

                sns.histplot(data=data, x=self.numeric_variable, ax=axs[row][col], stat=stat, kde=add_kde_line)
                
                axs[row][col].set_title(f"Distribution of {self.numeric_variable}\nfor {self.grouping_variable}={category}", fontsize=title_fontsize, weight="bold")
                
                axs[row][col].set_xlabel(f"{self.numeric_variable} for group {category}", fontsize=axis_fontsize, weight="bold")
                
                axs[row][col].set_ylabel(f"{stat} of {self.numeric_variable}", fontsize=axis_fontsize, weight="bold")

            else:
                axs[row][col].axis('off')

        plt.tight_layout()

        save_path = f"./{save_name}.{save_format}"

        self.relative_file_paths["Histograms"] = save_path

        plt.savefig(fname=save_path, format=save_format)
    
    def _comparison_plot_setup(self, transform_type, include_all_groups_plot, as_array):
        
        if transform_type == "log":
            transform_variable = f"Log_{self.numeric_variable}"
            self.data[transform_variable] = np.log(self.data[self.numeric_variable])
        
        
        if as_array:
                        # List of (data, data_name) tuples. data = numpy array, data_name=name of group 
            plot_data = [(self.data.loc[self.data[self.grouping_variable] == group_name, self.numeric_variable].to_numpy(),
                          self.data.loc[self.data[self.grouping_variable] == group_name, transform_variable].to_numpy(),
                          group_name) for group_name in self.group_names]
            
            if include_all_groups_plot:
                 plot_data.extend([(self.data[self.numeric_variable].to_numpy(),
                                    self.data[transform_variable].to_numpy(),
                                    "combined groups")])
        else:
            # List of (data, data_name) tuples. data = numpy array, data_name=name of group 
            plot_data = [(self.data.loc[self.data[self.grouping_variable] == group_name, [self.numeric_variable]],
                          self.data.loc[self.data[self.grouping_variable] == group_name, [transform_variable]],
                          group_name) for group_name in self.group_names]
            
            
            if include_all_groups_plot:
                 plot_data.extend([(self.data[[self.numeric_variable]], self.data[[transform_variable]], "all groups combined")])
            

        
        num_rows = len(plot_data)
        num_cols = 2
        
        plot_info = {"plot_data":plot_data, "transformed_variable":transform_variable, "num_rows":num_rows, "num_cols":num_cols}
                
        return plot_info
    
    def create_transform_comparison_histograms(self, transform_type="log", include_all_groups_plot=True, overlay_normal_curves=True,
                                               overlay_curve_color="#FE019A", round_digits=3, title_fontsize=16, axis_fontsize=12, kde=False,
                                               stat='density', save_name="hist_transform_comparison", save_format="png", fig_scale_width=7,
                                               fig_scale_height=5, seaborn_style="darkgrid"):
        
        plot_info = self._comparison_plot_setup(transform_type=transform_type, include_all_groups_plot=include_all_groups_plot, as_array=False)
        
        sns.set_style(seaborn_style)
        
        fig, axs = plt.subplots(nrows=plot_info["num_rows"],
                                ncols=plot_info["num_cols"],
                                figsize=(fig_scale_width*plot_info["num_cols"], fig_scale_height*plot_info["num_rows"]),
                                squeeze=False)
        
        for index, plot_data in enumerate(plot_info["plot_data"]):
            
            row = index
            
            data_original, data_transformed, category = plot_data
            
            pair = [(data_original, "original", self.numeric_variable),
                    (data_transformed, f"{transform_type} transformed", plot_info["transformed_variable"])]
            
            for col, data_info in enumerate(pair):
                
                data, context, col_name = data_info
                
                sns.histplot(data=data, x=col_name, ax=axs[row][col], stat=stat, kde=kde)
                
                if overlay_normal_curves:
                    
                    mu, sd = stats.norm.fit(data[col_name])
                    
                    x_min, x_max = axs[row][col].get_xlim()
                    x_norm = np.linspace(x_min, x_max, 2_000)
                
                    y_norm = stats.norm.pdf(x_norm, loc=mu, scale=sd)
                    
                    skew = stats.skew(data[col_name].to_numpy())
                    kurt = stats.kurtosis(data[col_name].to_numpy())
                    
                    axs[row][col].plot(x_norm, y_norm, color=overlay_curve_color)
                    
                    title = (f"Distribution of {context} {self.numeric_variable} data\nfor group={category}\n"
                             f"Nomal Fit, Mu:{round(mu, round_digits)}, SD:{round(sd, round_digits)}\n"
                             f"Skew:{round(skew, round_digits)}, Kurt:{round(kurt, round_digits)}")
                
                else:
                    
                    skew = stats.skew(data[col_name].to_numpy())
                    kurt = stats.kurtosis(data[col_name].to_numpy())
                    
                    title = (f"Distribution of {context} {self.numeric_variable} data\nfor group={category}\n"
                             f"Skew:{round(skew, round_digits)}, Kurt:{round(kurt, round_digits)}")
                    
                    
                axs[row][col].set_title(title,
                                        fontsize=title_fontsize,
                                        weight="bold")
            
                axs[row][col].set_xlabel(f"{context} {self.numeric_variable} for group={category}", fontsize=axis_fontsize, weight="bold")

                axs[row][col].set_ylabel(f"{stat} of {self.numeric_variable}", fontsize=axis_fontsize, weight="bold")

        plt.tight_layout()

        save_path = f"./{save_name}.{save_format}"

        self.relative_file_paths["Histogram_Transform"] = save_path
                
        plt.savefig(fname=save_path, format=save_format)
                
    def create_transform_comparison_qqplots(self, transform_type="log", loc=0, scale=1, reference_line_type="q", fit=False, figsize_scale_factor=5, 
                                            include_all_groups_plot=True, title_fontsize=16, axis_fontsize=12, kde=False,
                                            save_name="qqplot_transform_comparison", save_format="png", seaborn_style="darkgrid"):

        sns.set_style(seaborn_style)
        
        plot_info = self._comparison_plot_setup(transform_type=transform_type, include_all_groups_plot=include_all_groups_plot, as_array=True)
        
        fig, axs = plt.subplots(nrows=plot_info["num_rows"],
                                ncols=plot_info["num_cols"],
                                figsize=(figsize_scale_factor*plot_info["num_cols"], figsize_scale_factor*plot_info["num_rows"]),
                                squeeze=False)
        
        
        for index, plot_data in enumerate(plot_info["plot_data"]):
            
            row = index
            
            data_original, data_transformed, category = plot_data
            
            pair = [(data_original, "original", self.numeric_variable),
                    (data_transformed, f"{transform_type} transformed", plot_info["transformed_variable"])]
            
            for col, data_info in enumerate(pair):
                
                data, context, col_name = data_info
                
                figure = qqplot(data=data, ax=axs[row][col], line=reference_line_type, fit=fit, loc=loc, scale=scale)
                
                axs[row][col].set_title(f"{context} {self.grouping_variable}={category}\nQ-Q Plot", fontsize=title_fontsize, weight="bold")
                axs[row][col].set_xlabel("Theoretical Quantiles", fontsize=axis_fontsize, weight="bold")
                axs[row][col].set_ylabel("Sample Quantiles", fontsize=axis_fontsize, weight="bold")
                
        plt.tight_layout()

        save_path = f"./{save_name}.{save_format}"

        self.relative_file_paths["QQ_Transform"] = save_path
                
        plt.savefig(fname=save_path, format=save_format)
        
    
    def create_boxplots(self, seaborn_style="darkgrid", figsize=(14,10), log_transform=False, add_annots=True,
                        round_digits=3, annot_txt_yoffset=0.14, title_fontsize=24, axis_fontsize=18,
                        tick_labelsize="xx-large", tick_label_rotation=0, save_name="boxplots", save_format="png",
                        annot_fontsize=14):
        
        sns.set_style(seaborn_style)
        
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        
        if log_transform: 
            y_variable = f"Log_{self.numeric_variable}"
            self.data[y_variable] = np.log(self.data[self.numeric_variable])
            save_name = f"log_transformed_{save_name}"
        else:
            y_variable = self.numeric_variable
        
        sns.boxplot(x=self.grouping_variable, y=y_variable, data=self.data, ax=axs)
        
        if add_annots:
            
            group_medians = self.data.groupby(by=self.grouping_variable)[y_variable].median()
            group_means = self.data.groupby(by=self.grouping_variable)[y_variable].mean()

            group_medians.index = group_medians.index.map(str)
            group_means.index = group_means.index.map(str)
            
            for xticklabel in axs.get_xticklabels():
                
                label = xticklabel.get_text()
                x, y = xticklabel.get_position()
                
                text = (f"median = {round(group_medians[label],round_digits)}\n"
                        f"mean = {round(group_means[label], round_digits)}")
                
                axs.text(x, group_medians[label]+annot_txt_yoffset, text, ha="center", color="w", weight="semibold", fontsize=annot_fontsize)
        
        
        axs.set_title(f"Distribution of {y_variable} for various {self.grouping_variable}", fontsize=title_fontsize, weight="bold")
        axs.set_ylabel(f"{y_variable}", fontsize=axis_fontsize, weight="bold")
        axs.set_xlabel(f"{self.grouping_variable}", fontsize=axis_fontsize, weight="bold", labelpad=15)
        axs.tick_params(labelsize=tick_labelsize, labelrotation=tick_label_rotation)
        
        plt.tight_layout()

        save_path = f"./{save_name}.{save_format}"

        self.relative_file_paths["Boxplots"] = save_path
        
        plt.savefig(fname=save_path, transparent=False, format=save_format)