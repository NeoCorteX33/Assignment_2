import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_algorithm_performance_summary(summary_df, primary_metric, figsize=(12, 8), title=None, palette='viridis'):
    """
    Plots a bar chart of the median performance for a primary metric from the summary_df,
    including 95% confidence intervals as error bars.

    Args:
        summary_df (pd.DataFrame): DataFrame returned by NestedCVRunner.compare_algorithms().
                                   It should have estimators as index and metrics as columns,
                                   including columns for '[metric_name]_CI_low' and '[metric_name]_CI_high'.
        primary_metric (str): The name of the metric to plot (e.g., 'AUC', 'F1').
        figsize (tuple, optional): Size of the figure. Defaults to (12, 8).
        title (str, optional): Title for the plot. If None, a default title is generated.
                               Defaults to None.
        palette (str or list, optional): Seaborn color palette for the bars.
                                         Defaults to 'viridis'.
    """
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        print("Error: summary_df is empty or not a DataFrame. Cannot generate plot.")
        return

    if primary_metric not in summary_df.columns:
        print(f"Error: Primary metric '{primary_metric}' not found in summary_df columns.")
        print(f"Available metrics: {summary_df.columns.tolist()}")
        return

    ci_low_col = f"{primary_metric}_CI_low"
    ci_high_col = f"{primary_metric}_CI_high"

    if ci_low_col not in summary_df.columns or ci_high_col not in summary_df.columns:
        print(f"Error: Confidence interval columns ('{ci_low_col}', '{ci_high_col}') not found for metric '{primary_metric}'.")
        print("Make sure compare_algorithms was run and generated these columns.")
        return

    # Data for plotting (summary_df is already sorted by compare_algorithms)
    estimators = summary_df.index
    median_values = summary_df[primary_metric]
    ci_low = summary_df[ci_low_col]
    ci_high = summary_df[ci_high_col]

    # Calculate error bar lengths (asymmetric)
    # Error below median: median - ci_low
    # Error above median: ci_high - median
    error_below = median_values - ci_low
    error_above = ci_high - median_values
    asymmetric_error = [error_below.values, error_above.values]

    # Create the plot
    plt.figure(figsize=figsize)
    bars = sns.barplot(x=estimators, y=median_values, palette=palette, capsize=0.1) # capsize for error bar caps

    # Add error bars
    # bars.patches gives access to the bars created by seaborn
    # We need to iterate through them to add custom error bars if sns.barplot doesn't handle asymmetric well directly
    # For simplicity, let's use plt.errorbar directly on top or ensure sns.barplot's yerr is used correctly.
    # A more direct way with matplotlib:
    x_coords = np.arange(len(estimators))
    plt.bar(x_coords, median_values, yerr=asymmetric_error, capsize=5, color=sns.color_palette(palette, len(estimators)), alpha=0.8)


    # Add text annotations for median values on top of the bars
    for i, val in enumerate(median_values):
        if pd.notna(val):
            plt.text(x_coords[i], val + 0.01 * plt.ylim()[1], f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='black')
        else: # Handle NaN median values if any
            plt.text(x_coords[i], 0, 'N/A', ha='center', va='bottom', fontsize=9, color='black')


    # Set plot title and labels
    if title is None:
        plot_title = f'Median Performance of Algorithms for {primary_metric}\n(with 95% Confidence Intervals)'
    else:
        plot_title = title

    plt.title(plot_title, fontsize=15, pad=20)
    plt.xlabel('Estimator', fontsize=12)
    plt.ylabel(f'Median {primary_metric}', fontsize=12)
    plt.xticks(ticks=x_coords, labels=estimators, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(bottom=max(0, ci_low.min() - 0.05 if pd.notna(ci_low.min()) else 0),
             top=min(1, ci_high.max() + 0.05 if pd.notna(ci_high.max()) else 1) if primary_metric in ['AUC', 'PRAUC', 'F1', 'Balanced_Accuracy'] else None)


    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()


def plot_single_model_metrics_summary(
    summary_df,
    estimator_name,
    metric_names, # Pass the list of base metric names, e.g., from runner.metric_names
    figsize=(10, 7),
    title=None,
    palette='coolwarm_r' # Using a diverging palette can be nice here
):
    """
    Plots a bar chart of all specified metrics for a single estimator,
    including 95% confidence intervals as error bars.

    Args:
        summary_df (pd.DataFrame): DataFrame returned by NestedCVRunner.compare_algorithms().
                                   It should have estimators as index and metrics as columns,
                                   including columns for '[metric_name]_CI_low' and '[metric_name]_CI_high'.
        estimator_name (str): The name of the specific estimator to plot metrics for.
        metric_names (list): A list of the base metric names to plot (e.g., ['AUC', 'F1', 'MCC']).
        figsize (tuple, optional): Size of the figure. Defaults to (10, 7).
        title (str, optional): Title for the plot. If None, a default title is generated.
                               Defaults to None.
        palette (str or list, optional): Seaborn color palette for the bars.
                                         Defaults to 'coolwarm_r'.
    """
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        print("Error: summary_df is empty or not a DataFrame. Cannot generate plot.")
        return
    if estimator_name not in summary_df.index:
        print(f"Error: Estimator '{estimator_name}' not found in summary_df index.")
        print(f"Available estimators: {summary_df.index.tolist()}")
        return

    # Select the data for the chosen estimator
    model_performance = summary_df.loc[estimator_name]

    # Prepare data for plotting
    plot_metrics = []
    median_values = []
    ci_low_values = []
    ci_high_values = []

    for metric in metric_names:
        if metric in model_performance and f"{metric}_CI_low" in model_performance and f"{metric}_CI_high" in model_performance:
            median = model_performance[metric]
            ci_low = model_performance[f"{metric}_CI_low"]
            ci_high = model_performance[f"{metric}_CI_high"]

            if pd.notna(median) and pd.notna(ci_low) and pd.notna(ci_high):
                plot_metrics.append(metric)
                median_values.append(median)
                ci_low_values.append(ci_low)
                ci_high_values.append(ci_high)
            else:
                print(f"Warning: Missing median or CI data for metric '{metric}' for estimator '{estimator_name}'. Skipping this metric.")
        else:
            print(f"Warning: Metric '{metric}' or its CI columns not found for estimator '{estimator_name}'. Skipping this metric.")

    if not plot_metrics:
        print(f"No valid metric data to plot for estimator '{estimator_name}'.")
        return

    # Calculate error bar lengths (asymmetric)
    error_below = np.array(median_values) - np.array(ci_low_values)
    error_above = np.array(ci_high_values) - np.array(median_values)
    asymmetric_error = [error_below, error_above]

    # Create the plot
    plt.figure(figsize=figsize)
    x_coords = np.arange(len(plot_metrics))

    # Create bars with error bars
    colors = sns.color_palette(palette, len(plot_metrics))
    bars = plt.bar(x_coords, median_values, yerr=asymmetric_error, capsize=5, color=colors, alpha=0.8)

    # Add text annotations for median values on top of the bars
    for i, val in enumerate(median_values):
        plt.text(x_coords[i], val + 0.01 * plt.ylim()[1], f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='black')

    # Set plot title and labels
    if title is None:
        plot_title = f'Performance Metrics for {estimator_name}\n(Median with 95% Confidence Intervals)'
    else:
        plot_title = title

    plt.title(plot_title, fontsize=15, pad=20)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Median Score', fontsize=12)
    plt.xticks(ticks=x_coords, labels=plot_metrics, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust y-axis limits, common for metrics between 0 and 1
    # Consider the full range of CIs for y-limits
    min_y_val = np.nanmin(ci_low_values) if ci_low_values else 0
    max_y_val = np.nanmax(ci_high_values) if ci_high_values else 1
    plt.ylim(bottom=max(0, min_y_val - 0.05), top=min(1.05, max_y_val + 0.05))


    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()