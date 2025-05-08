from pathlib import Path
from scipy import stats
import math
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


def data_basics(data_path:Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform basic exploratory data analysis on the breast cancer dataset and display results in a structured format.

    Args:
        data_path (str or Path): Path to the dataset to be analysed
        
    Returns:
        tuple of (pd.DataFrame, pd.DataFrame): Preprocessed dataset and it's descriptive stats 
    """
    print("="*80)
    print(f"DATASET ANALYSIS OF: {Path(data_path).name}")
    print("="*80)
    
    # Load data and drop id column
    raw_data = pd.read_csv(data_path)
    raw_data.drop(columns=['id'], inplace=True)
    
    # Get basic dataset info
    data_shape = raw_data.shape
    missing_vals = raw_data.isnull().sum()
    duplicated_samples = raw_data.duplicated().sum()
    
    # Display information with clear formatting
    print(f"\n📊 DATASET DIMENSIONS: {data_shape[0]} rows x {data_shape[1]} columns\n")
    
    print("📋 COLUMN DATA TYPES:")
    
    # Check if only the diagnosis column is categorical and the rest are numerical
    dtypes = raw_data.dtypes
    categorical_cols = dtypes[dtypes == 'object'].index.tolist()
    numerical_cols = dtypes[dtypes != 'object'].index.tolist()
    
    if len(categorical_cols) == 1 and len(numerical_cols) > 0:
        common_num_dtype = dtypes[numerical_cols].unique()
        if len(common_num_dtype) == 1:
            print(f"\n🏷️ {categorical_cols[0]} is the only categorical column (type: {dtypes[categorical_cols[0]]})")
            print(f"📊 All the other {len(numerical_cols)} numerical columns are of type: {common_num_dtype[0]}")
    
    print(f"\n🔍 DUPLICATE SAMPLES: {duplicated_samples}")

    # Check the difference in class sizes
    if 'diagnosis' in raw_data.columns:
        print("\n⚖️ DIAGNOSIS CLASS DISTRIBUTION:")
        class_dist = raw_data['diagnosis'].value_counts(normalize=True).mul(100).round(2)
        class_counts = raw_data['diagnosis'].value_counts()
        for cls, pct in class_dist.items():
            print(f"  - {cls}: {class_counts[cls]} samples ({pct}%)")
    
    # Check for missing values
    print("\n❓ MISSING VALUES:")
    missing_summary = missing_vals[missing_vals > 0]
    if len(missing_summary) > 0:
        print(missing_summary)
    else:
        print("No missing values found.")
    
    descriptive_stats = raw_data.describe()
    
    return raw_data, descriptive_stats


def imputed_check(df: pd.DataFrame) -> None:
    """
    Imputing data using mean feature value and comparing distributions
    before and after imputation using the Kolmogorov-Smirnov test
    
    Args:
        original_df (pd.DataFrame): Original dataframe with missing values
        imputed_df (pd.DataFrame): Dataframe after imputation
        exclude_cols (list): Columns to exclude from comparison
    """
    # Create a copy to avoid modifying the original dataset
    df_imputed = df.copy()

    # Get columns with missing values
    miss_val_cols = df_imputed.columns[df_imputed.isnull().any()].tolist()
    if len(miss_val_cols) == 0:
        print("No missing values to impute.")
        return
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed[miss_val_cols] = imputer.fit_transform(df_imputed[miss_val_cols])
    print("Imputing missing values...")
    
    print("Checking for distribution differences after imputation with KS statistic:")
    print("-" * 50)
    
    p_values = []
    for idx, col in enumerate(miss_val_cols):
        # Get original and imputed values (dropping NA from original)
        orig_values = df_imputed[col].dropna()
        imp_values = df_imputed[col]
        
        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(orig_values, imp_values)
        p_values.append(p_value)
        
    for p in p_values:
        if p < 0.05:
            print(f"Found significant difference in distribution (p={p:.4f})")
        else:
            print(f"No significant difference in distribution (p={p:.4f})")

    print("Imputation complete.")
    print("-" * 50)


def group_features(df:pd.DataFrame, base_features:list, target_column:str ='diagnosis') -> None:
    """
    Generates grouped box plots for base features, showing mean, se, and worst values,
    separated by the target variable.

    Args:
        df (pd.DataFrame): The input dataframe containing the features and target.
        target_column (str): The name of the target column (e.g., 'diagnosis').
        base_features (list): A list of the base feature names (e.g., ['radius', 'texture', ...]).
    """
    n_features = len(base_features)
    n_cols = 2
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axes = axes.flatten()

    # Define the suffixes for the feature types
    suffixes = ['_mean', '_se', '_worst']

    for i, base_feature in enumerate(base_features):
        feature_cols = [f"{base_feature}{suffix}" for suffix in suffixes]
        # Check if all the required feature columns exist
        if not all(col in df.columns for col in feature_cols):
            print(f"Warning: Skipping '{base_feature}' as not all columns ({', '.join(feature_cols)}) were found.")
            if i < len(axes):
                 axes[i].axis('off')
            continue

        # Select the data for the current base feature and the target
        plot_data = df[feature_cols + [target_column]]

        # Transform the data from wide to long format suitable for grouping in boxplot
        melted_data = plot_data.melt(id_vars=[target_column],
                                     value_vars=feature_cols,
                                     var_name='Feature Type',
                                     value_name='Value')

        # Clean up feature type names for better readability
        melted_data['Feature Type'] = melted_data['Feature Type'].str.replace(base_feature + '_', '', regex=False)

        # Create the grouped box plot
        sns.boxplot(x='Feature Type', y='Value', hue=target_column, data=melted_data, ax=axes[i], order=['mean', 'se', 'worst'])
        axes[i].set_title(f'Distribution of {base_feature.capitalize()} (Mean, SE, Worst)')
        axes[i].set_xlabel('Measurement Type')
        axes[i].set_ylabel('Value')
        axes[i].tick_params(axis='x', rotation=0)

    # Turn off any remaining unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def _encode_target(df:pd.DataFrame, target_column:str ='diagnosis', positive_class_label:str= 'M') -> tuple[pd.Series,str]:
    """
    Encodes a binary categorical target column to 0/1.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
        positive_class_label (str): The label considered 'positive' (maps to 1).

    Returns:
        tuple: (pd.Series, str) - The encoded target Series and its name.

    Raises:
        ValueError: If target column not found, not binary, or positive label mismatch.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Handle categorical encoding
    unique_values = df[target_column].unique()
    if len(unique_values) != 2:
        raise ValueError(f"Target column '{target_column}' must be binary for encoding, found {len(unique_values)} unique values.")

    # Ensure positive_class_label is provided for categorical
    if positive_class_label is None:
         raise ValueError("Argument 'positive_class_label' must be provided for categorical target encoding.")

    if positive_class_label not in unique_values:
         raise ValueError(f"Positive class label '{positive_class_label}' not found in target column '{target_column}'. Available values: {unique_values}")

    negative_class_label = [val for val in unique_values if val != positive_class_label][0]
    encoded_target = df[target_column].apply(lambda x: 1 if x == positive_class_label else 0)
    encoded_target_name = f"{target_column}_encoded"
    print(f"Encoding target '{target_column}': {positive_class_label}=1, {negative_class_label}=0")
    
    return encoded_target, encoded_target_name


def correlation_analysis(df:pd.DataFrame, target_column:str='diagnosis', num_important:int=15) -> None:
    """
    Generate visualizations for both target correlations and feature-feature correlations.
    
    Args:
        df (pd.DataFrame): Input dataframe with features and target
        target_column (str): Name of the target column
        num_important (int): Number of top pairs/features to display
    """
    plot_df = df.copy()

    # Encode the categorical target column to numerical
    encoded_target, _ = _encode_target(plot_df, target_column, 'M')
    plot_df[target_column] = encoded_target
    
    # Calculate the correlation matrix
    corr_matrix = plot_df.corr()

    # Check that the correlation matrix is correct     
    if corr_matrix is None or corr_matrix.empty:
        print("Error: Correlation matrix is missing or empty. Cannot generate plots.")
        return

    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        print("Error: Correlation matrix is not square.")
        return
    
    # Check if target is in the correlation matrix
    if target_column not in corr_matrix.columns:
        print(f"Error: Target column '{target_column}' not found in correlation matrix.")
        print("Make sure you're using a correlation matrix that includes the encoded target.")
        return
    
    # Get correlations with target only
    target_corrs = corr_matrix[target_column].drop(target_column)
    target_corrs_abs = target_corrs.abs().sort_values(ascending=False)
    
    # Remove self-correlations
    abs_corr = corr_matrix.abs().unstack()
    sort_corr = abs_corr.sort_values(ascending=False)
    sort_corr = sort_corr[sort_corr.index.get_level_values(0) != sort_corr.index.get_level_values(1)]
    
    # Take every other pair to avoid duplicates
    unique_pairs = sort_corr[::2]
    
    # Make sure we don't request more pairs than available
    num_to_show = min(num_important, len(unique_pairs))
    top_pairs = unique_pairs.head(num_to_show)
    
    top_feature_pairs = np.unique(top_pairs.index.get_level_values(0).tolist() + 
                            top_pairs.index.get_level_values(1).tolist())
    
    # Create the heatmap plot with all features
    plt.figure(figsize=(18, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 10})
    plt.title("Feature correlations heatmap including diagnosis", fontsize=16)
    plt.xticks(fontsize=13, rotation=90)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.show()

    # Plot the correlations with the target
    plt.figure(figsize=(12, 8))
    features_to_show = min(num_important, len(target_corrs_abs))
    best_target = target_corrs[target_corrs_abs.head(features_to_show).index]
    colors = ['g' if c >= 0 else 'r' for c in best_target]
    
    plt.barh(
        best_target.index,
        best_target,
        color=colors
    )
    
    plt.title(f'Top {features_to_show} Features Correlated with {target_column}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # Plot the pairplot of the most correlated feature pairs
    # Use plot_df which has the encoded target for proper visualization
    pairplot_df = plot_df[list(top_feature_pairs) + [target_column]].copy()
    
    g = sns.pairplot(pairplot_df, hue=target_column, corner=True, height=3)
    sns.move_legend(g, "upper right", bbox_to_anchor=(1.2, 1), 
                    title=target_column, fontsize=12, title_fontsize=14)
    for ax in g.axes.flat:
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            ax.tick_params(labelsize=10)
    plt.suptitle(f"Pair Plots of the {len(top_feature_pairs)} Most Correlated Feature Pairs", y=1.02, fontsize=24)
    plt.show()


def pca_analysis(df: pd.DataFrame, n_components:int, target_column:str= 'diagnosis') -> None:
    """
    Implements PCA on the dataset and visualizes the results with class separation.
    
    Args:
        df_imputed (pd.DataFrame): Input dataframe with cancer data
        target_column (str): Name of the target column
        n_components (int): Number of PCA components to retain
    """
    df_copy = df.copy()

    if target_column in df_copy.columns:
        y = df_copy[target_column]
        X = df_copy.drop(columns=[target_column])
    else:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    feature_names = X.columns

    # Impute features
    X_imputed = SimpleImputer(strategy='mean').fit_transform(X)
    print("Imputing missing values...")

    # Scale features
    print("Standardizing features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Apply PCA
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate total explained variance
    explained_variance = pca.explained_variance_ratio_
    total_variance = sum(explained_variance) * 100
    print(f"Total explained variance with {n_components} components: {total_variance:.2f}%")
    explained_variance_1d = explained_variance[0] * 100
    explained_variance_2d = explained_variance[1] * 100
    explained_variance_3d = explained_variance[2] * 100
    total_variance_2d = explained_variance_1d + explained_variance_2d
    total_variance_3d = total_variance_2d + explained_variance_3d
    
    # Create 2D scatter plot
    fig = plt.figure(figsize=(12, 10))
    categories = y.unique()
    colors = ['#1f77b4', '#ff7f0e']
    markers = ['o', 's']
    
    for i, category in enumerate(categories):
        mask = y == category
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                    s=60, alpha=0.8, label=f"{category}",
                    color=colors[i] if colors else None,
                    marker=markers[i] if markers else 'o',
                    edgecolor='w', linewidth=0.5)

    plt.title(f'PCA of Breast Cancer Dataset: {(total_variance_2d):.1f}% Variance Explained', fontsize=16)
    plt.xlabel(f'PC1 ({explained_variance_1d:.1f}%)', fontsize=14)
    plt.ylabel(f'PC2 ({explained_variance_2d:.1f}%)', fontsize=14)    
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#1f77b4', '#ff7f0e']
    markers = ['o', 's']
    
    for i, category in enumerate(categories):
        mask = y == category
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                  s=50, alpha=0.8, label=f"{category}",
                  color=colors[i] if colors else None,
                  marker=markers[i] if markers else 'o',
                  edgecolor='w', linewidth=0.5)
        
    ax.set_title(f'3D PCA of Breast Cancer Dataset: {total_variance_3d:.1f}% Variance Explained', fontsize=16)
    ax.set_xlabel(f'PC1 ({explained_variance_1d:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_variance_2d:.1f}%)', fontsize=12)
    ax.set_zlabel(f'PC3 ({explained_variance_3d:.1f}%)', fontsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Show the explained variance plot for the specified components
    if pca.n_components_ > 2:
        plt.figure(figsize=(12, 10))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='skyblue')
        plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red', linewidth=2)
        plt.title('Explained Variance by Principal Components', fontsize=14)
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Explained Variance Ratio', fontsize=12)
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Show feature contributions to PC1 and PC2
    if pca.n_components_ >= 2:
        components_df = pd.DataFrame(
            pca.components_[:2].T,
            columns=[f'PC{i+1}' for i in range(2)],
            index=feature_names
        )
        abs_components = abs(components_df)
        plt.figure(figsize=(12, 10))
        
        for i, pc in enumerate(['PC1', 'PC2']):
            plt.subplot(1, 2, i+1)
            top_features = abs_components[pc].sort_values(ascending=False).head(10).index
            coefs = components_df.loc[top_features, pc]
            colors = ['red' if c < 0 else 'green' for c in coefs]
            plt.barh(top_features, coefs, color=colors)
            plt.xlabel('Component Coefficient', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'Top 10 Features Contributing to {pc}', fontsize=14)
            plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()