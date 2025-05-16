# nested_cv.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from atom import ATOMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score, f1_score,
    recall_score, precision_score, average_precision_score, confusion_matrix, fbeta_score
)
from scipy.stats import bootstrap
from numpy import median as np_median

class rnCV:
    def __init__(self, dataset:pd.DataFrame, estimators:list, target_column:str='diagnosis', positive_class_label:str='M', R=10, N=5, K=3,  random_seed=33):
        """Performs Repeated Nested Cross-Validation (rnCV) using ATOM for the inner loop and scikit-learn for the outer loop.

        Attributes:
            dataset (pd.DataFrame): The full dataset including features and target.
            target_column (str): The name of the target variable column.
            positive_class_label (any): The label representing the positive class in the target column ('M').
            estimators (list): A list of model names supported by ATOM (e.g., ["LR", "RF", "LGB"]).
            R (int): Number of repeats (rounds) for the outer loop.
            N (int): Number of folds for the outer loop.
            K (int): Number of folds for the inner loop (used by ATOM's internal CV).
            random_seed (int): Seed for reproducibility.
            results (list): A list to store dictionaries of metrics from each outer fold evaluation.
            metric_names (list): List of metric names to calculate.
        """
        # Check input dataset, feature selection method and estimators for correct formats
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Dataset must be a pandas DataFrame.")
        if target_column not in dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        if positive_class_label not in dataset[target_column].unique():
            raise ValueError(f"Positive class label '{positive_class_label}' not found in target column.")
        if not isinstance(estimators, list) or len(estimators) == 0:
            raise ValueError("Estimators must be a non-empty list.")
        
        # Main parameters
        self.dataset = dataset.copy()
        self.target_column = target_column
        self.positive_class_label = positive_class_label
        self.estimators = estimators
        self.R = R
        self.N = N
        self.K = K
        self.random_seed = random_seed

        # Store the results from each fold of the outer loop
        self.results = []
        self.confusion_matrices = {}
        # Define metrics to calculate on the outer test sets
        self.metric_names = self.metric_names = [
            'MCC', 'AUC', 'PRAUC', 'Balanced_Accuracy', 'F1', 'F2',
            'Recall (Sensitivity)', 'Specificity', 'Precision (PPV)', 'NPV'
        ]
        # Encode the target variable
        print(f"Encoding target '{self.target_column}': {self.positive_class_label}=1, B=0")
        self.dataset[self.target_column] =  self.dataset[self.target_column].apply(lambda x: 1 if x == self.positive_class_label else 0)


    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculates a predifined set of classification metrics.
        
        Args:
            y_true (np.array): True labels (0 or 1)
            y_pred (np.array): Predicted labels (0 or 1)
            y_prob (np.array): Predicted probabilities for the positive class (class 1)
            
        Returns:
            dict: Dictionary containing the calculated metrics, or None for not calculated metrics.
            np.array: Confusion matrix values.
        """
        metrics = {}
        conf_matrix = None

        try:
            # Get confusion matrix values
            conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
            tp, fn, fp, tn = conf_matrix.ravel()

            # Matthews Correlation Coefficient
            metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

            # Area Under the ROC Curve (needs probabilities)
            try:
                metrics['AUC'] = roc_auc_score(y_true, y_prob)
            except ValueError: # Handle cases with only one class present/predicted
                metrics['AUC'] = None
            # Area Under the Precision-Recall Curve (also needs probabilities)
            try:
                metrics['PRAUC'] = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics['PRAUC'] = None
            # Balanced Accuracy
            metrics['Balanced_Accuracy'] = balanced_accuracy_score(y_true, y_pred)
            # F1 Score
            metrics['F1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            # F2 Score (weighted towards recall)
            metrics['F2'] = fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
            # Recall (Sensitivity or True Positive Rate)
            metrics['Recall (Sensitivity)'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            # Specificity (True Negative Rate)
            metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            # Precision (Positive Predictive Value)
            metrics['Precision (PPV)'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            # Negative Predictive Value
            metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            for m_name in self.metric_names:
                if m_name not in metrics:
                    metrics[m_name] = None
        
        return metrics, conf_matrix
    
    
    def run_nested_cv(self, n_trials=50, inner_metric='roc_auc', custom_hp_params=None):
        """
    Executes the Repeated Nested Cross-Validation pipeline.

    Args:
        n_trials (int): Number of trials for Optuna hyperparameter search within ATOM. Defaults to 50.
        inner_metric (str): The metric ATOM should optimize during hyperparameter tuning (inner loop).
                            Common choices: 'f1', 'roc_auc', 'accuracy', 'balanced_accuracy'. Defaults to 'roc_auc'.
    """
        print(f"Starting Repeated Nested Cross-Validation (R={self.R}, N={self.N}, K={self.K})")
        print(f"Inner loop optimization metric: {inner_metric}")
        print(f"Hyperparameter tuning trials per inner loop: {n_trials}")
        print("-" * 60)

        # Separate features and target variable
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        # --- Outer Loop Setup ---
        # Use Repeated Stratified K-Folds to handdle repeats and stratification
        outer_cv = RepeatedStratifiedKFold(
            n_splits=self.N,
            n_repeats=self.R,
            random_state=self.random_seed
        )
        
        # Keep track of the current repeat and fold number
        outer_fold_counter = 0
        total_outer_folds = self.R * self.N
        
        # --- Run Outer Loop ---
        for i, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X, y)):
            current_repeat = (i // self.N) + 1
            current_fold = (i % self.N) + 1
            outer_fold_counter += 1
            print(f'Processing: Repeat {current_repeat}/{self.R}, in Outer Fold {current_fold}/{self.N} from {total_outer_folds} total folds')

            # Get the outer training and test sets
            X_train_outer, y_train_outer = X.iloc[train_outer_idx], y.iloc[train_outer_idx]
            X_test_outer, y_test_outer = X.iloc[test_outer_idx], y.iloc[test_outer_idx]
            print(f'Outer trainig set shape: {X_train_outer.shape}, Outer test set shape: {X_test_outer.shape}')
            
            # --- Inner Loop Setup ---
            # Instantiate the ATOMClassifier with the specified estimators
            # Be very careful to provide ONLY the outer training data to prevent data leakage from the outer test set
            print(f'Initializing ATOMClassifier for HP tuning ...')
            try:
                atom = ATOMClassifier(
                    X_train_outer, y_train_outer,   # Outer training data only
                    device= 'cpu',
                    engine= 'sklearn',
                    verbose= 0,
                    random_state= self.random_seed + outer_fold_counter # Ensure different seed for each outer fold
                )

                # ATOM handles inner cross-validation and hyperparameter tuning
                print(f'Running ATOM for models {self.estimators} ...')
                print(f'Running inner loop with {self.K} folds for hyperparameter tuning ...')
                
                # Feature selection can be done here if needed
                atom.impute(strat_num='mean', verbose=1)
                atom.scale(strategy='robust', verbose=1)

                # Run the inner loop with the specified number of trials
                atom.run(
                    models=self.estimators,
                    metric=inner_metric,
                    n_trials=n_trials,
                    ht_params= {'cv': self.K},
                    est_params= custom_hp_params
                )

                # --- Evaluate on the Outer Test Set ---
                print("Evaluating tuned models on (processed) outer test set...")

                # Get all trained models as a dictionary {name: model}
                trained_models = atom._get_models()

                for i, model in enumerate(trained_models):
                    model_name = atom.models[i]  # Get the string name for the model
                    print(f"Evaluating {model_name}...")
                    print(f"Evaluating model {i}: {atom.models[i]}...")
                    y_prob_outer, y_pred_outer = None, None
                
                    # Get predictions
                    y_pred_outer = model.predict(X_test_outer)

                    # Get probabilities and handle different return types
                    y_prob_raw = model.predict_proba(X_test_outer)

                    # Convert to numpy array if it's a DataFrame
                    if hasattr(y_pred_outer, 'values'):
                        print(f"Converting predictions DataFrame to numpy array, shape: {y_pred_outer.shape}")
                        y_pred_outer = y_pred_outer.values
                    
                    # Convert to numpy array if it's a DataFrame
                    if hasattr(y_prob_raw, 'values'):
                        print(f"Converting DataFrame to numpy array, shape: {y_prob_raw.shape}")
                        y_prob_raw = y_prob_raw.values
                    
                    # Extract positive class probabilities (typically column 1)
                    if len(y_prob_raw.shape) > 1 and y_prob_raw.shape[1] > 1:
                        # Multi-column case - get positive class probability (column 1)
                        y_prob_outer = y_prob_raw[:, 1]
                    else:
                        # Single column case - flatten
                        y_prob_outer = y_prob_raw.flatten()
                        
                    # Calculate metrics if predictions succeeded
                    if y_pred_outer is not None and y_prob_outer is not None:
                        fold_metrics, conf_matrix = self._calculate_metrics(y_test_outer, y_pred_outer, y_prob_outer)
                        # Store the confusion matrix with a unique key
                        matrix_key = f"{current_repeat}_{current_fold}_{model_name}"
                        self.confusion_matrices[matrix_key] = conf_matrix
                    else:
                        fold_metrics = {m_name: None for m_name in self.metric_names}

                    # Store all calculated results
                    result_record = {
                        'Repeat': current_repeat,
                        'Outer_Fold': current_fold,
                        'Estimator': model_name,
                        **fold_metrics
                    }
                    self.results.append(result_record)

            except Exception as e:
                print(f"[Error] An exception occurred during ATOM processing or evaluation for outer fold {current_fold} in repeat {current_repeat}: {e}")
            print("\n" + "=" * 60)
        
        print("Repeated Nested Cross-Validation Finished!")
        print(f"Total results collected: {len(self.results)}")
        print("=" * 60)

        
    def get_results_df(self):
        """
        Converts the stored results list into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the performance metrics
                        for each estimator across all repeats and outer folds.
                        Returns an empty DataFrame if no results were collected.
        """
        if not self.results:
            print("Warning: No results found. Run `run_nested_cv` first.")
            return pd.DataFrame()
        return pd.DataFrame(self.results)


    def _calculate_median_ci(self, data_points, confidence_level=0.95):
            """
            Calculates the median and bootstrap confidence interval for a list of data points.
            """
            if not data_points or len(data_points) < 2:
                return np.nan, (np.nan, np.nan)
            data_array = np.array(data_points)
            median_val = np_median(data_array)
            try:
                res = bootstrap((data_array,), np_median, confidence_level=confidence_level, method='percentile')
                return median_val, (res.confidence_interval.low, res.confidence_interval.high)
            except Exception:
                return median_val, (np.nan, np.nan)


    def plot_confusion_matrices(self, normalize=True, figsize=(16, 12), cmap='Blues'):
        """
        Plots the saved confusion matrices for all estimators.
        
        Args:
            normalize (bool): Whether to normalize the confusion matrices. Defaults to True.
            figsize (tuple): Figure size for the plot. Defaults to (16, 12).
            cmap (str): Colormap for the heatmap. Defaults to 'Blues'.
        """
        if not self.confusion_matrices:
            print("No confusion matrices found. Run run_nested_cv first.")
            return
        
        # Group confusion matrices by estimator
        estimator_matrices = {}
        for key, matrix in self.confusion_matrices.items():
            if matrix is None:
                continue
            
            # Extract estimator name from the key
            _, _, estimator = key.split('_', 2)
            
            if estimator not in estimator_matrices:
                estimator_matrices[estimator] = []
            estimator_matrices[estimator].append(matrix)
        
        # Calculate average confusion matrix for each estimator
        avg_matrices = {}
        for estimator, matrices in estimator_matrices.items():
            if matrices:
                avg_matrices[estimator] = np.mean(matrices, axis=0)
        
        # Plot the average confusion matrices
        n_estimators = len(avg_matrices)
        if n_estimators == 0:
            print("No valid confusion matrices to plot.")
            return
        
        # Calculate grid layout
        n_cols = min(3, n_estimators)
        n_rows = (n_estimators + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, (estimator, matrix) in enumerate(avg_matrices.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Normalize if requested
            if normalize:
                matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
            else:
                fmt = 'd'
            
            # Plot the confusion matrix
            sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, 
                        xticklabels=['Negative (0)', 'Positive (1)'],
                        yticklabels=['Positive (1)', 'Negative (0)'], ax=ax)
            
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title(f'Confusion Matrix - {estimator}')
        
        i = len(avg_matrices)
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.suptitle("Average Confusion Matrices Across All Folds", y=1.02, fontsize=16)
        plt.show()
        
        return avg_matrices


    def compare_algorithms(self, primary_metric='AUC', sort_ascending=False):
        """
        Analyzes the collected results, calculates median performance for each metric,
        and prints a summary comparison of the algorithms.

        Args:
            primary_metric (str): The main metric to sort the comparison table by.
                                  Must be one of the calculated metric names. Defaults to 'AUC'.
            sort_ascending (bool): Whether to sort the primary metric in ascending order.
                                   Defaults to False (higher is better).

        Returns:
            pd.DataFrame: DataFrame containing the median metrics for each algorithm.
        """
        results_df = self.get_results_df()
        if results_df.empty:
            print("Cannot compare algorithms: No results available.")
            return pd.DataFrame()

        if primary_metric not in results_df.columns:
            available_metrics = [col for col in results_df.columns if col not in ['Repeat', 'Outer_Fold', 'Estimator']]
            print(f"Warning: Primary metric '{primary_metric}' not found in results.")
             
            if not available_metrics:
                print("Error: No metric columns found in results.")
                return pd.DataFrame()
            
            # Fallback to the first available metric
            primary_metric = available_metrics[0] 
            print(f"Using fallback primary metric: '{primary_metric}'")

        print("\n--- Algorithm Performance Comparison (Median over all outer folds) ---")
        summary_data = []
        unique_estimators = results_df['Estimator'].unique()
        for estimator in unique_estimators:
            estimator_results = results_df[results_df['Estimator'] == estimator]
            row = {'Estimator': estimator}
            for metric in self.metric_names:
                if metric in estimator_results.columns:
                    data_points = estimator_results[metric].dropna().tolist()
                    median_val, (ci_low, ci_high) = self._calculate_median_ci(data_points)
                    row[metric] = median_val
                    row[f"{metric}_CI_low"] = ci_low
                    row[f"{metric}_CI_high"] = ci_high
                else:
                    row[metric] = np.nan
                    row[f"{metric}_CI_low"] = np.nan
                    row[f"{metric}_CI_high"] = np.nan
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        if summary_df.empty:
            print("No summary data could be generated for algorithm comparison.")
            return pd.DataFrame()

        summary_df = summary_df.set_index('Estimator')
        summary_df = summary_df.sort_values(by=primary_metric, ascending=sort_ascending)

        print("Median performance metrics with 95% Confidence Intervals:")
        for estimator in summary_df.index:
            print(f"\nEstimator: {estimator}")
            for metric in self.metric_names:
                median_val = summary_df.loc[estimator, metric]
                ci_low = summary_df.loc[estimator, f"{metric}_CI_low"]
                ci_high = summary_df.loc[estimator, f"{metric}_CI_high"]
                if pd.notna(median_val):
                    print(f"  {metric:<20}: {median_val:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")
                else:
                    print(f"  {metric:<20}: N/A")

        if not summary_df.empty and primary_metric in summary_df.columns:
            winner_algorithm = summary_df.index[0]
            winner_score = summary_df.iloc[0][primary_metric]
            winner_ci_low = summary_df.iloc[0][f"{primary_metric}_CI_low"]
            winner_ci_high = summary_df.iloc[0][f"{primary_metric}_CI_high"]

            print("\n--- Winner Declaration ---")
            print(f"Based on the highest median '{primary_metric}':")
            print(f"Winner Algorithm: {winner_algorithm}")
            
            if pd.notna(winner_score):
                 print(f"  Median {primary_metric}: {winner_score:.3f} (95% CI: [{winner_ci_low:.3f}, {winner_ci_high:.3f}])")
            else:
                 print(f"  Median {primary_metric}: N/A for winner.")
        else:
            print("No algorithms to compare or declare a winner (possibly due to missing primary metric data).")
        print("-" * 60)
        
        return summary_df
    

    def train_final_model(self, winner_model_name:str, inner_metric:str="roc_auc", n_trials:int=50, custom_hp_params= None, save_dir:str='../models', save_name:str='winner.pkl'):
        """Trains the final model using the entire dataset with the optimal hyperparameters found.
        Args:
            winner_model_name (str): The name of the single winning algorithm (e.g., "RF", "LGB").
            n_trials (int): Number of trials for Optuna hyperparameter search.
            custom_hyperparameters (dict): Custom hyperparameters for the winning model.
            save_dir (str): Directory to save the final model.
            save_name (str): Name of the saved model file.
        """

        # Check if only one model is selected
        if len(self.estimators) > 1:
            raise ValueError("Multiple models were selected. Please select only one model for final training.")
        if not isinstance(winner_model_name, str):
            raise ValueError("Model name must be a string.")

        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        print(f"Training final model '{winner_model_name}' on the entire dataset...")

        # Initialize the ATOMClassifier with the entire dataset
        print(f"Initializing ATOMClassifier for final model training ...")

        try:
            atom = ATOMClassifier(
                X, y,
                device= 'cpu',
                engine= 'sklearn',
                verbose= 0,
                random_state= self.random_seed
            )

            # ATOM handles hyperparameter tuning
            print(f"Running ATOM for model {winner_model_name} ...")
            print(f"Running inner loop with {self.K} folds for hyperparameter tuning ...")
            
            # Feature selection can be done here if needed
            atom.impute(strat_num='mean', verbose=1)
            atom.scale(strategy='robust', verbose=1)

            # Run the inner loop with the specified number of trials
            atom.run(
                models=[winner_model_name],
                metric=inner_metric,
                n_trials=n_trials,
                ht_params={'cv': self.K},
                est_params= custom_hp_params
            )

            # # Get the best model
            # best_model = atom._get_models()[0]

            # Save the trained model
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            save_path = os.path.join(save_dir, save_name)
            atom.save(save_path, save_data=False)

        except Exception as e:
            print(f"[Error] An exception occurred during final model training: {e}")
            return None, None
        
        print(f"Final model saved to: {save_path}")
    

    def predict_unseen(self, independent_test_set:pd.DataFrame, winner_model_path:str):
        """Makes predictions on an unseen dataset using the best trained model and asseses the performance.
        Args:
            independent_test_set (pd.DataFrame): The unseen dataset for predictions.
            winner_model_path (str): Path to the saved model file.
        
        Returns:
            pd.DataFrame: DataFrame containing the predictions and performance metrics.
        """

        # Check if the independent test set has a correct format
        try:
            train_columns = self.dataset.drop(columns=[self.target_column]).columns

            # Verify that the independent test set has the same columns as the training set
            missing_columns = set(train_columns) - set(independent_test_set.columns)
            if missing_columns:
                raise ValueError(f"Independent test set is missing columns: {missing_columns}")
            
            # Check the target column
            if self.target_column not in independent_test_set.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in independent test")
            
            # Make a copy so the original set is not altered
            X_test = independent_test_set[train_columns].copy()

            # Encoding the target variable in the same way with training
            y_test = independent_test_set[self.target_column].apply(
                lambda x: 1 if x == self.positive_class_label else 0
            )
            
            print("Loading winner model")
            atom = ATOMClassifier.load(winner_model_path, data=(X_test, y_test))

            print("Making predictions on the independent test set")
            # Get the model and make the predictions
            final_model = atom._get_models()[0]
            model_name = atom.models

            # Get predictions
            y_pred = final_model.predict(X_test)

            # Get probabilities and handle different return types
            y_prob = final_model.predict_proba(X_test)

            # Convert to numpy array if it's a DataFrame
            if hasattr(y_pred, 'values'):
                print(f"Converting predictions DataFrame to numpy array, shape: {y_pred.shape}")
                y_pred = y_pred.values
            
            # Convert to numpy array if it's a DataFrame
            if hasattr(y_prob, 'values'):
                print(f"Converting DataFrame to numpy array, shape: {y_prob.shape}")
                y_prob = y_prob.values
                    
            # Extract positive class probabilities (typically column 1)
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                # Multi-column case - get positive class probability (column 1)
                y_prob = y_prob[:, 1]
            else:
                # Single column case - flatten
                y_prob = y_prob.flatten()

            # Calculate metrics if predictions succeeded
            metrics = {}
            if y_pred is not None and y_prob is not None:
                metrics, conf_matrix = self._calculate_metrics(y_test, y_pred, y_prob)
                matrix_key = f"WINNER model: {model_name}"
                self.confusion_matrices[matrix_key] = conf_matrix
                
            # Store metrics with proper format
            result_record = {
                'Repeat': 'Test',
                'Outer_Fold': 'Final',
                'Estimator': model_name,
                **metrics  # Unpack all metrics
            }
            self.results = [result_record]  # Reset and store only test set results

            # Print metrics
            print("\nTest Set Performance:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.3f}")

        except Exception as e:
            print(f"Error in predicting unseen data: {e}")