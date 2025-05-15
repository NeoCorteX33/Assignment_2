# src/custom_hyperparameters.py

import numpy as np

def get_custom_hyperparameter_spaces_selective():
    """
    Defines custom hyperparameter spaces for only Logistic Regression (LR)
    and Support Vector Machine (SVM), compatible with ATOM's `ht_params`
    argument for the Optuna engine.

    Other models will use ATOM's default search spaces.

    Returns:
        dict: A dictionary where keys are "LR" and "SVM"
              and values are dictionaries defining their hyperparameter search space
              for Optuna.
    """
    spaces = {
        "LR": {  # Logistic Regression with Elastic Net
            "penalty": ("categorical", ["elasticnet"]),
            "solver": ("categorical", ["saga"]),
            "C": ("float", 1e-4, 1e2, "log"),
            "l1_ratio": ("float", 0.01, 0.99),
            "max_iter": ("int", 500, 5000),
            "class_weight": ("categorical", ["balanced", None])
        },
        "SVM": {  # Support Vector Machine
            "C": ("float", 1e-3, 1e3, "log"),
            "kernel": ("categorical", ["linear", "rbf"]),
            "gamma": ("float", 1e-4, 1e1, "log"),
            "class_weight": ("categorical", ["balanced", None])
        }
        # Note: GNB, LDA, RF, LGB are intentionally omitted here.
        # ATOM will use its default search spaces for them.
    }
    return spaces

# --- Example of how to get the spaces (if you run this file directly) ---
if __name__ == '__main__':
    selective_spaces = get_custom_hyperparameter_spaces_selective()
    print("--- Logistic Regression Space ---")
    print(selective_spaces.get("LR")) # Use .get() in case a key is missing
    print("\n--- SVM Space ---")
    print(selective_spaces.get("SVM"))
    print("\n--- Attempt to get RF space (should be None) ---")
    print(selective_spaces.get("RF"))