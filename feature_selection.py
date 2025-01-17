from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def apply_chi2_selection(X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val, n_features=35):
    """
    Apply Chi-Squared Test for feature selection and return transformed data

    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        X_val_scaled: Scaled validation features
        y_train: Training labels
        y_test: Test labels
        y_val: Validation labels
        n_features: Number of features to select (default=30)
    """
    print("Starting Chi-Squared Feature Selection...")
    print(f"Initial number of features: {X_train_scaled.shape[1]}")

    # Ensure all feature values are non-negative for Chi-Squared Test
    X_train_non_negative = np.abs(X_train_scaled)
    X_test_non_negative = np.abs(X_test_scaled)
    X_val_non_negative = np.abs(X_val_scaled)

    # Apply Chi-Squared Test
    chi2_selector = SelectKBest(score_func=chi2, k=n_features)
    chi2_selector.fit(X_train_non_negative, y_train)

    # Get selected features
    selected_features = chi2_selector.get_support(indices=True)

    # Transform the data
    X_train_selected = chi2_selector.transform(X_train_non_negative)
    X_test_selected = chi2_selector.transform(X_test_non_negative)
    X_val_selected = chi2_selector.transform(X_val_non_negative)

    # Print results
    print("\nChi-Squared Results:")
    print("-" * 50)
    print(f"Selected {len(selected_features)} features")

    # Get feature scores
    feature_scores = chi2_selector.scores_[selected_features]

    # Create DataFrame with feature indices and scores
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Score': feature_scores
    })

    # Sort by score
    feature_importance_df = feature_importance_df.sort_values('Score', ascending=False)

    # Plot feature scores
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance_df)), feature_importance_df['Score'])
    plt.xlabel('Selected Features')
    plt.ylabel('Score')
    plt.title('Feature Scores of Selected Features (Chi-Squared)')
    plt.xticks(range(len(feature_importance_df)), feature_importance_df['Feature'], rotation=45)
    plt.tight_layout()
    plt.show()

    # Print selected features and their scores
    print("\nSelected Features and Their Scores:")
    print("-" * 50)
    for idx, row in feature_importance_df.iterrows():
        print(f"Feature {int(row['Feature'])}: {row['Score']:.4f}")

    return {
        'X_train_selected': X_train_selected,
        'X_test_selected': X_test_selected,
        'X_val_selected': X_val_selected,
        'selected_features': selected_features,
        'feature_scores': feature_importance_df,
        'chi2_selector': chi2_selector
    }

def apply_rfe_selection(X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val, n_features=35):
    """
    Apply RFE feature selection and return transformed data
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        X_val_scaled: Scaled validation features
        y_train: Training labels
        y_test: Test labels
        y_val: Validation labels
        n_features: Number of features to select (default=30)
    """
    print("Starting RFE Feature Selection...")
    print(f"Initial number of features: {X_train_scaled.shape[1]}")
    
    # Initialize XGBoost and RFE
    xgb = XGBClassifier(random_state=42,n_jobs= -1,tree_method='hist', max_depth=6,n_estimators=100)
    rfe = RFE(estimator=xgb, 
              n_features_to_select=n_features,
              step=2)  # Remove 2 feature at a time
    
    # Fit RFE
    rfe = rfe.fit(X_train_scaled, y_train)
    
    # Get selected features
    selected_features = np.where(rfe.support_)[0]
    
    # Transform the data
    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]
    X_val_selected = X_val_scaled[:, selected_features]
    
    print("\nRFE Results:")
    print("-" * 50)
    print(f"Selected {len(selected_features)} features")

    xgb_after = XGBClassifier(random_state=42)
   
    
    # Get feature importances from XGBoost
    xgb_after.fit(X_train_selected, y_train)
    importances = xgb_after.feature_importances_
    
    # Create DataFrame with feature indices and importances
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance_df)), feature_importance_df['Importance'])
    plt.xlabel('Selected Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance of Selected Features (RFE)')
    plt.xticks(range(len(feature_importance_df)), feature_importance_df['Feature'], rotation=45)
    plt.tight_layout()
    plt.show()
   
   # Print selected features and their scores
    print("\nSelected Features and Their Importance Scores:")
    print("-" * 50)
    for idx, row in feature_importance_df.iterrows():
        print(f"Feature {int(row['Feature'])}: {row['Importance']:.4f}")

    return {
        'X_train_selected': X_train_selected,
        'X_test_selected': X_test_selected,
        'X_val_selected': X_val_selected,
        'selected_features': selected_features,
        'feature_importances': feature_importance_df,
        'rfe': rfe
    }


def select_features(X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val, classification_type, feature_selection_type):
    if feature_selection_type == "f":
        chi2_results = apply_chi2_selection(
            X_train_scaled, 
            X_test_scaled,
            X_val_scaled,
            y_train, 
            y_test,
            y_val,
            n_features=30
        )
        X_train_selected = chi2_results['X_train_selected']
        X_test_selected = chi2_results['X_test_selected']
        X_val_selected = chi2_results['X_val_selected']

        print("\nFeature Selection Complete!")
        print(f"Final number of features: {X_train_selected.shape[1]}")

    elif feature_selection_type == "w":
        rfe_results = apply_rfe_selection(
            X_train_scaled, 
            X_test_scaled,
            X_val_scaled,
            y_train, 
            y_test,
            y_val,
            n_features=30
        )
        
        # Get transformed datasets
        X_train_selected = rfe_results['X_train_selected']
        X_test_selected = rfe_results['X_test_selected']
        X_val_selected = rfe_results['X_val_selected']
        print("\nFeature Selection Complete!")
        print(f"Final number of features: {X_train_selected.shape[1]}")


    return X_train_selected, X_test_selected, X_val_selected
