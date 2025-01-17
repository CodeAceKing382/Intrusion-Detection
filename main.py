from data_preprocessing import load_and_preprocess_data
from feature_selection import select_features
from model_training import train_model
from evaluation import evaluate_model

def get_user_input():
    print("\n=== INTRUSION DETECTION SYSTEM ===")
    print("\nPlease select your options:")
    
    # Get classification type
    while True:
        print("\n1. Classification Type:")
        print("   [b] Binary Classification")
        print("   [m] Multi-class Classification")
        classification = input("Enter your choice (b/m): ").lower()
        if classification in ['b', 'm']:
            break
        print("Invalid input! Please enter 'b' or 'm'")
    
    # Get feature selection type
    while True:
        print("\n2. Feature Selection Method:")
        print("   [f] Feature-based/Chi-squared")
        print("   [w] Wrapper-based/RFE")
        feature_selection = input("Enter your choice (f/w): ").lower()
        if feature_selection in ['f', 'w']:
            break
        print("Invalid input! Please enter 'f' or 'w'")
    
    return classification, feature_selection

def main():
    # Get user inputs
    classification, feature_selection = get_user_input()
    
    print("\n=== Starting INTRUSION DETECTION Pipeline ===")
    print(f"Selected Options:")
    print(f"- Classification: {'Binary' if classification == 'b' else 'Multi-class'}")
    print(f"- Feature Selection: {'Feature-based/Chi-squared' if feature_selection == 'f' else 'Wrapper-based/RFE'}\n")
    
    # Step 1: Data Preprocessing
    X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val = load_and_preprocess_data(
        classification_type=classification
    )
    
    # Step 2: Feature Selection
    X_train_selected, X_test_selected, X_val_selected = select_features(
        X_train_scaled, X_test_scaled, X_val_scaled,
        y_train, y_test, y_val,
        classification_type=classification,
        feature_selection_type=feature_selection
    )
    
    # Step 3: Model Training
    all_labels, all_preds = train_model(
        X_train_selected, X_test_selected, X_val_selected,
        y_train, y_test, y_val,
        classification_type=classification
    )
    
    # Step 4: Model Evaluation
    evaluate_model(all_labels, all_preds)
    
    print("\n=== Pipeline Completed Successfully ===\n")

if __name__ == "__main__":
    main()