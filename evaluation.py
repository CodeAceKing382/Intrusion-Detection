from sklearn.metrics import accuracy_score, classification_report
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)

def evaluate_model(all_labels,all_preds):
    # Assuming `all_labels` and `all_preds` are available (as arrays or lists)
    start_time = time.time()  # Start timer for prediction time

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')  # 'macro' or 'micro' based on requirement
    recall = recall_score(all_labels, all_preds, average='weighted')        # Adjust as needed
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Specificity and NPV for each class
    specificities = []
    npvs = []
    for i in range(conf_matrix.shape[0]):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        fn = conf_matrix[i, :].sum() - conf_matrix[i, i]
        tp = conf_matrix[i, i]
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        specificities.append(specificity)
        npvs.append(npv)

    # MCC
    mcc = matthews_corrcoef(all_labels, all_preds)

    # Prediction time
    end_time = time.time()
    prediction_time = end_time - start_time

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Specificities: {specificities}")
    print(f"NPVs: {npvs}")
    print(f"MCC: {mcc:.4f}")
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    print(f"confusion_matrix: {conf_matrix}")


