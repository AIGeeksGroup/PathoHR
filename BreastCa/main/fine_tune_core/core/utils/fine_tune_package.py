import torch
import tqdm
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score



def write_dict_to_config_file(config_dict, json_file_path):
    """
    Write a dictionary to a configuration file.
    Args:
        config_dict (dict): The dictionary to be written to the config file.
        config_file_path (str): The path to the configuration file.
    Returns:
        None
    """
    with open(json_file_path, 'w') as jsonfile:
        json.dump(config_dict, jsonfile, indent=4)
def load_json_to_dict(json_file_path):
    with open(json_file_path, 'r') as j:
        log_dict = json.load(j)
    return log_dict
        
def calculate_metrics(y_true, y_pred, pred_scores):
    """
    Calculate and print various evaluation metrics.
    
    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - y_scores: Target scores (for AUC).
    """
    #print(y_true.shape,y_pred.shape,pred_scores) 
    # (64,) (64,) tangle & (64,) (1,) resnet50
    if len(np.unique(y_true)) > 2:
        # multi-class 
        auc = roc_auc_score(y_true, pred_scores, multi_class="ovr", average="macro",)
    else :
        # regular 
        auc = roc_auc_score(y_true, pred_scores[:, 1]) # only send positive class score)

       
    #print(auc)
    bacc = balanced_accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return conf_matrix, auc, bacc, accuracy, precision, recall, f1