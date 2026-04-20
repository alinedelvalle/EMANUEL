from sklearn.metrics import *
from skmultilearn.dataset import load_from_arff


METRICS = ["dataset",
        "fold",
		"normalize",
		"meka",
		"weka",
        "accuracy_score",
        "average_precision_score",
        "balanced_accuracy_score",
        "cohen_kappa_score",
        "f1_score_macro",
        "f1_score_micro",
        "f1_score_weighted",
        "matthews_corrcoef",
        "precision_score",
        "recall_score",
        "roc_auc_score",
        "coverage_error",
        "label_ranking_average_precision_score",
        "label_ranking_loss",
        "model_size"]


def calculate_score(metric, y_true, y_pred, **kwargs):
	try:
		return metric(y_true, y_pred, **kwargs)
	except:
		return -1.0
     

def collect_and_persist_results(dataset, k, normalize, meka, weka, y_test, y_pred, model_size):
    
    this_result = {
		"dataset":                                  dataset,
        "fold":                                     k,
		"normalize":                                normalize,
		"meka":                                     meka,
		"weka":                                     weka,
        "accuracy_score":                           calculate_score(accuracy_score, y_test, y_pred),
        "average_precision_score":                  calculate_score(average_precision_score, y_test, y_pred),
        "balanced_accuracy_score":                  calculate_score(balanced_accuracy_score, y_test, y_pred),
        "cohen_kappa_score":                        calculate_score(cohen_kappa_score, y_test, y_pred),
        "f1_score_macro":                           calculate_score(f1_score, y_test, y_pred, average="macro"),
        "f1_score_micro":                           calculate_score(f1_score, y_test, y_pred, average="micro"),
        "f1_score_weighted":                        calculate_score(f1_score, y_test, y_pred, average="weighted"),
        "matthews_corrcoef":                        calculate_score(matthews_corrcoef, y_test, y_pred),
        "precision_score":                          calculate_score(precision_score, y_test, y_pred),
        "recall_score":                             calculate_score(recall_score, y_test, y_pred),
        "roc_auc_score":                            calculate_score(roc_auc_score, y_test, y_pred),
        "coverage_error":                           calculate_score(coverage_error, y_test, y_pred),
        "label_ranking_average_precision_score":    calculate_score(label_ranking_average_precision_score, y_test, y_pred),
        "label_ranking_loss":                       calculate_score(label_ranking_loss, y_test, y_pred),
        "model_size":                               model_size,
    }
	
    return this_result