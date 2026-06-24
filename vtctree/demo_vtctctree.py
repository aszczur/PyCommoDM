import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score
from vtctree import (VerifyingTemporalCutsDecisionTreeClassifier, TemporalCutsQualityMeasure,
                     BestTemporalCutSelectionMethod, CutsConflictsResolvingMethod)

# Load training and test datasets
path_tr = "data/tr_data.csv"
path_tst = "data/tst_data.csv"
data_tr = pd.read_csv(path_tr, sep=';', dtype='float64', low_memory=False)
print('data loaded: ', path_tr)
data_tst = pd.read_csv(path_tst, sep=';', dtype='float64', low_memory=False)
print('data loaded: ', path_tst)

# Determine the number of time points per time window
id_tw_column = 0
id_tp_column = 1
time_windows = data_tr.groupby(data_tr.columns[id_tw_column])[data_tr.columns[id_tp_column]].nunique()
tp_count = time_windows.iloc[0]

# VTC-Tree configuration
quality_measure = TemporalCutsQualityMeasure.DISC_PAIR_FROM_DIFF_CLASSES
quality_sel_method = BestTemporalCutSelectionMethod.MAX_QUALITY
alpha_start = 0.7
alpha_step = 0.05
alpha_step_count = 6
vcuts_max_count = 14 # max number of v_cuts in node
min_vc_quality_ratio = 0.5

# Train VTC-Tree classifier
vtree_classifier = VerifyingTemporalCutsDecisionTreeClassifier()
vtree_classifier.fit(data_tr, alpha_start=alpha_start, alpha_step=alpha_step, alpha_step_count=alpha_step_count,
                     quality_measure=quality_measure, quality_sel_method=quality_sel_method,
                     min_vc_quality_ratio=min_vc_quality_ratio, vcuts_max_count=vcuts_max_count)
# Display the learned tree structure
vtree_classifier.print_tree()

# Predict class labels and class probabilities
conflicts_resolv_methods = [CutsConflictsResolvingMethod.CRM_P, CutsConflictsResolvingMethod.CRM_T]
crm = conflicts_resolv_methods[0]
labels_pred = vtree_classifier.predict(data_tst, crm)
print(labels_pred)
proba_pred = vtree_classifier.predict_proba(data_tst, crm)

# Extract one decision value per time window
no_column = data_tst.shape[1]
labelsALL = data_tst.iloc[:, no_column - 1].to_numpy(dtype=float)
labels_origin = labelsALL[0::tp_count].tolist() # get dec from first tp of each time window
print(labels_origin)

# Model evaluation
report = classification_report(labels_origin, labels_pred)
print(report)
conf_matrix = confusion_matrix(labels_origin, labels_pred)
print(conf_matrix)
accuracy = accuracy_score(labels_origin, labels_pred)
print(f'Accuracy: {accuracy:.5f}')
balanced_acc = balanced_accuracy_score(labels_origin, labels_pred)
print(f"Accuracy balanced: {balanced_acc:.3f}\n")
class_recall = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
print("recall:", class_recall)

# Prepare data for AUC computation
y_true = labels_origin
y_proba_arr = np.array(proba_pred)
classes = np.unique(y_true)

# Compute AUC scores
if len(classes) > 2:  # multiclass
    auc_macro = roc_auc_score(y_true, y_proba_arr, multi_class="ovr", average="macro")
    auc_per_class = roc_auc_score(y_true, y_proba_arr, multi_class="ovr", average=None)
else:  # binary
    prob_class1 = y_proba_arr[:, 1]
    auc_macro = roc_auc_score(y_true, prob_class1)
    auc_per_class = [auc_macro]
print(f"AUC (macro): {auc_macro:.3f}")
if len(classes) > 2:
    for cls, auc in zip(classes, auc_per_class):
        print(f"AUC (class {cls}) = {auc:.3f}")

