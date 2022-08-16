# Imports --------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sys import platform
from handle_data import load_pickle
from decision_trees import train_random_forest
from optimize_pathways import br_0_1
from sklearn import metrics

INTERMEDIATE_FILE = '../20_Intermediate_Files/'
INPUT_DATA = '../00_Data/'
SAVE_DATA = '../05_pickled_data/'


#load the liu training dataset
liu_mut = load_pickle(SAVE_DATA+'liu_mut.pickle')
liu_clin = load_pickle(SAVE_DATA + 'liu_clin.pickle')

#train RF predictor using the training dataset
liu_mut = load_pickle(SAVE_DATA+'liu_mut.pickle')
liu_clin = load_pickle(SAVE_DATA + 'liu_clin.pickle')
d = pd.read_csv(INPUT_DATA + 'GO_test_genes_dict_intersection.csv')
GO_dict_intersection = {k: v[v.notna()].to_dict() for k,v in d.items()}

GO_test_dict_import = pd.read_csv(INPUT_DATA + 'GO_test_genes_dict_intersection.csv', index_col = 0)
new_test_gene_dict = dict()
for col in GO_test_dict_import.columns:
    GO_dict_intersection[col] = list(GO_test_dict_import[col].dropna())


pathway = 'GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION'
rf_pred = train_random_forest(GO_dict_intersection,pathway,train_mut=liu_mut,train_clin=liu_clin,random_state = 100)

liu_labels =  br_0_1(liu_clin['BR'])
liu_scores = rf_pred.predict_proba(liu_mut[GO_dict_intersection[pathway]])[:, 1]
metrics.roc_auc_score(liu_labels, liu_scores)

#load validation dataset
riaz_mut_val = load_pickle(SAVE_DATA + 'riaz_mut.pickle')
riaz_clin_val = load_pickle(SAVE_DATA + 'riaz_clin.pickle')

#apply the model to riaz validation dataset
riaz_labels =  br_0_1(riaz_clin_val['BR'])
riaz_scores = rf_pred.predict_proba(riaz_mut_val[GO_dict_intersection[pathway]])[:, 1]
metrics.roc_auc_score(riaz_labels, riaz_scores)

#load hugo test dataset
hugo_mut_test = load_pickle(SAVE_DATA + 'hugo_mut.pickle')
hugo_clin_test = load_pickle(SAVE_DATA + 'hugo_clin.pickle')

#apply the model to hugo test dataset
hugo_labels =  br_0_1(hugo_clin_test['BR'])
hugo_scores = rf_pred.predict_proba(hugo_mut_test[GO_dict_intersection[pathway]])[:, 1]
metrics.roc_auc_score(hugo_labels, hugo_scores)

#load hugo test dataset
hugo_mut_test = load_pickle(SAVE_DATA + 'hugo_mut.pickle')
hugo_clin_test = load_pickle(SAVE_DATA + 'hugo_clin.pickle')

#apply the model to hugo test dataset
hugo_labels =  br_0_1(hugo_clin_test['BR'])
hugo_scores = rf_pred.predict_proba(hugo_mut_test[GO_dict_intersection[pathway]])[:, 1]
metrics.roc_auc_score(hugo_labels, hugo_scores)