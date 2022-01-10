# Imports --------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from scipy.stats import spearmanr
import random

from sys import platform

# Path/constants -------------------------------------------------------------------------------------------------------

INTERMEDIATE_FILE = '../20_Intermediate_Files/'
SAVE_DATA = '../05_pickled_data/'

LOAD_DATA = False  ##Run once and change to False to save time later
# Data -----------------------------------------------------------------------------------------------------------------

# Functions  -----------------------------------------------------------------------------------------------------------

def br_0_1(br_labels):
    '''
    This is a function to turn the br labels into 0 or 1
    :param br_labels: the br labels of the data, a pandas series
    :return: a pandas series of 0 and 1, 0 for negative br outcome, 1 for positive br outcome
    '''
    return br_labels.replace(to_replace=['CR', 'PR'], value=1).replace(to_replace=['PD', 'SD', 'MR','NE'], value=0)


def br_outcome_roc_auc_score(mutations_count_pathway, br_labels):
    '''
    This is a function to give the roc_auc_score for a given classifier based on the BR labels and mutations in a single pathway
    :param br_labels: is the input from the BR column of the datasets, these have 5 values. Values are converted into either positive or negative
    :param mutations_pathway_count: is the number of mutations in a single given pathway per patient. This function processes pathways one at a time.
    :return: the roc auc score for a given pathway based on total mutation count for the pathway
    '''

    # convert BR values into 1 or 0. CR and PR are 1, the rest are 0
    br_posneg = br_labels.replace(to_replace=['CR', 'PR','R'], value=1).replace(to_replace=['PD', 'SD', 'MR','NE','NR'], value=0)
    return roc_auc_score(br_posneg, mutations_count_pathway)



# now define a function that returns the c_index based on survival time, censor status, and a single pathways mutations

def survival_outcome_c_index(mutations_count_pathway, survival_time, censor_status):
    '''
    This is a function to give the c_index based on the survival time, censor information, and mutations in a single pathway
    :param survival_time: the survival time in months (though actual label of time does not seem relevant, just make sure they are uniform in the dataset)
    :param censor_status: whether the patient is alive or dead. Alive patients will be censored (0 in event_observed for c_index)
    :param mutation_pathway_count: the number of mutations in a single given pathway. This function can only process pathways one at a time.
    :return: the c-index of the pathway
    '''

    # assign censor values of 0 to alive and 1 to dead
    censor_0_1 = censor_status.replace(to_replace=['Dead',True], value=0).replace(to_replace=['Alive',False], value=1)
    return concordance_index(event_times=survival_time, predicted_scores=mutations_count_pathway,event_observed=censor_0_1)



def optimization_score_function(optimize_on, mutation_number, br_labels, survival_time, censor_status, response = 1):
    '''
    This is a function to calculate an optimization score using either the roc_auc_score or c_index functions
    :param optimize_on: the function to use for optimization, currently should be either 'roc_auc_score' or 'c_index'
    :param br_labels: the input from the BR column of the datasets, these have 5 values. Values are converted into either positive or negative. Only need this parameter when optimizing using roc_auc_scores
    :param survival_time: the survival time in months (though actual label of time does not seem relevant, just make sure they are uniform in the dataset). Only need this parameter when optimizing on c_index.
    :param censor_status: whether the patient is alive or dead. Alive patients will be censored (0 in event_observed for c_index). Only need this parameter when optimizing on c_index.
    :param response: use 1 if predicting response 0 if predicting resistance
    :return: either the roc_auc_score or c_index for the pathway
    '''

    if optimize_on == 'roc_auc_score':
    #get optimization score using br_outcome_roc_auc_score function. Add to list for later analysis
        optimization_param = br_outcome_roc_auc_score(mutation_number, br_labels)


    elif optimize_on == 'c_index':
        #or if using c_index, get the c_index. Add to list for later analysis
        optimization_param = survival_outcome_c_index(mutation_number, survival_time, censor_status)

    if response == 0:
        optimization_param = 1- optimization_param

    return optimization_param

def select_pathways():
    '''
    this function selects the top pathways from the genetic algorithm and forward selected results
    :return: a list of the top pathways for both the genetic algorithm and forward selected methods

    '''
    GA_res = pd.read_csv('../25_Results/tables/GA_GO_results.csv')
    FS_res = pd.read_csv('../25_Results/tables/FS_GO_results.csv')

    l1 = list(GA_res['auc_liu_GO_GA_response'])
    l2 = list(GA_res['auc_riaz_GO_GA_response'])

    q1 = np.quantile(l1, 0.97)
    p1 = [list(GA_res['GO_pathways'])[i] for i in range(len(l1)) if l1[i] >= q1 and l2[i] > l1[i] * 0.9]

    l11 = list(FS_res['auc_liu_GO_FS_response'])
    l22 = list(FS_res['auc_riaz_GO_FS_response'])
    l33 = list(FS_res['auc_liu_GO_FS_response_randomized'])
    l44 = list(FS_res['auc_riaz_GO_FS_response_randomized'])
    q2 = np.quantile(l11, 0.97)
    q3 = np.quantile(l33, 0.97)

    p11 = [list(FS_res['GO_pathways'])[i] for i in range(len(l11)) if
           (l11[i] >= q2 and l22[i] > l11[i] * 0.9) or (l33[i] >= q3 and l44[i] > l33[i] * 0.9)]
    # p11 = [list(FS_res['GO_pathways'])[i] for i in range(len(l11)) if l11[i]>=0.65 and l22[i]>l11[i]*0.9 and l33[i]>=0.65 and l44[i]>l33[i]*0.9 ]

    # pt = list(set(p11)&set(p1))
    pt = list(set(p1 + p11))

    return pt

def pathway_pd_to_dict(ppd):
    '''

    :param ppd:
    :return:
    '''
    d1 = ppd.to_dict('list')
    d2 = {list(d1.keys())[i]: [j for j in list(d1.items())[i][1] if type(j) == str] for i in range(len(d1.keys()))}
    return d2

def select_pathways_trees():
    '''
    this function selects the top pathways for the tree methods
    :return: a list of the top pathways in both the gradient boosting and random forest methods (turns out that the GB pathways are all in the RF pathways)


    '''
    GO_gb = pd.read_csv('../20_Intermediate_Files/GO_Gradient_Boosting_All_Pathways.csv').drop(columns = 'Unnamed: 0')
    all_pathway_random_forests = pd.read_csv('../20_Intermediate_Files/GO_Random_Forest_All_Pathways.csv').drop(columns = 'Unnamed: 0')
    #get the quantile for the gb scores on the training set
    q1 = np.quantile(GO_gb['train_gradient_boosting_roc_auc_score'].fillna(0), 0.68)
    #and quantile for the random forest on the training dataset. Can change quantile if necessary
    q2 = np.quantile(all_pathway_random_forests['train_roc_auc_score'].fillna(0), 0.68)
    #find the values where the training is greater than the quantile and the validation is greater than 0.9 below the quantile.
    #can test these numbers and change if need more pathways
    GB_top = [list(GO_gb['pathway'])[i] for i in range(len(GO_gb['train_gradient_boosting_roc_auc_score'])) if GO_gb['train_gradient_boosting_roc_auc_score'][i] >= q1 and GO_gb['validation_gradient_boosting_roc_auc_score'][i] > GO_gb['train_gradient_boosting_roc_auc_score'][i] * 0.9]
    RF_top = [list(all_pathway_random_forests['pathway'])[i] for i in range(len(all_pathway_random_forests['train_roc_auc_score'])) if all_pathway_random_forests['train_roc_auc_score'][i] >= q2 and all_pathway_random_forests['validation_roc_auc_score'][i] > all_pathway_random_forests['train_roc_auc_score'][i] * 0.9]

    GB_top
    combined_pathways = list(set(GB_top + RF_top))
    return combined_pathways
