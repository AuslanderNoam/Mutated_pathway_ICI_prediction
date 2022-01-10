#this is a python script to create the decision trees

# Imports --------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from optimize_pathways import *
from feature_selection_pathway_mutation_load import *
from handle_data import *
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
plt.rcParams["font.family"] = "Arial"
plt.rcParams['pdf.fonttype'] = 42

# Data -----------------------------------------------------------------------------------------------------------------

liu_mut = load_pickle(SAVE_DATA+'liu_mut.pickle')
liu_clin = load_pickle(SAVE_DATA + 'liu_clin.pickle')
riaz_mut_val = load_pickle(SAVE_DATA + 'riaz_mut.pickle')
riaz_clin_val = load_pickle(SAVE_DATA + 'riaz_clin.pickle')
GO_dict_intersection = load_pickle(SAVE_DATA + 'GO_dict_intersection.pickle')

# Decision Tree Functions -----------------------------------------------------------------------------------------------------------------

def train_random_forest(pathway_dictionary,pathway,train_mut=liu_mut,train_clin=liu_clin,max_depth = 5, min_samples_split = 2,random_state = 100):
    '''

    :param pathway_dictionary:
    :param pathway:
    :param train_mut:
    :param train_clin:
    :param max_depth:
    :param min_samples_split:
    :param random_state:
    :return:
    '''
    rforest = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                     random_state=random_state)
    rforest = rforest.fit(train_mut[pathway_dictionary[pathway]], br_0_1(train_clin['BR']))
    return rforest

def train_gradient_boosting(pathway_dictionary, pathway, train_mut=liu_mut, train_clin=liu_clin,max_depth = 2,
                            learning_rate = 0.1, loss = 'deviance', n_estimators = 100, random_state = 100):
    '''

    :param pathway_dictionary:
    :param pathway:
    :param train_mut:
    :param train_clin:
    :param max_depth:
    :param learning_rate:
    :param loss:
    :param n_estimators:
    :param random_state:
    :return:
    '''
    gbc = GradientBoostingClassifier(loss = loss, n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, random_state = random_state)
    gbc = gbc.fit(train_mut[pathway_dictionary[pathway]],br_0_1(train_clin['BR']))

    return gbc


def random_forest_multiple_pathway(pathway_list, pathway_dictionary, train_clin, train_mut, val_clin,val_mut, test_clin = None, test_mut = None, max_depth = 5, min_samples_split = 2, csv_path_name = False, random_state = 100):
    '''
    This is a function to carry out random forest classifications iterating over multiple pathways
    :param pathway_list: the list of the pathways to carry out rfc on
    :param pathway_dictionary: a dictionary with keys as pathways and values as genes associated with that pathway
    :param train_clin: the training clinical values, should have a column called 'BR' which is the response variable.
    :param train_mut: the training mutation values, a dataframe of patients as rows and genes as columns, entries indicate whether the gene is mutated
    :param val_clin: the validation clinical data
    :param val_mut: the validation mutation data
    :params: the rest of the parameters are the random forest classifier parameters, please see documentation for RandomForestClassifier from sklearn
    :params csv_path_name: the path and name to save a csv of the output scores, if left False will not save as csv
    :return: A pandas dataframe with a column for pathways, the training roc auc score, and the validation roc auc score for each pathway
    '''

    train_roc_list = []
    val_roc_list = []
    test_roc_list = []
    new_pathway_list = []
    #run classifier and fit to data
    for pathway in pathway_list:
        #just check if the pathway is actually populated, if not then skip that pathway
        if not pathway_dictionary[pathway]:
            train_roc_list.append(np.nan)
            val_roc_list.append(np.nan)
            new_pathway_list.append(pathway)
            continue
        #run the random forest and fit on the training data
        rforest = train_random_forest(pathway_dictionary,pathway,train_mut=train_mut,train_clin=train_clin,max_depth = max_depth, min_samples_split = min_samples_split, random_state = random_state)

        #obtain the roc_auc_scores for both the training and validation data
        train_roc = roc_auc_score(br_0_1(train_clin['BR']), rforest.predict_proba(train_mut[pathway_dictionary[pathway]])[:,1])
        val_roc = roc_auc_score(br_0_1(val_clin['BR']), rforest.predict_proba(val_mut[pathway_dictionary[pathway]])[:,1])
        if test_mut is not None:
            test_roc = roc_auc_score(br_0_1(test_clin['BR']), rforest.predict_proba(test_mut[pathway_dictionary[pathway]])[:,1])
            test_roc_list.append(test_roc)
        #store these values in a list for constructing the output df
        train_roc_list.append(train_roc)
        val_roc_list.append(val_roc)
        new_pathway_list.append(pathway)
    #make the output df
    if test_mut is not None:
        output_df = pd.DataFrame([new_pathway_list, train_roc_list, val_roc_list,test_roc_list]).transpose()
        output_df.columns = ['pathway','train_roc_auc_score', 'validation_roc_auc_score','test_roc_auc_score']
    else:
        output_df = pd.DataFrame([new_pathway_list, train_roc_list, val_roc_list]).transpose()
        output_df.columns = ['pathway','train_roc_auc_score', 'validation_roc_auc_score']

    #export as csv if given the csv and path name (or just csv name)
    if csv_path_name != False:
        output_df.to_csv(csv_path_name)

    return output_df

def gradient_boosting_multiple_pathway(pathway_list, pathway_dictionary, train_clin, train_mut, val_clin, val_mut, test_clin = None, test_mut = None, max_depth = 2, learning_rate = 0.1, loss = 'deviance', n_estimators = 100, csv_path_name = False, random_state = 100):
    '''
    This is a function to carry out gradient boosting classifications iterating over multiple pathways
    :param pathway_list: the list of the pathways to carry out rfc on
    :param pathway_dictionary: a dictionary with keys as pathways and values as genes associated with that pathway
    :param train_clin: the training clinical values, should have a column called 'BR' which is the response variable.
    :param train_mut: the training mutation values, a dataframe of patients as rows and genes as columns, entries indicate whether the gene is mutated
    :param val_clin: the validation clinical data
    :param val_mut: the validation mutation data
    :params: the rest of the parameters are the gradient booster parameters, please see documentation for RandomForestClassifier from sklearn
    :params csv_path_name: the path and name to save a csv of the output scores, if left False will not save as csv
    '''

    train_roc_list = []
    val_roc_list = []
    test_roc_list = []
    new_pathway_list = []
    #run classifier and fit to data
    for pathway in pathway_list:
        #just check if the pathway is actually populated
        if not pathway_dictionary[pathway]:
            train_roc_list.append(np.nan)
            val_roc_list.append(np.nan)
            new_pathway_list.append(pathway)
            continue
        #fit the gradient booster
        gbc = train_gradient_boosting(pathway_dictionary, pathway, train_mut=train_mut, train_clin=train_clin, max_depth=2,
                                learning_rate=0.1, loss='deviance', n_estimators=100, random_state=100)
        #obtain the roc_auc_scores for both the training and validation data

        train_roc = roc_auc_score(br_0_1(train_clin['BR']), gbc.predict_proba(train_mut[pathway_dictionary[pathway]])[:,1])
        val_roc = roc_auc_score(br_0_1(val_clin['BR']), gbc.predict_proba(val_mut[pathway_dictionary[pathway]])[:,1])
        if test_mut is not None:
            test_roc = roc_auc_score(br_0_1(test_clin['BR']), gbc.predict_proba(test_mut[pathway_dictionary[pathway]])[:,1])
            test_roc_list.append(test_roc)
        #store these values in a list for constructing the output df
        train_roc_list.append(train_roc)
        val_roc_list.append(val_roc)
        new_pathway_list.append(pathway)
    #make the output df
    if test_mut is not None:
        output_df = pd.DataFrame([new_pathway_list, train_roc_list, val_roc_list,test_roc_list]).transpose()
        output_df.columns = ['pathway','train_roc_auc_score', 'validation_roc_auc_score','test_roc_auc_score']
    else:
        output_df = pd.DataFrame([new_pathway_list, train_roc_list, val_roc_list]).transpose()
        output_df.columns = ['pathway','train_roc_auc_score', 'validation_roc_auc_score']

    #export as csv if given the csv and path name (or just csv name)
    if csv_path_name != False:
        output_df.to_csv(csv_path_name)

    return output_df
