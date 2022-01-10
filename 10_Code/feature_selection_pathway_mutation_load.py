# This is a python script to carry out preliminary EDA and PCA on cancer data from Riaz and Liu

# Imports --------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import random
from optimize_pathways import *
from handle_data import *


# Path/constants -------------------------------------------------------------------------------------------------------

INTERMEDIATE_FILE = '../20_Intermediate_Files/'
SAVE_DATA = '../05_pickled_data/'

LOAD_DATA = False  ##Run once and change to False to save time later
# Data -----------------------------------------------------------------------------------------------------------------

liu_mut = load_pickle(SAVE_DATA+'liu_mut.pickle')
liu_clin = load_pickle(SAVE_DATA + 'liu_clin.pickle')
GO_dict_intersection = load_pickle(SAVE_DATA + 'GO_dict_intersection.pickle')

# Functions  -----------------------------------------------------------------------------------------------------------


def greedy_forward_selector(pathway, pathway_dictionary, number_of_genes, patient_mutation_data, optimize_on, br_labels = False, survival_time = False, censor_status = False, resistance = False):
    ''' This is a function to find the top genes per pathway by using a greedy forward selection method
    :param pathway is the pathway to use to select the top ten genes
    :param pathway_dictionary is a dictionary of the all of the pathways as keys and their associated genes as values
    :param number_of_genes is the number of genes to return. If there are less than that number of genes in the pathway, all genes will be returned.
    :param patient_mutation_data is the dataframe of patient mutation information by gene. Gene information is column, patient data is index
    :param optimize_on is the optimization parameter (either roc_auc_score or c_index) that will be maximized to use to select the genes
    :param br_labels is the input from the BR column of the datasets, these have 5 values. Values are converted into either positive or negative. Only need this parameter when optimizing using roc_auc_scores
    :param survival_time: is the survival time in months (though actual label of time does not seem relevant, just make sure they are uniform in the dataset). Only need this parameter when optimizing on c_index.
    :param censor_status: is whether the patient is alive or dead. Alive patients will be censored (0 in event_observed for c_index). Only need this parameter when optimizing on c_index.
    :param resistance: False is if predicting respones True if predicting resistance
    :return: a list of the pathway genes selected, and the roc auc score for those genes
    '''


    assert optimize_on in ['roc_auc_score', 'c_index'], 'Optimize_on only supports roc_auc_score or c_index'

    #get the genes of the pathway
    pathway_genes = pathway_dictionary[pathway]

    pathway_genes = list(set(patient_mutation_data.keys())& set(pathway_genes))
    #carry out a test to determine if the pathway has less genes than the number given to return. If true, will end function and return all genes


    #start an empty list for the genes to return, and a zero current optimization score so that we always get at least one gene at the end
    #if looking for resistance, use a 1 instead of 0
    genes_to_return = []
    if resistance == True:
        current_optimization = 1
    else:
        current_optimization = 0

    #iterate over the number of genes to return. If the length of the genes is greater than the actual number of genes in the pathway, default to the number of genes in the pathway
    if number_of_genes >= len(pathway_genes):
        number_of_genes = len(pathway_genes)

    for round_number in range(0, number_of_genes):
        # print(i)
        #for each iteration, find the genes that provide the best results



        previous_best_optimization = current_optimization

        for pat_gene in pathway_genes:

            #copy the genes_to_return list, and append the current testing gene
            genes_to_test = [i for i in genes_to_return]
            genes_to_test.append(pat_gene)
            #get the combined mutations for the genes for each patient
            mutation_number = mutation_load(genes_to_test, patient_mutation_data = patient_mutation_data)
            #mutation_number = patient_mutation_data[genes_to_test].sum(1) ##Check the shape

            #now carry out the selected optimization parameter
            optimization_param = optimization_score_function(optimize_on = optimize_on, mutation_number = mutation_number, br_labels = br_labels, survival_time = survival_time, censor_status = censor_status)

            #test and see if the current optimization_param is larger than the largest optimization parameter
            #if checking for resistance, then reverse the >
            if resistance == True:
                if optimization_param < current_optimization:
                    current_optimization = optimization_param
                    current_best_gene = pat_gene

            else:
                if optimization_param > current_optimization:
                    current_optimization = optimization_param
                    current_best_gene = pat_gene

        #now test if the current optimization parameter is larger than the previous optimization parameter. If it is not, then break the loop
        #do the opposite for resistance, check if the current_optimization is less than the previous optimization
        if resistance == True:
            if current_optimization < previous_best_optimization:
                genes_to_return.append(current_best_gene)
            else:
                break

        else:
            if current_optimization > previous_best_optimization:
                genes_to_return.append(current_best_gene)
            else:
                break

    #this is a final check to ensure that the optimization score is correct and matches which should have been generated above, in the loop
    mutation_number = mutation_load(genes_to_return, patient_mutation_data = patient_mutation_data)

    optimization_confirmation = optimization_score_function(optimize_on = optimize_on, mutation_number = mutation_number, br_labels = br_labels, survival_time = survival_time, censor_status = censor_status)
    optimization_confirmation = optimization_confirmation

    assert abs(optimization_confirmation - current_optimization) < 0.05, "Optimizations are not the same, problem when checking optimizations: "+pathway

    #finally, return the list of genes for the pathway and the associated optimization parameter
    return genes_to_return, optimization_confirmation


def probabilistic_forward_selector(pathway, pathway_dictionary, number_of_genes, patient_mutation_data, optimize_on, br_labels = False, survival_time = False, censor_status = False, resistance = False):
    ''' This is a function to find the top genes per pathway by using a greedy forward selection method
    :param pathway is the pathway to use to select the top ten genes
    :param pathway_dictionary is a dictionary of the all of the pathways as keys and their associated genes as values
    :param number_of_genes is the number of genes to return. If there are less than that number of genes in the pathway, all genes will be returned.
    :param patient_mutation_data is the dataframe of patient mutation information by gene. Gene information is column, patient data is index
    :param optimize_on is the optimization parameter (either roc_auc_score or c_index) that will be maximized to use to select the genes
    :param br_labels is the input from the BR column of the datasets, these have 5 values. Values are converted into either positive or negative. Only need this parameter when optimizing using roc_auc_scores
    :param survival_time is the survival time in months (though actual label of time does not seem relevant, just make sure they are uniform in the dataset). Only need this parameter when optimizing on c_index.
    :param censor_status is whether the patient is alive or dead. Alive patients will be censored (0 in event_observed for c_index). Only need this parameter when optimizing on c_index.
    :param response 1 is if predicting respones 0 if predicting resistance
    :return:
    '''
    #this assert checks if the optimize input was carrect
    assert optimize_on in ['roc_auc_score', 'c_index'], 'Optimize_on only supports roc_auc_score or c_index'

    #get the genes of the pathway
    pathway_genes = pathway_dictionary[pathway]

    pathway_genes = list(set(patient_mutation_data.keys())& set(pathway_genes))


    #start an empty list for the genes to return, and a zero current optimization score so that we always get at least one gene at the end
    #if looking for resistance, use a 1 instead of 0
    genes_to_return = []
    if resistance == True:
        current_optimization = 1
    else:
        current_optimization = 0

    #iterate over the number of genes to return. If the length of the number_of_genes variable is greater than the actual number of genes in the pathway, default to the number of genes in the pathway
    if number_of_genes >= len(pathway_genes):
        number_of_genes = len(pathway_genes)
    #this counter_token is used to keep track of the while loop. Specifically, if the probabalistic model does not select a gene that increases roc auc score, will repeat the loop
    counter_token = 0
    while counter_token <= number_of_genes:

        #for each iteration, find the genes that provide the best results
        previous_best_optimization = current_optimization

        #reset check token for later checking if we actually pulled a gene
        check_token = 0
        #set this to False, will become True if there are any genes which increase our score.
        genes_increase = False

        #iterate over the genes in the pathway
        for pat_gene in pathway_genes:

            #duplicate the genes we will return, and append the new gene for checking
            genes_to_test = [i for i in genes_to_return]
            genes_to_test.append(pat_gene)
            #get the combined mutations for the genes for each patient
            mutation_number = mutation_load(genes_to_test, patient_mutation_data = patient_mutation_data)

            #now carry out the selected optimization parameter
            optimization_param = optimization_score_function(optimize_on = optimize_on, mutation_number = mutation_number, br_labels = br_labels, survival_time = survival_time, censor_status = censor_status)

            #test and see if the current optimization_param is larger than the largest optimization parameter
            #if checking for resistance, check if the current optimization parameter is smaller
            if resistance == True:


                if optimization_param < current_optimization:
                    #set the genes_increase marker to True, we found a gene that decreases the score
                    genes_increase = True
                    #if it is larger, and probability is set to True, then current_best_gene will be updated using a probability.
                    #the probability is set by taking 1/(number of genes wanted + the number of rounds already completed)


                    probability_set = 1/(number_of_genes+counter_token)
                    #now if the random number is less than or equal to the probability set, take the gene, otherwise do not take the gene
                    if random.random() <= probability_set:
                        current_optimization = optimization_param
                        current_best_gene = pat_gene
                        check_token = 1

            else:

                if optimization_param > current_optimization:
                    #set the genes_increase marker to True, we found a gene that increases the score
                    genes_increase = True
                    #if it is larger, and probability is set to True, then current_best_gene will be updated using a probability.
                    #the probability is set by taking 1/(number of genes wanted + the number of rounds already completed)


                    probability_set = 1/(number_of_genes+counter_token)
                    #now if the random number is less than or equal to the probability set, take the gene, otherwise do not take the gene
                    if random.random() <= probability_set:
                        current_optimization = optimization_param
                        current_best_gene = pat_gene
                        check_token = 1



        #first check if we actually took one of the genes. If not, and there are genes which increase the optimization parameter, then repeat the loop.
        if (genes_increase == True) and (check_token == 0):
            continue

        #this next part will execute if genes_increase == False or check_token == 1.
        #if genes_increase == False, it should mean we found no genes which increased the score
        #if check_token == 1, that means we pulled a gene, and so we need to check if it is better than the previous best optimization
        #if we have a gene now, and it increases our (or decreases for resistance) score, then we append it to our list
        if resistance == True:
            if current_optimization < previous_best_optimization:
                genes_to_return.append(current_best_gene)
            else:
                break

        else:
            if current_optimization > previous_best_optimization:
                genes_to_return.append(current_best_gene)
            else:
                break

        #if we did not find a gene which increases our score, break the while loop
        #and then set the counter_token if the loop is not broken
        counter_token += 1

    #this now checks if the optimization is correct for the new pathway. Should be the same as given above
    mutation_number = mutation_load(genes_to_return, patient_mutation_data = patient_mutation_data)
    #mutation_number = patient_mutation_data.loc[:,genes_to_return].sum(1)
    optimization_confirmation = optimization_score_function(optimize_on = optimize_on, mutation_number = mutation_number, br_labels = br_labels, survival_time = survival_time, censor_status = censor_status)

    assert abs(optimization_confirmation - current_optimization) < 0.05, "Optimizations are not the same, problem when checking optimizations"

    #finally, return the list of genes for the pathway and the associated optimization parameter
    return genes_to_return, optimization_confirmation


def mutation_load(gene_list, patient_mutation_data):
    '''
    This is a function to generate the mutation load of the selected genes on the patients. Currently, this is the sum of the mutations in the genes
    :param gene_list: a list of the genes to select
    :param patient_mutation_data: the patient mutation data
    :return: the sum of the mutations per patient
    '''
    return patient_mutation_data[list(set(gene_list))].sum(1)


def mutations_per_pathway_per_patient(mut_df, pathways_dict):

    '''
    This is a function to take in the patient mutation data and pathway dictionary and combine the two into a dataframe of the number of mutations per patient per pathway (pathway columns, patient rows)
    The function mutation_load is a more efficient way to do something similar on a single pathway
    :param mut_df: the mutation dataframe per patient. Patients are rows, genes are columns, entries are whether that gene has a mutation
    :param pathways_dict: a dictionary of each pathway as keys and their associated genes as values
    :return: a dataframe of mutations per patient per pathway
    '''

    # take the pathway dictionary and, for each pathway, sum the nuber of mutations based on the mutation dataframe.
    #output the result as a dataframe

    mut_pat_all = np.array([mutation_load(gene_list = list(pathways_dict.values())[k],patient_mutation_data = mut_df) for k in range(len(pathways_dict.values()))])

    #mut_pat_all = np.array([mut_df[(list(set(list(pathways_dict.values())[k])))].sum(1) for k in range(len(pathways_dict.values()))])
    pathway_df = pd.DataFrame(mut_pat_all, columns=mut_df.index)
    pathway_df.index = pathways_dict.keys()

    return pathway_df

def scores_df_generator(patient_information_df, mut_df, pathways_dict, save_as_csv = False, path_and_name_to_save_csv = None):

    '''
    This is a function to take in the patient mutation data, the patient information data, and the gene pathway data, and output a dataframe of the pathway, c_index, and roc_auc_score
    :param mut_df is the dataframe of mutation information per gene per patient.
    :param patient_information_df is the dataframe of the patient's clinical information
    :param pathways_dict is a dictionary of each pathway as keys and their associated genes as values
    :param save_as_csv set to True if you want to export as csv
    :param path_and_name_to_save_csv set to the path and name for the csv
    :return A dataset of pathways and their associated c_index and roc_auc_scores. Optionally a csv of the same.
    '''
    #get the mutations per pathway per patient dataframe
    pathway_mut_df = mutations_per_pathway_per_patient(mut_df, pathways_dict)

    #start new lists for recording scores
    c_index_list = []
    ras_list = []
    pathway_list = []

    #now run both the roc_auc_score function and the c_index function over all requested pathways
    for pathway in pathway_mut_df.index:
        if 'survival' in patient_information_df.keys():
            c_index_score = survival_outcome_c_index(mutations_count_pathway = pathway_mut_df.loc[pathway,:], survival_time = patient_information_df.loc[:,'survival'], censor_status = patient_information_df.loc[:,'vital_status'])
        else:
            c_index_score = [0.5 for i in range(patient_information_df.shape[0])]
        ras = br_outcome_roc_auc_score(mutations_count_pathway = pathway_mut_df.loc[pathway,:], br_labels = patient_information_df.loc[:,'BR'])

        c_index_list.append(c_index_score)
        ras_list.append(ras)
        pathway_list.append(pathway)

    #combine mutations and pathways into a single dataframe

    combined_df = pd.DataFrame([pathway_list, c_index_list,ras_list]).transpose()
    combined_df.columns = ['pathway','c_index','roc_auc_score']

    if save_as_csv == True:
        assert path_and_name_to_save_csv != None, "Please provide path and name for csv export"
        combined_df.to_csv(path_and_name_to_save_csv)

    return combined_df

def eval_pathway_mutation_load(mut_df_1,clin_df_1,pathways_dict,mut_df_2 = None,clin_df_2 = None, save_as_csv=False,path_and_name_to_save_csv_df_1=None, path_and_name_to_save_csv_df_2 = None, analyze_both = False, correlations = False):
    '''

    This function takes in one to two datasets, both mutation and clinical data for both, and returns the scores for each pathway.
    It can also return the correlations and pvalues for roc_auc_score and c_index for both pathways
    This can be used to evaluate new mutation pathways
    :param mut_df_1: the mutation data for the 1st dataset
    :param clin_df_1: the clinical data for the 1st dataset
    :param pathway_dict: the pathway dictionary, with pathways as keys and genes as values
    :param val_mut: the mutation data for the 2nd dataset (optional based on analyze_both)
    :param val_clin: the clinical data for the 2nd dataset (optional based on analyze_both)

    :param save_as_csv: Boolean for saving as csv, must provide path_and_name for the datasets you have if True
    :param path_and_name_to_save_csv_traning: path and name for saving the training csv results
    :param path_and_name_to_save_csv_validation: path and name for saving the validation csv results
    :param analyze_both: True if you want to look at both training and validation simultaneously, false if you just want to test one set of data
    :param correlations: Boolean, if True returns the spearmans rank correlations in addition to the other results, otherwise does not return correlations
    :return: depending on the input, can return training scores, training and validation scores, or training and validation scores with spearmans rank correlations and associated parameters
    '''


    scores_for_training = scores_df_generator(mut_df=mut_df_1, patient_information_df=clin_df_1,
                                         pathways_dict=pathways_dict, save_as_csv=save_as_csv,
                                         path_and_name_to_save_csv=path_and_name_to_save_csv_df_1)


    if analyze_both == True:

        scores_for_validation = scores_df_generator(mut_df=mut_df_2, patient_information_df=clin_df_2,
                                              pathways_dict=pathways_dict, save_as_csv=save_as_csv,
                                              path_and_name_to_save_csv=path_and_name_to_save_csv_df_2)

    if correlations == True:

        rho_c_index, pval_c_index = spearmanr(scores_for_validation.loc[:, 'c_index'], scores_for_training.loc[:, 'c_index'])
        rho_ras, pval_ras = spearmanr(scores_for_validation.loc[:, 'roc_auc_score'], scores_for_training.loc[:, 'roc_auc_score'])

        return scores_for_training,scores_for_validation, rho_c_index, pval_c_index, rho_ras, pval_ras
    if analyze_both == True:
        return scores_for_training, scores_for_validation
    else:
        return scores_for_training

def filter_pathways_for_feature_selection(scores_training,scores_validation,filterby, n_pathways = 100,diff=0, sort_by = 'training', return_combined = False):
    '''
    This function takes in the scores for the validation and training sets and returns the top and bottom pathways of those datasets (they must be the same direction, either above 0.5 or below 0.5, to be returned).
    This was ultimately not used for selecting pathways for feature selection
    :param scores_training: this is the scores of the training dataset.
    :param score_validation: this is the scores of the validation dataset
    :param filterby: this is the column to use to filter and sort, either roc_auc_score or c_index in the current implementation (9/28/2021)
    :param n_pathways: The final number of pathways to return, will return the top n pathways by sorting and bottom n pathways (so total pathways will be double what was put)
    :param diff: This is used to further filter the pathways, choosing only the pathways that have a validation and training set difference smaller than the specified amount.
    :param sort_by: 'both','training','validation' sort by either the training, validation, or both scores
    :param return_all: Boolean to return either just the top and bottom same direction pathways or the entire dataframe
    :return: The top n_pathways above 0.5 direction and bottom n_pathways below 0.5 direction
    '''
    combined_scores = pd.DataFrame(
        [scores_validation['pathway'], scores_training[filterby], scores_validation[filterby]]).transpose()

    combined_scores.columns = ['pathway', 'training_'+filterby, 'validation_'+filterby]
    combined_scores=combined_scores.set_index('pathway')

    combined_scores['diff'] = combined_scores.loc[:, 'validation_'+filterby] - combined_scores.loc[:, 'training_'+filterby]

    top_differences = combined_scores[abs((combined_scores['diff']) < diff)]
    top_differences = top_differences.sort_values(by='training_'+filterby, axis=0, ascending=True)

    top_differences=top_differences[(top_differences['training_'+filterby] > 0.5) == (top_differences['validation_'+filterby] > 0.5)]

    if sort_by == 'both':
        sorting = ['training_'+filterby, 'validation_'+filterby]
    elif sort_by =='training':
        sorting = ['training_'+filterby]
    elif sort_by == 'validation':
        sorting = ['validation_'+filterby]

    sorted_scores = top_differences.sort_values(
        by=sorting, axis=0, ascending=False)

    above_n = min(sorted_scores[sorted_scores['training_'+filterby]>0.5].shape[0],n_pathways)
    below_n = min(sorted_scores[sorted_scores['training_'+filterby] < 0.5].shape[0], n_pathways)



    if return_combined == False:
        pat_return_top = sorted_scores.head(above_n).reset_index()
        pat_return_below = sorted_scores.tail(below_n).reset_index()
        return pat_return_top, pat_return_below
    else:
        pat_return = pd.concat([sorted_scores.head(above_n), sorted_scores.tail(below_n)]).reset_index()
        return pat_return

def full_pathway_top_bottom_scores(train_mut, train_clin, val_mut, val_clin, pathway_dict, n_pathways = 100, diff = 0, sort_by = 'training', return_which = 'both'):
    '''
    This is a function to generate the top and bottom combined pathways for later analysis
    :param train_mut: the training mutation DataFrame
    :param train_clin: the training clinical DataFrame
    :param val_mut: the validation mutation dataframe
    :param val_clin: the validation clinical dataframe
    :param pathway_dict: the pathway dictionary to analyze
    :param n_pathways: number of pathways to return for each optimization metric and each direction (so n_pathways*4 max pathways)
    :param diff: the difference in the pathways to select, change if you want to select only pathways that differ by a certain amount
    :param sort_by: sort the dataframe and select the top amounts either by sorting on the training, validation, or both
    :param return_which: return 'roc_auc_score', 'c_index', or both combined dataframe
    :return: returns a dataframe of the top pathways and a dataframe of the bottom pathways, with chosen evaluation parameters
    '''
    train_score, val_score = eval_pathway_mutation_load(train_mut=train_mut,train_clin=train_clin,val_mut=val_mut,val_clin=val_clin,GO_dict=pathway_dict)
    top_pathways_ras, bottom_pathways_ras = filter_pathways_for_feature_selection(scores_training = train_score,scores_validation = val_score,filterby = 'roc_auc_score', n_pathways = 100,diff=0, sort_by = 'training', return_combined = False)
    top_pathways_c_index, bottom_pathways_c_index = filter_pathways_for_feature_selection(scores_training = train_score,scores_validation = val_score,filterby = 'c_index', n_pathways = 100,diff=0, sort_by = 'training', return_combined = False)


    if return_which == 'both':
        top_pathways_combined = top_pathways_c_index.append(top_pathways_ras).reset_index()
        bottom_pathways_combined = bottom_pathways_c_index.append(bottom_pathways_ras).reset_index()
        return top_pathways_combined, bottom_pathways_combined

    elif return_which == 'roc_auc_score':
        return top_pathways_ras, bottom_pathways_ras
    elif return_which == 'c_index':
        return top_pathways_c_index, bottom_pathways_c_index
    else:
        print('Provide either both, roc_auc_score, or c_index as values for return_which')
        return

def multiple_pathway_forward_selector(pathway_list, pathway_dict, number_of_genes, patient_mutation_data, optimize_on, br_labels, survival_time, censor_status, probability = False, resistance = False, csv_save_pathway_scores_name = None, csv_save_pathway_dictionary_name = None):
    '''
    This is a function to carry out forward selection on multiple pathways. It will return a dataframe of pathways and the new forward selected scores for both roc_auc_score and c_index
    It will also return a dictionary for pathways by c_index and roc_auc_score with the selected genes for each optimization type
    :param pathwayList: is the pathway list to itearte over to select the n genes for each pathway
    :param pathway_dictionary is a dictionary of the ALL of the pathways as keys and ALL their associated genes as values
    :param number_of_genes is the number of genes to return per pathway. If there are less than that number of genes in the pathway, all genes will be returned.
    :param patient_mutation_data is the dataframe of patient mutation information by gene. Gene information is column, patient data is index
    :param optimize_on is the optimization parameter (either roc_auc_score or c_index) that will be maximized to use to select the genes
    :param br_labels is the input from the BR column of the datasets, these have 5 values. Values are converted into either positive or negative. Only need this parameter when optimizing using roc_auc_scores
    :param survival_time: is the survival time in months (though actual label of time does not seem relevant, just make sure they are uniform in the dataset). Only need this parameter when optimizing on c_index.
    :param censor_status: is whether the patient is alive or dead. Alive patients will be censored (0 in event_observed for c_index). Only need this parameter when optimizing on c_index.
    :param probability: whether to use the greedy or probabilistic forward selector. True if you want to use the probabilistic, false if you want the greedy
    :param resistance: False is if predicting respones True if predicting resistance
    :param csv_save_pathway_scores_name: name of csv for the pathways and scores dataframe, will not export if None
    :param csv_save_pathway_dictionary_name: name of csv for the pathways and genes dictionary, will not export if None
    :returns: a dataframe of the pathways and their optimization scores, and a dictionary of the pathways and their new genes
    '''

    #first do roc_auc_score greedy or probabilistic forward selectors
    new_pathways_dict_ras = {}
    scores_list_ras = []
    pathways_list_ras = []

    for pathway in pathway_list:

        #need a check to see if there is an empty list. If it is empty, then append nan as the score and append the pathway to the list, then continue the loop
        if not pathway_dict[pathway]:
            scores_list_ras.append(np.nan)
            pathways_list_ras.append(pathway)
            continue


        if probability == False:
            #if not carrying out probability selection, then find the new genes and scores from the greedy_forward_selector function, and append those to lists and dictionaries for later use
            new_pathway_genes, new_pathway_score = greedy_forward_selector(pathway = pathway, pathway_dictionary = pathway_dict, number_of_genes = number_of_genes, patient_mutation_data = patient_mutation_data, optimize_on = optimize_on, br_labels = br_labels, survival_time = survival_time, censor_status = censor_status, resistance = resistance)
            new_pathways_dict_ras[pathway] = new_pathway_genes
            scores_list_ras.append(new_pathway_score)
            pathways_list_ras.append(pathway)
        else:
            #same thing as previous, but if you want to use the probabilistic_forward_selector function instead
            new_pathway_genes, new_pathway_score = probabilistic_forward_selector(pathway, pathway_dictionary = pathway_dict, number_of_genes = number_of_genes, patient_mutation_data = patient_mutation_data, optimize_on = optimize_on, br_labels = br_labels, survival_time = survival_time, censor_status = censor_status, resistance = resistance)
            new_pathways_dict_ras[pathway] = new_pathway_genes
            scores_list_ras.append(new_pathway_score)
            pathways_list_ras.append(pathway)

    #create the dataframe from the scores and pathways
    pathways_ras_scores = pd.DataFrame([pathways_list_ras, scores_list_ras]).transpose()
    pathways_ras_scores.columns = ['pathways','forward_selected_' + optimize_on +'_scores']

    #save as csv. The dictionaries will also be saved as csv. Use the function read_csv_ditionary in handle_data.py to read in the dictionary again
    if csv_save_pathway_scores_name != None:
        pathways_ras_scores.to_csv(csv_save_pathway_scores_name)

    if csv_save_pathway_dictionary_name != None:
        pd.DataFrame(dict([(key,pd.Series(value)) for key,value in new_pathways_dict_ras.items() ])).to_csv(csv_save_pathway_dictionary_name)

    return pathways_ras_scores, new_pathways_dict_ras

def evaluate_fitness(current_pop_array, all_genes, mut_data, br_data):
    '''
    This is a function to evaluate the fitness of the entire population, by each pop (member of the population)
    :param current_pop_array: the current population array, with 1 for gene presence or 0 for gene absence
    :param mut_data: the training mutation data
    :param br_data: the training br data, should be a series or array or list
    :return: returns a list of fitness scores per row
    '''
    #iterate over the array and calculate mutations_per_pathway_per patient and then fitness, based on roc_auc_score
    fitness_score_list = []
    for row in current_pop_array:
        #get the pathway genes for this pop
        pathway_information = list([j for i,j in zip(row, all_genes) if i>0])
        #then calculate the score
        mutation_count = mutation_load(pathway_information, mut_data)
        fitness_score = br_outcome_roc_auc_score(mutations_count_pathway = mutation_count, br_labels = br_data)
        fitness_score_list.append(fitness_score)

    return fitness_score_list

def reproduction_crossover(reproduction_array, number_of_pops):
    '''
    This takes the reproduction array and produces a new evaluation array by creating 'offspring' of two randomly selected individuals
    The new offspring will have randomly selected genes from the parent
    :param reproduction_array: a numpy array of the pops to reproduce
    :param number_of_pops: the final size of the population
    :return: an array of number_of_pops rows, half of which are the reproductive progeny of the parents in the reproduction array
    '''
    #calculate the number of children to produce
    total_children = number_of_pops - reproduction_array.shape[0]

    output_array = reproduction_array.copy()
    #iterate over the number of children, randomly select parents, and then randomly select genes from those parents
    for children in range(0,total_children):
        #create empty list for the children's genes
        children_gene = []
        #find parents randomly
        parents_indices = np.random.choice(list(range(0,reproduction_array.shape[0])), size = 2, replace = False)
        #select parents out of the reproduction array, with replacement
        parents = reproduction_array[parents_indices,:]
        #and select the genes for the children from the parents
        probs = [np.random.choice([0,1], p = [0.5,0.5]) for i in range(len(parents[0,:]))]

        children_gene = [parents[0,i] if probs[i] == 0 else parents[1,i] for i in range(len(parents[0,:]))]

        #then append the children to the output array
        output_array = np.append(output_array,[children_gene], axis = 0)

    return output_array

def genetic_algorithm(pathway_genes, mut_data, br_data, number_of_pops = 20,  number_of_selection_iterations = 10, probability = 0.1, resistance = False):
    '''
    This is a genetic algorithm to determine the best genes in a pathway to optimize BR roc_auc_score
    :param pathway_genes: A list of genes in the pathway being tested
    :param mut_data: mutation data for the patients
    :param br_data: the BR data for the patients
    :param number_of_pops: the number of pops in the population array to select, default is 20
    :param number_of_selection_iterations: the number of generations of the genetic algorithm, AKA the number of selection loops
    :param probability: the probability of a pop initializing with a gene. Each gene has the same probability of being initialized.
    :return: The genes selected in the pathway, and the fitness score of those selected genes
    '''
    #randomly create a population array
    population_array = np.random.choice([0,1], size = (number_of_pops, len(pathway_genes)), p = [1-probability, probability])
    #iterate, which is esentially the number of 'generations' to run the genetic algorithm
    for iteration in range(0,number_of_selection_iterations):

        #generate a fitness list
        fitness_list = evaluate_fitness(current_pop_array = population_array, all_genes = pathway_genes, mut_data = mut_data, br_data = br_data)
        #get the samples above the median
        median_score = np.median(fitness_list)
        #create an array of parents for reproduction, if they are above the median score
        #get ones below median score if you want resistance
        if resistance == False:
            reproduction_array = np.array([i for i,j in zip(population_array, fitness_list) if j >= median_score])
        if resistance == True:
            reproduction_array = np.array([i for i,j in zip(population_array, fitness_list) if j <= median_score])
        #if reproduction array is greater than half the population array, break the loop, we have iterated too much
        if np.shape(reproduction_array)[0] > np.shape(population_array)[0]/2:
            break

        #repopulate the array with children and the parents
        population_array = reproduction_crossover(reproduction_array = reproduction_array, number_of_pops = number_of_pops)

    #generate a final fitness list
    final_fitness_list = evaluate_fitness(current_pop_array = population_array, all_genes = pathway_genes, mut_data = mut_data, br_data = br_data)
    #select the top sample.  If it converges, then there will be multiple top samples of the same pathway with the same genes, so take the first
    #reverse this if resistance = True
    if resistance == False:
        top_sample = [i for i,j in zip(population_array, final_fitness_list) if j == max(final_fitness_list)]
        top_sample = top_sample[0]
    if resistance == True:
        top_sample = [i for i,j in zip(population_array, final_fitness_list) if j == min(final_fitness_list)]
        top_sample = top_sample[0]
    #get the genes for the top sample
    gene_names_top_sample = list([j for i,j in zip(top_sample, pathway_genes) if i>0])
    #return gene names and the fitness score for that new pathway
    #return the max final fitness list if resistance == false
    #Return the min if resistance == True
    if resistance == False:
        return gene_names_top_sample, max(final_fitness_list)
    if resistance == True:
        return gene_names_top_sample, min(final_fitness_list)

def multiple_pathway_genetic_algorithm(pathways_list, pathways_dict, mut_data, br_data, number_of_pops = 20, number_of_selection_iterations = 10, probability = 0.1, resistance = False, save_as_csv = False, csv_save_pathway_scores_name = None,csv_save_pathway_dictionary_name = None):
    '''
    This is a function to run multiple genetic algorithms over all of the pathways in a list. It will return the best pathway definition to maximize BR AUC score.

    :param pathways_list: A list of pathways to test
    :param pathways_dict: a dictionary of the pathways as keys and the each pathway's genes as values
    :param mut_data: mutation data for the patients
    :param br_data: the BR data for the patients
    :param number_of_pops: the number of pops in the population array to select, default is 20
    :param number_of_selection_iterations: the number of generations of the genetic algorithm, AKA the number of selection loops
    :param probability: the probability of a pop initializing with a gene. Each gene has the same probability of being initialized.
    :param resistance: whether to maximize or minimize the fitness score. Resistance == False to maximize, Resistance = False to minimize
    :return: a dataframe with a column of pathways and a column of fitness values per pathway, and a dictionary with pathways as keys and the new pathway's selected genes as values
    '''

    new_pathway_dict = {}
    new_scores_list = []
    pathways_names = []

    #iterate over each pathway, run the genetic algorithm, and save the new genes and scores
    for pathway in pathways_list:
        new_genes, new_score = genetic_algorithm(pathway_genes = pathways_dict[pathway], mut_data = mut_data, br_data = br_data, number_of_pops = 20,  number_of_selection_iterations = 10, probability = 0.1, resistance = resistance)
        new_pathway_dict[pathway] = new_genes
        new_scores_list.append(new_score)
        pathways_names.append(pathway)

    #create dataframe with pathways and scores
    pathway_scores_df = pd.DataFrame([pathways_names, new_scores_list]).transpose()
    pathway_scores_df.columns = ['pathway','roc_auc_score']

    #output the dataframe and dictionary to a csv
    if save_as_csv == True:
        if csv_save_pathway_scores_name != None:
            pathway_scores_df.to_csv(csv_save_pathway_scores_name)
        #the dictionary will be output as a csv, can be reconstituted into dictionary using read_csv_dictionary from handle_data.py file
        if csv_save_pathway_dictionary_name != None:
            pd.DataFrame(dict([(key,pd.Series(value)) for key,value in new_pathway_dict.items() ])).to_csv(csv_save_pathway_dictionary_name)

    return pathway_scores_df, new_pathway_dict


def shared_genes(pathways, pathway_dictionary, pathway_gene_list = False):
    '''
    This function takes a list of pathways and a pathway dictionary and outputs a dataframe of the number of pathways that have the gene and a binary numpy array that indicates whether the gene is in the pathway
    :param pathways: a list of pathways
    :param pathway_dictionary: A dictionary of the pathways and their genes, usually the result of some feature selection on the genes
    :param pathway_gene_list: a list of genes to examine in the pathways. If provided, will only look at those genes when outputting array or dataframe.
    :return: a pandas dataframe with genes in one column and the number of pathways which has that gene as rows
    :return: a numpy array with columns as genes and rows as pathways, 1 indicates presence of the gene in that pathway
    '''
    top_genes_list = []
    #if pathway dictionary is provided, iterate over the pathways and create a list of all of the genes.
    if type(pathway_gene_list) == bool:
        for k in pathways:
            new_list = list(set([x for x in pathway_dictionary[k]]))
            top_genes_list.extend(new_list)
    else:
        top_genes_list = pathway_gene_list

    #remove duplicates
    top_genes_list=list(set(top_genes_list))
    #now iterate over the pathways and find the genes which are present in that pathway, and create a numpy array
    boolean_gene_list = []
    for pathway in pathways:
        boolean_list = list([0 if x not in pathway_dictionary[pathway] else 1 for x in top_genes_list])
        boolean_gene_list.append(boolean_list)

    boolean_gene_array = np.array(boolean_gene_list)

    #also make into a dataframe
    pathway_count = boolean_gene_array.sum(axis = 0)
    output_df = pd.DataFrame([top_genes_list,pathway_count]).transpose()
    output_df.columns = ['gene','number_of_pathways']

    return output_df, boolean_gene_array



def train_all_pathway_mutation_load_predictors(dictionary = GO_dict_intersection):
    '''
    :return:
    '''
    GO_top_pathways_new_scores, GO_top_pathways_new_dict = multiple_pathway_forward_selector(pathway_list =  dictionary.keys(), pathway_dict = dictionary, number_of_genes = 10, optimize_on = 'roc_auc_score', patient_mutation_data = liu_mut, br_labels = liu_clin.loc[:,'BR'], survival_time = liu_clin.loc[:, 'survival'], censor_status = liu_clin.loc[:, 'vital_status'], probability = False, resistance = False, csv_save_pathway_scores_name = '../20_Intermediate_Files/GO_top_pathway_new_scores_names.csv', csv_save_pathway_dictionary_name = '../20_Intermediate_Files/GO_top_pathway_gene_names.csv')

    ##Run greedy forward to find response pathways
    GO_top_pathways_new_scores_resistance, GO_top_pathways_new_dict_resistance = multiple_pathway_forward_selector(
        pathway_list = dictionary.keys(),
        pathway_dict = dictionary,
        number_of_genes = 10,
        optimize_on = 'roc_auc_score',
        patient_mutation_data = liu_mut,
        br_labels = liu_clin.loc[:,'BR'],
        survival_time = liu_clin.loc[:, 'survival'],
        censor_status = liu_clin.loc[:, 'vital_status'],
        probability = False,
        resistance = True,
        csv_save_pathway_scores_name = '../20_Intermediate_Files/GO_top_pathway_new_scores_names_resistance.csv',
        csv_save_pathway_dictionary_name = '../20_Intermediate_Files/GO_top_pathway_gene_names_resistance.csv'
    )

    ##Run prob forward to find response pathways
    GO_top_pathways_new_scores_randomized, GO_top_pathways_new_dict_randomized = multiple_pathway_forward_selector(
        pathway_list = dictionary.keys(),
        pathway_dict = dictionary,
        number_of_genes = 10,
        optimize_on = 'roc_auc_score',
        patient_mutation_data = liu_mut,
        br_labels = liu_clin.loc[:,'BR'],
        survival_time = liu_clin.loc[:, 'survival'],
        censor_status = liu_clin.loc[:, 'vital_status'],
        probability = True,
        resistance = False,
        csv_save_pathway_scores_name = '../20_Intermediate_Files/GO_top_pathway_new_scores_names_randomized.csv',
        csv_save_pathway_dictionary_name = '../20_Intermediate_Files/GO_top_pathway_gene_names_randomized.csv'
    )


    ######GA:
    ##Run GA to find response pathways
    GO_genetic_pathway_scores, GO_genetic_dict = multiple_pathway_genetic_algorithm(
        pathways_list = list(dictionary.keys()),
        pathways_dict = dictionary,
        mut_data = liu_mut,
        br_data = liu_clin.loc[:,'BR'],
        save_as_csv = True,
        csv_save_pathway_scores_name = '../20_Intermediate_Files/GO_genetic_algorithm_pathways_scores.csv',
        csv_save_pathway_dictionary_name = '../20_Intermediate_Files/GO_genetic_algorithm_pathway_dictionary.csv'
    )
