#These are the scripts for running the genetic algorithm selector maximization
#this looks like it should be folded into another .py, not sure which though


# Imports --------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from optimize_pathways import *
from feature_selection_pathway_mutation_load import *
from handle_data import *

# Data -----------------------------------------------------------------------------------------------------------------

liu_mut = load_pickle(SAVE_DATA+'liu_mut.pickle')
liu_clin = load_pickle(SAVE_DATA + 'liu_clin.pickle')
GO_dict_intersection = load_pickle(SAVE_DATA + 'GO_dict_intersection.pickle')


# Scripts -----------------------------------------------------------------------------------------------------------------


def train_all_pathway_mutation_load_predictors():
    '''

    :return:
    '''
    GO_top_pathways_new_scores, GO_top_pathways_new_dict = multiple_pathway_forward_selector(pathway_list =  GO_dict_intersection.keys(), pathway_dict = GO_dict_intersection, number_of_genes = 10, optimize_on = 'roc_auc_score', patient_mutation_data = liu_mut, br_labels = liu_clin.loc[:,'BR'], survival_time = liu_clin.loc[:, 'survival'], censor_status = liu_clin.loc[:, 'vital_status'], probability = False, resistance = False, csv_save_pathway_scores_name = '../20_Intermediate_Files/GO_top_pathway_new_scores_names.csv', csv_save_pathway_dictionary_name = '../20_Intermediate_Files/GO_top_pathway_gene_names.csv')

    ##Run greedy forward to find response pathways
    GO_top_pathways_new_scores, GO_top_pathways_new_dict= multiple_pathway_forward_selector(
        pathway_list = GO_dict_intersection.keys(),
        pathway_dict = GO_dict_intersection,
        number_of_genes = 10,
        optimize_on = 'roc_auc_score',
        patient_mutation_data = liu_mut,
        br_labels = liu_clin.loc[:,'BR'],
        survival_time = liu_clin.loc[:, 'survival'],
        censor_status = liu_clin.loc[:, 'vital_status'],
        probability = False,
        resistance = False,
        csv_save_pathway_scores_name = '../20_Intermediate_Files/GO_top_pathway_new_scores_names.csv',
        csv_save_pathway_dictionary_name = '../20_Intermediate_Files/GO_top_pathway_gene_names.csv'
    )

    ##Run prob forward to find response pathways
    GO_top_pathways_new_scores_randomized, GO_top_pathways_new_dict_randomized = multiple_pathway_forward_selector(
        pathway_list = GO_dict_intersection.keys(),
        pathway_dict = GO_dict_intersection,
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
        pathways_list = list(GO_dict_intersection.keys()),
        pathways_dict = GO_dict_intersection,
        mut_data = liu_mut,
        br_data = liu_clin.loc[:,'BR'],
        save_as_csv = True,
        csv_save_pathway_scores_name = '../20_Intermediate_Files/GO_genetic_algorithm_pathways_scores.csv',
        csv_save_pathway_dictionary_name = '../20_Intermediate_Files/GO_genetic_algorithm_pathway_dictionary.csv'
    )
