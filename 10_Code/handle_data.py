# This is a python script to carry out preliminary EDA and PCA on cancer data from Riaz and Liu

# Imports --------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


# Path/constants -------------------------------------------------------------------------------------------------------

INTERMEDIATE_FILE = '../20_Intermediate_Files/'
SAVE_DATA = '../05_pickled_data/'

# Data -----------------------------------------------------------------------------------------------------------------
def save_pickle(var, name):
    '''

    :param var:
    :param name:
    :return:
    '''
    with open(SAVE_DATA+name+'.pickle', "wb") as handle: pickle.dump(var, handle)

def load_pickle(path):
    '''

    :param path:
    :return:
    '''
    with open(SAVE_DATA+path, "rb") as handle: ret = pickle.load(handle)
    return ret

def load_save_data():
    ##LOAD LIU
    liu_clin = pd.read_csv('../00_Data/liu_clin.csv')
    liu_clin = liu_clin.set_index('Patient')
    liu_mut = pd.read_csv('../00_Data/liu_mut_corrected_genes.csv')
    liu_mut = liu_mut.set_index('sample_id')
    GET_INTERSECT = lambda x: list(set(liu_mut.keys()) & set(x))

    ##LOAD RIAZ
    #import riaz_mut and riaz_clin
    riaz_mut = pd.read_csv('../00_Data/riaz_mut.csv')
    riaz_mut = riaz_mut.set_index('sample_id')
    riaz_clin = pd.read_csv('../00_Data/riaz_clin.csv')
    riaz_clin = riaz_clin.set_index('Patient')
    riaz_clin = riaz_clin.rename(columns={'Response': 'BR'})

    GO = pd.read_csv('../00_Data/GO.csv',low_memory=False)
    # drop the link column
    GO_drop = GO.drop(columns=['link'])
    # tranpsose
    GO_drop_t = GO_drop.transpose()
    # and set first row to columns and drop superfluous row
    GO_drop_t.columns = GO_drop_t.iloc[0]
    GO_drop_t = GO_drop_t.drop(index='pathway')

    # dictionary mapping from pathway to genes, remove nans
    GO_dict = {GO_drop_t.keys()[i]: [v for v in list(set(GO_drop_t[GO_drop_t.keys()[i]])) if v!='nan'] for i in range(len(GO_drop_t.keys()))}
    # returning genes also in Liu
    GO_dict_intersection = {}

    for pathway in GO_dict.keys():
            GO_dict_intersection[pathway] = list(set(liu_mut.keys()) & set(riaz_mut.keys()) & set(GO_dict[pathway]))

    #now load in manually curated (MC) pathway and find the intersection with the two datasets, just like the GO_dict
    MC = pd.read_csv('../00_Data/pathways.csv')

    MC_dict = {MC.keys()[i]: [v for v in list(set(MC[MC.keys()[i]])) if v!='nan'] for i in range(len(MC.keys()))}

    MC_dict_intersection = {}

    for pathway in MC_dict.keys():
        MC_dict_intersection[pathway] = list(set(liu_mut.keys()) & set(riaz_mut.keys()) & set(MC_dict[pathway]))


    ##Only 5 things we are using really, this helps making sure only these are used
    save_pickle(liu_mut,'liu_mut')
    save_pickle(liu_clin, 'liu_clin')
    save_pickle(riaz_mut, 'riaz_mut')
    save_pickle(riaz_clin, 'riaz_clin')
    save_pickle(GO_dict_intersection, 'GO_dict_intersection')
    save_pickle(MC_dict_intersection, 'MC_dict_intersection')

def load_save_msk():
    msk_clin = pd.read_csv('../00_Data/msk_tmb_clin.csv')
    msk_clin = msk_clin.set_index('SAMPLE_ID')
    msk_mut = pd.read_csv('../00_Data/msk_tmb_mut.csv')
    msk_mut = msk_mut.set_index('Tumor_Sample_Barcode')
    msk_clin['death'] = msk_clin['vital_status'] == 'Alive'
    save_pickle(msk_mut, 'msk_mut')
    save_pickle(msk_clin, 'msk_clin')

def read_csv_dictionary(path_name):
    '''
    This function reads in a csv of the dictionary of pathways and associated genes and outputs a dictionary
    :param path_name: the path and name of csv to recover as a pathway dictionary
    '''
    new_dict = pd.read_csv(path_name)
    try:
        new_dict = new_dict.drop(columns = ['Unnamed: 0'])
    except:
        pass
    # and set first row to columns and drop superfluous row

    # dictionary mapping from pathway to genes, remove nans
    return {new_dict.keys()[i]: [v for v in list(set(new_dict[new_dict.keys()[i]])) if type(v) is not float] for i in range(len(new_dict.keys()))}

def extract_cancers_msk(msk_clin,msk_mut, cancer_names):
    """
    Extract mutation and clinical data for a specific cancer type.

    :param cancer_data: tcga data

    :param cancer_names: cancer study name abbreviations of TCGA

    :param fields: select all or less of 'clinical_table', 'mutation_table','arm_table'

    :return: copied data
    ToDo: assert clinical table is added
    """
    clin = msk_clin.copy()
    mut = msk_mut.copy()


    sample_ids = sorted(set(
            [clin.index[j] for j, i in enumerate(clin['type']) if
             i in cancer_names]))
    # extract sample_ids based on the specific disease
    assert sample_ids

    mut = mut.loc[sample_ids,:]
    clin = clin.loc[sample_ids, :]
    clin['vital_status'] = clin['death']

    return clin,mut

def new_pathway_test_intersection(pathways_to_intersect, pathway_dictionary, test_genes_list):
    '''
    This is a function to take in the pathways and subset the genes in that pathway to only the genes present in the test dataset
    This is in preparation for checking the test dataset. This works for both the original pathways and pathways from the FS and GA algorithms
    :param pathways_to_intersect: a list of the pathways to intersect using this function
    :param pathway_dictionary: a dictionary of the genes is the pathways (can also be constructed from the FS or GA algorithms)
    :param test_genes_list: a list of the genes present in the test dataset
    :return: a dictionary of the pathways and their genes
    '''
    new_intersection_dictionary = {}
    fraction_list = []

    for pathway in pathways_to_intersect:
        new_intersection = list(np.unique(set(pathway_dictionary[pathway])&set(test_genes_list)))
        new_intersection_dictionary[pathway] = new_intersection
        #get the fraction of genes which were removed in the pathway
        if len(set(pathway_dictionary[pathway])) == 0:
            fraction = np.nan
        else:
            fraction = len(new_intersection)/len(set(pathway_dictionary[pathway]))
        fraction_list.append(fraction)

    return new_intersection_dictionary, fraction_list


def load_save_hugo():
    '''
    This makes the pickle for the hugo test datasets, and the dictionaries for the intersections.
    The intersection dictionaries were created in the file test_genes_train_val_analysis.py
    '''
    hugo_mut = pd.read_csv('../00_Data/hugo_mut.csv')
    hugo_mut = hugo_mut.set_index('sample_id')
    hugo_clin = pd.read_csv('../00_Data/hugo_clin.csv')
    hugo_clin = hugo_clin.set_index('sample_id')
    #rename response column to 'BR'
    hugo_clin = hugo_clin.rename(columns = {'Response':'BR'})
    GO_dict_intersection = load_pickle(SAVE_DATA + 'GO_dict_intersection.pickle')

    GO_test_genes_dict_intersection, f_list = new_pathway_test_intersection(list(GO_dict_intersection.keys()),GO_dict_intersection,hugo_mut.columns)

    save_pickle(GO_test_genes_dict_intersection, 'GO_test_genes_dict_intersection')
    save_pickle(hugo_mut,'hugo_mut')
    save_pickle(hugo_clin, 'hugo_clin')


def process_data_main():
    load_save_data()
    load_save_msk()
    load_save_hugo()
#     #ToDo: Make main function that runs all data processing and saving
