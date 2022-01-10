#this is a python file to investigate neural networks

# Imports --------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from optimize_pathways import *
from feature_selection_pathway_mutation_load import *
from handle_data import *
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

# Data -----------------------------------------------------------------------------------------------------------------

liu_mut = load_pickle(SAVE_DATA+'liu_mut.pickle')
liu_clin = load_pickle(SAVE_DATA + 'liu_clin.pickle')
riaz_mut_val = load_pickle(SAVE_DATA + 'riaz_mut.pickle')
riaz_clin_val = load_pickle(SAVE_DATA + 'riaz_clin.pickle')
GO_dict_intersection = load_pickle(SAVE_DATA + 'GO_dict_intersection.pickle')

# Functions -----------------------------------------------------------------------------------------------------------------

def save_model(model,model_path):
    '''
    saves Keras model : lets save in a specific directory for saved models
    :param model:
    :param path_to_save:
    '''
    model.save(model_path+'.h5')

def load_model(model_path):
    """
    Loads Keras model.
    :param model_path: Path to H5 model.
    :return: Keras model.
    """
    model_loaded = load_model(model_path)
    return model_loaded

def prep_data_lstm(mut_data,clin_data,pathways_dict,pathway,sorted_data = None):
    '''

    :param mut_data:
    :param clin_data:
    :param pathways_dict:
    :param pathway:
    :param sorted_data:
    :return:
    '''
    if sorted_data is None:
        sorted_data = mutation_data_sorter(mut_data, pathways_dict, pathway)

    X = np.array(mut_data[sorted_data.columns])
    x1 = X.reshape(X.shape[0], 1, X.shape[1])
    y = np.array([float(i == 'CR' or i == 'PR') for i in clin_data['BR']])
    y = y.reshape(y.shape[0], 1)
    y1 = keras.utils.to_categorical(y, num_classes=2)

    return x1,y1,sorted_data

def prep_data_fnn(mut_data,clin_data,pathways_dict,pathway):
    '''

    :param mut_data:
    :param clin_data:
    :param pathways_dict:
    :param pathway:
    :return:
    '''
    x1 = np.array(mut_data[pathways_dict[pathway]])
    y = np.array([float(i == 'CR' or i == 'PR') for i in clin_data['BR']])
    y1 = y.reshape(y.shape[0], 1)
    return x1,y1

def train_fnn(pathways_dict, pathway, train_mut=liu_mut, train_clin=liu_clin):
    '''

    :param pathways_dict:
    :param pathway:
    :param train_mut:
    :param train_clin:
    :return:
    '''
    model2 = Sequential()
    optimizer = Adam()
    x1, y1 = prep_data_fnn(train_mut,train_clin,pathways_dict,pathway)
    model2.add(Dense(5, input_dim=x1.shape[1], activation='sigmoid'))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model2.fit(x1, y1, epochs=100, batch_size=27, verbose=0)

    return model2

def train_lstm(pathways_dict,pathway,train_mut=liu_mut,train_clin=liu_clin):
    '''

    :param pathways_dict:
    :param pathway:
    :param train_mut:
    :param train_clin:
    :return:
    '''

    x1,y1,sorted_data = prep_data_lstm(train_mut,train_clin,pathways_dict, pathway)

    layer1 = LSTM(5, input_shape=(1, np.array(sorted_data).shape[1]))
    output = Dense(2, activation="softmax", name="Output")
    optimizer = Adam()
    model = Sequential()
    model.add(layer1)
    model.add(output)
    model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x1, y1, epochs=100, batch_size=27, verbose=0)

    return model,sorted_data

def mutation_data_sorter(mutation_data, pathway_dictionary, pathway):
    '''
    This function takes a dataframe of mutation data, a pathway, and a pathway dictionary and outputs a dataframe with the genes sorted according to how many mutatios that gene had between all of the patients
    :param mutation_data: the mutation data to organize
    :param pathway_dictionary: a dictionary of pathways as keys and genes in that pathway as values
    :param pathway: the pathway to sort
    :return: a dataframe with the genes with the highest number of mutations over the patients in the left column, and genes with sequentially less mutations in the remaining columns
    '''
    xtest = mutation_data[pathway_dictionary[pathway]]
    gene_numbers = xtest.sum(0)
    sorted_genes = gene_numbers.sort_values(ascending = False)
    xfinal = xtest[sorted_genes.index]

    return xfinal

def lstm_multiple_pathways(pathways, pathways_dict, train_mut, train_clin, val_mut, val_clin, test_mut = None, test_clin = None, csv_path_name = None, return_model_and_curve = False, save_model_path = None, save_model_ending = ''):
    '''
    This is a function to run multiple pathways through lstm neural networks.
    :param pathways: a list of pathways to run through the lstm models
    :param train_mut: the training mutation data
    :param train_clin: the traning clinical data
    :param val_mut: the validation mutation data
    :param val_clin: the validation clinical data
    :param test_clin: optional test dataset clinical data
    :param test_mut: optional test dataset mutation data
    :param csv_path_name: optional name for saving the final dataframe to a csv
    :param return_model_and_curve: opional boolean to return the model and fpr/tpr values, only works on the last pathway
    :param save_model_path: if provided, will save the model with this path and pathway as the name
    :param save_model_end: if save_model_path is provided, the model path and name will be save_model_path + pathway + save_model_end. Used for robustness analysis
    :return: a dataframe of the pathways and the scores for the sorted LSTM model
    '''

    score_list_train = []
    score_list_val = []
    score_list_test = []
    completed_pathway_list = []
    for pathway in pathways:

        model,sorted_data = train_lstm(pathways_dict,pathway,train_mut=liu_mut,train_clin=liu_clin)

        x1,y1,sorted_data = prep_data_lstm(train_mut, train_clin, pathways_dict, pathway,sorted_data = sorted_data)
        #predict using x1 to obtain the training auc score, and append to list for later use
        ypred = model.predict(x1)
        auc_lstm = br_outcome_roc_auc_score(ypred[:,1],train_clin['BR'])
        score_list_train.append(auc_lstm)

        x1_val, y1_val,sorted_data = prep_data_lstm(val_mut, val_clin, pathways_dict, pathway,sorted_data = sorted_data)
        val_pred1 = model.predict(x1_val)
        auc_lstm_val = br_outcome_roc_auc_score(val_pred1[:,1],val_clin['BR'])
        score_list_val.append(auc_lstm_val)

        #if test data is provided: calc scores for the test dataset
        if test_mut is not None:
            x1_test, y1_test,sorted_data = prep_data_lstm(test_mut, test_clin, pathways_dict, pathway,sorted_data = sorted_data)
            test_pred1 = model.predict(x1_test)
            auc_lstm_test1 = br_outcome_roc_auc_score(test_pred1[:, 1], test_clin['BR'])
            score_list_test.append(auc_lstm_test1)

        #save the model
        if save_model_path is not None:
            save_model_full_name = save_model_path + '_' + pathway + '_' + save_model_ending
            save_model(model = model,model_path = save_model_full_name)

        fpr_train, tpr_train, thresh_train = roc_curve(br_0_1(train_clin['BR']), ypred[:,1])
        fpr_val, tpr_val, thresh_val = roc_curve(br_0_1(val_clin['BR']), val_pred1[:,1])
        #and keep a record of the completed pathways
        completed_pathway_list.append(pathway)

    #now make a complete dataframe
    if test_mut is not None:
        output_df = pd.DataFrame([completed_pathway_list, score_list_train, score_list_val, score_list_test]).transpose()
        output_df.columns = ['pathway', 'auc_lstm_roc_auc_score_training','auc_lstm_roc_auc_score_validation','auc_lstm_roc_auc_score_test']
    else:
        output_df = pd.DataFrame([completed_pathway_list, score_list_train, score_list_val]).transpose()
        output_df.columns = ['pathway', 'auc_lstm_roc_auc_score_training','auc_lstm_roc_auc_score_validation']

    #output as a csv if given the csv name and path
    if csv_path_name is not None:
        output_df.to_csv(csv_path_name)
    #and check if you want to return the model and fpr and tpr. only works for one pathway, the last pathway ToDo:Explain?
    if return_model_and_curve is True:
        return output_df, model, fpr_train, tpr_train, fpr_val, tpr_val

    #otherwise, just output the pathways and scores
    return output_df

def fnn_multiple_pathways(pathways, pathways_dict, train_mut, train_clin, val_mut, val_clin, test_mut = None, test_clin = None, csv_path_name = None, return_model_and_curve = False, save_model_path = None, save_model_ending = ''):
    '''
    This is a function to run multiple pathways through forward neural networks.
    :param pathways: a list of pathways to run through the lstm models
    :param train_mut: the training mutation data
    :param train_clin: the traning clinical data
    :param val_mut: the validation mutation data
    :param val_clin: the validation clinical data
    :param test_clin: optional test dataset clinical data
    :param test_mut: optional test dataset mutation data
    :param csv_path_name: optional name for saving the final dataframe to a csv
    :param return_model_and_curve: opional boolean to return the model and fpr/tpr values, only works on the last pathway
    :param save_model_path: if provided, will save the model with this path and pathway as the name
    :param save_model_end: if save_model_path is provided, the model path and name will be save_model_path + pathway + save_model_end. Used for robustness analysis
    :return: a dataframe of the pathways and the scores for the forward neural network model
    '''

    score_list_train = []
    score_list_val = []
    score_list_test = []
    completed_pathway_list = []
    for pathway in pathways:

        model2=train_fnn(pathways_dict, pathway, train_mut=train_mut, train_clin=train_clin)

        x1, y1 = prep_data_fnn(train_mut, train_clin, pathways_dict, pathway)
        ypred2 = model2.predict(x1)
        auc_feed_forward = br_outcome_roc_auc_score(ypred2,train_clin['BR'])
        score_list_train.append(auc_feed_forward)

        val_x2,val_y2 = prep_data_fnn(val_mut, val_clin, pathways_dict, pathway)
        val_pred2 = model2.predict(val_x2)
        auc_lstm_val2 = br_outcome_roc_auc_score(val_pred2,val_clin['BR'])
        score_list_val.append(auc_lstm_val2)

        #if test provided: evaluate on test data
        if test_mut is not None:
            test_x2,test_y2 = prep_data_fnn(test_mut, test_clin, pathways_dict, pathway)
            test_pred2 = model2.predict(test_x2)
            auc_lstm_test2 = br_outcome_roc_auc_score(test_pred2,test_clin['BR'])
            score_list_test.append(auc_lstm_test2)

        completed_pathway_list.append(pathway)

        #save model
        if save_model_path is not None:
            save_model_full_name = save_model_path + '_' + pathway + '_' + save_model_ending
            save_model(model = model2, model_path = save_model_full_name)


        # #if you want to plot the roc curve for a single pathway. This will always return for only the last pathway, so use it for only one pathway
        fpr_train, tpr_train, thresh_train = roc_curve(br_0_1(train_clin['BR']), ypred2)
        fpr_val, tpr_val, thresh_val = roc_curve(br_0_1(val_clin['BR']), val_pred2)

    #now make a complete dataframe
    if test_mut is not None:
        output_df = pd.DataFrame([completed_pathway_list, score_list_train, score_list_val, score_list_test]).transpose()
        output_df.columns = ['pathway', 'auc_FNN_roc_auc_score_training','auc_FNN_roc_auc_score_validation','auc_FNN_roc_auc_score_test']
    else:
        output_df = pd.DataFrame([completed_pathway_list, score_list_train, score_list_val]).transpose()
        output_df.columns = ['pathway', 'auc_FNN_roc_auc_score_training','auc_FNN_roc_auc_score_validation']

    if csv_path_name != None:
        output_df.to_csv(csv_path_name)
    if return_model_and_curve == True:
        return output_df, model2, fpr_train, tpr_train, fpr_val, tpr_val

    return output_df


# Scripts -----------------------------------------------------------------------------------------------------------------


#this is the script to run on the HPC
def hpc_run():
    fnn_multiple_pathways(pathways = list(GO_dict_intersection.keys()), pathways_dict = GO_dict_intersection,train_mut = liu_mut, train_clin = liu_clin, val_mut = riaz_mut_val, val_clin = riaz_clin_val, csv_path_name = '../20_Intermediate_Files/GO_FNN_roc_auc_scores.csv' )
    lstm_multiple_pathways(pathways = list(GO_dict_intersection.keys()), pathways_dict = GO_dict_intersection,train_mut = liu_mut, train_clin = liu_clin, val_mut = riaz_mut_val, val_clin = riaz_clin_val, csv_path_name = '../20_Intermediate_Files/GO_Sorted_LSTM_roc_auc_scores.csv' )
