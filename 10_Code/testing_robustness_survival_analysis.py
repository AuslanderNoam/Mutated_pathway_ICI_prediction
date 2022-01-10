# this is a script to look at survival and proportional hazard

# Imports --------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from handle_data import *
from decision_trees import *
from neural_network import *
from optimize_pathways import *
from lifelines import CoxPHFitter
from feature_selection_pathway_mutation_load import *
from lifelines.statistics import logrank_test
from sklearn import metrics


# Data --------------------------------------------------------------------------------------------------------------

liu_mut = load_pickle(SAVE_DATA+'liu_mut.pickle')
liu_clin = load_pickle(SAVE_DATA + 'liu_clin.pickle')
riaz_mut_val = load_pickle(SAVE_DATA + 'riaz_mut.pickle')
riaz_clin_val = load_pickle(SAVE_DATA + 'riaz_clin.pickle')
hugo_mut_test = load_pickle(SAVE_DATA + 'hugo_mut.pickle')
hugo_clin_test = load_pickle(SAVE_DATA + 'hugo_clin.pickle')
msk_clin = load_pickle(SAVE_DATA + 'msk_clin.pickle')
msk_mut = load_pickle(SAVE_DATA + 'msk_mut.pickle')

GO_dict_intersection = load_pickle(SAVE_DATA + 'GO_dict_intersection.pickle')
GO_test_genes_dict_intersection = load_pickle(SAVE_DATA + 'GO_test_genes_dict_intersection.pickle')
GA_dict_GO = read_csv_dictionary('../20_Intermediate_Files/GO_genetic_algorithm_pathway_dictionary.csv')
FS_dict_GO = read_csv_dictionary('../20_Intermediate_Files/GO_top_pathway_gene_names.csv')
FS_prob_dict_GO = read_csv_dictionary('../20_Intermediate_Files/GO_top_pathway_gene_names_randomized.csv')

CANCER_DICT = {'melanoma':['SKCM'],'lung':['LUAD','LUSC'],'bladder':['BLCA'],'renal':['KIRC'],'head&neck':['HNSC'],'esophagus':['ESCA'],'glioma':['LGG'],'colon':['COAD'],'pancanc':['SKCM', 'LUAD', 'LUSC', 'BLCA', 'KIRC', 'HNSC', 'ESCA', 'LGG', 'COAD'],'subset1':['SKCM', 'COAD','BLCA','KIRC'],'subset2':['LUAD', 'LUSC','HNSC','ESCA','LGG']}

# Functions -----------------------------------------------------------------------------------------------------------------

def calc_proportional_hazard(score,s,cenc, convert=True, strat = None, thr = None):
    '''Later
    evaluates the conditional survival of a depending on b with PH
        :param first_event - list of first_event (mutation or arm)
        :param first_event - list of second_event (mutation or arm)
        :param s - survival time
        :param cenc - censoring information. Can also be the strings alive or dead.
        :para one_sided - one sided test, only evaluate results where hazard_ratios < 1
        :return: results - PH results
        cas - nonempty of PH failed to run
        cox_df - PH structure
        pv - PH pvalue
          '''

    if convert:
        death = [1 if i =='Alive' else 0 for i in cenc]
    else:
        death = cenc

    survival = s
    if strat is None:
        cox_df = pd.DataFrame(np.array([survival, death, score]).transpose(),
                          columns=['survival', 'death', 'g1'])
    else:
        cox_df = pd.concat([pd.DataFrame(np.array([survival, death, score]).transpose(),
                          columns=['survival', 'death', 'g1']),strat],axis=1)

    if thr is None:
        val = select_threshold_logrank(score,survival,cenc,convert=convert)
    else:
        val = thr

    if val==0 or sum(cox_df['g1'] < val)<10 or sum(cox_df['g1'] >= val)<10:
        s1 = cox_df['survival'][cox_df['g1'] > val]  # high
        d1 = cox_df['death'][cox_df['g1'] > val]

        s0 = cox_df['survival'][cox_df['g1'] <= val]
        d0 = cox_df['death'][cox_df['g1'] <= val]
    else:
        s1 = cox_df['survival'][cox_df['g1'] >= val]  # high
        d1 = cox_df['death'][cox_df['g1'] >= val]

        s0 = cox_df['survival'][cox_df['g1'] < val]
        d0 = cox_df['death'][cox_df['g1'] < val]

    results_lr = logrank_test(s0, s1, event_observed_A=d0, event_observed_B=d1)

    cph = CoxPHFitter()

    ##We either have that or we test PH assumptions seperately
    try:
        if strat is not None:
            cph.fit(cox_df, duration_col='survival', event_col='death',strata=list(strat.keys()))
        else:
            cph.fit(cox_df, duration_col='survival', event_col='death')

        pv = float(cph.summary.p)
        hazard_ratio = cph.hazard_ratios_.g1
    except:
        pv=1
        hazard_ratio = 1

    return pv,results_lr.p_value,cox_df,hazard_ratio

def select_threshold_logrank(score,survival,cenc, convert=True, minsmp = 10):
    '''

    :param score:
    :param s:
    :param cenc:
    :param convert:
    :return:
    '''

    if convert:
        death = [1 if i =='Alive' else 0 for i in cenc]
    else:
        death = cenc

    cox_df = pd.DataFrame(np.array([survival, death, score]).transpose(),
                          columns=['survival', 'death', 'g1'])

    vals = np.unique(score)

    lrpv = []
    for v in range(len(vals)):
        s1 = cox_df['survival'][cox_df['g1'] >= vals[v]]  # high
        d1 = cox_df['death'][cox_df['g1'] >= vals[v]]

        s0 = cox_df['survival'][cox_df['g1'] < vals[v]]
        d0 = cox_df['death'][cox_df['g1'] < vals[v]]
        results_lr = logrank_test(s0, s1, event_observed_A=d0, event_observed_B=d1)
        if not pd.isna(results_lr.p_value) and len(s0)>minsmp and len(s1)>minsmp:
            lrpv.append(results_lr.p_value)
        else:
            lrpv.append(1)

    if len(np.unique(vals)) > 1:
        thr = vals[np.argmin(lrpv)]
    else:
        thr = vals[len(vals)//2]

    return thr


def get_mutation_load_selected(mut_data,pathway,feature_selection = 'GA',retrain=False, spec_dict = None):
    '''

    :param mut_data:
    :param pathway:
    :param feature_selection:
    :return:
    '''

    assert feature_selection in ['GA','FS','FSR','ML']
    if spec_dict is not None:
        GA_dict_GO_inter = new_pathway_test_intersection([pathway], spec_dict, list(mut_data.keys()))[0]
        scores = mutation_load(GA_dict_GO_inter[pathway], patient_mutation_data=mut_data)
    else:
        if feature_selection=='GA':
            if not retrain:
                dict = GA_dict_GO
            else:
                dict = read_csv_dictionary('../20_Intermediate_Files/GA_DICT_MSK.csv')

            GA_dict_GO_inter = new_pathway_test_intersection([pathway], dict, list(mut_data.keys()))[0]
            scores = mutation_load(GA_dict_GO_inter[pathway], patient_mutation_data=mut_data)

        if feature_selection=='FS':
            if not retrain:
                dict = FS_dict_GO
            else:
                dict = read_csv_dictionary('../20_Intermediate_Files/FS_DICT_MSK.csv')
            FS_dict_GO_inter = new_pathway_test_intersection([pathway], dict, list(mut_data.keys()))[0]
            scores = mutation_load(FS_dict_GO_inter[pathway], patient_mutation_data=mut_data)

        if feature_selection=='FSR':
            if not retrain:
                dict = FS_prob_dict_GO
            else:
                dict = read_csv_dictionary('../20_Intermediate_Files/FSR_DICT_MSK.csv')
            FS_prob_dict_GO_inter = new_pathway_test_intersection([pathway], dict, list(mut_data.keys()))[0]
            scores = mutation_load(FS_prob_dict_GO_inter[pathway], patient_mutation_data=mut_data)

        if feature_selection == 'ML':
            scores = mut_data.sum(1)

    return list(scores)

def get_classifier_scores(mut_data,pathway,classifier = 'RF',spec_dict=None,random_state=None):
    '''

    :param mut_data:
    :param pathway:
    :param classifier:
    :return:
    '''

    assert classifier in ['RF', 'GB', 'LSTM', 'FNN']
    if spec_dict is not None:
        GO_dict_intersection_new = spec_dict
    else:
        GO_dict_intersection_new = new_pathway_test_intersection([pathway], GO_dict_intersection, list(mut_data.keys()))[0]

    if classifier == 'RF':
        model = train_random_forest(GO_dict_intersection_new, pathway,random_state=random_state)
        scores = model.predict_proba(mut_data[GO_dict_intersection_new[pathway]])[:, 1]
    if classifier == 'GB':
        model = train_gradient_boosting(GO_dict_intersection_new, pathway,random_state=random_state)
        scores = model.predict_proba(mut_data[GO_dict_intersection_new[pathway]])[:, 1]
    if classifier == 'LSTM':
        model,sorted_data = train_lstm(GO_dict_intersection_new, pathway)
        X = np.array(mut_data[sorted_data.columns])
        x1 = X.reshape(X.shape[0], 1, X.shape[1])
        scores = model.predict(x1)[:,1]
    if classifier == 'FNN':
        model = train_fnn(GO_dict_intersection_new, pathway)
        x1 = np.array(mut_data[GO_dict_intersection_new[pathway]])
        scores = model.predict(x1).T[0]

    return list(scores)


def get_scores(mut_data,pathway,classifier,retrain=False,spec_dict=None,random_state=None):
    '''

    :param mut_data:
    :param pathway:
    :param classifier:
    :return:
    '''
    assert classifier in ['RF', 'GB', 'LSTM', 'FNN','GA','FS','FSR','ML']

    if classifier in ['RF', 'GB', 'LSTM', 'FNN']:
        scores = get_classifier_scores(mut_data,pathway,classifier=classifier,spec_dict=spec_dict,random_state=random_state)
    if classifier in ['GA','FS','FSR','ML']:
        scores = get_mutation_load_selected(mut_data,pathway,feature_selection=classifier,retrain=retrain,spec_dict=spec_dict)

    return scores

def show_cancer_types_msk():
    print('Cancer types included: '+",".join(list(CANCER_DICT.keys())))
    return list(CANCER_DICT.keys())


def get_strat(clin_data,stratn):

    if stratn is not None:
        strat = clin_data[stratn]
        strat.index=[i for i in range(len(strat.index))]
    else:
        strat=None
    return strat


def get_pvalues_pathway_scores(pathways, classifier,mut_data,clin_data,convert=True,retrain=False,stratn=None):
    pvalue_cox_list = []
    pvalue_lr_list = []
    hazard_list = []


    strat=get_strat(clin_data,stratn)

    for pathway in pathways:
        if classifier=='ML' and 'TMB_SCORE' in clin_data.keys():
            scores = list(clin_data['TMB_SCORE'])
        else:
            scores = get_scores(mut_data,pathway,classifier,retrain=retrain)
        pvalue_train_cox, pvalue_train_lr, cox_df_train, train_hazard_ratio = calc_proportional_hazard(scores, clin_data['survival'], clin_data['vital_status'],convert=convert, strat = strat)

        pvalue_cox_list.append(pvalue_train_cox)
        pvalue_lr_list.append(pvalue_train_lr)
        hazard_list.append(train_hazard_ratio)

    combined_df = pd.DataFrame()
    combined_df['pathway'] = pathways
    combined_df['pvalue_ph'] = pvalue_cox_list
    combined_df['pvalue_log_rank'] = pvalue_lr_list
    combined_df['hr'] = hazard_list

    return combined_df

def eval_msk_survival(pathways, classifier,cancer,retrain=False,stratn=None):
    '''

    :param pathways:
    :param classifier:
    :param cancer: from show_cancer_types_msk()
    :param mut_data:
    :param clin_data:
    :return:
    '''

    assert cancer in show_cancer_types_msk()
    clin, mut = extract_cancers_msk(msk_clin, msk_mut, CANCER_DICT[cancer])
    combined_df = get_pvalues_pathway_scores(pathways, classifier, mut, clin,convert=False,retrain=retrain,stratn=stratn)

    return combined_df

def retrain_ml_pathway_msk(pathways_list = ['GO_LEUKOCYTE_DIFFERENTIATION','GO_IMMUNE_RESPONSE','GO_REGULATION_OF_I_KAPPAB_KINASE_NF_KAPPAB_SIGNALING','GO_REGULATION_OF_TRANSCRIPTION_FROM_RNA_POLYMERASE_II_PROMOTER']):
    '''
    Retrains ML predictors with MSK genes and saves dictionaries
    '''
    returned_result, new_pathways_for_scoring = multiple_pathway_genetic_algorithm(pathways_list=pathways_list,
                                                                                   pathways_dict=GO_dict_intersection,
                                                                                   mut_data=liu_mut,
                                                                                   br_data=liu_clin.loc[:, 'BR'],
                                                                                   save_as_csv=True,
                                                                                   csv_save_pathway_dictionary_name='../20_Intermediate_Files/GA_DICT_MSK.csv')

    returned_result, new_pathways_for_scoring = multiple_pathway_forward_selector(pathway_list=pathways_list,
                                                                                  pathway_dict=GO_dict_intersection,
                                                                                  number_of_genes=10,
                                                                                  optimize_on='roc_auc_score',
                                                                                  patient_mutation_data=liu_mut,
                                                                                  br_labels=liu_clin.loc[:, 'BR'],
                                                                                  survival_time=liu_clin.loc[:,
                                                                                                'survival'],
                                                                                  censor_status=liu_clin.loc[:,'vital_status'],
                                                                                  probability=False,
                                                                                  resistance=False,
                                                                                  csv_save_pathway_dictionary_name='../20_Intermediate_Files/FS_DICT_MSK.csv')

    returned_result, new_pathways_for_scoring = multiple_pathway_forward_selector(pathway_list=pathways_list,
                                                                                  pathway_dict=GO_dict_intersection,
                                                                                  number_of_genes=10,
                                                                                  optimize_on='roc_auc_score',
                                                                                  patient_mutation_data=liu_mut,
                                                                                  br_labels=liu_clin.loc[:, 'BR'],
                                                                                  survival_time=liu_clin.loc[:,
                                                                                                'survival'],
                                                                                  censor_status=liu_clin.loc[:,
                                                                                                'vital_status'],
                                                                                  probability=True,
                                                                                  resistance=False,
                                                                                  csv_save_pathway_dictionary_name='../20_Intermediate_Files/FSR_DICT_MSK.csv')


def retrain_feature_sel(classifier,pathways,dict=GO_test_genes_dict_intersection):
    assert classifier in ['GA', 'FS', 'FSR']
    if classifier=='GA':
        GO_genetic_pathway_scores, GO_dict = multiple_pathway_genetic_algorithm(
            pathways_list=pathways,
            pathways_dict=dict,
            mut_data=liu_mut,
            br_data=liu_clin.loc[:, 'BR'],
            save_as_csv=True,
        )
    if classifier=='FSR':
        GO_top_pathways_new_scores_randomized, GO_dict = multiple_pathway_forward_selector(
            pathway_list=pathways,
            pathway_dict=dict,
            number_of_genes=10,
            optimize_on='roc_auc_score',
            patient_mutation_data=liu_mut,
            br_labels=liu_clin.loc[:, 'BR'],
            survival_time=liu_clin.loc[:, 'survival'],
            censor_status=liu_clin.loc[:, 'vital_status'],
            probability=True,
            resistance=False,
        )
    if classifier=='FS':
        GO_top_pathways_new_scores_randomized, GO_dict = multiple_pathway_forward_selector(
            pathway_list=pathways,
            pathway_dict=dict,
            number_of_genes=10,
            optimize_on='roc_auc_score',
            patient_mutation_data=liu_mut,
            br_labels=liu_clin.loc[:, 'BR'],
            survival_time=liu_clin.loc[:, 'survival'],
            censor_status=liu_clin.loc[:, 'vital_status'],
            probability=False,
            resistance=False,
        )
    return GO_dict

def robustness_analysis(classifier,pathways,reps=5):
    res_riaz = {}
    res_hugo = {}
    for p in pathways:
        auc1=[]
        auc2=[]
        for rep in range(reps):
            if classifier in ['GA', 'FS', 'FSR']:
                GO_dict = retrain_feature_sel(classifier,pathways)
            else:
                GO_dict = GO_test_genes_dict_intersection

            l1 = br_0_1(riaz_clin_val['BR'])
            sc1 = get_scores(riaz_mut_val,p,classifier,retrain=False,spec_dict=GO_dict,random_state=rep)
            auc1.append(metrics.roc_auc_score(l1, sc1))

            l2 = br_0_1(hugo_clin_test['BR'])
            sc2 = get_scores(hugo_mut_test, p, classifier, retrain=False, spec_dict=GO_dict, random_state=rep)
            auc2.append(metrics.roc_auc_score(l2, sc2))

        res_riaz[p] = auc1
        res_hugo[p] = auc2

    return res_riaz,res_hugo


# Scripts --------------------------------------------------------------------------------------------------------------




p_ml = ['GO_LEUKOCYTE_DIFFERENTIATION','GO_IMMUNE_RESPONSE','GO_REGULATION_OF_I_KAPPAB_KINASE_NF_KAPPAB_SIGNALING','GO_REGULATION_OF_TRANSCRIPTION_FROM_RNA_POLYMERASE_II_PROMOTER']
p_class = ['GO_ATPASE_BINDING','GO_HORMONE_MEDIATED_SIGNALING_PATHWAY','GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION','GO_REGULATION_OF_T_CELL_PROLIFERATION']

ml_list = ['GA','FS','FSR','ML']
classifier_list = ['RF', 'GB', 'LSTM', 'FNN']
cancer_list = ['melanoma','lung','bladder','renal','head&neck','esophagus','glioma','colon','pancanc']
