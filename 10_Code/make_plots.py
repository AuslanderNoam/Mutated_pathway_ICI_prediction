#These are the code for the figures

# Imports --------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from optimize_pathways import *
from feature_selection_pathway_mutation_load import *
from handle_data import *
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.facecolor'] = 'none'
import seaborn as sns
from scipy.stats import spearmanr
import seaborn as sns; sns.set_theme(color_codes=True)
import scipy
from statannot import add_stat_annotation
from lifelines import CoxPHFitter
import numpy as np
import lifelines
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import export_graphviz
from subprocess import call
import graphviz
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from neural_network import *
from matplotlib.pyplot import figure
from lifelines.statistics import logrank_test
from sklearn import metrics
from testing_robustness_survival_analysis import calc_proportional_hazard,CANCER_DICT,msk_clin,msk_mut,get_scores,get_strat,select_threshold_logrank,robustness_analysis
import matplotlib.image as mpimg
import zepid
from zepid.graphics import EffectMeasurePlot


# Data -----------------------------------------------------------------------------------------------------------------

liu_mut = load_pickle(SAVE_DATA+'liu_mut.pickle')
liu_clin = load_pickle(SAVE_DATA + 'liu_clin.pickle')
liu_clin['death'] = [i for i in liu_clin['vital_status']]
riaz_mut_val = load_pickle(SAVE_DATA + 'riaz_mut.pickle')
riaz_clin_val = load_pickle(SAVE_DATA + 'riaz_clin.pickle')
hugo_mut_test = load_pickle(SAVE_DATA + 'hugo_mut.pickle')
hugo_clin_test = load_pickle(SAVE_DATA + 'hugo_clin.pickle')
GO_dict_intersection = load_pickle(SAVE_DATA + 'GO_dict_intersection.pickle')
MC_dict_intersection = load_pickle(SAVE_DATA + 'MC_dict_intersection.pickle')

# Functions -----------------------------------------------------------------------------------------------------------------

flatten = lambda l: [item for sublist in l for item in sublist]




def plot_roc(pathway,classifier='RF',retrain=False, spec_dict = None):
    sns.set_style("ticks")
    GO_test_genes_dict_intersection = load_pickle(SAVE_DATA + 'GO_test_genes_dict_intersection.pickle')

    hugo_mut_test = load_pickle(SAVE_DATA + 'hugo_mut.pickle')
    hugo_clin_test = load_pickle(SAVE_DATA + 'hugo_clin.pickle')

    scores_tr = get_scores(liu_mut, pathway, classifier=classifier, retrain=retrain,spec_dict=spec_dict)
    l_tr = br_0_1(liu_clin['BR'])
    scores_val = get_scores(riaz_mut_val, pathway, classifier=classifier, retrain=retrain,spec_dict=spec_dict)
    l_val = br_0_1(riaz_clin_val['BR'])

    scores_ts = get_scores(hugo_mut_test, pathway, classifier=classifier, retrain=retrain,spec_dict=GO_test_genes_dict_intersection)
    l_ts = br_0_1(hugo_clin_test['BR'])

    plt.figure(0).clf()
    fpr, tpr, thresh = metrics.roc_curve(l_tr, scores_tr)
    auc = metrics.roc_auc_score(l_tr, scores_tr)
    plt.plot(fpr, tpr, "-b",label="Liu, AUC=" + str(auc))
    plt.legend(loc="lower right")
    fpr, tpr, thresh = metrics.roc_curve(l_val, scores_val)
    auc = metrics.roc_auc_score(l_val, scores_val)
    plt.plot(fpr, tpr, "-r",label="Riaz, AUC=" + str(auc))
    plt.legend(loc="lower right")
    fpr, tpr, thresh = metrics.roc_curve(l_ts, scores_ts)
    auc = metrics.roc_auc_score(l_ts, scores_ts)
    plt.plot(fpr, tpr, "-g", label="Hugo, AUC=" + str(auc))
    plt.legend(loc="lower right")

    plt.plot([0,0.5,1], [0,0.5,1],"--k")

    return plt

def plot_roc_val_rf_gb(pathway,retrain=False):
    sns.set_style("ticks")
    scores_val1 = get_scores(riaz_mut_val, pathway, classifier='RF', retrain=retrain)
    scores_val2 = get_scores(riaz_mut_val, pathway, classifier='GB', retrain=retrain)
    l_val = br_0_1(riaz_clin_val['BR'])

    plt.figure(0).clf()
    fpr, tpr, thresh = metrics.roc_curve(l_val, scores_val1)
    auc = metrics.roc_auc_score(l_val, scores_val1)
    plt.plot(fpr, tpr, "-b",label="RF, AUC=" + str(auc))
    plt.legend(loc="lower right")
    fpr, tpr, thresh = metrics.roc_curve(l_val, scores_val2)
    auc = metrics.roc_auc_score(l_val, scores_val2)
    plt.plot(fpr, tpr, "-r",label="GB, AUC=" + str(auc))
    plt.legend(loc="lower right")

    plt.plot([0,0.5,1], [0,0.5,1],"--k")
    return plt

def make_surv_plot_msk(cancer,pathway,classifier,stratn=None):

    clin, mut = extract_cancers_msk(msk_clin, msk_mut, CANCER_DICT[cancer])
    plot_KM(mut,clin,pathway,classifier,stratn=stratn)

def plot_KM(mut,clin,pathway,classifier,titletxt='',savePath='',convert=False,retrain=True,stratn=None,show_table=False):
    '''
    Later
    Plots Kaplan Meier curves for 2 events
    '''
    figure(figsize=(6, 4), dpi=80)
    sns.set_style("ticks")
    scores = get_scores(mut, pathway, classifier=classifier,retrain=retrain)
    val = select_threshold_logrank(scores, clin['survival'], clin['death'], convert=convert)

    strat = get_strat(clin, stratn)
    pval,pval2,cox_df,hazard_ratio = calc_proportional_hazard(scores,clin['survival'],clin['death'], convert=convert,strat=strat,thr=val)

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

    kaplan_meier1 = KaplanMeierFitter()
    kaplan_meier0 = KaplanMeierFitter()

    crv0 = pathway+' low'
    crv1 = pathway+' high'

    kaplan_meier1.fit(s1, d1, label=crv1)
    axid = kaplan_meier1.plot_survival_function()  #low
    kaplan_meier0.fit(s0, d0, label=crv0)#low
    kaplan_meier0.plot_survival_function(ax=axid)
    if show_table:
        add_at_risk_counts(kaplan_meier1, kaplan_meier0, ax=axid)

    mytext = "CPH P=%.2e, log-rank p=%.2e" % (pval,pval2)
    plt.tight_layout()

    axid.set_title(titletxt)
    axid.set_ylabel('survival probability')
    axid.text(0.1, 0.1, mytext)


    if len(savePath) > 0:
        fig = axid.get_figure()
        fig.savefig(savePath)

# Scripts -----------------------------------------------------------------------------------------------------------------
#
#add in select pathways for RF and GB, get top 3 perent performing pathways, should be about 15-20. Then should be less than a 10 percent drop

def make_fig_1a():
    GA_res = pd.read_csv('../25_Results/tables/GA_GO_results.csv')
    FS_res = pd.read_csv('../25_Results/tables/FS_GO_results.csv')
    pt = select_pathways()

    A1 = GA_res.set_index('GO_pathways')
    A2 = FS_res.set_index('GO_pathways')

    traind = pd.concat([A1.loc[pt]['auc_liu_GO_GA_response'],A2.loc[pt]['auc_liu_GO_FS_response'],A2.loc[pt]['auc_liu_GO_FS_response_randomized']],axis=1)
    vald = pd.concat([A1.loc[pt]['auc_riaz_GO_GA_response'],A2.loc[pt]['auc_riaz_GO_FS_response'],A2.loc[pt]['auc_riaz_GO_FS_response_randomized']],axis=1)

    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(12, 8))

    traind.plot.bar(stacked=False,ax=axes[0],width=0.66)
    vald.plot.bar(stacked=False,ax=axes[1],width=0.66)

    g1 = pd.read_csv('../20_Intermediate_Files/GO_genetic_algorithm_pathway_dictionary.csv')
    g2 = pd.read_csv('../20_Intermediate_Files/GO_top_pathway_gene_names.csv')
    g3 = pd.read_csv('../20_Intermediate_Files/GO_top_pathway_gene_names_randomized.csv')

    g11=g1[pt]
    g22=g2[pt]
    g33=g3[pt]

    d1 = pathway_pd_to_dict(g11)
    d2 = pathway_pd_to_dict(g22)
    d3 = pathway_pd_to_dict(g33)

    gene1 = list(set(flatten(d1.values())))
    gene2 = list(set(flatten(d2.values())))
    gene3 = list(set(flatten(d3.values())))

    gc = list(set(gene1)&set(gene2)&set(gene3))

    PATS1=[];PATS2=[];PATS3=[]
    for pat in pt:
        ptm1 = [];ptm2 = [];ptm3 = []
        for gene in gc:
            ptm1.append(1) if gene in d1[pat] else ptm1.append(0)
            ptm2.append(1) if gene in d2[pat] else ptm2.append(0)
            ptm3.append(1) if gene in d3[pat] else ptm3.append(0)

        PATS1.append(ptm1)
        PATS2.append(ptm2)
        PATS3.append(ptm3)


    P1 = np.array(PATS1)
    P2 = np.array(PATS2)
    P3 = np.array(PATS3)

    # THR = 3
    THR = 3

    x=P1+P2+P3
    x2 = x[:,[i for i in range(len(np.sum(x,axis=0))) if np.sum(x,axis=0)[i]>THR]]
    nm2=[gc[i] for i in range(len(np.sum(x,axis=0))) if np.sum(x,axis=0)[i]>THR]
    PP = pd.DataFrame(x2,columns=nm2,index=pt)
    g = sns.clustermap(PP.T,col_cluster=False,cmap="Blues")

    return pt

def make_corrplot(df):

    corr = df.corr()
    f, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, mask=mask, cmap=cmap,vmin=-1,vmax=1)


def make_fig_1c():

    GA_res = pd.read_csv('../25_Results/tables/GA_GO_results.csv')
    GA_res=GA_res.set_index('GO_pathways')

    FS_res = pd.read_csv('../25_Results/tables/FS_GO_results.csv')
    FS_res=FS_res.set_index('GO_pathways')

    ALL_GO = GA_res.merge(FS_res)[['auc_liu_GO_GA_response','auc_liu_GO_FS_response','auc_liu_GO_FS_response_randomized','auc_riaz_GO_GA_response', 'auc_riaz_GO_FS_response','auc_riaz_GO_FS_response_randomized']]
    ALL_GO=ALL_GO.rename({'auc_liu_GO_GA_response':'Genetic Algorithm training','auc_liu_GO_FS_response':'Forward Greedy training','auc_liu_GO_FS_response_randomized':'Forward Randomized training','auc_riaz_GO_GA_response':'Genetic Algorithm validation', 'auc_riaz_GO_FS_response':'Forward Greedy validation','auc_riaz_GO_FS_response_randomized':'Forward Randomized validation'})

    make_corrplot(ALL_GO)


def get_pd_box(l1,l11):
    v1 = np.quantile(l1, 0.95)
    v2 = np.quantile(l1, 0.05)
    ids = [i for i in range(len(l11)) if l1[i] > v1 or l1[i] < v2]

    X = [[l11[i] for i in ids], ['high training' if l1[i] > v1 else 'low training' for i in ids]]

    GA = pd.DataFrame(np.array(X).T, columns=['AUC', 'group'])
    GA['AUC'] = pd.to_numeric(GA['AUC'])
    return GA


def make_fig_1b():

    GA_res = pd.read_csv('../25_Results/tables/GA_GO_results.csv')
    FS_res = pd.read_csv('../25_Results/tables/FS_GO_results.csv')
    ALL_GO = GA_res.merge(FS_res,left_on='GO_pathways',right_on='GO_pathways')[
        ['auc_liu_GO_GA_response', 'auc_liu_GO_FS_response', 'auc_liu_GO_FS_response_randomized', 'auc_riaz_GO_GA_response',
         'auc_riaz_GO_FS_response', 'auc_riaz_GO_FS_response_randomized','GO_pathways']]

    l1 = list(ALL_GO['auc_liu_GO_GA_response'])
    l2 = list(ALL_GO['auc_liu_GO_FS_response'])
    l3 = list(ALL_GO['auc_liu_GO_FS_response_randomized'])
    l11 = list(ALL_GO['auc_riaz_GO_GA_response'])
    l22 = list(ALL_GO['auc_riaz_GO_FS_response'])
    l33 = list(ALL_GO['auc_riaz_GO_FS_response_randomized'])




    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12, 4))
    GA = get_pd_box(l1,l11)
    sns.boxplot(x='group', y='AUC',  data=GA,ax=axes[0]).set_title('Genetic Algorithm')
    add_stat_annotation(ax=axes[0], data=GA, x='group', y='AUC',
        box_pairs=[("low training", "high training")],
        test='Mann-Whitney', text_format='star', loc='outside', verbose=2)

    FSG = get_pd_box(l2,l22)
    sns.boxplot(x='group', y='AUC',  data=FSG,ax=axes[1]).set_title('Forward Greedy')
    add_stat_annotation(ax=axes[1], data=FSG, x='group', y='AUC',
        box_pairs=[("low training", "high training")],
        test='Mann-Whitney', text_format='star', loc='outside', verbose=2)

    FSR = get_pd_box(l3,l33)
    sns.boxplot(x='group', y='AUC',  data=FSR,ax=axes[2]).set_title('Forward Randomized ')
    add_stat_annotation(ax=axes[2], data=FSR, x='group', y='AUC',
        box_pairs=[("low training", "high training")],
        test='Mann-Whitney', text_format='star', loc='outside', verbose=2)



def make_fig_2a():
    sns.set_style("ticks")

    combined_pathways = select_pathways_trees()

    l0 = br_0_1(liu_clin['BR'])
    l1 = br_0_1(riaz_clin_val['BR'])
    AUC0=[];AUC1=[]
    AUC00 = [];
    AUC11 = []

    for i in range(len(combined_pathways)):
        sc0 = get_scores(liu_mut, combined_pathways[i], 'RF')
        auc0 = metrics.roc_auc_score(l0, sc0)
        AUC0.append(auc0)
        sc0 = get_scores(liu_mut, combined_pathways[i], 'GB')
        auc0 = metrics.roc_auc_score(l0, sc0)
        AUC00.append(auc0)

        sc1=get_scores(riaz_mut_val, combined_pathways[i], 'RF')
        auc1 = metrics.roc_auc_score(l1, sc1)
        AUC1.append(auc1)
        sc1 = get_scores(riaz_mut_val, combined_pathways[i], 'GB')
        auc1 = metrics.roc_auc_score(l1, sc1)
        AUC11.append(auc1)
    x=[AUC0,AUC00]
    traind = pd.DataFrame(np.array([AUC0,AUC00]).T,columns=['RF','GB'])
    traind.index = combined_pathways
    vald = pd.DataFrame(np.array([AUC1, AUC11]).T, columns=['RF', 'GB'])
    vald.index = combined_pathways

    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(12, 8))

    traind.plot.bar(stacked=False,ax=axes[0],width=0.66)
    vald.plot.bar(stacked=False,ax=axes[1],width=0.66)
    plt.savefig('../30_Figures/Fig2a.pdf')
    return traind,vald

def make_fig_2b():
    plt0 = plot_roc_val_rf_gb('GO_REGULATION_OF_T_CELL_PROLIFERATION')
    plt0.savefig('../30_Figures/Fig2b1.pdf')
    plt1 = plot_roc_val_rf_gb('GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION')
    plt1.savefig('../30_Figures/Fig2b2.pdf')


def make_fig_2c():
    t_cell_rf = RandomForestClassifier(max_depth=5, min_samples_split=2, random_state=100)
    t_cell_rf = t_cell_rf.fit(liu_mut[GO_dict_intersection['GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION']],
                              br_0_1(liu_clin['BR']))
    x = [t_cell_rf.estimators_[k].feature_importances_ for k in range(100)]
    a = np.array(x)
    df = pd.DataFrame(a, columns=GO_dict_intersection['GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION'])
    g = sns.clustermap(df, row_cluster=False, cmap="Purples",xticklabels=1)
    g.savefig('../30_Figures/Fig2c.pdf')


def make_fig_2d(n_tree=None):
    #this makes a figure of a tree from one of the random forest trees for 'GO_REGULATION_OF_T_CELL_PROLIFERATION' pathway

    #recreate the trees for the pathway you want to examine using the same random seed.
    #want to eaxmine the 'GO_REGULATION_OF_T_CELL_PROLIFERATION' pathway
    t_cell_rf = RandomForestClassifier(max_depth = 5, min_samples_split = 2, random_state = 100)
    t_cell_rf = t_cell_rf.fit(liu_mut[GO_dict_intersection['GO_REGULATION_OF_T_CELL_PROLIFERATION']],br_0_1(liu_clin['BR']))

    feat_imp = t_cell_rf.feature_importances_
    # best_feaus = [GO_dict_intersection['GO_REGULATION_OF_T_CELL_PROLIFERATION'][i] for i in range(len(feat_imp)) if feat_imp[i]>0.03]
    best_feaus = [i for i in range(len(feat_imp)) if
                  feat_imp[i] > 0.025]
    agreement_f = [np.mean([t_cell_rf.estimators_[k].feature_importances_[i]/np.count_nonzero(t_cell_rf.estimators_[k].feature_importances_) for i in best_feaus]) for k in range(100)]

    n_ft = [np.count_nonzero(t_cell_rf.estimators_[k].feature_importances_) for k in range(100)]

    if n_tree is None:
        n_tree = agreement_f.index(max(agreement_f))
    np.mean([t_cell_rf.estimators_[n_tree].feature_importances_[i] for i in best_feaus])
    #I just want to check. Looks correct
    #t_cell_rf_roc_train = roc_auc_score(br_0_1(liu_clin['BR']), t_cell_rf.predict_proba(liu_mut[GO_dict_intersection['GO_REGULATION_OF_T_CELL_PROLIFERATION']])[:,1])
    #t_cell_rf_roc_val = roc_auc_score(br_0_1(riaz_clin_val['BR']), t_cell_rf.predict_proba(riaz_mut_val[GO_dict_intersection['GO_REGULATION_OF_T_CELL_PROLIFERATION']])[:,1])

    t_cell_rf_estimator = t_cell_rf.estimators_[n_tree]
    #tree.plot_tree(t_cell_rf_estimator)
    t_cell_rf_estimator.export_graphviz(t_cell_rf_estimator, out_file='../30_Figures/t_cell_rf_tree.dot', feature_names = GO_dict_intersection['GO_REGULATION_OF_T_CELL_PROLIFERATION'], class_names = ['Non-Response','Response'],rounded = True, proportion = False,precision = 2, filled = True)
    call(['dot', '-Tpdf', '../30_Figures/t_cell_rf_tree.dot', '-o', '../30_Figures/t_cell_rf_tree_%s.pdf'%(n_tree), '-Gdpi=600'])


    return feat_imp,agreement_f,n_ft

def make_lstm_fnn_roc_curve(pathway):
    '''
    Note: Rerunning this function may result in a slightly different graph due to variations in recreating the neural network
    The general robustness of the graph was confirmed
    '''

    GO_fnn_df1, new_fnn_model, fpr1_fnn_train, tpr1_fnn_train, fpr1_fnn_val, tpr1_fnn_val = fnn_multiple_pathways(pathways = [pathway], pathways_dict = GO_dict_intersection,train_mut = liu_mut, train_clin = liu_clin, val_mut = riaz_mut_val, val_clin = riaz_clin_val, return_model_and_curve = True)
    GO_lstm_df1,new_lstm_model,fpr1_lstm_train, tpr1_lstm_train,fpr1_lstm_val, tpr1_lstm_val = lstm_multiple_pathways(pathways = [pathway], pathways_dict = GO_dict_intersection,train_mut = liu_mut, train_clin = liu_clin, val_mut = riaz_mut_val, val_clin = riaz_clin_val, return_model_and_curve = True)



    #val_x1 = np.array(riaz_mut_val[GO_dict_intersection[pathway]])
    #val_x1_reshaped = val_x1.reshape(val_x1.shape[0],1,val_x1.shape[1])
    #val_pred1 = new_model.predict(val_x1_reshaped)

    #val_x1 = np.array(riaz_mut_val[GO_dict_intersection[pathway]])
    #val_x1_reshaped = val_x1.reshape(val_x1.shape[0],1,val_x1.shape[1])
    #val_pred1 = new_model.predict(val_x1_reshaped)
    #auc_lstm_val = round(br_outcome_roc_auc_score(val_pred1[:,1],riaz_clin_val['BR']),3)


    #fpr1, tpr1, thresh1 = roc_curve(br_0_1(riaz_clin_val['BR']), val_pred1[:,1])

    plt.plot(fpr1_lstm_train, tpr1_lstm_train, color='darkorange', lw=2, label='ROC curve for LSTM training (auc = '+str(round(GO_lstm_df1.iloc[0,1],3)) + ')')
    plt.plot(fpr1_lstm_val, tpr1_lstm_val, color='yellow', lw=2, label='ROC curve for LSTM validation (auc = '+str(round(GO_lstm_df1.iloc[0,2],3)) + ')')
    plt.plot(fpr1_fnn_train, tpr1_fnn_train, color='blue', lw=2, label='ROC curve for FNN training (auc = '+str(round(GO_fnn_df1.iloc[0,1],3)) + ')')
    plt.plot(fpr1_fnn_val, tpr1_fnn_val, color='teal', lw=2, label='ROC curve for FNN validation (auc = '+str(round(GO_fnn_df1.iloc[0,2],3)) + ')')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LSTM ROC for ' + pathway)
    plt.legend(loc="lower right")
    plt.savefig('../30_Figures/LSTM_FNN_Combined_ROC_Curve_' + pathway + '.pdf')
    plt.show()


def fig_2e():
    #this will make three roc_auc curve plots. These plots are for the three neural network pathways that overlapped with the decision tree method pathways that had over 0.7 auc score
    '''
    Note: Rerunning this function may result in a slightly different graph due to variations in recreating the neural network
    The general robustness of the graph was confirmed
    '''

    nn_pathways = ['GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION','GO_REGULATION_OF_T_CELL_PROLIFERATION','GO_HORMONE_MEDIATED_SIGNALING_PATHWAY']
    for pathway in nn_pathways:
        make_lstm_fnn_roc_curve(pathway)


def make_box_dict(res_riaz,res_hugo,mv=0.85):
    sns.set_style("ticks")
    d1=pd.DataFrame(flatten([res_riaz[i] for i in list(res_riaz.keys())]), columns=['AUC'])
    d1['pathway'] = flatten([[list(res_riaz.keys())[i] for v in range(len(list(res_riaz.values())[0]))] for i in
                             range(len(list(res_riaz.keys())))])
    d1['dataset'] = ['riaz' for i in range(d1.shape[0])]

    d2 = pd.DataFrame(flatten([res_hugo[i] for i in list(res_hugo.keys())]), columns=['AUC'])
    d2['pathway'] = flatten([[list(res_hugo.keys())[i] for v in range(len(list(res_hugo.values())[0]))] for i in
                             range(len(list(res_hugo.keys())))])

    d2['dataset'] = ['hugo' for i in range(d2.shape[0])]


    a = pd.concat([d1, d2])


    ax1=sns.boxplot(x='pathway', y='AUC', hue='dataset', data=a,palette="Reds")
    sns.stripplot(x='pathway', y='AUC', hue='dataset',ax=ax1, data=a,jitter = True, split = True, linewidth = 0.5,palette="Reds")

    plt.setp(ax1.get_xticklabels(), rotation=90)
    plt.ylim(0.35,mv)


def make_fig_3a():
    plt0 = plot_roc('GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION', classifier='RF', retrain=False)
    plt0.savefig('../30_Figures/Fig3a1.pdf')
    plt1 = plot_roc('GO_REGULATION_OF_T_CELL_PROLIFERATION', classifier='RF', retrain=False)
    plt1.savefig('../30_Figures/Fig3a2.pdf')



def make_fig_3b():
    p_class = ['GO_ATPASE_BINDING', 'GO_HORMONE_MEDIATED_SIGNALING_PATHWAY', 'GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION',
               'GO_REGULATION_OF_T_CELL_PROLIFERATION']


    res_riaz, res_hugo = robustness_analysis('RF', p_class, reps=50)
    make_box_dict(res_riaz, res_hugo)




def make_fig_4abd():
    make_surv_plot_msk('melanoma', 'GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION', 'RF', stratn=['type', 'sex'])
    make_surv_plot_msk('melanoma', 'GO_LEUKOCYTE_DIFFERENTIATION', 'GA', stratn=['type', 'sex'])
    make_surv_plot_msk('subset1', 'GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION', 'RF', stratn=['type', 'sex'])


def get_cph_spec(canc,pathway='GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION'):
    clin, mut = extract_cancers_msk(msk_clin, msk_mut, CANCER_DICT[canc])
    scores = get_scores(mut, pathway, classifier='RF',retrain=True)
    strat = get_strat(clin, ['type'])
    # pval,pval2,cox_df,hazard_ratio = calc_proportional_hazard(scores,clin['survival'],clin['death'], convert=False,strat=strat)
    pval,pval2,cox_df,hazard_ratio = calc_proportional_hazard(scores,clin['survival'],clin['death'], convert=False)

    cox_df = cox_df.rename({'g1': canc},axis='columns')
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='survival', event_col='death')
    return cph,pval,pval2

def make_fig_4c():
    php = [];
    plr = []
    sns.set_style("ticks")
    ax1 = plt.subplot(8, 1, 1)
    cph, pval, pval2 = get_cph_spec('melanoma')
    ax1 = cph.plot()
    ax1.set_xlim(-30, 35)
    php.append(pval)
    plr.append(pval2)

    ax1 = plt.subplot(8, 1, 2)
    cph, pval, pval2 = get_cph_spec('colon')
    ax1 = cph.plot()
    ax1.set_xlim(-30, 35)
    php.append(pval)
    plr.append(pval2)

    ax1 = plt.subplot(8, 1, 3)
    cph, pval, pval2 = get_cph_spec('bladder')
    ax1 = cph.plot()
    ax1.set_xlim(-30, 35)
    php.append(pval)
    plr.append(pval2)

    ax1 = plt.subplot(8, 1, 4)
    cph, pval, pval2 = get_cph_spec('renal')
    ax1 = cph.plot()
    ax1.set_xlim(-30, 35)
    php.append(pval)
    plr.append(pval2)

    ax1 = plt.subplot(8, 1, 5)
    cph, pval, pval2 = get_cph_spec('lung')
    ax1 = cph.plot()
    ax1.set_xlim(-30, 35)
    php.append(pval)
    plr.append(pval2)

    ax1 = plt.subplot(8, 1, 6)
    cph, pval, pval2 = get_cph_spec('esophagus')
    ax1 = cph.plot()
    ax1.set_xlim(-30, 35)
    php.append(pval)
    plr.append(pval2)

    ax1 = plt.subplot(8, 1, 7)
    cph, pval, pval2 = get_cph_spec('glioma')
    ax1 = cph.plot()
    ax1.set_xlim(-30, 35)
    php.append(pval)
    plr.append(pval2)

    ax1 = plt.subplot(8, 1, 8)
    cph, pval, pval2 = get_cph_spec('head&neck')
    ax1 = cph.plot()
    ax1.set_xlim(-30, 35)
    php.append(pval)
    plr.append(pval2)

def get_number_of_genes_in_rf_hugo():
    #This is a small script to generate what percentage of genes are shared for three pathways in hugo vs the training and test datasets
    hugo_mut_test.columns
    new_pathway_test_intersection(['GO_REGULATION_OF_LEUKOCYTE_PROLIFERATION','GO_REGULATION_OF_T_CELL_PROLIFERATION','GO_HORMONE_MEDIATED_SIGNALING_PATHWAY'], GO_dict_intersection, hugo_mut_test.columns)

def make_supp_table_5():
    #this will make the supplementary table 5, which is the number of genes present in the different pathways
    #first, get the datasets

    liu_mut_raw = pd.read_csv('../00_Data/liu_mut_corrected_genes.csv', index_col = 0)
    riaz_mut_raw = pd.read_csv('../00_Data/riaz_mut.csv', index_col = 0)
    hugo_mut_raw = pd.read_csv('../00_Data/hugo_mut.csv', index_col = 0)
    msk_mut_raw = pd.read_csv('../00_Data/msk_tmb_mut.csv', index_col = 0)

    #liu_mut_raw
    #riaz_mut_raw
    #hugo_mut_raw
    #msk_mut_raw

    #next, generate the raw GO gene sets

    GO = pd.read_csv('../00_Data/GO.csv',low_memory=False)
    # drop the link column
    GO_drop = GO.drop(columns=['link'])
    # tranpsose
    GO_drop_t = GO_drop.transpose()
    # and set first row to columns and drop superfluous row
    GO_drop_t.columns = GO_drop_t.iloc[0]
    GO_drop_t = GO_drop_t.drop(index='pathway')

    # dictionary mapping from pathway to genes, remove nans
    GO_dict_raw = {GO_drop_t.keys()[i]: [v for v in list(set(GO_drop_t[GO_drop_t.keys()[i]])) if v!='nan'] for i in range(len(GO_drop_t.keys()))}

    #next, create dictionaries of the pathways with their associated genes. Do this for both the single entries and the intersection entries.
    #this is where the intersection happens for the indicated dictionaries

    GO_dict_liu = {}
    GO_dict_riaz = {}
    GO_dict_both = {}
    GO_dict_hugo = {}
    GO_dict_trio = {}
    GO_dict_msk = {}
    GO_dict_all = {}

    for pathway in GO_dict_raw.keys():
            GO_dict_liu[pathway] = list(set(liu_mut_raw.keys()) & set(GO_dict_raw[pathway]))

    for pathway in GO_dict_raw.keys():
            GO_dict_riaz[pathway] = list(set(riaz_mut_raw.keys()) & set(GO_dict_raw[pathway]))

    for pathway in GO_dict.keys():
            GO_dict_both[pathway] = list(set(liu_mut_raw.keys()) & set(riaz_mut_raw.keys()) & set(GO_dict_raw[pathway]))

    for pathway in GO_dict.keys():
            GO_dict_hugo[pathway] = list(set(GO_dict_raw[pathway]) & set(hugo_mut_raw.keys()))

    for pathway in GO_dict.keys():
            GO_dict_trio[pathway] = list(set(liu_mut_raw.keys()) & set(riaz_mut_raw.keys()) & set(GO_dict_raw[pathway]) & set(hugo_mut_raw.keys()))

    for pathway in GO_dict.keys():
            GO_dict_msk[pathway] = list(set(GO_dict_raw[pathway]) & set(msk_mut_raw.keys()))

    for pathway in GO_dict.keys():
            GO_dict_all[pathway] = list(set(liu_mut_raw.keys()) & set(riaz_mut_raw.keys()) & set(GO_dict_raw[pathway]) & set(hugo_mut_raw.keys()) & set(msk_mut_raw.keys()))

    #count the number of genes in the pathways

    liu_gene_number_list = [len(GO_dict_liu[pathway]) for pathway in GO_dict_liu.keys()]

    riaz_gene_number_list = [len(GO_dict_riaz[pathway]) for pathway in GO_dict_riaz.keys()]

    riaz_liu_intersect_gene_number_list = [len(GO_dict_both[pathway]) for pathway in GO_dict_both.keys()]

    hugo_gene_number_list =  [len(GO_dict_hugo[pathway]) for pathway in GO_dict_hugo.keys()]

    trio_gene_number_list = [len(GO_dict_trio[pathway]) for pathway in GO_dict_trio.keys()]

    msk_gene_number_list = [len(GO_dict_msk[pathway]) for pathway in GO_dict_msk.keys()]

    all_gene_number_list = [len(GO_dict_all[pathway]) for pathway in GO_dict_all.keys()]


    #now make the final dataframe

    supp_table_5 = pd.DataFrame()
    supp_table_5_extended = pd.DataFrame()

    row_label_list = ['training and validation shared genes', 'training, validation, and testing shared genes','training, validation, testing, and msk shared genes']
    extended_row_label_list = ['training and validation shared genes', 'training, validation, and testing shared genes','training, validation, testing, and msk shared genes','training only genes','validation only genes','testing only genes','msk only genes']

    supp_table_list_of_lists = [riaz_liu_intersect_gene_number_list,trio_gene_number_list,all_gene_number_list]
    supp_table_list_of_lists_extended = [riaz_liu_intersect_gene_number_list,trio_gene_number_list,all_gene_number_list,liu_gene_number_list, riaz_gene_number_list, hugo_gene_number_list, msk_gene_number_list]

    for label_number in range(len(row_label_list)):
        supp_table_5[row_label_list[label_number]] = supp_table_list_of_lists[label_number]

    for label_number in range(len(extended_row_label_list)):
        supp_table_5_extended[extended_row_label_list[label_number]] = supp_table_list_of_lists_extended[label_number]

    supp_table_5_t = supp_table_5.transpose()
    supp_table_5_extended_t = supp_table_5_extended.transpose()

    supp_table_5_t.columns = GO_dict_raw.keys()
    supp_table_5_extended_t.columns = GO_dict_raw.keys()

    #export to csv
    supp_table_5_t.to_csv('../40_Supplementary_Tables/supplementary_table_5.csv')
    supp_table_5_extended_t.to_csv('../40_Supplementary_Tables/supplementary_table_5_extended.csv')


def make_plots_main():
    make_fig_1a()
    make_fig_1b()
    make_fig_1c()
    make_fig_2a()
    make_fig_2b()
    make_fig_2c()
    fig_2e()
    make_fig_3a()
    make_fig_3b()
    make_fig_4abd()
    make_fig_4c()
