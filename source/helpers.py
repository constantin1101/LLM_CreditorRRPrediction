import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from textwrap import wrap
import numpy as np
#from nam.utils import *

def set_sizes(sc):
    SMALL_SIZE = 10*sc
    MEDIUM_SIZE = 12*sc
    BIGGER_SIZE = 14*sc

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    plt.rc('axes', titlesize=MEDIUM_SIZE)   
    plt.rc('axes', linewidth=2) 

def print_influence(ax, name, influence, ymin=None, ymax=None, g=None, color=None):
    if len(influence) == 2:
        if len(influence['values']) > 2:
            sns.regplot(ax=ax, x=influence['values'], y=influence['scores'], color=color, scatter_kws={'s': 2}, line_kws={'lw':4}) #linewidth=7,
            #ax.set_xlim((2000,4000))
            #ax.set_xticks((2000, 3000, 4000))
        else:
            sns.barplot(ax=ax, data=influence, x='values', y='scores', color=color)
            ax.bar_label(ax.containers[0], fmt='%.2f')
        # sns.barplot(ax=ax, data=influence, x='values', y='scores', color=color)
        # ax.bar_label(ax.containers[0], fmt='%.2f')

        
        ax.set_ylim((ymin, ymax))
        ax.set_yticks([-30, -15, 0, 15, 30])
        ax.set_xlabel('Values')
        ax.set_ylabel('Score')
        ax.grid(linestyle='-', linewidth=0.5)
    elif len(influence) == 3:
        # wont happen!
        sns.heatmap(influence['scores'], cmap='Blues', vmin=ymin, vmax=ymax)
        ax.set_xlabel("\n".join(wrap(name.split(' x ')[0], 20)))
        ax.set_ylabel("\n".join(wrap(name.split(' x ')[1], 20)))
    ax.set_title("\n".join(wrap(name, 15)), weight='bold')
    if ax.is_first_col(): ax.set_ylabel('Score')
    else: ax.set_ylabel('')
    ax.set_xlabel('')

    return ax

def plot_feature_influences(names_list, influences_list, df_global_imp, palette, ymin = -50, ymax = 50, rows=4, figsize=(20,20)):
    fig = plt.figure(figsize=figsize)
    for ind, (name, influence) in enumerate(zip(names_list, influences_list)): 
        cat = df_global_imp.loc[df_global_imp['col'] == name, 'category'].values[0]
        color = palette[cat]
        ax = plt.subplot(int(len(names_list) / rows)+1, rows, ind+1)
        ax = print_influence(ax, name, influence, ymin, ymax, color=color)
        ax.grid(axis='y', linestyle='-', linewidth=0.5, which='minor')
    return fig


def get_most_important_features(feature_names, feature_importances, final_cil_vars, final_cil_vars_str, filter_combined=True):
    dict_global_imp = {'col': feature_names, 'imp': feature_importances}
    df_global_imp = pd.DataFrame(dict_global_imp)

    df_global_imp['category'] = ""
    for cat_column, cat in zip(final_cil_vars, final_cil_vars_str):
        df_global_imp.loc[df_global_imp['col'].isin(cat_column), 'category'] = cat 
    df_global_imp.loc[df_global_imp['col'].str.contains(' x '), 'category'] = 'combined'
    if filter_combined:
        df_global_imp = df_global_imp.drop(df_global_imp[df_global_imp['category'] == 'combined'].index)
    df_global_imp['imp'] = df_global_imp['imp'] /df_global_imp['imp'].abs().max()

    return df_global_imp

def print_global_importances(df_global_imp,
                        palette,
                        num_examples,
                        plot_labels,
                        figsize=(10,10)):
    fig_feature_imp = plt.figure(figsize=figsize)
    ax = sns.barplot(data=df_global_imp,
                    x='imp',
                    y='col',
                    hue='category',
                    dodge=False,
                    order = df_global_imp.sort_values(
                            by='imp', 
                            ascending=False)['col'][:num_examples],
                    palette=palette
                )
    h, l = ax.get_legend_handles_labels()
    x = dict(zip(palette.keys(), plot_labels))
    l = [x[k] for k in l]
    ax.legend(h, l, title='Category', loc='lower right')
    ax.set_xlabel('Relative importance')
    ax.set_ylabel('Feature')
    ax.grid(axis='x', linestyle='-', linewidth=1, which='both')

    return fig_feature_imp

def get_influence_ebm_global(variable: str, ebm_global):
    ind = ebm_global._internal_obj['overall']['names'].index(variable)
    influence_data = ebm_global._internal_obj['specific'][ind]
    if 'names' in ebm_global._internal_obj['specific'][ind].keys():
        values = influence_data['names']
        if len(values) > 2:
            values = values[:-1]
        scores = influence_data['scores']
        return {'values': values, 'scores': scores}
    else: 
        values_x = influence_data['left_names'][:-1]
        values_y = influence_data['right_names'][:-1]
        scores = influence_data['scores']
        return {'values_x': values_x, 'values_y': values_y,  'scores': scores}

def prediction_test_plot(model, X_test, y_test, num_examples=100):
    y_pred = model.predict(X_test[0:num_examples]) 
    fig = plt.figure(figsize=(10,5))
    plt.plot(y_test[0:num_examples], label = 'Real data', marker="x")
    plt.plot(y_pred, label = 'Predicted data', marker="x")
    plt.grid()
    plt.title('Prediction')
    plt.legend()
    
    return fig


def get_feature_influences_nam(dataset, nam_model):
    unique_features, _ = dataset.ufo, dataset.single_features
    mean_pred, feat_data_contrib = calc_mean_prediction(nam_model, dataset)
    feat_data_contrib_pairs = list(feat_data_contrib.items())
    feat_data_contrib_pairs.sort(key=lambda x: x[0])
    mean_pred_pairs = list(mean_pred.items())
    mean_pred_pairs.sort(key=lambda x: x[0])

    influences_list = []
    names_list = []
    for feature_name, feature_value in feat_data_contrib_pairs:
        unique_feat_data = unique_features[feature_name]
        if len(unique_feat_data) == 2:
            influences_list.append({'values': [str(float(i)) for i in unique_feat_data],'scores': feature_value})
        elif len(unique_feat_data)> 2:
            influences_list.append({'values': np.array(unique_feat_data),'scores': feature_value})
        names_list.append(feature_name)

    return names_list, influences_list


def compute_mean_feature_importance_nam(mean_pred, avg_hist_data):
    mean_abs_score = {}
    for k in avg_hist_data:
        try:
            mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - mean_pred[k]))
        except:
            continue
    feature_names, feature_importances = zip(*mean_abs_score.items())
    return feature_names, feature_importances


def select_features(names_list, influences_list, selected_features_list):
    df_d = pd.DataFrame({'names': names_list, 'influences': influences_list})
    df_d = df_d.loc[df_d['names'].isin(selected_features_list)]
    df_d = df_d.sort_values(by='names')
    return df_d['names'], df_d['influences']



