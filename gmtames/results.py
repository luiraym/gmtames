'''
gmtames
results.py

Raymond Lui
17-Nov-2022
'''




import os
import re
import pathlib
import pickle
import copy
import functools

import pandas as pd
import numpy as np

from sklearn.metrics import balanced_accuracy_score as balanced_accuracy_score_orig
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from gmtames.data import STRAIN_LIST
from gmtames.nn import RANDOM_SEED
from gmtames.nn import FILTER_VALUE




np.random.seed(RANDOM_SEED)




# BLOCK 1

def sensitivity_score(y_true, y_pred):
    return classification_report(y_true, np.rint(y_pred), output_dict=True)['1.0']['recall']


def specificity_score(y_true, y_pred):
    return classification_report(y_true, np.rint(y_pred), output_dict=True)['0.0']['recall']


def balanced_accuracy_score(y_true, y_pred):
    return balanced_accuracy_score_orig(y_true, np.rint(y_pred))


def getMetricName(metric):
    name = re.sub('score', '', metric.__name__.replace('_', ' ')).strip()
    
    if name == 'roc auc': name = name.upper()
    else: name = name.capitalize()

    return name




# BLOCK 2

def calculateMetrics(metrics, y_true, y_pred):
    assert type(metrics) == list, "Specified metric functions must be wrapped in a list"

    metrics = pd.Series(
        [metric(y_true, y_pred) for metric in metrics],
        index=[getMetricName(metric) for metric in metrics],
        name='Observed'
    )

    return metrics


def sampleResample(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    positives = y_true[y_true == 1.0]
    negatives = y_true[y_true == 0.0]

    n_positives = len(positives)
    n_negatives = len(negatives)
    assert n_positives + n_negatives == len(y_true) # Check there are no other labels

    # Generate resample indices for each label
    positives_id = positives.iloc[np.random.randint(0, n_positives, size=n_positives)].index
    negatives_id = negatives.iloc[np.random.randint(0, n_negatives, size=n_negatives)].index
    
    resample = []
    for y in [y_true, y_pred]:
        resample.append(
            pd.concat([
                y.loc[positives_id],
                y.loc[negatives_id]
            ])
        )

    return resample


def generateResamples(*data_args, n_bootstraps):
    # Create a generator instead of storing a big list of n samples
    return (sampleResample(*data_args) for _ in range(n_bootstraps))




# BLOCK 3

def calculateMetricEstimates(metrics, *task_data, n_bootstraps):
    estimates = [
        calculateMetrics(metrics, *resample) for resample in generateResamples(*task_data, n_bootstraps=n_bootstraps)
    ]

    return pd.DataFrame({i_bootstrap: estimate for i_bootstrap, estimate in enumerate(estimates)})


def calculateConfidenceIntervals(estimates, level):
    alpha = 1 - level
    half_alpha = 0.5 * alpha
    percentiles = [half_alpha, 1 - half_alpha]
    intervals = estimates.apply(lambda row: row.quantile(percentiles), axis=1)
    intervals.columns = ['%.1f%%' % (p * 100) for p in percentiles]
    
    return intervals








def loadTestPredictions(f):
    test_predictions = pickle.load(open(f, 'rb'))

    # Retrieve task list from file name
    task_list = f.name.replace('_test_predictions.pkl', '').split('_TA') # Use '_TA' as delimiter so '_S9' strains don't break
    for idx, task in enumerate(task_list):
        if not task.startswith('TA'):
             task_list[idx] = 'TA' + task # Add back 'TA' string
    
    y_true = pd.DataFrame(test_predictions['y_true'], columns=task_list, index=test_predictions['gmtamesqsar_id'])
    y_pred = pd.DataFrame(test_predictions['y_pred'], columns=task_list, index=test_predictions['gmtamesqsar_id'])

    # Separate out strain-specific test predictions
    strain_dataset_dict = {}
    for strain in task_list:
        # Start by splitting out strain-specific y_trues and removing any missing values (from MTL)
        y_true_split = y_true[strain].to_frame().rename(columns={strain: 'y_true'})
        missing_values = y_true_split[y_true_split['y_true'] == -2]
        y_true_strain = y_true_split.drop(missing_values.index)
        
        # Left merge in y_pred instances corresponding to labelled y_true instances
        y_pred_split = y_pred[strain].to_frame().rename(columns={strain: 'y_pred'})
        y_true_pred = y_true_strain.merge(right=y_pred_split, how='left', left_index=True, right_index=True)
        strain_dataset_dict[strain] = y_true_pred.sort_index()

    return strain_dataset_dict, task_list


def calculateResults(path_to_output):
    # STEP 1: PREPARE DIRECTORIES AND METRICS
    path_to_results = pathlib.Path(path_to_output / 'results/')
    path_to_results.mkdir(exist_ok=True)

    path_to_test_predictions = path_to_output / ('test_predictions/')
    test_predictions_dir = os.scandir(path_to_test_predictions)

    metrics = [
        balanced_accuracy_score,
        roc_auc_score,
        sensitivity_score,
        specificity_score,
    ]


    # STEP 2: CALCULATE FULL EXPERIMENT RESULTS FROM TEST PREDICTIONS (-> SUPPLEMENTARY INFO)
    experiment_results_list = []
    experiment_results_name_list = []

    for f in test_predictions_dir:
        strain_dataset_dict, task_list = loadTestPredictions(f)
        task_list_mod = [task.replace('_', '+') for task in task_list]
        
        grouping_results_list = []
        for task in task_list:
            task_observed = calculateMetrics(metrics, strain_dataset_dict[task]['y_true'], strain_dataset_dict[task]['y_pred'])  # Series[metrics]
            task_estimated = calculateMetricEstimates(metrics, strain_dataset_dict[task]['y_true'], strain_dataset_dict[task]['y_pred'], n_bootstraps=1000)  # DataFrame[metrics, bootstrap estimates]
            task_intervals_95 = calculateConfidenceIntervals(task_estimated, level=0.95)  # DataFrame[metrics, lower/upper bounds]
            task_intervals_83 = calculateConfidenceIntervals(task_estimated, level=0.83)  # DataFrame[metrics, lower/upper bounds]

            task_results = pd.concat([task_observed, task_intervals_95, task_intervals_83], axis=1)  # DataFrame[metrics, statistics]
            task_results = task_results.stack()  # Series[(metrics, statistics)]
            grouping_results_list.append(task_results)

        grouping_results = pd.concat(grouping_results_list, keys=task_list_mod, axis=1)  # DataFrame[(metrics, statistics), tasks]
        grouping_results = grouping_results.T  # DataFrame[tasks, (metrics, statistics)]
        experiment_results_list.append(grouping_results)
        experiment_results_name_list.append(', '.join(task_list_mod))

    experiment_results = pd.concat(experiment_results_list, keys=experiment_results_name_list, axis=0)  # DataFrame[(grouping, task), (metrics, statistics)]
    experiment_results.columns.names = ['Metric', 'Statistic']
    experiment_results.index.names = ['Strain task grouping', 'Strain task']
    experiment_results = experiment_results.swaplevel('Strain task grouping', 'Strain task')  # DataFrame[(task, grouping), (metrics, statistics)]
    experiment_results = experiment_results.sort_index(level='Strain task')

    for task in experiment_results.index.unique(level='Strain task'):
        for grouping in experiment_results.loc[task].index:
            grouping_list = grouping.split(', ')
            
            if len(grouping_list) == 1.0: architecture = 'ST'
            if len(grouping_list) == 16.0: architecture = 'uMT'
            if len(grouping_list) > 1.0 and len(grouping_list) < 16.0: architecture = 'gMT'
            experiment_results.loc[(task, grouping), 'Learning architecture'] = architecture
            
    # Specify stable sort algorithm to ensure consecutive sorts are carried over
    experiment_results.sort_values([('Balanced accuracy', 'Observed'), ('ROC AUC', 'Observed'), ('Sensitivity', 'Observed'), ('Specificity', 'Observed')], ascending=False, kind='stable', inplace=True)
    experiment_results.sort_values('Learning architecture', key=lambda x: x.map({'ST': 0, 'uMT': 1, 'gMT': 2}), kind='stable', inplace=True)
    experiment_results.sort_index(level='Strain task', sort_remaining=False, kind='stable', inplace=True)
    
    experiment_results.set_index('Learning architecture', append=True, inplace=True)
    experiment_results = experiment_results.swaplevel('Strain task grouping', 'Learning architecture')  # DataFrame[(task, architecture, grouping), (metrics, statistics)]
    
    def computeSignificance(df):
        metric = df.columns.unique(level='Metric')[0]
        for task in df.index.unique(level='Strain task'):
            st_upper_83 = df.loc[(task, 'ST'), (metric, '91.5%')].values[0]
            st_upper_95 = df.loc[(task, 'ST'), (metric, '97.5%')].values[0]
            for architecture, grouping in df.loc[task].index:
                significance = 'NS'  # Reset significance to "Not Significant" and only reassign var if fulfils the below conditions
                if architecture == 'ST': significance = '-'
                else:
                    mt_lower_83 = df.loc[(task, architecture, grouping), (metric, '8.5%')]
                    mt_lower_95 = df.loc[(task, architecture, grouping), (metric, '2.5%')]
                    if mt_lower_83 > st_upper_83: significance = '*'  # Check 83% first then overwrite with 95% if applicable
                    if mt_lower_95 > st_upper_95: significance = '**'  # Don't use elif as that will evaluate the 83% condition, stop if true, and not proceed with the 95% condition
                df.loc[(task, architecture, grouping), (metric, 'Significance')] = significance
        return df
    experiment_results = experiment_results.groupby(axis='columns', level='Metric', group_keys=False).apply(computeSignificance)  # https://github.com/dask/dask/issues/4592

    experiment_results.to_csv(path_to_results / 'gmtames_full_experiment_results.csv')

    
    # STEP 3: PARSE FULL EXPERIMENT RESULTS INTO ... 
    curated_results = copy.deepcopy(experiment_results)
    curated_results.rename(columns={
        'Balanced accuracy': 'Balanced accuracy full',
        'ROC AUC': 'ROC AUC full',
        'Sensitivity': 'Sensitivity full',
        'Specificity': 'Specificity full'
    }, level='Metric', inplace=True)

    curated_results.reset_index('Strain task grouping', inplace=True)  # Convert grouping index to column so it is not lost when grouped together
    curated_results = curated_results.groupby(level=['Strain task', 'Learning architecture'], sort=False).first()  # Turn sort off in groupby to preserve architecture order.
    curated_results.set_index('Strain task grouping', append=True, inplace=True)

    def _parseStatistics(df):
        metric_full = df.columns.unique(level='Metric')[0]
        metric = metric_full.rstrip(' full')
        for indices in df.index:
            parsed = '%.3f (%.3f-%.3f)' % (
                df.loc[indices, (metric_full, 'Observed')],
                df.loc[indices, (metric_full, '2.5%')],
                df.loc[indices, (metric_full, '97.5%')]
            )
            significance = df.loc[indices, (metric_full, 'Significance')]
            if '*' in significance: parsed = parsed + significance
            df.loc[indices, metric] = parsed
        return df


    # STEP 3A: ... CURATED STRAIN RESULTS (-> TABLES 2, 3)
    strain_results = curated_results.groupby(axis='columns', level='Metric', group_keys=False).apply(_parseStatistics)
    
    strain_results = strain_results.loc[:, ['Balanced accuracy', 'ROC AUC']]
    strain_results.columns = strain_results.columns.droplevel('Statistic')

    substitution_results = strain_results.loc[['TA100', 'TA100+S9', 'TA102', 'TA102+S9', 'TA104', 'TA104+S9', 'TA1535', 'TA1535+S9'], :]
    frameshift_results = strain_results.loc[['TA1537', 'TA1537+S9', 'TA1538', 'TA1538+S9', 'TA97', 'TA97+S9', 'TA98', 'TA98+S9'], :]

    substitution_results.to_csv(path_to_results / 'gmtames_curated_substitution_results.csv')
    frameshift_results.to_csv(path_to_results / 'gmtames_curated_frameshift_results.csv')


    # STEP 3B: ... INTO CURATED AVERAGE RESULTS (-> TABLE 4)
    average_results = curated_results.groupby(level='Learning architecture', sort=False).mean(numeric_only=True)
    
    overall_results = curated_results.reset_index('Learning architecture')
    overall_results.sort_values([('Balanced accuracy full', 'Observed'), ('ROC AUC full', 'Observed'), ('Sensitivity full', 'Observed'), ('Specificity full', 'Observed')], ascending=False, kind='stable', inplace=True)
    overall_results = overall_results.groupby(level='Strain task', sort=False).first()  # Checkpoint - best learning architecture for each strain (to be bolded in Tables 2, 3)
    overall_results['Learning architecture'] = 'Overall'
    overall_results.set_index('Learning architecture', append=True, inplace=True)
    overall_results = overall_results.groupby(level='Learning architecture').mean(numeric_only=True)

    average_results = pd.concat([average_results, overall_results])
        
    def _computeSignificanceMod(df):
        metric = df.columns.unique(level='Metric')[0]
        st_upper_83 = df.loc['ST', (metric, '91.5%')]
        st_upper_95 = df.loc['ST', (metric, '97.5%')]
        for architecture in df.index:
            significance = 'NS'
            if architecture == 'ST': significance = '-'
            else:
                mt_lower_83 = df.loc[architecture, (metric, '8.5%')]
                mt_lower_95 = df.loc[architecture, (metric, '2.5%')]
                if mt_lower_83 > st_upper_83: significance = '*'
                if mt_lower_95 > st_upper_95: significance = '**'
            df.loc[architecture, (metric, 'Significance')] = significance
        return df
    average_results = average_results.groupby(axis='columns', level='Metric', group_keys=False).apply(_computeSignificanceMod)
    average_results = average_results.groupby(axis='columns', level='Metric', group_keys=False).apply(_parseStatistics)

    average_results = average_results.loc[:, ['Balanced accuracy', 'ROC AUC']]
    average_results.columns = average_results.columns.droplevel('Statistic')
    
    average_results.to_csv(path_to_results / 'gmtames_curated_average_results.csv')
