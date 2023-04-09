'''
gmtames
data.py

Raymond Lui
13-July-2022
'''




import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




PATH_TO_MASTER_DATASETS = 'gmtames/data/master_datasets/gmtamesQSAR_'

PATH_TO_BASE_DATASETS = 'gmtames/data/base_datasets/gmtamesQSAR_'

PATH_TO_AD_DATASETS = 'gmtames/data/ad_datasets/gmtamesAD_'

STRAIN_LIST = [
    'TA100', 'TA100_S9', 'TA102', 'TA102_S9', 'TA104', 'TA104_S9', 'TA1535', 'TA1535_S9',
    'TA1537', 'TA1537_S9', 'TA1538', 'TA1538_S9', 'TA97', 'TA97_S9', 'TA98', 'TA98_S9'
]

BASE_DATASET_TYPE_LIST = ['train', 'val', 'test']

MODELLING_DATASET_TYPE_LIST = ['train', 'val', 'trainval', 'test']




def generateBaseDatasets():
    # Read master datasets from file
    master_endpoints = pd.read_csv(PATH_TO_MASTER_DATASETS  + 'endpoints.csv')
    master_fingerprints = pd.read_csv(PATH_TO_MASTER_DATASETS + 'fingerprints.csv')
  
    for strain in STRAIN_LIST:
        for t in BASE_DATASET_TYPE_LIST:
            # Extract base dataset for respective strain and train/val/test from the master endpoint dataset
            base_dataset = master_endpoints.loc[
                (master_endpoints['Strain'] == strain) & 
                (master_endpoints['TrainValTest'] == t.capitalize())
            ]

            # Clean up base dataset by renaming endpoint column after respective strain then dropping meta columns
            base_dataset = base_dataset.rename(columns={'Endpoint': strain})
            base_dataset = base_dataset.drop(columns=['Strain', 'TrainValTest'])

            # Add fingerprints to base dataset by left merging with master fingerprint dataset based on MultitaskAmesQSAR_ID column
            # Clean up base dataset by moving strain endpoint column back to the end after appended fingerprints
            base_dataset = base_dataset.merge(right=master_fingerprints, how='left', on='gmtamesQSAR_ID')
            endpoint_col = base_dataset.pop(strain)
            base_dataset.insert(len(base_dataset.columns), endpoint_col.name, endpoint_col)

            # Write base dataset to file
            base_dataset.to_csv(PATH_TO_BASE_DATASETS + strain + '_' + t + '.csv', index=False)

    return None


def generateModellingDatasets(selected_tasks):
    task_list = selected_tasks.split(',')
    last_n_cols = -(len(task_list))

    assert len(task_list) != 0, 'Need to specify at least one strain as a task'
    assert set(task_list).issubset(STRAIN_LIST), 'Check all specified tasks are valid strains'

    # Read from file the base datasets for the selected strains and respective train/test ID
    # Concatenate/stack atop each other the strain-specific base datasets for each train and test ID
    # Store the raw modelling train and test datasets as items in a dictionary
    modelling_datasets = {}
    for t in MODELLING_DATASET_TYPE_LIST:
        if t in ['train', 'val', 'test']:
            modelling_datasets[t] = pd.concat([pd.read_csv(PATH_TO_BASE_DATASETS + strain + '_' + t + '.csv') for strain in task_list])
    
        elif t in ['trainval']:
            trainval_type_list = ['train', 'val']
            trainval_dataset = pd.concat([pd.read_csv(PATH_TO_BASE_DATASETS + strain + '_' + t_ + '.csv') for strain in task_list for t_ in trainval_type_list])
            modelling_datasets[t] = trainval_dataset.sort_values('gmtamesQSAR_ID')
    
    for t, dataset in modelling_datasets.items():
        # Consolidate each raw train/val/test dataset by grouping together MultitaskAmesQSAR_ID duplicates
        ## as_index=False to keep MultitaskAmesQSAR_IDs a separate column and not integrated as DataFrame indices
        ## sort=False to prevent alphabetical sorting of MultitaskAmesQSAR_IDs (i.e. 1,10,100,1000,2,20,200,2000,...,9,90,900)
        # Take the first value of the each grouping since the MultitaskAmesQSAR_IDs should be the same
        dataset = dataset.groupby(by='gmtamesQSAR_ID', as_index=False, sort=False).first()

        # One hot encode endpoints
        dataset.iloc[:, last_n_cols:] = dataset.iloc[:, last_n_cols:].replace('Positive', 1).replace('Negative', 0).fillna(np.nan)

        # Overwrite the raw train/val/test dataset with new dataset
        modelling_datasets[t] = dataset

    return modelling_datasets, task_list


def generateApplicabilityDomainDatasets():
    for strain in STRAIN_LIST:
        modelling_datasets, task_list = generateModellingDatasets(strain)
        
        for dataset in modelling_datasets:
            ad_dataset = modelling_datasets[dataset].drop(columns=['gmtamesQSAR_ID', strain])
            ad_dataset.to_csv(PATH_TO_AD_DATASETS + strain + '_' + dataset + '.csv', index=False, header=False)

    return None


def describeBaseDatasets(path_to_output):
    path_to_output.mkdir(exist_ok=True)

    task_list_mod = [strain.replace('_', '+') for strain in STRAIN_LIST]
    final_stats = []
    
    for strain in STRAIN_LIST:
        modelling_datasets, task_list = generateModellingDatasets(strain)
        strain_stats = []
        
        total_dataset = pd.concat([modelling_datasets['train'], modelling_datasets['val'], modelling_datasets['test']])
        n_total_size = total_dataset[strain].count()
        n_total_pos = total_dataset[strain].sum()
        n_total_neg = n_total_size - n_total_pos
        total_stats = pd.Series([n_total_size, (n_total_neg / n_total_pos)], index=['Size', 'CB'])
        strain_stats.append(total_stats)

        for dataset in modelling_datasets:
            n_size = modelling_datasets[dataset][strain].count()
            n_pos = modelling_datasets[dataset][strain].sum()
            n_neg = n_size - n_pos
            dataset_stats = pd.Series([n_size, (n_neg / n_pos)], index=['Size', 'CB'])
            strain_stats.append(dataset_stats)

        strain_stats = pd.concat(strain_stats, keys=['Total', 'Training', 'Validation', 'Train+Validation', 'Test'])
        final_stats.append(strain_stats)

    final_stats = pd.concat(final_stats, keys=task_list_mod, axis=1) 
    published_stats = final_stats.T.loc[:, ['Total', 'Training', 'Validation', 'Test']]
    published_stats.index.name = 'Strain'
    published_stats.columns.names = ['Dataset', 'Statistic']

    ad_datasets = ['val', 'test']
    final_ad = []
    for strain in STRAIN_LIST:
        strain_ad = []
        for dataset in ad_datasets:
            dataset_ad = pd.read_csv('gmtames/data/ad_datasets/ad_results/%s_%s.csv' % (strain, dataset))
            dataset_ad = dataset_ad.loc[:, ['Approach', 'Test inside AD', 'Test outside AD']]
            dataset_ad = dataset_ad.set_index('Approach')
            dataset_ad['AD'] = (dataset_ad['Test inside AD'] / (dataset_ad['Test inside AD'] + dataset_ad['Test outside AD'])) * 100
            dataset_ad.loc['Mean'] = dataset_ad.mean()
            dataset_ad = dataset_ad.loc['Distance from centroid', 'AD']
            strain_ad.append(dataset_ad)
        final_ad.append(strain_ad)
    
    final_ad = pd.DataFrame(final_ad, index=task_list_mod, columns=ad_datasets)

    published_stats['Validation', 'AD'] = final_ad['val']
    published_stats['Test', 'AD'] = final_ad['test']
    def _parseStats(df):
        dataset_type = df.columns.unique(level='Dataset')[0]
        for indices in df.index:
            if dataset_type == 'Total' or dataset_type == 'Training':
                parsed = '%i (1:%.0f)' % (  # Float interpolation with 0 precision rounds up/down vs. integer interpolation only rounds down
                    df.loc[indices, (dataset_type, 'Size')],
                    df.loc[indices, (dataset_type, 'CB')]
                )
            if dataset_type == 'Validation' or dataset_type == 'Test':
                parsed = '%i (1:%.0f, %.0f%%)' % (
                    df.loc[indices, (dataset_type, 'Size')],
                    df.loc[indices, (dataset_type, 'CB')],
                    df.loc[indices, (dataset_type, 'AD')]
                )
            df.loc[indices, dataset_type] = parsed
        return df
    published_stats = published_stats.groupby(axis='columns', level='Dataset', group_keys=False).apply(_parseStats)
    published_stats = published_stats.loc[:, [('Total', 'Size'), ('Training', 'Size'), ('Validation', 'Size'), ('Test', 'Size')]]
    published_stats.columns = published_stats.columns.droplevel('Statistic')

    published_stats.to_csv(path_to_output / 'gmtamesQSAR_descriptive_stats.csv')

    return None


def correlateBaseDatasets(save_heatmap=None):
    path_to_output = save_heatmap
    path_to_output.mkdir(exist_ok=True)
    
    # Load modelling dataset for all strains
    modelling_datasets, task_list = generateModellingDatasets(','.join(STRAIN_LIST))
    
    # Combine train/val/test modelling datasets and extract strain task labels
    concat_datasets = pd.concat([modelling_datasets['train'], modelling_datasets['val'], modelling_datasets['test']])
    grouped_datasets = concat_datasets.groupby(by='gmtamesQSAR_ID', as_index=False, sort=False).first()
    strain_task_labels = grouped_datasets.iloc[:, -(len(STRAIN_LIST)):]
    
    # Compute correlation matrix for strain tasks
    strain_task_correlation = strain_task_labels.corr()

    # Compute overlap matrix for strain tasks (https://stackoverflow.com/questions/18233864)
    mask = pd.notnull(strain_task_labels)
    l = len(strain_task_labels.columns)
    strain_task_overlap = np.zeros((l, l))
    for i in range(l):
        strain_total = mask.iloc[:, i].sum()
        for j in range(i + 1):  # range(i+1) loops in 0.027s since it only iterates through half the matrix vs range(l) in 0.047s
            strain_task_overlap[i, j] = strain_task_overlap[j,i] = (mask.iloc[:, i] & mask.iloc[:, j]).sum()
    strain_task_overlap = pd.DataFrame(strain_task_overlap, index=strain_task_labels.columns, columns=strain_task_labels.columns)

    # Compute relative overlap matrix for strain tasks
    strain_task_overlap_rel = copy.deepcopy(strain_task_overlap)
    for i in range(l):
        total = strain_task_overlap_rel.iloc[i, i]
        for j in range(l):  # range(i+1) and df[i,j]=df[j,i] won't work here since this is not a symmentrical matrix
            val = strain_task_overlap_rel.iloc[i, j]
            strain_task_overlap_rel.iloc[i, j] = (val / total) * 100

    # Generate correlation and overlap heatmaps for strain tasks
    if save_heatmap is not None:
        data = [strain_task_overlap_rel, strain_task_correlation]
        ranges = [[0, 100], [0.0, 1.0]]
        cbars = [np.arange(0, 110, 10), np.arange(0.0, 1.1, 0.1)]
        annotations = ['%i%%', '%.2f']
        titles = ['A. Strain task dataset overlap', 'B. Strain task correlation']

        fig, axs = plt.subplots(1, 2, figsize=(35, 15))
        for col in range(2):
            ax = axs[col]
            im = ax.imshow(data[col], cmap='RdYlBu_r', vmin=ranges[col][0], vmax=ranges[col][1])

            ax.set_xticks(range(len(data[col])))
            ax.set_yticks(range(len(data[col])))
            ax.set_xticklabels([s.replace('_', '+') for s in data[col].columns], rotation=45, horizontalalignment='left', weight='semibold', size='x-large')
            ax.set_yticklabels([s.replace('_', '+') for s in data[col].index], weight='semibold', size='x-large')
            ax.xaxis.set_ticks_position('top')
            ax.set_title(titles[col], weight='black', size=20, loc='left', pad='40')

            cax = ax.inset_axes([1.02, 0.0, 0.03, 1.0])
            cbar = fig.colorbar(im, ticks=cbars[col], ax=ax, cax=cax)
            if col == 0: cbar.ax.set_yticklabels(['%i%%' % i for i in cbar.get_ticks()])
            cbar.ax.tick_params(labelsize='x-large')

            for i in range(len(data[col].index)):
                for j in range(len(data[col].columns)):
                    annotation = annotations[col] % data[col].iloc[i, j]
                    if col == 0:
                        annotation += '\n(%i)' % strain_task_overlap.iloc[i, j]
                        if data[col].iloc[i, j] >= 80 or data[col].iloc[i, j] <= 20:
                            ax.text(j, i, annotation, size='large', ha='center', va='center', color='w')
                        else:
                            ax.text(j, i, annotation, size='large', ha='center', va='center')
                    if col == 1:
                        if data[col].iloc[i, j] >= 0.8 or data[col].iloc[i, j] <= 0.2:
                            ax.text(j, i, annotation, size='large', ha='center', va='center', color='w')
                        else:
                            ax.text(j, i, annotation, size='large', ha='center', va='center')

        plt.savefig(path_to_output / 'gmtamesQSAR_heatmaps.svg', bbox_inches='tight')

    return strain_task_correlation
