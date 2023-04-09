'''
gmtames
mtg.py

Raymond Lui
1-September-2022
'''




import logging
from timeit import default_timer as timer

import optuna

from gmtames.data import generateModellingDatasets
from gmtames.nn import RANDOM_SEED
from gmtames.nn import SEARCH_SPACE
from gmtames.nn import checkDevice
from gmtames.nn import loadModellingDataset
from gmtames.nn import computeOptunaObjective
from gmtames.nn import testFinalModel




logger = logging.getLogger('gmtames.mtg')




def runMTGExperiment(args_tasks, path_to_output, device):
    # Start MTG experiment
    logger.info('>>>> RUNNING MECHANISTIC TASK GROUPING EXPERIMENT')
    timer_start = timer()

    # Generate modelling datasets from base datasets
    modelling_datasets, tasks = generateModellingDatasets(args_tasks)

    # Check if device on which to load data and run neural networks is valid
    device = checkDevice(device)
    logger.info('Device confirmed:|%s' % device)

    # Load in train and val datasets
    _, train_dataloader = loadModellingDataset(modelling_datasets, tasks, 'train', device)
    val_dataloader_with_id = loadModellingDataset(modelling_datasets, tasks, 'val', device)
     
    # Set up sampler for Optuna optimisation
    grid_search_sampler = optuna.samplers.GridSampler(search_space=SEARCH_SPACE, seed=RANDOM_SEED)

    # Run Optuna hyperparameter optimisation for neural networks
    study = optuna.create_study(sampler=grid_search_sampler, direction='maximize', study_name='_'.join(tasks))
    study.optimize(lambda trial: computeOptunaObjective(trial, train_dataloader, val_dataloader_with_id, tasks, path_to_output, device))
    
    # Retrieve optimisation results
    logger.info('Best trial number:|%s' % study.best_trial.number)
    logger.info('Best trial objective:|%s' % study.best_trial.value)
    logger.info('Best trial params:|%s' % study.best_trial.params)
    
    hyperparam_dict = study.best_trial.params
    hyperparam_dict['best_trial_num'] = study.best_trial.number

    # Load in test dataset
    _, trainval_dataloader = loadModellingDataset(modelling_datasets, tasks, 'trainval', device)
    test_dataloader_with_id = loadModellingDataset(modelling_datasets, tasks, 'test', device)
    
    # Test optimised neural network
    logger.info('Final neural network hyperparams:|%s' % hyperparam_dict)
    testFinalModel(hyperparam_dict, trainval_dataloader, test_dataloader_with_id, tasks, path_to_output, device)

    # Finish MTG experiment
    timer_end = timer()
    study_timing = timer_end - timer_start
    logger.info('Mechanistic task grouping experiment timing:|%s' % study_timing)
    logger.info('DONE')
    
    return study
