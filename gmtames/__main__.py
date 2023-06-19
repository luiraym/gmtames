'''
gmtames
__main__.py

Raymond Lui
9-July-2022
'''




import argparse
import pathlib
import logging
import logging.config

from gmtames.logging import generateLoggingConfig

from gmtames.data import generateBaseDatasets
from gmtames.data import generateApplicabilityDomainDatasets
from gmtames.data import describeBaseDatasets
from gmtames.data import correlateBaseDatasets
from gmtames.mtg import runMTGExperiment
from gmtames.results import calculateResults




def parseArgs():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='mode', required=True, help='select mode to run')

    # Data generation and analysis mode
    data_mode = subparser.add_parser('data', help='generate, describe, and plot base datasets')
    data_mode.add_argument('--generateBaseDatasets', action='store_true', help='generate base datasets from master datasets')
    data_mode.add_argument('--generateApplicabilityDomainDatasets', action='store_true')
    data_mode.add_argument('--describeBaseDatasets', action='store_true', help='calculate descriptive statistics for base datasets')
    data_mode.add_argument('--correlateBaseDatasets', action='store_true', help='compute strain task correlation matrix for base datasets')
    data_mode.add_argument('--testsplit', required=True, help='specify algorithm used to split out the test set')
    data_mode.add_argument('--output', required=True, help='specify path to output folder')

    # Mechanistic task grouping experiment mode
    mtg_mode = subparser.add_parser('mtg', help='run mechanistic task grouping experiments')
    mtg_mode.add_argument('--tasks', required=True, help='specify task(s) to create n-task neural network')
    mtg_mode.add_argument('--testsplit', required=True, help='specify algorithm used to split out the test set')
    mtg_mode.add_argument('--output', required=True, help='specify path to output folder')
    mtg_mode.add_argument('--device', default='cpu', help='specify "cuda:_" device; default "cpu"')

    # Compute bootstrapped confidence intervals for test predictions
    results_mode = subparser.add_parser('results', help='calculate metrics and bootstrap stats from test predictions')
    results_mode.add_argument('output', help='specify path to output folder which contains test predictions folder')

    args = parser.parse_args()
    
    return args


def parsePathToOutput():
    path_to_output_folder = pathlib.Path('output/')
    path_to_output_folder.mkdir(exist_ok=True)

    path_to_output = path_to_output_folder / args.output
    path_to_output.mkdir(exist_ok=True)

    return path_to_output


def startLogging():
    if args.mode == 'mtg':
        logging_config = generateLoggingConfig(path_to_output)
        logging.config.dictConfig(logging_config)
        logger = logging.getLogger('gmtames')

        logger.info('>>>> START LOGGING A GMTAMES EXPERIMENT')
        for arg, value in vars(args).items():
            logger.info('%s:|%s' % (arg, value))
        logger.info('DONE')

        return logger


def main():
    if args.mode == 'data':
        if args.generateBaseDatasets: generateBaseDatasets(args.testsplit)
        if args.generateApplicabilityDomainDatasets: generateApplicabilityDomainDatasets(args.testsplit)
        if args.describeBaseDatasets: describeBaseDatasets(args.testsplit, path_to_output)
        if args.correlateBaseDatasets: correlateBaseDatasets(args.testsplit, save_heatmap=path_to_output)

    if args.mode == 'mtg':
        runMTGExperiment(args.tasks, args.testsplit, path_to_output, args.device)

    if args.mode == 'results':
        calculateResults(path_to_output)




if __name__ == '__main__':
    try:
        args = parseArgs()
        path_to_output = parsePathToOutput()
        logger = startLogging()
        main()

    except Exception as e:
        logger.exception(e)
        print(e)
