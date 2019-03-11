import os
import pickle
import argparse
from collections import namedtuple, defaultdict

import numpy as np

import plotting_config as plotting_config

IterationResults = namedtuple('IterationResults',
                              ['iteration',
                               'x_tested', 'y_tested',
                               'x_remaining', 'y_remaining',
                               'x_next', 'y_next',
                               'x_best', 'y_best',
                               'test_stats', 'remaining_stats',
                               'acquisition'
                               ])
Experiment = namedtuple('Experiment', ['config', 'results'])

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str,
                    default='power-plant',
                    help='Data set to analyse for')

parser.add_argument('-e', '--experiment', type=str,
                    default='weight_pruning_hyperprior3',
                    help='experiment to analyse for')

parser.add_argument('--rmse', action='store_true',
                    help='Perform search on the RMSE instead')

parser.add_argument('-v', '--version', type=str,
                    default='0.1.0',
                    help='Experiment version to analyse for')

parser.add_argument('--outdir', type=str, help='Directory to get output from',
                    default='output')

args = parser.parse_args()

outdir = os.path.join(args.outdir, args.experiment, args.data) #f'/home/mjhutchinson/Documents/MachineLearning/gpar/output/searches/{args.experiment}/{args.data}'

dirs = os.listdir(outdir)

configs = [pickle.load(open(outdir + dir + '/config.pkl', 'rb')) for dir in dirs]

def match(file_args):
    if not args.experiment == file_args.experiment:
        return False
    if not args.data == file_args.data:
        return False
    if not args.rmse == file_args.rmse:
        return False
    if not args.version == file_args.version:
        return False
    return True

configs = [config for config in configs if match(config)]

experiments = [Experiment(config, pickle.load(open(config.outdir + 'results.pkl', 'rb'))) for config in configs]

experiment_types = defaultdict(list)

for experiment in experiments:
    experiment_types[(experiment.config.acquisition if not experiment.config.random else "Random", experiment.config.final)].append(experiment.results)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

plt.figure(figsize=plotting_config.full_width_square)
plt.title(f'Search methods on {args.data} comparison \n Searching for optimal {"RMSE" if args.rmse else "Log Likelihood"}')

for idx, key in enumerate(experiment_types.keys()):
    results = experiment_types[key]
    if results is not []:
        iteration_results = np.zeros([len(results[0]), len(results)])
        iterations = np.zeros([len(results[0]), len(results)])
        for i in range(len(results[0])):
            for j in range(len(results)):
                iteration_results[i, j] = np.squeeze(results[j][i].y_best)
                iterations[i, j] = np.size(results[j][i].y_tested)

        print(iterations, iteration_results)

        # This meaning does not work if the results come at different numbers of points acquired for this search type
        plt.plot(iterations[:, 0], np.mean(iteration_results, axis=1), c=colors[idx], label=f'{key[0]} {"final only" if key[1] else "all"}')
        # plt.fill_between(iterations[:, 0], np.min(iteration_results, axis=1), np.max(iteration_results, axis=1), color=colors[idx], alpha=0.3)

        # plt.plot(iterations[:, 0], np.min(iteration_results, axis=1), c=colors[idx], linestyle='dashed')
        # plt.plot(iterations[:, 0], np.max(iteration_results, axis=1), c=colors[idx], linestyle='dashed')

        # plt.plot(iterations[:, 0], np.mean(iteration_results, axis=1) + np.std(iteration_results, axis=1), c=colors[idx], linestyle='dashed')
        # plt.plot(iterations[:, 0], np.mean(iteration_results, axis=1) - np.std(iteration_results, axis=1), c=colors[idx], linestyle='dashed')

        plt.fill_between(iterations[:, 0],
                         np.mean(iteration_results, axis=1) + np.std(iteration_results, axis=1),
                         np.mean(iteration_results, axis=1) - np.std(iteration_results, axis=1),
                         color=colors[idx], alpha=0.2)



plt.legend()
plt.show()