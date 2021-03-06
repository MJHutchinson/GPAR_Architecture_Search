import os
import pickle
import argparse
from collections import namedtuple, defaultdict

import numpy as np
from tqdm import tqdm

import plotting
from plotting_config import *
import plotting_config

final_thesis_dir = '/home/mjhutchinson/Documents/University/4th Year/4th Year Project/Final Thesis/Thesis-LaTeX/Chapter5/Figs'

IncrementalIterationResults = namedtuple('IncrementalIterationResults',
                              [
                                'iteration',
                                'x',
                                'y',
                                'y_tested_flags',
                                'y_next_flags',
                                'x_test',
                                'test_stats',
                                'acquisition_value',
                                'x_best', 'y_best'
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

parser.add_argument('-n', '--name', type=str,
                    required=True,
                    help='Experiment version to analyse for')

parser.add_argument('-f', '--full', action='store_true',
                    help='Generate plots of each iteration for the experiment')

parser.add_argument('--outdir', type=str, help='Directory to get output from',
                    default='output')

parser.add_argument('--experiment_extra', type=str, help='identifier added to end of experiment name')

args = parser.parse_args()

print(args)

outdir = os.path.join(args.outdir, 'searches', args.experiment, args.data, args.name) #f'/home/mjhutchinson/Documents/MachineLearning/gpar/output/searches/{args.experiment}/{args.data}'

dirs = [dir for dir in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, dir))]

def match(file_args):
    if not args.experiment == file_args.experiment+args.experiment_extra:
        return False
    if not args.data == file_args.data:
        return False
    if not args.rmse == file_args.rmse:
        return False
    # if not args.name == file_args.name:
    #     return False
    return True

match_dirs = []
for dir in dirs:
    try:
        config = pickle.load(open(os.path.join(outdir, dir, 'config.pkl'), 'rb'))
        if match(config):
            match_dirs.append(dir)
    except:
        pass

dirs = match_dirs

experiments = [Experiment(pickle.load(open(os.path.join(outdir, dir, 'config.pkl'), 'rb')), pickle.load(open(os.path.join(outdir, dir, 'results.pkl'), 'rb'))) for dir in dirs]

experiment_types = defaultdict(list)

dump_dict = {}

for experiment in experiments:
    experiment_types[(experiment.config.acquisition if not experiment.config.random else "Random", experiment.config.final)].append(experiment.results)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(text_width, text_height/2.1))
ax1.set_ylabel('Mean best search \n value found')
ax2.set_ylabel('Standard deviation \n of best values found')
ax2.set_xlabel('Percentage of points in space sampled')
# ax3.set_title('Percentage of points sampled')

for idx, key in enumerate(experiment_types.keys()):
    results = experiment_types[key]
    if results is not []:

        lens = [len(results[i]) for i in range(len(results))]
        min_len = min(lens)

        iteration_results = np.zeros([min_len, len(results)])
        iterations = np.zeros([min_len, len(results)])

        for i in range(min_len):
            for j in range(len(results)):
                iteration_result = np.squeeze(results[j][i].y_best)
                if len(iteration_result.shape) == 2:
                    iteration_results[i, j] = iteration_result[:, -1]
                elif len(iteration_result.shape) == 1:
                    iteration_results[i, j] = iteration_result[-1]
                else:
                    iteration_results[i, j] = iteration_result

                iterations[i, j] = results[j][i].y_tested_flags.sum()
                # iterations[i, j] = (results[j][i].y_tested_flags * np.array([1000,3000,26000])).sum()

        # print(iterations, iteration_results)

        f_max = results[0][0].y[:, -1].max()
        total_points = results[0][0].y.size
        # total_points = results[0][0].y.size * 10000

        ax1.axhline(f_max, label='Maxima value')

        mean = np.mean(iteration_results, axis=1)
        std = np.std(iteration_results, axis=1)

        mean_std = std / np.sqrt(iteration_results.shape[0])

        # This meaning does not work if the results come at different numbers of points acquired for this search type
        ax1.plot(iterations[:, 0]/total_points, mean, c=colors[idx], label=f'{key[0]} {"final only" if key[1] else "all"}')
        ax1.fill_between(iterations[:, 0]/total_points, mean + mean_std, mean - mean_std, color=colors[idx], alpha=0.3)


        ax2.plot(iterations[:, 0]/total_points, std, c=colors[idx], label=f'{key[0]} {"final only" if key[1] else "all"}')

        dump_dict[key] = {
            'iterations': iterations[:, 0],
            'mean': mean,
            'mean_std': mean_std,
            'std': std,
            'best': f_max,
            'total_points': total_points
        }

        # plt.fill_between(iterations[:, 0], np.min(iteration_results, axis=1), np.max(iteration_results, axis=1), color=colors[idx], alpha=0.3)

        # plt.plot(iterations[:, 0], np.min(iteration_results, axis=1), c=colors[idx], linestyle='dashed')
        # plt.plot(iterations[:, 0], np.max(iteration_results, axis=1), c=colors[idx], linestyle='dashed')

        # plt.plot(iterations[:, 0], np.mean(iteration_results, axis=1) + np.std(iteration_results, axis=1), c=colors[idx], linestyle='dashed')
        # plt.plot(iterations[:, 0], np.mean(iteration_results, axis=1) - np.std(iteration_results, axis=1), c=colors[idx], linestyle='dashed')
        #
        # plt.fill_between(iterations[:, 0],
        #                  np.mean(iteration_results, axis=1) + np.std(iteration_results, axis=1),
        #                  np.mean(iteration_results, axis=1) - np.std(iteration_results, axis=1),
        #                  color=colors[idx], alpha=0.2)

plt.legend()
savefig(outdir + '/search_results')
savefig(os.path.join(final_thesis_dir, args.data, 'search_comparison' + args.experiment_extra), png=False, pdf=True)
plt.close()

pickle.dump({'name': args.name, 'data': dump_dict}, open(os.path.join(outdir, 'combined_results.pkl'), 'wb'))

seed_dict = defaultdict(list)

for experiment in experiments:
    seed_dict[experiment.config.seed].append(experiment)

for seed in seed_dict.keys():
    results = seed_dict[seed]

    if results is not []:
        fig, ax = plt.subplots(1, 1, figsize=(text_width, text_height/2.1))
        ax.set_title(f'Seed {seed}')
        ax.set_xlabel('Percentage of points in space sampled')

        for j, result in enumerate(results):
            iteration_results = np.zeros([len(result.results)])
            iterations = np.zeros([len(result.results)])

            for i in range(len(result.results)):
                iteration_result = np.squeeze(result.results[i].y_best)
                if len(iteration_result.shape) == 2:
                    iteration_results[i] = iteration_result[:, -1]
                elif len(iteration_result.shape) == 1:
                    iteration_results[i] = iteration_result[-1]
                else:
                    iteration_results[i] = iteration_result

                iterations[i] = result.results[i].y_tested_flags.sum()

            f_max = result.results[-1].y[:, -1].max()

            total_points = result.results[-1].y.size

            plt.axhline(f_max, label='Maxima value')
            plt.plot(iterations/total_points, iteration_results, c=colors[j], label=f'{result.config.acquisition} {"final only" if result.config.final else "all"}')

        plt.legend()
        savefig(outdir + f'/search_results_{seed}')
        plt.close()


if args.full:
    for experiment in tqdm(experiments, desc='Looping experiments'):

        fig_dir = os.path.join(outdir, os.path.split(experiment.config.outdir)[-1])

        ## make incremental gif

        # images = [img for img in os.listdir(fig_dir) if img]

        for iteration in tqdm(experiment.results, desc=f'Looping iterations: {experiment.config.outdir.split("/")[-1]}'):
            if iteration.x_test is None:
                continue

            plotting.plot_iteration_incremental(**iteration._asdict(), fig_dir=fig_dir)
            pass

        f_bests = np.squeeze([np.squeeze(result.y_best) for result in experiment.results])
        acquired = np.squeeze([result.y_tested_flags.sum() for result in experiment.results])
        f_max = experiment.results[-1].y[:, -1].max()

        plotting.plot_search_results(acquired, f_bests, f_max, fig_dir)
