import os
import pickle
import argparse
from collections import namedtuple, defaultdict

import numpy as np
from tqdm import tqdm

import plotting
import plotting_config as plotting_config

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

args = parser.parse_args()

print(args)

outdir = os.path.join(args.outdir, 'searches', args.experiment, args.data, args.name) #f'/home/mjhutchinson/Documents/MachineLearning/gpar/output/searches/{args.experiment}/{args.data}'

dirs = [dir for dir in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, dir))]

def match(file_args):
    if not args.experiment == file_args.experiment+'_incremental':
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

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=plotting_config.full_width_square)
plt.suptitle(f'Search methods on {args.data} comparison \n Searching for optimal {"RMSE" if args.rmse else "Log Likelihood"}')
ax1.set_title('Mean best search value found')
ax2.set_title('Standard deviation of best values found')

for idx, key in enumerate(experiment_types.keys()):
    results = experiment_types[key]
    if results is not []:
        iteration_results = np.zeros([len(results[0]), len(results)])
        iterations = np.zeros([len(results[0]), len(results)])
        for i in range(len(results[0])):
            for j in range(len(results)):
                iteration_result = np.squeeze(results[j][i].y_best)
                if len(iteration_result.shape) == 2:
                    iteration_results[i, j] = iteration_result[:, -1]
                elif len(iteration_result.shape) == 1:
                    iteration_results[i, j] = iteration_result[-1]
                else:
                    iteration_results[i, j] = iteration_result

                iterations[i, j] = results[j][i].y_tested_flags.sum()

        # print(iterations, iteration_results)

        f_max = results[0][0].y[:, -1].max()

        ax1.axhline(f_max, label='Maxima value')

        # This meaning does not work if the results come at different numbers of points acquired for this search type
        ax1.plot(iterations[:, 0], np.mean(iteration_results, axis=1), c=colors[idx], label=f'{key[0]} {"final only" if key[1] else "all"}')
        ax2.plot(iterations[:, 0], np.std(iteration_results, axis=1), c=colors[idx], label=f'{key[0]} {"final only" if key[1] else "all"}')

        dump_dict[key] = {
            'iterations': iterations[:, 0],
            'mean': list(np.mean(iteration_results, axis=1)),
            'std': list(np.std(iteration_results, axis=1)),
            'best': f_max
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
plotting_config.savefig(outdir + '/search_results')
plt.close()

pickle.dump({'name': args.name, 'data': dump_dict}, open(os.path.join(outdir, 'combined_results.pkl'), 'wb'))

seed_dict = defaultdict(list)

for experiment in experiments:
    seed_dict[experiment.config.seed].append(experiment)

for seed in seed_dict.keys():
    results = seed_dict[seed]

    if results is not []:
        fig, ax = plt.subplots(1, 1, figsize=plotting_config.full_width_square)
        ax.set_title(f'Seed {seed}')

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

            plt.axhline(f_max, label='Maxima value')
            plt.plot(iterations, iteration_results, c=colors[j], label=f'{result.config.acquisition} {"final only" if result.config.final else "all"}')

        plt.legend()
        plotting_config.savefig(outdir + f'/search_results_{seed}')
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
