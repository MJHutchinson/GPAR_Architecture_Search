import os
import pickle
import argparse
from collections import namedtuple, defaultdict

import numpy as np
import matplotlib.pyplot as plt

import plotting
from plotting_config import *
import plotting_config

final_thesis_dir = '/home/mjhutchinson/Documents/University/4th Year/4th Year Project/Final Thesis/Thesis-LaTeX/Chapter5/Figs'

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str,
                    default='synthetic',
                    help='Data set to analyse for')

parser.add_argument('-e', '--experiment', type=str,
                    default='synthetic',
                    help='experiment to analyse for')

parser.add_argument('--outdir', type=str, help='Directory to get output from',
                    default='output')

args = parser.parse_args()

print(args)

outdir = os.path.join(args.outdir, 'searches', args.experiment, args.data)

dirs = [dir for dir in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, dir))]
results_files = [os.path.join(outdir, dir, 'combined_results.pkl') for dir in dirs if os.path.isfile(os.path.join(outdir, dir, 'combined_results.pkl'))]
results = [pickle.load(open(file, 'rb')) for file in results_files]

# compare_fig, compare_ax = plt.subplots(1,1,figsize=(text_width, text_width/1.5))


def custom_key_sort(input):
    try:
        key = float(input['name'].split('_')[0])
    except ValueError:
        key = 50

    return key


fig, (mean_ax, std_ax) = plt.subplots(2,1, figsize=(text_width * (text_width / 6.98), text_height/1.05))

for j, key in enumerate(sorted(results[0]['data'].keys())):

    final_mean = []
    final_sd = []
    samples = []

    for i, result in enumerate(sorted(results, key=lambda x: custom_key_sort(x), reverse=True)):
        mean_ax.plot(result['data'][key]['iterations'] / result['data'][key]['total_points'], result['data'][key]['mean'], c=colors[j], linestyle=linestyles[i], label=key[0] + " " + " ".join(result['name'].replace('_', ' ').split(' ')[:2]))
        mean_ax.fill_between(result['data'][key]['iterations'] / result['data'][key]['total_points'], result['data'][key]['mean'] + result['data'][key]['mean_std'], result['data'][key]['mean'] - result['data'][key]['mean_std'], color=colors[j], alpha=0.1  )
        std_ax.plot(result['data'][key]['iterations'] / result['data'][key]['total_points'], result['data'][key]['std'], c=colors[j], linestyle=linestyles[i], label=result['name'].replace('_', ' '))

        final_mean.append(result['data'][key]['mean'][-1])
        final_sd.append(result['data'][key]['mean'][-1])
        samples.append(result['name'].split('_')[0])

    # compare_ax.scatter(samples, final_mean, label=key)

# mean_ax.legend()


mean_lims = {
    'bostonHousing': [-2.57, -2.43],
    'concrete': [-3.1, -2.95],
    'energy': [-0.8, -0.45],
    'kin8nm': [1.26, 1.32],
    'power-plant': [-2.820, -2.795],
    'wine-quality-red': [-0.96, -0.94],
    'yacht': [-1, -0.4],
    'synthetic': [1.3, 1.7]
}

std_lims = {
    'bostonHousing': [0, 0.08],
    'concrete': [0, 0.075],
    'energy': [0, 0.1],
    'kin8nm': [0, 0.05],
    'power-plant': [-0.001, 0.01],
    'wine-quality-red': [-0.001, 0.01],
    'yacht': [0, 0.4],
    'synthetic': [0, 0.3]
}

mean_ax.set_ylim(mean_lims[args.data])
std_ax.set_ylim(std_lims[args.data])

mean_ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=0)

savefig_handle(fig, os.path.join(outdir, f'search_algorithm_comparison'), pdf=True, svg=False)
savefig_handle(fig, os.path.join(final_thesis_dir, args.data, 'search_algorithm_comparison'), png=False, pdf=True, svg=False)

# savefig_handle(compare_fig, os.path.join(outdir, f'{args.data}-compare'), pdf=True, svg=False)