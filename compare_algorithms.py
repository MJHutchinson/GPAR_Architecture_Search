import os
import pickle
import argparse
from collections import namedtuple, defaultdict

import numpy as np
import matplotlib.pyplot as plt

import plotting
from plotting_config import *



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

compare_fig, compare_ax = plt.subplots(1,1,figsize=(text_width, text_width/1.5))


def custom_key_sort(input):
    try:
        key = float(input['name'].split('_')[0])
    except ValueError:
        key = 50

    return key

for key in sorted(results[0]['data'].keys()):
    fig = plt.figure()

    final_mean = []
    final_sd = []
    samples = []

    for i, result in enumerate(sorted(results, key=lambda x: custom_key_sort(x))):
        plt.plot(result['data'][key]['iterations'], result['data'][key]['mean'], c=colors[i], label=result['name'].replace('_', ' '))

        final_mean.append(result['data'][key]['mean'][-1])
        final_sd.append(result['data'][key]['mean'][-1])
        samples.append(result['name'].split('_')[0])

    compare_ax.scatter(samples, final_mean, label=key)

    plt.legend()
    savefig(os.path.join(outdir, f'{args.data}-{key}'), pdf=False, svg=False)