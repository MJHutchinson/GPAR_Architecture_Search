import os
import pickle
import argparse
from collections import namedtuple, defaultdict

import numpy as np
import matplotlib.pyplot as plt

import plotting
import plotting_config as pc



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

for key in sorted(results[0]['data'].keys()):
    fig = plt.figure()
    for i, result in enumerate(sorted(results, key=lambda x: float(x['name'].split('_')[0]))):
        plt.plot(result['data'][key]['iterations'], result['data'][key]['mean'], c=pc.colors[i], label=result['name'].replace('_', ' '))
    plt.legend()
    pc.savefig(os.path.join(outdir, f'{args.data}-{key}'), pdf=False, svg=False)
