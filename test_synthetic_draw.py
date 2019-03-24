import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from gpar import GPARRegressor
from stheno import B, GP, EQ, Delta, model

import plotting_config

B.epsilon = 1e-6  # Set regularisation a bit higher to ensure robustness.

seed = 0

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
parser.add_argument('-n', '--noise', type=float, default=0.01, help='Initial noise in the function')
parser.add_argument('-l', '--latent', action='store_true', help='Sample from the latent function (no noise)')
parser.add_argument('-f', '--fit', action='store_true', help='Fit GPAR to the sampled function and plot')
parser.add_argument('--show', action='store_true', help='Show plots at end of fit')

args = parser.parse_args()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

def transform_x(x):
    return np.stack((x[:, 0], np.log10(x[:, 1])), axis=0).T

x1 = np.linspace(1,5,5)
x2 = np.concatenate([np.linspace(1, 9, 9), np.linspace(10,98,45), np.linspace(100,980,45)])
xx1, xx2 = np.meshgrid(x1, x2)
x = np.stack([np.ravel(xx1), np.ravel(xx2)], axis=1)

model = GPARRegressor(scale=[2., .3], scale_tie=True,
                              linear=True, linear_scale=10., linear_with_inputs=False,
                              nonlinear=False, nonlinear_with_inputs=False,
                              markov=1,
                              replace=True,
                              noise=args.noise)

n = 10

y = model.sample(transform_x(x), p=n, latent=args.latent)
plt.figure(figsize=(20, 10))

cs = ['tab:red', 'tab:blue', 'tab:green', 'tab:pink', 'tab:cyan']

for i in range(y.shape[1]):
    plt.subplot(2, 5, i + 1)
    plt.title('Output {}'.format(i + 1))

    for j, c in enumerate(cs):
        inds = x[:, 0] == j+1

        plt.semilogx(x[inds, 1], y[inds, i], label='{} layers'.format(j+1), c=c)

plt.legend()

plotting_config.savefig(os.path.join(os.curdir, 'output', 'synthetic_seed_tests', f'{args.seed}_{"latent" if args.latent else f"obs_{args.noise}"}'), pdf=False, svg=False)

plt.figure(figsize=(20, 10))

for j, c in enumerate(cs):
    inds = x[:, 0] == j+1

    plt.plot(y[inds, :].T, label='{} layers'.format(j+1), c=c)

plt.legend()

if args.fit:
    print('Fitting')
    model.fit(transform_x(x), y, progressive=True, trace=False, iters=100)

    print('Sampling')
    y_samples = np.array(model.sample(transform_x(x), num_samples=50, latent=True, posterior=True))
    means = y_samples.mean(axis=0)
    stds = y_samples.std(axis=0)
    lowers = np.percentile(y_samples, 2.5, axis=0)
    uppers = np.percentile(y_samples, 100 - 2.5, axis=0)

    print('Plotting')
    plt.figure(figsize=(20, 10))

    for i in range(y.shape[1]):
        plt.subplot(2, 5, i + 1)
        plt.title('Output {}'.format(i + 1))

        for j, c in enumerate(cs):
            inds = x[:, 0] == 1 + j

            plt.semilogx(x[inds, 1], means[inds, i],
                     label='{} layers)'.format(1 + j), c=c, ls='--')
            plt.fill_between(x[inds, 1], lowers[inds, i], uppers[inds, i], color=c, alpha = 0.3)

    plotting_config.savefig(os.path.join(os.curdir, 'output', 'synthetic_seed_tests',
                                         f'{args.seed}_{"latent" if args.latent else f"obs_{args.noise}"}_fit'), pdf=False,
                            svg=False)

if args.show:
    plt.show()