import os
import sys
import time
import pickle
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stheno import B
from scipy.stats import norm

from gpar.regression import GPARRegressor
from plotting_config import *

B.epsilon = 1e-5  # Set regularisation a bit higher to ensure robustness.

def transform_x(x):
    return np.stack((x[:, 0], np.log10(x[:, 1])), axis=0).T


# Parse arguments.
parser = argparse.ArgumentParser()

# Experiment arguments
parser.add_argument('-d', '--data', type=str,
                    default='concrete',
                    help='name of data set')
parser.add_argument('-e', '--experiment', type=str,
                    default='weight_pruning_hyperprior3',
                    help='name of the experiment to run on')
parser.add_argument('-i', '--iters', type=int, default=100,
                    help='number of iterations')
parser.add_argument('-q', '--quick', action='store_true',
                    help='quick test on the first three outputs')
parser.add_argument('--rmse', action='store_true',
                    help='predict the RMSE instead')
parser.add_argument('--final', action='store_true',
                    help='fit using only the final observations')
parser.add_argument('--validsmall', action='store_true',
                    help='split data in half to create a validation set')
parser.add_argument('--joint', action='store_true',
                    help='also train jointly')
parser.add_argument('--subexperiment', type=str, default='None', help='Save figs in a subfolder')
parser.add_argument('--seed', type=int, default=0, help='random number seed to use for the experiment')

# Model arguments

parser.add_argument('--input_linear',       action='store_true',  help='Use linear kernal on inputs')
parser.add_argument('--output_linear',      action='store_true',  help='Use linear kernal on outputs')
parser.add_argument('--output_nonlinear',   action='store_true',  help='Use nonlinear kernal on outputs')
parser.add_argument('--scale_tie',          action='store_true',  help='Tie scales for input between outputs')
parser.add_argument('--markov',             type=int,             default=None,   help='Treat outputs as markov order'
                                                                                       ' n. 0 sets no markov structure')

args = parser.parse_args()

print(args)

if args.markov == 0: args.markov = None


data_dir = '/home/mjhutchinson/Documents/MachineLearning/architecture_search_gpar/data'  # The data is located here.
fig_dir = f'/home/mjhutchinson/Documents/MachineLearning/architecture_search_gpar/output/fits/{args.experiment}/{args.subexperiment + "/" if args.subexperiment is not None else ""}'
os.makedirs(fig_dir, exist_ok=True)

fig_path = os.path.join(fig_dir, f'{args.data}-'
                                    f'{"rmse" if args.rmse else "loglik"}-'
                                    f'{"final" if args.final else "all"}-'
                                    f'{"validsmall" if args.validsmall else "validbig"}-'
                                    f'{"joint-" if args.joint else ""}'
                                    f'{"scale_tie-" if args.scale_tie else ""}'
                                    f'{"input_linear-" if args.input_linear else ""}'
                                    f'{"output_linear-" if args.output_linear else ""}'
                                    f'{"output_nonlinear-" if args.output_nonlinear else ""}'
                                    f'markov_{args.markov}-'
                                    f'{args.seed}'
                        )

if os.path.isfile(fig_path + '.png'):
    print('Experiment already run. Exiting')
    sys.exit(0)

np.random.seed(args.seed)


model = GPARRegressor(scale=[1., .5],
                      scale_tie=args.scale_tie,
                      linear=args.output_linear, linear_scale=10.,
                      input_linear=args.input_linear, input_linear_scale=10.,
                      nonlinear=args.output_nonlinear,
                      markov=args.markov,
                      replace=True,
                      noise=0.01)

# model2 = GPARRegressor(scale=[1., .5], scale_tie=True,
#                           linear=True, linear_scale=10., input_linear=False,
#                           nonlinear=False,
#                           markov=1,
#                           replace=True,
#                           noise=0.01)

# Load data.
formatting_args = (data_dir,
                   args.data,
                   args.experiment,
                   'rmse' if args.rmse else 'loglik')
x = np.genfromtxt('{}/{}/'
                  '{}/x_{}.txt'
                  ''.format(*formatting_args))
y = np.genfromtxt('{}/{}/'
                  '{}/f_{}.txt'
                  ''.format(*formatting_args))

if args.final:
    y = (y[:, -1])[:, np.newaxis]

# if args.valid:
#     def gen_dataset(x, y):
#         # inds = x[:, 1].argsort(axis=0)
#         # x, y = x[inds], y[inds]
#         # inds = x[:, 0].argsort(axis=0, kind='mergesort')
#         # x, y = x[inds], y[inds]
#
#         # return x[::2, :],y[::2, :],x[1::2, :],y[1::2, :]
#         inds = np.arange(len(x))
#         np.random.shuffle(inds)
#         middle = int(len(x)/2.)
#         #middle = 10
#         return x[:middle, :], y[:middle, :], x[middle:, :], y[middle:, :]
# else:
#     def gen_dataset(x, y):
#         return x,y,None,None

def gen_dataset(x, y):
    # inds = x[:, 1].argsort(axis=0)
    # x, y = x[inds], y[inds]
    # inds = x[:, 0].argsort(axis=0, kind='mergesort')
    # x, y = x[inds], y[inds]

    # return x[::2, :],y[::2, :],x[1::2, :],y[1::2, :]
    inds = np.arange(len(x))
    np.random.shuffle(inds)
    x, y = x[inds], y[inds]

    if args.validsmall:
        middle = 12
    else:
        middle = int(len(x)/2.)

    return x[:middle, :], y[:middle, :], x[middle:, :], y[middle:, :]

x, y, x_valid, y_valid = gen_dataset(x, y)

if args.quick:
    y = y[:, :3]
min_layers = int(min(x[:, 0]))
max_layers = int(max(x[:, 0]))

# Print some information about the data.
print('Number of outputs:', y.shape[1])
print('Layers:')
print('   min:', min_layers)
print('   max:', max_layers)

# Normalise data.
y_means = y.mean(axis=0, keepdims=True)
y_stds = y.std(axis=0, keepdims=True)
y = (y - y_means) / y_stds


def unnormalise(y):
    return y * y_stds + y_means


# Create some test inputs.
n = 100
x_tests = [np.stack((i * np.ones(n),
                     np.linspace(min(x[:, 1]), int(max(x[:, 1]) * 1.5), n)), axis=0).T
           for i in range(min_layers, max_layers + 1)]

# Create GPAR.
# model = GPARRegressor(scale=[1., .5], scale_tie=True,
#                       linear=True, linear_scale=10., linear_with_inputs=False,
#                       nonlinear=False, nonlinear_with_inputs=False,
#                       markov=1,
#                       replace=True,
#                       noise=0.01)

# model = GPARRegressor(scale=args.synthetic_scales, scale_tie=True,
#                           linear=True, linear_scale=10., input_linear=False,
#                           nonlinear=False,
#                           markov=1,
#                           replace=True,
#                           noise=args.noise)


# time the fitting
start = time.time()

# Fit model and print hyperparameters.
model.fit(transform_x(x), y,
          iters=args.iters, trace=False, fix=True)
if args.joint:
    model.fit(transform_x(x), y,
              iters=args.iters, trace=False, fix=False)
print('Hyperparameters:')
for k, v in model.get_variables().items():
    print(k)
    print('   ', v)

# Predict.
# preds = []
# for i, x_test in enumerate(x_tests):
#     print('Predicting {}/{}'.format(i + 1, max_layers - min_layers + 1))
#     preds.append(model.predict(transform_x(x_test),
#                                num_samples=50,
#                                latent=True,
#                                credible_bounds=True))

end = time.time()

samples = []
for i, x_test in enumerate(x_tests):
    print('Predicting {}/{}'.format(i + 1, max_layers - min_layers + 1))
    samples.append(model.sample(transform_x(x_test),
                                num_samples=50,
                                latent=True,
                                posterior=True))



# Unnormalise data and predictions.
y = unnormalise(y)
samples = [unnormalise(sample) for sample in samples]
preds = [(sample.mean(axis=0),
          sample.std(axis=0),
          np.percentile(sample, 2.5, axis=0),
          np.percentile(sample, 100 - 2.5, axis=0)
          ) for sample in samples]



# Compute metrics of training
if x_valid is not None:
    print('Predicting validation points')
    samples = unnormalise(model.sample(transform_x(x_valid),
                                num_samples=50,
                                latent=True,
                                posterior=True))


    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)

    valid_rmse = np.mean((y_valid - means) ** 2) ** 0.5
    valid_loglik = model.logpdf(transform_x(x_valid), y_valid) / len(x_valid)
else:
    valid_rmse = 0
    valid_loglik = 0

print('Predicting training points')
samples = unnormalise(model.sample(transform_x(x),
                            num_samples=50,
                            latent=True,
                            posterior=True))


means = np.mean(samples, axis=0)
stds = np.std(samples, axis=0)

train_rmse = np.mean((y - means) ** 2) ** 0.5
train_loglik = model.logpdf(transform_x(x), y) / len(x)

# Plot the result.
if args.final:
    plt.figure(figsize=(text_width, text_width/2))
    plt.title(f'\n Validation RMSE: {valid_rmse:.6f}, Validation loglik: {valid_loglik:.6f}'
              f'\n Training   RMSE: {train_rmse:.6f}, Training   loglik: {train_loglik:.6f}')

    cs = ['tab:red', 'tab:blue', 'tab:green']
    cs_valid = ['tab:pink', 'tab:cyan', 'tab:olive']
    for i in range(y.shape[1]):

        for j, (x_test, (means, sdev, lowers, uppers), c, c_valid) \
                in enumerate(zip(x_tests, preds, cs, cs_valid)):
            inds = x[:, 0] == min_layers + j
            inds_valid = x_valid == min_layers + j

            plt.semilogx(x_test[:, 1], means[:, i],
                         label='Prediction ({} layers)'.format(min_layers + j), c=c, ls='--')
            plt.fill_between(x_test[:, 1], lowers[:, i], uppers[:, i], color=c, alpha=0.3)

            plt.scatter(x[inds, 1], y[inds, i],
                        label='Train points ({} layers)'.format(min_layers + j), c=c, zorder=10)
            # plt.scatter(x_valid[inds, 1], y_valid[inds, i],
            #             label='Train points ({} layers)'.format(j), c=c_valid, zorder=9)

else:
    plt.figure(figsize=(text_height, text_width))
    plt.suptitle(f'Validation RMSE: {valid_rmse:.3f}, Validation loglik: {valid_loglik:.3f}'
                 f'\nTraining   RMSE: {train_rmse:.3f}, Training   loglik: {train_loglik:.3f}')


    cs = ['tab:red', 'tab:blue', 'tab:green']
    cs_valid = ['tab:pink', 'tab:cyan', 'tab:olive']
    for i in range(y.shape[1]):
        plt.subplot(1, 3, i + 1)
        plt.title('Output {}'.format(i + 1))

        for j, (x_test, (means, sdev, lowers, uppers), c, c_valid) \
                in enumerate(zip(x_tests, preds, cs, cs_valid)):
            inds = x[:, 0] == min_layers + j
            inds_valid = x_valid == min_layers + j

            plt.semilogx(x_test[:, 1], means[:, i],
                     label='Prediction ({} layers)'.format(min_layers + j), c=c, ls='--')
            plt.fill_between(x_test[:, 1], lowers[:, i], uppers[:, i], color=c, alpha = 0.3)

            plt.scatter(x[inds, 1], y[inds, i],
                        label='Train points ({} layers)'.format(min_layers + j), c=c, zorder=10)
            # plt.scatter(x_valid[inds, 1], y_valid[inds, i],
            #             label='Train points ({} layers)'.format(j), c=c_valid, zorder=9)

plt.tight_layout()
plt.legend()
plt.subplots_adjust(top=0.85)
savefig(fig_path, png=True, pdf=True)

# plt.show()
plt.close()

if not os.path.isfile(os.path.join(fig_dir, 'results.pkl')):
    dataframe = pd.DataFrame(columns=['dataset', 'rmse_fit', 'final_only', 'joint_trained','validation_set_small',
                                      'scale_tie', 'input_linear_kernal', 'output_linear_kernal',
                                      'output_nonlinear_kernal', 'markov',
                                      'validation_loglikelihood', 'validation_rmse',
                                      'train_loglikelihood', 'train_rmse',
                                      'wall_time', 'seed'
                                      ])
else:
    dataframe = pickle.load(open(os.path.join(fig_dir, 'results.pkl'), 'rb'))


dataframenew = dataframe.append({'dataset': args.data,
                              'rmse_fit': args.rmse,
                              'final_only': args.final,
                              'joint_trained': args.joint,
                              'validation_set_small': args.validsmall,
                              'scale_tie': args.scale_tie,
                              'input_linear_kernal': args.input_linear,
                              'output_linear_kernal': args.output_linear,
                              'output_nonlinear_kernal': args.output_nonlinear,
                              'markov': args.markov,
                              'validation_loglikelihood': valid_loglik,
                              'validation_rmse': valid_rmse,
                              'train_loglikelihood': train_loglik,
                              'train_rmse': train_rmse,
                              'wall_time': end-start,
                              'seed': args.seed
                              },
                             ignore_index=True)


pickle.dump(dataframenew, open(os.path.join(fig_dir, 'results.pkl'), 'wb'))
