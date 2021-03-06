import os
import sys
import shutil
import pickle
import datetime
import argparse
from collections import namedtuple

import yaml
import torch
import numpy as np
from stheno import B

import plotting as plotting
from gpar.regression import GPARRegressor

### Function of this script
# This script performs architecture search over a dummy space by using a single GP + bayes opt techniques to perform
# acquisition on the final performance of the architecture.

## Initialise

# Pick initial point set - how many, strategy

## Loop

# Train gpar model - gpar params, training regime

# Thompson sample - from remaining points in set

# Pick new points to train


B.epsilon = 1e-6  # Set regularisation a bit higher to ensure robustness.
trace = False  # if to print out optimisation trace
version = '0.3.0' # current version of the algorithm
np.random.seed(0) # fix random seed

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str,
                    default='power-plant',
                    help='name of data set')

parser.add_argument('-e', '--experiment', type=str,
                    default='weight_pruning_hyperprior3',
                    help='name of the experiment to run on')

parser.add_argument('-a', '--acquisition', type=str,
                    default='EI',
                    help='Acquisition function to use in the search. Valid are EI (Expected Improvement), PI ('
                         'Probability of Improvement) or SD (mu + std)')

parser.add_argument('-r', '--random', action='store_true',
                    help='If present, perform a random search')

parser.add_argument('-i', '--iters', type=int, default=100,
                    help='number of iterations')

parser.add_argument('--rmse', action='store_true',
                    help='Perform search on the RMSE instead')

parser.add_argument('--final', action='store_true',
                    help='Use a GP over the final test points only')

parser.add_argument('--force', action='store_true',
                    help='Force redo experiment')

parser.add_argument('--joint', action='store_true',
                    help='also train jointly')

parser.add_argument('-n', '--initial_points',
                    default=5, type=int,
                    help='Number of initial points to select as seed point (randomly drawn)')

parser.add_argument('-t', '--thompson_samples',
                    default=4, type=int,
                    help='Number of points to sample per iteration')

parser.add_argument('-ts', '--samples_per_thompson', default=0, type=int,
                    help='Number of samples to use per thompson sample for acquisition, if 0, use all 50')

parser.add_argument('-s', '--seed', type=int, default=0,
                    help='Random seed to use for the search')

parser.add_argument('-fs', '--function_seed', type=int, default=0,
                    help='Random seed to use for the synthetic function')

parser.add_argument('-m', '--max_evals', type=int, default=45,
                    help='Iterations to run the search for, accounting for thompson samples')

parser.add_argument('-p', '--plot', action='store_true',
                    help='Plot results as we go')

parser.add_argument('--noise', type=float, default=0.01,
                    help='noise to model on the synthetic function, if the synthetic function is used')

parser.add_argument('--datadir', type=str,
                    help='Directory the data is located in',
                    default='data/')

parser.add_argument('--outdir', type=str,
                    help='Directory to put output in',
                    default='output')

parser.add_argument('--name', type=str, required=True, help='Name to give the experiment')

args = parser.parse_args()

args.version = version

print(args)

# parse args

acquisition_function = None
if args.acquisition == "EI":
    from acquisition_utils import expected_improvement
    acquisition_function = expected_improvement
elif args.acquisition == "PI":
    from acquisition_utils import probability_improvement
    acquisition_function = probability_improvement
elif args.acquisition == "SD":
    from acquisition_utils import standard_dev_improvement
    acquisition_function = standard_dev_improvement

if acquisition_function is None and not args.random:
    raise ValueError('Invalid acquisition function specified. Use the script help to find valid functions')

if args.random:
    args.acquisition = "Random"

if args.random and not args.final:
    print("Cannot have random search with multiple outputs")
    sys.exit(0)

# Define location of data and output directories
data_dir = args.datadir #'/home/mjhutchinson/Documents/MachineLearning/architecture_search_gpar/data/'

# Define the transformation to apply to the x data
def transform_x(x):
    return np.stack((x[:, 0], np.log10(x[:, 1])), axis=0).T


if args.data == 'synthetic':
    np.random.seed(args.function_seed)
    torch.manual_seed(args.function_seed)
    args.experiment = 'synthetic'

    x1 = np.linspace(1, 5, 5)
    x2 = np.concatenate([np.linspace(1, 9, 9), np.linspace(10, 98, 45), np.linspace(100, 980, 45)])
    xx1, xx2 = np.meshgrid(x1, x2)
    x = np.stack([np.ravel(xx1), np.ravel(xx2)], axis=1)

    model = GPARRegressor(scale=[2., .3], scale_tie=True,
                          linear=True, linear_scale=10., linear_with_inputs=False,
                          nonlinear=False, nonlinear_with_inputs=False,
                          markov=1,
                          replace=True,
                          noise=args.noise)

    n = 10

    y = model.sample(transform_x(x), p=n, latent=False)

else:
    # Load data.
    x = np.genfromtxt(os.path.join(data_dir, args.data, args.experiment, f'x_{"rmse" if args.rmse else "loglik"}.txt'))
    y = np.genfromtxt(os.path.join(data_dir, args.data, args.experiment, f'f_{"rmse" if args.rmse else "loglik"}.txt'))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.final:
    y = y[:, -1][:, np.newaxis]

num_function_samples = max(50, args.thompson_samples * args.samples_per_thompson)

outdir = os.path.join(args.outdir, 'searches', args.experiment, args.data, args.name,
                      f'{"rmse" if args.rmse else "loglik"}-{"random" if args.random else args.acquisition}-{"final" if args.final else "all"}-{args.seed}')
args.outdir = outdir

# check if exists and make folder
if os.path.isdir(outdir):
    if args.force:
        shutil.rmtree(outdir)
    else:
        print('Experiment already run. Exiting')
        sys.exit(0)

os.makedirs(outdir)

# Find min and max layers in the data
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

# Define the function to unnormalise y data
def unnormalise(y):
    return y * y_stds + y_means


# Create some test inputs.
n = 100
x_test = np.concatenate([np.stack((i * np.ones(n),
                                   np.linspace(min(x[:, 1]), int(max(x[:, 1]) * 1.0), n)), axis=0).T
                         for i in range(min_layers, max_layers + 1)])

# Generate initial set
# Assumes same layer widths for each num layers
layer_sizes = np.unique(x[np.where(x[:, 0] == min_layers)][:, 1])

IterationResults = namedtuple('IterationResults',
                              ['iteration',
                               'x_tested', 'y_tested',
                               'x_remaining', 'y_remaining', 'remaining_stats', 'remaining_acquisition',
                               'x_next', 'y_next', 'next_acquisition',
                               'x_best', 'y_best',
                               'x_test', 'test_stats'
                               ])


iteration_results = []

x_tested = None
y_tested = None

x_remaining = x
y_remaining = y

# Perform initial random search
while y_tested is None or len(y_tested) < args.initial_points:
    random_ind = np.random.choice(np.arange(len(x_remaining)), 1)

    x_next = x_remaining[random_ind]
    y_next = y_remaining[random_ind]

    if x_tested is None:
        x_tested = x_remaining[random_ind]
        y_tested = y_remaining[random_ind]
    else:
        x_tested = np.concatenate([x_tested, x_remaining[random_ind]], axis=0)
        y_tested = np.concatenate([y_tested, y_remaining[random_ind]], axis=0)

    x_remaining = np.delete(x, random_ind, 0)
    y_remaining = np.delete(y, random_ind, 0)

    x_best, y_best = x_tested[y_tested[:, -1].argmax()], unnormalise(y_tested[y_tested[:, -1].argmax(), -1])

    iteration_results.append(
        IterationResults(
            -1,
            x_tested, unnormalise(y_tested),
            x_remaining, unnormalise(y_remaining), None, None,
            x_next, y_next, None,
            x_best, unnormalise(y_best),
            None, None
        )
    )

iteration = 1
# Iterate the search procedure. TODO: Figure better termination conditions
if not args.random:

    model = GPARRegressor(scale=[1., .5], scale_tie=True,
                          linear=True, linear_scale=10., linear_with_inputs=False,
                          nonlinear=False, nonlinear_with_inputs=False,
                          markov=1,
                          replace=True,
                          noise=0.01)

    while y_tested.shape[0] < args.max_evals:

        # Define the GPAR model for this iteration. TODO: reuse hyper parameters from previous run as a starting point?
        print(f'Running iteration {iteration}')

        # Fit the GPAR model. If joint, this means we want to co-fit the parameters after the first fitting
        # Potential looses some accuracy on each layer and sometimes goes unstable as a result
        print('\t Fitting model')
        model.fit(transform_x(x_tested), y_tested, trace=trace, progressive=True, iters=args.iters)
        if args.joint:
            model.fit(transform_x(x_tested), y_tested, trace=trace, progressive=False, iters=args.iters)

        # Sample the test points to allow us to plot the form of the GPAR function
        print('\t Predicting test points')
        test_samples = unnormalise(model.sample(transform_x(x_test),
                                                num_samples=num_function_samples,
                                                latent=True,
                                                posterior=True))

        # Compute statistics about the test points
        test_stats = [test_samples.mean(axis=0),
                      test_samples.std(axis=0),
                      np.percentile(test_samples, 2.5, axis=0),
                      np.percentile(test_samples, 100 - 2.5, axis=0)]

        # Sample the points remaining in the dataset - in reality this would be Thompson sampling of the network space
        print('\t Predicting remaining points')
        remaining_samples = unnormalise(model.sample(transform_x(x_remaining),
                                                     num_samples=num_function_samples,
                                                     latent=True,
                                                     posterior=True))

        # Statistics from the samples of remaining points
        remaining_stats = [remaining_samples.mean(axis=0),
                           remaining_samples.std(axis=0),
                           np.percentile(remaining_samples, 2.5, axis=0),
                           np.percentile(remaining_samples, 100 - 2.5, axis=0)]

        # calculate the acquisition function for the next iteration. TODO: make this more flexible
        # r_mean = remaining_stats[0]
        # r_std = remaining_stats[1]
        # r_delta = r_mean - y_best

        if args.random:
            remaining_acquisition = np.ones(remaining_stats[0][:, -1].shape)
            next_index = np.random.choice(np.arange(len(remaining_stats[0])), args.thompson_samples, replace=False)
        else:
            if args.samples_per_thompson:
                # list to store values in
                next_index = []
                # loop for number of samples we want
                for i in range(args.thompson_samples):
                    # Error check number of samples
                    if remaining_samples.shape[0] < args.thompson_samples * args.samples_per_thompson: raise ValueError('Not enough samples to compute all points')
                    # Select subset of full smaple space
                    sample_subset = remaining_samples[i*args.samples_per_thompson:(i+1)*args.samples_per_thompson, :]
                    # Compute acquisition on the subset
                    remaining_acquisition = acquisition_function(sample_subset, y_best)
                    # Sort the acquisitions and get the indicies
                    sorted_index = list(np.squeeze(remaining_acquisition).argsort(axis=0))
                    # Try to add to the next index until a valid next is added
                    next = sorted_index.pop()
                    while next in next_index:
                        next = sorted_index.pop()
                    next_index.append(next)

                next_index = np.array(next_index)
            else:
                remaining_acquisition = acquisition_function(remaining_samples, y_best)
                remaining_acquisition = remaining_acquisition[:, -1]
                # Select the top next indices to try
                next_index = np.argpartition(remaining_acquisition, len(remaining_acquisition) - args.thompson_samples)[-args.thompson_samples:]

        # Select the next points to try
        x_next = x_remaining[next_index]
        y_next = y_remaining[next_index]
        next_acquisition = remaining_acquisition[next_index]

        print('\t Plotting results')
        # Plot the next points in a nice format and save

        iteration_result = IterationResults(
                iteration,
                x_tested, unnormalise(y_tested),
                x_remaining, unnormalise(y_remaining), remaining_stats, remaining_acquisition,
                x_next, unnormalise(y_next), next_acquisition,
                x_best, unnormalise(y_best),
                x_test, test_stats
            )

        if args.plot:
            plotting.plot_iteration(**(iteration_result._asdict()),
                                    fig_dir=outdir)

        iteration_results.append(iteration_result)

        # Update the two sets of points available to the algorithm
        x_remaining = np.delete(x_remaining, next_index, axis=0)
        y_remaining = np.delete(y_remaining, next_index, axis=0)

        x_tested = np.concatenate([x_tested, x_next], axis=0)
        y_tested = np.concatenate([y_tested, y_next], axis=0)

        # update the best points found so far. TODO: Maybe look at making f/x_best the best from the GPAR function, rather than the data point?
        x_best, y_best = x_tested[y_tested[:, -1].argmax()], y_tested[y_tested[:, -1].argmax(), -1]

        iteration += 1

else:
    while y_tested.shape[0] < args.max_evals:
        print(f'Running iteration {iteration}')
        random_ind = np.random.choice(np.arange(len(x_remaining)), 1)

        x_next = x_remaining[random_ind]
        y_next = y_remaining[random_ind]

        x_tested = np.concatenate([x_tested, x_remaining[random_ind]], axis=0)
        y_tested = np.concatenate([y_tested, y_remaining[random_ind]], axis=0)

        x_remaining = np.delete(x, random_ind, 0)
        y_remaining = np.delete(y, random_ind, 0)

        x_best, y_best = x_tested[y_tested[:, -1].argmax()], unnormalise(y_tested[y_tested[:, -1].argmax()])

        iteration_results.append(
            IterationResults(
                -1,
                x_tested, unnormalise(y_tested),
                x_remaining, unnormalise(y_remaining), None, None,
                x_next, y_next, None,
                x_best, unnormalise(y_best),
                None, None
            )
        )

        iteration += 1


f_bests = np.squeeze([np.squeeze(result.y_best) for result in iteration_results])
acquired = np.squeeze([len(result.x_tested) for result in iteration_results])

if args.plot:
    plotting.plot_search_results(acquired, f_bests, outdir)

pickle.dump(args, open(os.path.join(outdir, 'config.pkl'), 'wb'))
yaml.dump(args, open(os.path.join(outdir, 'config.yaml'), 'w'))
pickle.dump(iteration_results, open(os.path.join(outdir, 'results.pkl'), 'wb'))
