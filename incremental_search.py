import os
import sys
import shutil
import pickle
import datetime
import argparse
from copy import deepcopy
from collections import namedtuple

import yaml
import torch
import numpy as np
from stheno import B

import plotting as plotting
from gpar.regression import GPARRegressor

### Function of this script
# This script performs architecture search over a dummy space by using a full GPAR model to checkpoint model performance
# and make decisions about what networks to train 

## Initialise

# Pick initial point set - how many, strategy

## Loop

# Train gpar model - gpar params, training regime

# Thompson sample - from remaining points in set

# Pick new points to train


B.epsilon = 1e-5  # Set regularisation a bit higher to ensure robustness.
trace = False  # if to print out optimisation trace
version = '0.3.0' # current version of the algorithm
np.random.seed(0) # fix random seed

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

parser.add_argument('-m', '--max_evals', type=int, default=50,
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

parser.add_argument('--synthetic_scales', nargs=2, type=float, metavar=('Depth Scale', 'Width Scale'),
                    help='2 argument list to determine the length scales of the generator function', default=[2., 0.5])


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

    x1 = np.linspace(1, 5, 5)
    x2 = np.concatenate([np.linspace(1, 9, 9), np.linspace(10, 98, 45), np.linspace(100, 980, 45)])
    xx1, xx2 = np.meshgrid(x1, x2)
    x = np.stack([np.ravel(xx1), np.ravel(xx2)], axis=1)

    if len(args.synthetic_scales) != 2:
        raise ValueError('Requires exactly 2 synthetic scale arguments')

    model = GPARRegressor(scale=args.synthetic_scales, scale_tie=True,
                          linear=True, linear_scale=10., input_linear=False,
                          nonlinear=False,
                          markov=None,
                          replace=True,
                          noise=args.noise)

    n = 3

    y = model.sample(transform_x(x), p=n, latent=False)

else:
    # Load data.
    x = np.genfromtxt(os.path.join(data_dir, args.data, args.experiment, f'x_{"rmse" if args.rmse else "loglik"}.txt'))
    y = np.genfromtxt(os.path.join(data_dir, args.data, args.experiment, f'f_{"rmse" if args.rmse else "loglik"}.txt'))

# Sort x by layer size and layer width
inds = x[:, 1].argsort()

x = x[inds]
y = y[inds]

inds = x[:, 0].argsort(kind='mergesort')

x = x[inds]
y = y[inds]


np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.final:
    y = y[:, -1][:, np.newaxis]

depth = y.shape[1]

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
y_normalised = (y - y_means) / y_stds

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

iteration_results = []

y_tested = np.zeros(y.shape, dtype=np.bool)
# y_tested = None
#
# x_remaining = x
# y_remaining = y

# Perform initial random search
while y_tested.sum() < (args.initial_points * depth):
    available_set = np.arange(y_tested.shape[0])[np.where(~y_tested[:, -1])]
    next_index = np.random.choice(available_set, 1)

    # Select the next points to try
    x_next_inds = next_index
    y_next_inds = tuple([np.ones(depth, dtype=np.int) * next_index, np.arange(depth)])
    # Create a helpful flagged array of the next values
    y_next = np.zeros(y.shape, dtype=np.bool)
    y_next[y_next_inds] = True

    # 'Test' the selected set of points
    y_tested[y_next_inds] = True

    y_temp = y.copy()
    y_temp[~y_tested] = -np.inf
    x_best, y_best = x[y_temp[:, -1].argmax()], y[y_temp[:, -1].argmax(), -1]

    iteration_result = IncrementalIterationResults(
        -1,
        x.copy(), y.copy(),
        y_tested.copy(),
        y_next.copy(),
        None,
        None,
        None,
        x_best.copy(),
        y_best.copy()
    )
    iteration_results.append(iteration_result)

iteration = 1
# Iterate the search procedure. TODO: Figure better termination conditions
if not args.random:

    # model = GPARRegressor(scale=[1., .5], scale_tie=True,
    #                       linear=True, linear_scale=10., input_linear=False,
    #                       nonlinear=False,  # missing non linear inputs now?
    #                       markov=1,
    #                       replace=True,
    #                       noise=0.01)

    while y_tested.sum() < (args.max_evals * depth):

        model = GPARRegressor(scale=[1., .5], scale_tie=True,
                              linear=True, linear_scale=10., input_linear=False,
                              nonlinear=False,  # missing non linear inputs now?
                              markov=1,
                              replace=True,
                              noise=0.01)

        # Define the GPAR model for this iteration. TODO: reuse hyper parameters from previous run as a starting point?
        print(f'Running iteration {iteration}')

        # Fit the GPAR model. If joint, this means we want to co-fit the parameters after the first fitting
        # Potential looses some accuracy on each layer and sometimes goes unstable as a result
        print('\t Fitting model')

        x_data = x.copy()
        y_data = y_normalised.copy()
        y_data[np.where(~y_tested)] = np.nan
        active_inds = np.where(y_tested[:, 0])
        x_data = x_data[active_inds]
        y_data = y_data[active_inds]

        model.fit(transform_x(x_data), y_data, trace=trace, fix=True, iters=args.iters)
        if args.joint:
            model.fit(transform_x(x_data), y_data, trace=trace, fix=False, iters=args.iters)

        # Sample the test points to allow us to plot the form of the GPAR function
        print('\t Predicting test points')
        test_samples = unnormalise(model.sample(transform_x(x_test),
                                                num_samples=num_function_samples,
                                                latent=True,
                                                posterior=True))
        print('Computing test stats')
        # Compute statistics about the test points
        test_stats = [test_samples.mean(axis=0),
                      test_samples.std(axis=0),
                      np.percentile(test_samples, 2.5, axis=0),
                      np.percentile(test_samples, 100 - 2.5, axis=0)]

        # Sample the points in the dataset - in reality this would be Thompson sampling of the network space
        print('\t Predicting remaining points')
        dataset_samples = unnormalise(model.sample(transform_x(x),
                                                   num_samples=num_function_samples,
                                                   latent=True,
                                                   posterior=True))

        # Statistics from the samples of points
        dataset_stats = [dataset_samples.mean(axis=0),
                           dataset_samples.std(axis=0),
                           np.percentile(dataset_samples, 2.5, axis=0),
                           np.percentile(dataset_samples, 100 - 2.5, axis=0)]

        if args.random:
            available_set = np.arange(y_tested.shape[0])[np.where(~y_tested[:, -1])]
            next_ind = np.random.choice(available_set, 1)

            x_next_inds = next_ind
            y_next_inds = tuple([np.ones(depth, dtype=np.int) * next_ind, np.arange(depth, dtype=np.int)])
        else:
            if args.samples_per_thompson:
                # list to store values in
                next_index = []
                # loop for number of samples we want
                for i in range(args.thompson_samples):
                    # Error check number of samples
                    if dataset_samples.shape[0] < args.thompson_samples * args.samples_per_thompson: raise ValueError('Not enough samples to compute all points')
                    # Select subset of full smaple space
                    sample_subset = dataset_samples[i * args.samples_per_thompson:(i + 1) * args.samples_per_thompson, :]
                    # Compute acquisition on the subset
                    acquisition_value = acquisition_function(sample_subset, y_best)
                    acquisition_value = acquisition_value[:, -1]
                    # Get which networks have been trained to completion
                    completed = y_tested[:, -1]
                    # Set acquisition for completed networks to -inf
                    acquisition_value[np.where(completed)] = -np.inf
                    # Sort the acquisitions and get the indices
                    sorted_index = list(np.squeeze(acquisition_value).argsort(axis=0))
                    # Try to add to the next index until a valid next is added
                    next = sorted_index.pop()
                    while next in next_index:
                        next = sorted_index.pop()
                    next_index.append(next)

                next_index = np.array(next_index)

                acquisition_value = acquisition_function(dataset_samples, y_best)
                acquisition_value = acquisition_value[:, -1]
                completed = y_tested[:, -1]
                # Set acquisition to nan where complected for plotting
                acquisition_value[np.where(completed)] = np.nan
            else:
                acquisition_value = acquisition_function(dataset_samples, y_best)
                acquisition_value = acquisition_value[:, -1]
                completed = y_tested[:, -1]
                # Set acquisition for completed networks to -inf
                acquisition_value[np.where(completed)] = -np.inf
                # Select the top next indices to try
                next_index = np.argpartition(acquisition_value, len(acquisition_value) - args.thompson_samples)[-args.thompson_samples:]
                # Set acquisition to nan where complected for plotting
                acquisition_value[np.where(completed)] = np.nan


        # Select the next points to try
        x_next_inds = next_index
        y_next_inds = tuple([next_index, y_tested.sum(axis=1)[next_index]])
        # Create a helpful flagged array of the next values
        y_next = np.zeros(y.shape, dtype=np.bool)
        y_next[y_next_inds] = True

        print('\t Plotting results')
        # Plot the next points in a nice format and save
        iteration_result = IncrementalIterationResults(
            iteration,
            x.copy(), y.copy(),
            y_tested.copy(),
            y_next.copy(),
            x_test,
            test_stats,
            acquisition_value.copy(),
            x_best.copy(),
            y_best.copy()
        )

        if args.plot:
            plotting.plot_iteration_incremental(**(iteration_result._asdict()),
                                                fig_dir=outdir)

        # Update the two sets of points available to the algorithm
        # 'Test' the selected set of points
        y_tested[y_next_inds] = True

        # update the best points found so far. TODO: Maybe look at making f/x_best the best from the GPAR function, rather than the data point?
        y_temp = y.copy()
        y_temp[~y_tested] = -np.inf
        x_best, y_best = x[y_temp[:, -1].argmax()].copy(), y[y_temp[:, -1].argmax(), -1].copy()

        iteration_results.append(iteration_result)

        iteration += 1

else:
    while y_tested.shape[0] < args.max_evals:
        print(f'Running iteration {iteration}')
        available_set = np.arange(y_tested.shape[0])[np.where(~y_tested[:, -1])]
        next_index = np.random.choice(available_set, 1)

        # Select the next points to try
        x_next_inds = next_index
        y_next_inds = tuple([np.ones(depth, dtype=np.int) * next_index, y_tested.sum(axis=1)[next_index]])
        # Create a helpful flagged array of the next values
        y_next = np.zeros(y.shape, dtype=np.bool)
        y_next[y_next_inds] = True

        # 'Test' the selected set of points
        y_tested[y_next_inds] = True

        x_best, y_best = x[y_tested[:, -1].nanargmax()], unnormalise(y_tested[y_tested[:, -1].nanargmax(), -1])

        # update the best points found so far. TODO: Maybe look at making f/x_best the best from the GPAR function, rather than the data point?
        y_temp = y.copy()
        y_temp[~y_tested] = -np.inf
        x_best, y_best = x[y_temp[:, -1].argmax()], y[y_temp[:, -1].argmax(), -1]

        iteration_result = IncrementalIterationResults(
            iteration,
            x.copy(), y.copy(),
            y_tested.copy(),
            y_next.copy(),
            None,
            None,
            None,
            x_best.copy(),
            y_best.copy()
        )

        iteration += 1

# Store final iteration results
iteration_result = IncrementalIterationResults(
    iteration,
    x.copy(), y.copy(),
    y_tested.copy(),
    None,
    x_test,
    None,
    None,
    x_best.copy(),
    y_best.copy()
)

pickle.dump(args, open(os.path.join(outdir, 'config.pkl'), 'wb'))
yaml.dump(args, open(os.path.join(outdir, 'config.yaml'), 'w'))
pickle.dump(iteration_results, open(os.path.join(outdir, 'results.pkl'), 'wb'))

f_bests = np.squeeze([np.squeeze(result.y_best) for result in iteration_results])
acquired = np.squeeze([result.y_tested_flags.sum() for result in iteration_results])
f_max = iteration_results[-1].y[:, -1].max()

# if args.plot:
plotting.plot_search_results(acquired, f_bests, f_max, outdir)
