import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from stheno import B
from scipy.stats import norm

from gpar.regression import GPARRegressor

B.epsilon = 1e-6  # Set regularisation a bit higher to ensure robustness.
data_dir = '/home/mjhutchinson/Documents/MachineLearning/gpar/data/'  # The data is located here.


def transform_x(x):
    return np.stack((x[:, 0], np.log10(x[:, 1])), axis=0).T


# Parse arguments.
parser = argparse.ArgumentParser()
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
parser.add_argument('--joint', action='store_true',
                    help='also train jointly')
args = parser.parse_args()

print(args)

fig_dir = f'/home/mjhutchinson/Documents/MachineLearning/gpar/output/{args.experiment}/'
os.makedirs(fig_dir, exist_ok=True)

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

def gen_dataset(x, y):
    # inds = x[:, 1].argsort(axis=0)
    # x, y = x[inds], y[inds]
    # inds = x[:, 0].argsort(axis=0, kind='mergesort')
    # x, y = x[inds], y[inds]
    #
    # return x[::2, :],y[::2, :],x[1::2, :],y[1::2, :]
    return x,y,x[1:2,:],y[1:2,:]


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
model = GPARRegressor(scale=[1., .5], scale_tie=True,
                      linear=True, linear_scale=10., linear_with_inputs=False,
                      nonlinear=False, nonlinear_with_inputs=False,
                      markov=1,
                      replace=True,
                      noise=0.01)

# Fit model and print hyperparameters.
model.fit(transform_x(x), y,
          iters=args.iters, trace=False, progressive=True)
if args.joint:
    model.fit(transform_x(x), y,
              iters=args.iters, trace=False, progressive=False)
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

print('Predicting validation points')
samples = unnormalise(model.sample(transform_x(x_valid),
                            num_samples=50,
                            latent=True,
                            posterior=True))


means = np.mean(samples, axis=0)[:, -1]
stds = np.std(samples, axis=0)[:, -1]
_y = y_valid[:, -1]

valid_rmse = np.mean((_y - means) ** 2) ** 0.5
valid_mean_loglik = np.mean(norm.logpdf(_y, loc=means, scale=stds))

print('Predicting training points')
samples = unnormalise(model.sample(transform_x(x),
                            num_samples=50,
                            latent=True,
                            posterior=True))


means = np.mean(samples, axis=0)[:, -1]
stds = np.std(samples, axis=0)[:, -1]
_y = y_valid[:, -1]

train_rmse = np.mean((_y - means) ** 2) ** 0.5
train_mean_loglik = np.mean(norm.logpdf(_y, loc=means, scale=stds))

# Plot the result.
plt.figure(figsize=(20, 10))
plt.suptitle(f'{args.data} {"rmse" if args.rmse else "loglik"} '
             f'\n Validation RMSE: {valid_rmse:.6f}, Validation mean loglik (Fianl layer, fitted Gaussian likelihood): {valid_mean_loglik:.6f}'
             f'\n Training   RMSE: {train_rmse:.6f}, Training   mean loglik (Fianl layer, fitted Gaussian likelihood): {train_mean_loglik:.6f}')


cs = ['tab:red', 'tab:blue', 'tab:green']
cs_valid = ['tab:pink', 'tab:cyan', 'tab:olive']
for i in range(y.shape[1]):
    plt.subplot(2, 5, i + 1)
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


fig_path = fig_dir + args.data + '-' + ('rmse' if args.rmse else 'loglik')
plt.tight_layout()
plt.legend()
plt.subplots_adjust(top=0.85)
plt.savefig(fig_path + '.png', format='png')
plt.savefig(fig_path + '.svg', format='svg')

# plt.show()
plt.close()