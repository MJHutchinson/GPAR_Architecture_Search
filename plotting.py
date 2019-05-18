from math import ceil

import numpy as np
import matplotlib.pyplot as plt

from plotting_config import *

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

def plot_iteration(iteration, x_tested, y_tested, x_remaining, y_remaining, remaining_acquisition, x_test, test_stats, x_next, next_acquisition, fig_dir, **kwargs):
    '''
    :param iteration: Iteration we are currently on of the optimisation
    :param x_tested: The points tested so far by the search procedure
    :param y_tested: The results of the points tested so far by the procedure
    :param x_remaining: Points left in the dummy set (if None, not plotted)
    :param y_remaining: Points values left in the dummy set (if None, not plotted)
    :param remaining_acquisition: the acquisition function of te points we might try next
    :param x_test: Points that have been sampled to plot mean/margins of GPAR
    :param test_stats: Statistics of the samples drawn from the GPAR model
    :param x_next: Points to be sampled in the next iteration
    :param next_acquisition: Acquisition function at next points
    :param fig_dir: directory to save figures in
    '''

    min_layers = int(min(min(x_remaining[:, 0]), min(x_tested[:, 0])))
    max_layers = int(max(max(x_remaining[:, 0]), max(x_tested[:, 0])))

    cs = ['xkcd:red', 'xkcd:blue', 'xkcd:green', 'xkcd:purple', 'xkcd:orange']
    cs_remaining = ['xkcd:pink', 'xkcd:sky blue', 'xkcd:light green', 'xkcd:lavender', 'xkcd:peach']

    outputs = int(y_tested.shape[1])
    if outputs < 5:
        rows = 1
        cols = outputs
    else:
        rows = ceil(outputs / 5.)
        cols = 5

    # plt.figure(figsize=(4 * cols, 5 * rows))
    plt.figure(figsize=(20, 10))
    plt.suptitle(f'Iteration {iteration}')

    for i in range(y_tested.shape[1]):
        plt.subplot(rows, cols, i + 1)
        plt.title('Output {}'.format(i + 1))

        for j in range(max_layers - min_layers + 1):
            inds_test = x_test[:, 0] == j + min_layers
            inds_tested = x_tested[:, 0] == j + min_layers
            inds_remaining = x_remaining[:, 0] == j + min_layers
            inds_next = x_next[:, 0] == j + min_layers

            mean_test = test_stats[0]
            std_test = test_stats[1]
            lowers_test = test_stats[2]
            uppers_test = test_stats[3]

            plt.semilogx(x_test[inds_test, 1], mean_test[inds_test, i],
                         label=f'Prediction ({j + min_layers} layers)', c=cs[j], ls='--')

            plt.fill_between(x_test[inds_test, 1], lowers_test[inds_test, i], uppers_test[inds_test, i], color=cs[j],
                             alpha=0.3)

            # plt.fill_between(x_test[inds_test, 1], mean_test[inds_test, i] - std_test[inds_test, i],
            #                  mean_test[inds_test, i] + std_test[inds_test, i], color=cs[j],
            #                  alpha=0.3)

            plt.scatter(x_remaining[inds_remaining, 1], y_remaining[inds_remaining, i],
                        label=f'Expected improvement: {j} layers', c=cs_remaining[j], zorder=9)

            # plt.errorbar(x_remaining[inds_remaining, 1], mean_rem[inds_remaining, i], yerr=std_rem[inds_remaining, i],
            #              fmt='o', label=f'Expected improvement: {j} layers', c=cs_remaining[j])

            plt.scatter(x_tested[inds_tested, 1], y_tested[inds_tested, i], c=cs[j],
                        label=f'Points tested: {j + min_layers} layers', zorder=10)

            axes = plt.gca()
            y_lim = axes.get_ylim()

            for x_n in x_next[inds_next]:
                plt.plot([x_n[1], x_n[1]], y_lim, c=cs[j], label='New point to try')

            axes.set_ylim(y_lim)

    plt.subplot(rows, cols, 1)
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    savefig(fig_dir + f'/iteration_{iteration}')
    plt.close('all')


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    for j in range(max_layers - min_layers + 1):

        inds_test = x_test[:, 0] == j + min_layers
        inds_tested = x_tested[:, 0] == j + min_layers
        inds_remaining = x_remaining[:, 0] == j + min_layers
        inds_next = x_next[:, 0] == j + min_layers

        mean_test = test_stats[0]
        std_test = test_stats[1]
        lowers_test = test_stats[2]
        uppers_test = test_stats[3]

        ax1.semilogx(x_test[inds_test, 1], mean_test[inds_test, -1],
                     label=f'Prediction ({j + min_layers} layers)', c=cs[j], ls='--')

        ax1.fill_between(x_test[inds_test, 1], lowers_test[inds_test, -1], uppers_test[inds_test, -1], color=cs[j],
                         alpha=0.3)

        # plt.fill_between(x_test[inds_test, 1], mean_test[inds_test, i] - std_test[inds_test, i],
        #                  mean_test[inds_test, i] + std_test[inds_test, i], color=cs[j],
        #                  alpha=0.3)

        ax1.scatter(x_remaining[inds_remaining, 1], y_remaining[inds_remaining, -1],
                    label=f'Expected improvement: {j} layers', c=cs_remaining[j], zorder=9)

        # plt.errorbar(x_remaining[inds_remaining, 1], mean_rem[inds_remaining, i], yerr=std_rem[inds_remaining, i],
        #              fmt='o', label=f'Expected improvement: {j} layers', c=cs_remaining[j])

        ax1.scatter(x_tested[inds_tested, 1], y_tested[inds_tested, -1], c=cs[j],
                    label=f'Points tested: {j + min_layers} layers', zorder=10)

        y_lim = ax1.get_ylim()

        for x_n in x_next[inds_next]:
            ax1.plot([x_n[1], x_n[1]], y_lim, c=cs[j], label='New point to try')

        ax1.set_ylim(y_lim)


        inds_remaining = x_remaining[:, 0] == j + min_layers

        ax2.scatter(x_remaining[inds_remaining, 1], remaining_acquisition[inds_remaining],
                    label=f'expected improv ({j + min_layers} layers)', c=cs[j])

        inds_next = x_next[:, 0] == j + min_layers

        for x_n, r_n in zip(x_next[inds_next], next_acquisition[inds_next]):
            ax2.scatter(x_n[1], r_n, c=cs_remaining[j], label='New point to try', zorder=10)

    plt.tight_layout()
    plt.legend()
    savefig(fig_dir + f'/iteration_{iteration}_expected_improv')
    plt.close('all')

def plot_search_results(aquisitions, f_bests, f_max, fig_dir):

    plt.figure(figsize=(text_width, text_width/1.5))
    plt.title('Evolution of best results found at number of evaluated points')
    plt.plot(aquisitions, f_bests, 'b')
    plt.axhline(f_max)

    plt.xlabel('Points evaluated')
    plt.ylabel('Maximum function value')

    savefig(fig_dir + '/search_results')

    plt.close('all')


def plot_iteration_incremental(iteration, x, y, y_tested_flags, y_next_flags, x_test, test_stats, acquisition_value, fig_dir, **kwargs):
    '''
    :param iteration: Iteration we are currently on of the optimisation
    :param x_tested: The points tested so far by the search procedure
    :param y_tested: The results of the points tested so far by the procedure
    :param remaining_acquisition: the acquisition function of te points we might try next
    :param x_test: Points that have been sampled to plot mean/margins of GPAR
    :param test_stats: Statistics of the samples drawn from the GPAR model
    :param x_next: Points to be sampled in the next iteration
    :param next_acquisition: Acquisition function at next points
    :param fig_dir: directory to save figures in
    '''

    min_layers = int(min(x[:, 0]))
    max_layers = int(max(x[:, 0]))

    cs = ['xkcd:red', 'xkcd:blue', 'xkcd:green', 'xkcd:purple', 'xkcd:orange']
    cs_remaining = ['xkcd:pink', 'xkcd:sky blue', 'xkcd:light green', 'xkcd:lavender', 'xkcd:peach']

    outputs = int(y.shape[1])
    if outputs < 5:
        rows = 1
        cols = outputs
    else:
        rows = ceil(outputs / 5.)
        cols = 5

    # plt.figure(figsize=(4 * cols, 5 * rows))
    plt.figure(figsize=(20, 10))
    plt.suptitle(f'Iteration {iteration}')

    x_remaining = x.copy()
    y_remaining = y.copy()
    y_remaining[np.where(y_tested_flags)] = np.nan

    x_tested = x.copy()
    y_tested = y.copy()
    y_tested[np.where(~y_tested_flags)] = np.nan

    for i in range(y.shape[1]):
        plt.subplot(rows, cols, i + 1)
        plt.title('Output {}'.format(i + 1))

        for j in range(max_layers - min_layers + 1):
            inds_test = np.where(x_test[:, 0] == j + min_layers)
            inds_tested = np.where((x[:, 0] == j + min_layers) & y_tested_flags[:, i])
            inds_remaining = np.where((x[:, 0] == j + min_layers) & ~y_tested_flags[:, i])

            mean_test = test_stats[0]
            std_test = test_stats[1]
            lowers_test = test_stats[2]
            uppers_test = test_stats[3]

            plt.semilogx(np.squeeze(x_test[inds_test, 1]), np.squeeze(mean_test[inds_test, i]),
                         label=f'Prediction ({j + min_layers} layers)', c=cs[j], ls='--')

            plt.fill_between(np.squeeze(x_test[inds_test, 1]), np.squeeze(lowers_test[inds_test, i]), np.squeeze(uppers_test[inds_test, i]), color=cs[j],
                             alpha=0.3, zorder=10+j)

            # plt.fill_between(x_test[inds_test, 1], mean_test[inds_test, i] - std_test[inds_test, i],
            #                  mean_test[inds_test, i] + std_test[inds_test, i], color=cs[j],
            #                  alpha=0.3)

            #plt.scatter(x_remaining[inds_remaining, 1], y_remaining[inds_remaining, i],
            #            label=f'Expected improvement: {j} layers', c=cs_remaining[j], zorder=j)

            # plt.errorbar(x_remaining[inds_remaining, 1], mean_rem[inds_remaining, i], yerr=std_rem[inds_remaining, i],
            #              fmt='o', label=f'Expected improvement: {j} layers', c=cs_remaining[j])

            plt.scatter(x_tested[inds_tested, 1], y_tested[inds_tested, i], c=cs[j],
                        label=f'Points tested: {j + min_layers} layers', zorder=20+j)


        for j in range(max_layers - min_layers + 1):
            inds_next = np.where((x[:, 0] == j + min_layers) & y_next_flags[:, i])

            axes = plt.gca()
            y_lim = axes.get_ylim()

            for x_n in x[inds_next]:
                plt.plot([x_n[1], x_n[1]], y_lim, c=cs[j], label='New point to try', zorder=30)

            axes.set_ylim(y_lim)

    plt.subplot(rows, cols, 1)
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    savefig(fig_dir + f'/iteration_{iteration}')
    plt.close('all')


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    for j in range(max_layers - min_layers + 1):
        inds_test = np.where(x_test[:, 0] == j + min_layers)
        inds_tested = np.where((x[:, 0] == j + min_layers) & y_tested_flags[:, i])
        inds_remaining = np.where((x[:, 0] == j + min_layers) & ~y_tested_flags[:, i])
        inds_next = np.where((x[:, 0] == j + min_layers) & y_next_flags.sum(axis=1))

        mean_test = test_stats[0]
        std_test = test_stats[1]
        lowers_test = test_stats[2]
        uppers_test = test_stats[3]

        ax1.semilogx(np.squeeze(x_test[inds_test, 1]), np.squeeze(mean_test[inds_test, -1]),
                     label=f'Prediction ({j + min_layers} layers)', c=cs[j], ls='--')

        ax1.fill_between(np.squeeze(x_test[inds_test, 1]), np.squeeze(lowers_test[inds_test, -1]), np.squeeze(uppers_test[inds_test, -1]), color=cs[j],
                         alpha=0.3)

        # plt.fill_between(x_test[inds_test, 1], mean_test[inds_test, i] - std_test[inds_test, i],
        #                  mean_test[inds_test, i] + std_test[inds_test, i], color=cs[j],
        #                  alpha=0.3)

        ax1.scatter(x_remaining[inds_remaining, 1], y_remaining[inds_remaining, -1],
                    label=f'Expected improvement: {j} layers', c=cs_remaining[j], zorder=9)

        # plt.errorbar(x_remaining[inds_remaining, 1], mean_rem[inds_remaining, i], yerr=std_rem[inds_remaining, i],
        #              fmt='o', label=f'Expected improvement: {j} layers', c=cs_remaining[j])

        ax1.scatter(x_tested[inds_tested, 1], y_tested[inds_tested, -1], c=cs[j],
                    label=f'Points tested: {j + min_layers} layers', zorder=10)

        y_lim = ax1.get_ylim()

        for x_n in x[inds_next]:
            ax1.plot([x_n[1], x_n[1]], y_lim, c=cs[j], label='New point to try')

        ax1.set_ylim(y_lim)


        inds_remaining = x_remaining[:, 0] == j + min_layers

        ax2.scatter(x_remaining[inds_remaining, 1], acquisition_value[inds_remaining],
                    label=f'expected improv ({j + min_layers} layers)', c=cs[j], zorder=1)

        for x_n, r_n in zip(x[inds_next], acquisition_value[inds_next]):
            ax2.scatter(x_n[1], r_n, c=cs_remaining[j], label='New point to try', zorder=10)

    plt.tight_layout()
    plt.legend()
    savefig(fig_dir + f'/iteration_{iteration}_expected_improv')
    plt.close('all')

    # This ASSUMES x is in the order we want, and each layer is in the same order, and that the individual
    # dimensions are independent - I.e all layer sizes are the same for all layer depths
    fig = plt.figure(figsize=(10,10))
    for j in range(max_layers - min_layers + 1):
        inds = x[:, 0] == min_layers + j

        x_set = x[inds]
        y_tested_flags_layer_set = y_tested_flags[inds]
        y_next_flags_layer_set = y_next_flags[inds]

        tested_locs = list(np.where(y_tested_flags_layer_set))
        tested_locs[0] = tested_locs[0] + 0.1*j
        next_locs = list(np.where(y_next_flags_layer_set))
        next_locs[0] = next_locs[0] + 0.1*j

        plt.scatter(tested_locs[0], tested_locs[1], c=cs[j], label=f'Tested points, layer {min_layers + j}')
        plt.scatter(next_locs[0], next_locs[1], c=cs_remaining[j], label=f'Next points, layer {min_layers + j}')

        plt.grid(b=False)
        plt.ylim([0, y.shape[1]])

    plt.xticks(np.arange(x_set[:, 1].size), x_set[:, 1])

    plt.tight_layout()
    plt.legend()
    savefig(fig_dir + f'/iteration_{iteration}_incremental')
    plt.close('all')