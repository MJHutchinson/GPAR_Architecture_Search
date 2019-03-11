from math import ceil

import matplotlib.pyplot as plt

import plotting_config

def plot_iteration(iteration, x_tested, y_tested, x_remaining, y_remaining, remaining_acquisition, x_test, test_stats, x_next, acquisition_next, fig_dir):
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
    :param acquisition_next: Acquisition function at next points
    :param fig_dir: directory to save figures in
    '''

    min_layers = int(min(min(x_remaining[:, 0]), min(x_tested[:, 0])))
    max_layers = int(max(max(x_remaining[:, 0]), max(x_tested[:, 0])))

    cs = ['tab:red', 'tab:blue', 'tab:green']
    cs_remaining = ['tab:pink', 'tab:cyan', 'tab:olive']

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
    plotting_config.savefig(fig_dir + f'/iteration_{iteration}')


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

        for x_n, r_n in zip(x_next[inds_next], acquisition_next[inds_next]):
            ax2.scatter(x_n[1], r_n, c=cs_remaining[j], label='New point to try')

    plt.tight_layout()
    plt.legend()
    plotting_config.savefig(fig_dir + f'/iteration_{iteration}_expected_improv')
    plt.close('all')

def plot_search_results(aquisitions, f_bests, fig_dir):

    plt.figure(figsize=(10,5))
    plt.title('Evolution of best results found at number of evaluated points')
    plt.plot(aquisitions, f_bests, 'b')

    plt.xlabel('Points evaluated')
    plt.ylabel('Maximum function value')

    plotting_config.savefig(fig_dir + '/search_results')

    plt.close('all')