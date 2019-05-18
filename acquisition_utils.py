import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def expected_improvement(f_samples, f_best):
    f_mean = np.mean(f_samples, axis=0)
    f_delta = f_mean - f_best
    f_std = np.std(f_samples, axis=0)
    Z = f_delta / f_std

    EI = f_delta * norm.cdf(Z) + f_std * norm.pdf(Z)

    # EI = f_delta + \
    #      f_std * norm.pdf(f_delta / f_std) - \
    #      np.abs(f_delta) * norm.cdf(-np.abs(f_delta) / f_std)

    # EI = np.clip(EI, a_min=0, a_max=None)

    return EI


def probability_improvement(f_samples, f_best):
    f_mean = np.mean(f_samples, axis=0)
    f_delta = f_mean - f_best
    f_std = np.std(f_samples, axis=0)

    return norm.cdf(f_delta / f_std)


def standard_dev_improvement(f_samples, f_best):
    f_mean = np.mean(f_samples, axis=0)
    f_delta = f_mean - f_best
    f_std = np.std(f_samples, axis=0)

    return f_delta + f_std


if __name__ == '__main__':
    x = np.linspace(-5, 5, 100)

    y = -x ** 2 / 5
    s = 5 * np.abs(np.sin(x))
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, y + s, ls='--')
    plt.plot(x, y - s, ls='--')

    ei = expected_improvement(y - (max(y)), s)
    plt.figure()
    plt.plot(x, ei)

    plt.show()
