import numpy as np
# for plotting
import matplotlib
# enable LaTeX style fonts
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# evaluation metrics
from sklearn.metrics import r2_score


def draw_rsquared(ax, x, y, fontsize):
    ax.annotate(r'$\mathbf{R^2}$ = %.4f' % r2_score(x, y),
                xy=(.05, .85),
                fontsize=fontsize + 1,
                xycoords='axes fraction',
                color='k')


def pretty_axis(ax, fontsize):
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.05)
    ax.grid()
    ax.set_aspect('equal')
    ax.tick_params(labelsize=fontsize - 2)


def draw_sublabel(ax, text, fontsize):
    ax.annotate(text,
                xy=(.85, .05),
                fontsize=fontsize + 1,
                xycoords='axes fraction',
                color='k')


def draw_lines(ax, x, y):

    x_ = np.linspace(0, 1)

    A = np.vstack([x, np.ones(len(x))]).T
    (m, c), res, r, s = np.linalg.lstsq(A, y, rcond=-1)
    std = np.sqrt(res[0] / len(y))

    ax.plot(x_, m * x_ + c, 'k', linestyle='-')
    ax.plot(x_, m * x_ + c + 1.96 * std, '--', color='gray')
    ax.plot(x_, m * x_ + c - 1.96 * std, '--', color='gray')



def plot_count_1x2(x_data_1, x_data_2, y_data, x1_label='', x2_label=''):

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax[0].scatter(x_data_1, y_data, marker='o', s=40, facecolors='none', edgecolors='k')
    ax[1].scatter(x_data_2, y_data, marker='o', s=40, facecolors='none', edgecolors='k')
    ax[0].set_ylabel('Count', fontsize=fontsize)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_xlim(0, 1.05)
    ax[0].set_xlabel(x1_label, fontsize=fontsize)
    ax[1].set_xlabel(x2_label, fontsize=fontsize)
    ax[0].tick_params(labelsize=fontsize-2)
    ax[1].tick_params(labelsize=fontsize-2)

    x = np.linspace(0, 1)
    A = np.vstack([x_data_1, np.ones(len(x_data_1))]).T
    (m, c), res, r, s = np.linalg.lstsq(A, y_data)
    std = np.sqrt(res[0] / len(y_data))

    ax[0].plot(x, m*x + c, 'k', linestyle='-')
    ax[0].plot(x, m * x + c + 1.96 * std, '--', color='gray')
    ax[0].plot(x, m * x + c - 1.96 * std, '--', color='gray')

    A = np.vstack([x_data_2, np.ones(len(x_data_2))]).T
    (m, c), res, r, s = np.linalg.lstsq(A, y_data)

    std = np.sqrt(res[0] / len(y_data))
    ax[1].plot(x, m*x + c, 'k', linestyle='-')
    ax[1].plot(x, m * x + c + 1.96 * std, '--', color='gray')
    ax[1].plot(x, m * x + c - 1.96 * std, '--', color='gray')

    ax[0].annotate(r'$\mathbf{R^2}$ = %.4f' % r2_score(x_data_1, y_data),
                xy=(.05, .85), fontsize=fontsize + 1, xycoords='axes fraction', color='k')
    ax[1].annotate(r'$\mathbf{R^2}$ = %.4f' % r2_score(x_data_2, y_data),
                xy=(.05, .85), fontsize=fontsize + 1, xycoords='axes fraction', color='k')

    ax[0].annotate(r'\textbf{a)}', xy=(.85, .05), fontsize=fontsize + 1, xycoords='axes fraction', color='k')
    ax[1].annotate(r'\textbf{b)}', xy=(.85, .05), fontsize=fontsize + 1, xycoords='axes fraction', color='k')

    ax[0].grid()
    ax[1].grid()
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    plt.tight_layout()

    return fig
