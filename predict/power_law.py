# for plotting
import matplotlib
import numpy as np

# evaluation metrics
from sklearn.metrics import r2_score


def build_X(x):
    x0 = np.ones(x.shape)
    x1 = x
    X = np.array([x0, x1])
    return X.T.reshape(-1, 2)


def train(X, y):
    sol1 = np.linalg.lstsq(X, y, rcond=-1) # Solve linear system (least-square solution)
    return sol1[0], sol1[1]


def predict(X, a):
    return X @ a


def power_law_prediction_ax(ax, xs, ys, x_lim_1, x_lim_2, fontsize):
    valid = np.logical_and(xs > 0, ys > 0)
    x_train = xs[valid].reshape(-1, 1)
    y_train = ys[valid].reshape(-1, 1)
    x_train = np.log(x_train)
    y_train = np.log(y_train)
    X = build_X(x_train)
    sol, res = train(X, y_train)
    a, b = np.exp(sol[0]), sol[1]
    y_pred = predict(X, sol)
    std = np.sqrt(res[0] / len(x_train))
    R2 = r2_score(y_train, y_pred)
    ax.scatter(xs, ys, marker='o', s=40, facecolors='none', edgecolors='k')
    ax.set_yscale('log')
    ax.set_xscale('log')
    x_ = np.logspace(x_lim_1, x_lim_2, 10)
    ax.loglog(x_, a*x_**b, '-', color='k')
    ax.loglog(x_, np.exp(1.96*std)*a*x_**b, '--',  color='gray')
    ax.loglog(x_, np.exp(-1.96*std)*a*x_**b, '--',  color='gray')
    ax.set_title(f"R2: {R2:.03}", fontsize=fontsize + 1)
    ax.grid()