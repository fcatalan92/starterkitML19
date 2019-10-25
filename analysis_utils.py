import random
import numpy as np
import pandas as pd
from pandas.core.index import Index
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from IPython.display import HTML

# Data Visualization Functions
###########################################################################################
def plot_distr(df, column=None, figsize=None, bins=10, **kwds):
    """Build a DataFrame and create two dataset for signal and bkg
    Draw histogram of the DataFrame's series comparing the distribution
    in `data1` to `data2`.
    X: data vector
    y: class vector
    column: string or sequence
        If passed, will be used to limit data to a subset of columns
    figsize : tuple
        The size of the figure to create in inches by default
    bins: integer, default 10
        Number of histogram bins to be used
    kwds : other plotting keyword arguments
        To be passed to hist function
    """

    data1 = df[df.y < 0.5]
    data2 = df[df.y > 0.5]

    if column is not None:
        if not isinstance(column, (list, np.ndarray, Index)):
            column = [column]
        data1 = data1[column]
        data2 = data2[column]

    if figsize is None:
        figsize = [20, 15]

    axes = data1.hist(column=column, color='tab:blue', alpha=0.5, bins=bins, figsize=figsize,
                      label="background", density=True, log=True, grid=False, **kwds)
    axes = axes.flatten()
    axes = axes[:len(column)]
    data2.hist(ax=axes, column=column, color='tab:orange', alpha=0.5, bins=bins, label="signal",
               density=True, log=True, grid=False, **kwds)[0].legend()
    for a in axes:
        a.set_ylabel('Counts (arb. units)')


def plot_corr(df, columns, title, figsize=(9, 8), **kwds):
    """Calculate pairwise correlation between features.
    Extra arguments are passed on to DataFrame.corr()
    """

    col = columns+['InvMass', 'y']
    df = df[col]

    if title == "signal":
        data = df[df.y > 0.5].drop('y', 1)
    elif title == "background":
        data = df[df.y < 0.5].drop('y', 1)

    corrmat = data.corr(**kwds)
    _, ax1 = plt.subplots(ncols=1, figsize=figsize)
    opts = {'cmap': plt.get_cmap("coolwarm"), 'vmin': -1, 'vmax': +1, 'snap': True}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)
    ax1.set_title(title, fontsize=22)

    labels = corrmat.columns.values
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels)), minor=False)
        ax.set_yticks(np.arange(len(labels)), minor=False)
        ax.set_xticklabels(labels, minor=False, ha='left', rotation=90, fontsize=15)
        ax.set_yticklabels(labels, minor=False, va='bottom', fontsize=15)

        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')

    plt.tight_layout()


def plot_roc(y_truth, model_decision):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_truth, model_decision)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right")
    plt.grid()


def plot_output_train_test(clf, x_train, y_train, x_test, y_test, columns=None,
                           raw=True, bins=50, figsize=(6, 5), ylim=(1e-5, 1e2), location='best', log=False, **kwds):
    prediction = []
    for x, y in ((x_train, y_train), (x_test, y_test)):
        d1 = clf.predict(x[y > 0.5][columns], output_margin=raw)
        d2 = clf.predict(x[y < 0.5][columns], output_margin=raw)
        prediction += [d1, d2]

    low = min(np.min(d) for d in prediction)
    high = max(np.max(d) for d in prediction)
    low_high = (low, high)

    plt.figure(figsize=figsize)
    plt.hist(prediction[0], color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True, label='S, train', log=log, **kwds)
    plt.hist(prediction[1], color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True, label='B, train', log=log, **kwds)

    hist, bins = np.histogram(prediction[2], bins=bins, range=low_high, density=True)
    scale = len(prediction[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S, test')

    hist, bins = np.histogram(prediction[3], bins=bins, range=low_high, density=True)
    scale = len(prediction[3]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B, test')

    plt.gcf().subplots_adjust(left=0.14)
    plt.xlabel("BDT output", fontsize=15)
    plt.ylabel("Arbitrary units", fontsize=15)
    if log:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(loc=location, frameon=False, fontsize=15)

    plt.tight_layout()


def plot_feature_imp(model, imp_list=None, line_pos=None):

    n_plots = len(imp_list)
    _, ax1 = plt.subplots(ncols=n_plots, figsize=(10, 6), squeeze=False)
    ax1 = ax1[0]

    for imp_type, axc in zip(imp_list, ax1):
        feat_imp = pd.Series(model.get_booster().get_score(importance_type=imp_type))
        feat_imp = feat_imp * 1. / feat_imp.sum()
        feat_imp.plot(ax=axc, kind='bar', fontsize=16)
        axc.set_ylabel(f'Relative {imp_type}', fontsize=16)
        axc.set_xticklabels(axc.get_xticklabels())

        if line_pos is not None:
            axc.axhline(y=line_pos, color='r', linestyle='-', linewidth=6)

    plt.tight_layout()


def plot_bdt_eff(threshold, eff_sig):
    plt.plot(threshold, eff_sig, 'r.', label='Signal efficiency')
    plt.legend()
    plt.xlabel('BDT Score')
    plt.ylabel('Efficiency')
    plt.title('Efficiency vs Score')
    plt.grid()

# Utility for tutorial
###########################################################################################
def show_solution(for_next=True):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' solution'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1, 2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}
            {js_hide_current}
        </script>
        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current,
        toggle_text=toggle_text
    )

    return HTML(html)
    