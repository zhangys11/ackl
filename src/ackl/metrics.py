import os
import sys
import math
import timeit
import numpy as np
import matplotlib.pyplot as plt
# import full name to avoid conflict with those display params
import IPython.core.display

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from cla.metrics import get_metrics, metric_polarity_dict, es_max

if __package__:
    from .kernels import kernel_names, kernel_hparams, kernel_formulas, \
    kernel_fullnames, kernel_dict, kernel_hparas_divide_n, \
    cosine_kernel
else:
    ROOT_DIR = os.path.dirname(__file__)
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    from kernels import kernel_names, kernel_hparams, kernel_formulas, \
    kernel_fullnames, kernel_dict, kernel_hparas_divide_n, \
    cosine_kernel

def acc(X, y, f):
    '''
    Use classification accuracy as a metric for kernel performance

    Parameters
    ----------
    X : expect an already-scaled data matrix
    y : target labels. support bianary classification.
    f : a kernelfunciton. Can use lambda to pass concrete kernel-specific parameters, e.g., 
        lambda x, y: laplacian_kernel(x, y, 2.0/N)
        lambda x, y: laplacian_kernel(x, y, 4.0/N)
    '''
    XK = f(X, X)  # convert mxn matrix to mxm matrix in the kernel space
    XK = np.nan_to_num(XK, copy=False, nan=0)  # replace NaN with 0

    try:
        clf = LogisticRegressionCV().fit(XK, y)
        return clf.score(XK, y)
    except Exception as e:
        print(e)
        return np.NaN


def kes(X, y, f):
    '''
    kernel effect size

    Parameter
    ---------
    f - a kernel function. Can use lambda to pass concrete kernel-specific parameters, e.g., 
        lambda x, y: laplacian_kernel(x, y, 2.0/N)
        lambda x, y: laplacian_kernel(x, y, 4.0/N)

    Return
    ------
    The max effect size of all the PLS components after kernel transform
    '''
    XK = f(X, X)  # kernel transform
    XK = np.nan_to_num(XK)

    try:
        lda = LinearDiscriminantAnalysis()
        X_dr = lda.fit_transform(XK, y)
    except:
        print('Exception in LDA. Try PLS.')
        try:
            pls = PLSRegression(n_components=len(XK), scale=False)
            X_dr = pls.fit(XK, y).transform(XK)
        except:
            print('Exception in PLS. Use PCA instead.')
            try:
                X_dr = PCA().fit_transform(XK)
            except:
                X_dr = XK  # do without DR

    X_dr = np.nan_to_num(X_dr)
    return es_max(X_dr, y)


def nmd(X, y, f, display=False):
    '''
    Computes the normalized mean difference between the between-class and within-class matrices.

    Parameter
    ---------
    f - a kernel function. Can use lambda to pass concrete kernel-specific parameters, e.g., 
        lambda x, y: laplacian_kernel(x, y, 2.0/N)
        lambda x, y: laplacian_kernel(x, y, 4.0/N)

    Return
    ------
    ratio
    '''

    labels = list(set(y))
    assert len(labels) == 2
    X1 = X[y == labels[0]]
    X2 = X[y == labels[1]]

    LT = np.nan_to_num(f(X1, X1))  # left-top
    RB = np.nan_to_num(f(X2, X2))  # right-bottom
    RT = np.nan_to_num(f(X1, X2))  # right-top
    LB = np.nan_to_num(f(X2, X1))  # left-bottom

    # for a standard kernel, RT equals LB.
    # print( npt.assert_almost_equal(RT,LB) )

    # we want RT LB significantly differ from LT RB.
    RT_LB = np.concatenate((RT.flatten(), LB.flatten()))
    LT_RB = np.concatenate((LT.flatten(), RB.flatten()))
    # print(RT_LB.shape, LT_RB.shape)

    mu1 = RT_LB.mean()
    mu2 = LT_RB.mean()
    std1 = RT_LB.std()
    std2 = LT_RB.std()
    m1 = len(RT_LB)
    m2 = len(LT_RB)
    pooled_std = math.sqrt(((m1-1)*std1**2 + (m2-1)*std2**2) / (m1-1 + m2-1))
    res = abs(mu1-mu2) / pooled_std

    # r1 = ( RT.sum() + LB.sum()) / ( LT.sum() + RB.sum())
    # r2 = ( np.log(RT).sum() + np.log(LB).sum()) / ( np.log(LT).sum() + np.log(RB).sum())
    # return r1,r2

    if display:
        plt.figure()
        plt.matshow(LT)
        plt.title("Left Top")
        plt.show()
        plt.matshow(RB)
        plt.title("Right Bottom")
        plt.show()
        plt.matshow(RT)
        plt.title("Right Top")
        plt.show()
        plt.matshow(LB)
        plt.title("Left Bottom")
        plt.show()

    return res


def binary_response_pattern(cmap='gray'):
    '''
    Generates the simplest response pattern for just two points 0 and 1

           0      1 
    0  k(0,0) k(0,1)
    1  k(1,0) k(1,1)

    Parameter
    ---------
    cmap : color map scheme
    '''
    X = np.array([0, 1]).reshape(-1, 1)
    _ = preview_kernels(X, np.array(
        [0, 1]), cmap, None, True, False, False, False, False)


def linear_response_pattern(n=10, dim=1, cmap='gray'):
    '''
    Generates a response pattern of pairwise-point-distances.
    See how each kernel responds to / varies with linearly arranged points.

    Parameter
    ---------
    n : sample size. 
    dim : how many dimensions to use for the sample data
        If dim = 1, will use n natural series, i.e., 1,2,3,4... 
        If dim = 2, will use two 2d series, i.e., (0,1),(0,2),(0,3),(0,4)... and (1,0),(2,0),(3,0),(4,0)...
    cmap : color map scheme
    '''

    # Generate a column vector of natural series, i.e., 1,2,3...
    X = np.array(range(1, n+1)).reshape(-1, 1)
    y = [0]*round(n/2) + [1]*(n-round(n/2))

    if dim == 2:
        zeros = np.array([0] * n).reshape(-1, 1)
        X = np.vstack((np.hstack((X, zeros)), np.hstack((zeros, X))))
        y = [0] * n + [1] * n

    _ = preview_kernels(X, np.array(y), cmap, None,
                        True, False, False, False, False)


def preview_kernels(X, y, cmap=None, hyper_param_optimizer=kes,
                    scale=False, metrics=True, logplot=False, 
                    scatterplot=True, embed_title=True, 
                    selected_kernel_names=None):
    '''
    Try various kernel types to evaluate the pattern after kernel-ops.  
    Each kernel uses their own default/empirical paramters.    
    In binary classification, because the first and last half samples belong to two classes respectively. 
    We want to get a contrasive pattern, i.e., (LT、RB) and （LB、RT）have blocks of different colors. 

    Parameters
    ----------
    X : an m-by-n data matrix. 
        Should be already rescaled to non-negative ranges (required by chi2 family) 
        and re-ordered by y. 
    y : class labels
    cmap : color map scheme to use. set None to use default, or set another scheme, e.g., 'gray', 'viridis', 'jet', 'rainbow', etc.
        For small dataset, we recommend 'gray'. For complex dataset, we recommend 'viridis'.
    hyper_param_optimizer : which optimizer to optimize the hyper parameters for each kernel. 
        For real-world dataset, use kes by default. For toy dataset, set None to disable optimizer.
    scale : whether do feature scaling
    metrics : whether calculate classifiability metrics.
    logplot : whether to output the log-scale plot in parallel.
    scatterplot : whether to ouptut the kernel heatmaps and the scatter plots after PCA / PLS, to check classifiability.
        The PLS tries to maximize the covariance between X and Y.
    embed_title : whether embed the title in the plots. If not, will generate the title in HTML.
    selected_kernel_names : a list of kernel names to be process. 
        If None or 'all', will use all kernels.
    '''

    all_dic_metrics = {}

    if scale:
        X = MinMaxScaler().fit_transform(X)

    # to be safe, perform a re-order
    if (y is not None and len(set(y)) == 2):
        labels = list(set(y))  # re-order X by y
        X = np.vstack((X[y == labels[0]], X[y == labels[1]]))
        y = [labels[0]] * np.sum(y == labels[0]) + \
            [labels[1]] * np.sum(y == labels[1])

    if selected_kernel_names is None or selected_kernel_names == 'all':
        selected_kernel_names = kernel_names

    for i, key in enumerate(selected_kernel_names):
        best_hparam = None
        best_metric = -np.inf
        title = str(i+1) + '. ' + \
            (kernel_fullnames[key] if key in kernel_fullnames else key)

        if hyper_param_optimizer is not None and key in kernel_hparams:
            for param in kernel_hparams[key]:
                if key in kernel_hparas_divide_n:
                    param = param/X.shape[1]  # divide hparam by data dim
                new_metric = hyper_param_optimizer(
                    X, y, lambda x, y: kernel_dict[key](x, y, param))
                if (new_metric > best_metric):
                    best_metric = new_metric
                    best_hparam = param
            title = str(i+1) + '. ' + (kernel_fullnames[key] if key in kernel_fullnames else key) \
                + ('(' + format(best_hparam, '.2g') +
                   ')') if best_hparam is not None else ''

        if hyper_param_optimizer is None or key not in kernel_hparams:
            kns = kernel_dict[key](X, X)
            metric_nmd = nmd(X, y, lambda x, y: kernel_dict[key](x, y))
        else:
            kns = kernel_dict[key](X, X, best_hparam)
            metric_nmd = nmd(X, y, lambda x, y: kernel_dict[key](x, y, param))

        ######## plot after kernel transforms ########
        
        if scatterplot:
            _, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(kns, cmap=cmap)
            ax[0].set_axis_off()
            if logplot:
                ax[1].imshow(1+np.log(kns), cmap=cmap)
                ax[1].set_axis_off()

            plt.axis('off')
            if embed_title:
                plt.title(title + '\n' +
                        kernel_formulas[key] + '\n' + "NMD = %.3g" % metric_nmd)
            else:
                # print(title)
                IPython.display.display(IPython.display.HTML('<h3>' + title + '</h3>' + '<p>' + kernel_formulas[key].replace('<', '&lt;')
                                                            .replace('>', '&gt;') + '</p><p>' + "NMD = %.3g" % metric_nmd + '</p>'))
            plt.show()

            ######## scatter plot after PCA ########
            kns = np.nan_to_num(kns)
            pca = PCA(n_components=2)  # keep the first 2 components
            X_pca = pca.fit_transform(kns)
            X_pca = np.nan_to_num(X_pca)
            plot_components_2d(X_pca, y)
            plt.title('PCA')
            plt.show()

            ######## scatter plot after LDA ########

            try:
                kns = np.nan_to_num(kns)
                lda = LinearDiscriminantAnalysis()
                X_lda = lda.fit(kns, y).transform(kns)
                X_lda = np.nan_to_num(X_lda)

                lda.score(kns, y)
                if (X_lda.shape[1] == 1):
                    X_lda = np.hstack((X_lda, np.zeros((X_lda.shape[0], 1))))
                plot_components_2d(X_lda, y)
                # the coefficient of determination or R squared method is the proportion of the variance in the dependent variable that is predicted from the independent variable.
                plt.title(
                    'LDA (ACC = ' + str(np.round(lda.score(kns, y), 3)) + ')')
                plt.show()
            except Exception as e:
                print('Exception : ', e)
                # print('X_pls = ', X_pls)
                # plt.title('PLS')

            ######## scatter plot after PLS ########

            try:
                ''' # using CV
                kns = np.nan_to_num(kns)
                pls = PLSRegression(n_components=2, scale=False)

                k_train, k_test, y_train, y_test = train_test_split(
                    kns, y, test_size=0.3)

                X_pls = pls.fit(k_train, y_train).transform(k_test)
                X_pls = np.nan_to_num(X_pls)

                pls.score(k_test, y_test)
                plot_components_2d(X_pls, y_test, legends=['C1', 'C2'])
                # the coefficient of determination or R squared method is the proportion of the variance in the dependent variable that is predicted from the independent variable.
                plt.title(
                    'PLS (R2 = ' + str(np.round(pls.score(k_test, y_test), 3)) + ')')
                plt.show()
                '''

                kns = np.nan_to_num(kns)
                pls = PLSRegression(n_components=2, scale=False)
                X_pls = pls.fit(kns, y).transform(kns)
                X_pls = np.nan_to_num(X_pls)

                pls.score(kns, y)
                plot_components_2d(X_pls, y)
                # the coefficient of determination or R squared method is the proportion of the variance in the dependent variable that is predicted from the independent variable.
                plt.title(
                    'PLS (R2 = ' + str(np.round(pls.score(kns, y), 3)) + ')')
                plt.show()
            except Exception as e:
                print('Exception : ', e)
                # print('X_pls = ', X_pls)
                # plt.title('PLS')

        ###### metrics ######
        if metrics:
            kns = np.nan_to_num(np.hstack((kns, np.array(y).reshape(-1, 1))),   # do nan filtering simultaneously for X and y
                                nan=0, posinf=kns.max(), neginf=kns.min())
            _, dic_metrics = get_metrics(kns[:, :-1], kns[:, -1].flatten())
            dic_metrics['NMD'] = metric_nmd
            all_dic_metrics[key] = dic_metrics

    return all_dic_metrics


def visualize_metric_dicts(dics, plot=True):
    '''
    Example
    -------
    dics = preview_kernels(X, y)
    html_str = generate_html_from_dicts(dics)
    display(HTML(html_str)) # use in jupyter notebook
    '''

    row_names = []
    column_names = []

    html_str = '<table><tr><th></th>'

    # use the 1st loop to get row and col names
    for kernel in dics:
        column_names.append(kernel)
        html_str += '<th>' + kernel + '</th>'
        for key in dics[kernel]:
            if key not in row_names:
                row_names.append(key)

    html_str += '<th>best kernel(s)</th>'
    html_str += '</tr>'

    # use the 2nd loop to fill in data
    for row in row_names:
        if row not in metric_polarity_dict:
            continue

        html_str += '<tr><td>' + row + '</td>'
        metrics = []
        for col in column_names:
            metrics.append(dics[col][row] if row in dics[col] else np.nan)
            html_str += '<td>' + \
                (str(round(dics[col][row], 3))
                 if row in dics[col] else '') + '</td>'

        metrics = np.nan_to_num(metrics, nan=np.nan,
                                posinf=np.nan, neginf=np.nan)

        best_metric_value = None
        try:
            if row == 'NMD':  # this is a kernel-specific metric, not listed in metric_polarity_dict
                best_metric_idx = np.nanargmax(metrics)
            else:
                best_metric_idx = metric_polarity_dict[row](metrics)

            # we may have multiple best values
            best_metric_value = metrics[best_metric_idx]
            best_metric_idxs = np.where(metrics == best_metric_value)
        except Exception as e:
            print(e)  # All-NaN slice encountered
            best_metric_idxs = []

        best_kernel_names = str(np.array(column_names)[best_metric_idxs])
        html_str += '<td>' + best_kernel_names + '</td>'
        html_str += '</tr>'

        if plot:
            plt.figure(figsize=(20, 3))
            plt.bar(column_names, metrics, alpha=0.7,
                    width=0.6, edgecolor='black', color='white', label=row)
            plt.bar(np.array(column_names)[best_metric_idxs], metrics[best_metric_idxs], alpha=0.7,
                    width=0.6, edgecolor='black', color='gold',
                    label="best kernels: " + best_kernel_names)
            plt.xticks(rotation=40)
            plt.legend()
            plt.show()

            print(row + "\nbest kernels: " + best_kernel_names +
                  "\nbest value: " + str(best_metric_value))

    html_str += '</table>'
    # display(HTML( html_str ))

    return html_str


def optimize_kernel_hparam(X, y, key, hparams=[], cmap=None):
    '''
    Paramters
    ---------
    X : need to re-order X by y first
    '''
    if hparams is None or len(hparams) <= 0:
        print('hparams is empty. Try built-in values.')
        if key not in kernel_hparams:
            print('Built-in hparams not found. Exit.')
            return

    for h in hparams:
        title = (kernel_fullnames[key] if key in kernel_fullnames else key) \
            + ('(' + format(h, '.2g') + ')')
        plt.imshow(kernel_dict[key](X, X, h), cmap=cmap)
        metric_str = "NMD = %.3g" % nmd(X, y, lambda x, y: kernel_dict[key](x, y, h)) \
            + "\tACC = %.3g" % acc(X, y, lambda x,
                                   y: kernel_dict[key](x, y, h))
        plt.axis('off')
        plt.title(title + '\n' + kernel_formulas[key] + '\n' + metric_str)
        plt.show()


def cosine_kernel_response_pattern(n=10, cmap='gray', logplot=False, embed_title=False):
    '''
    Special demo for the cosine kernel with 2D dataset. 
    We use equal-angle-interval input. While for other kernels, we use equal-interval input. 
    '''

    comment = 'Unlike other kernels, we use equal-angle-interval (e.g., $ (i/n)*\pi $) input for the cosine kernel.'
    X = []
    for i in range(n):
        theta = math.pi / n * i
        X.append([math.cos(theta), math.sin(theta)])

    X = np.array(X)

    y = [0]*round(n/2) + [1]*(n-round(n/2))
    y = np.array(y)

    kns = cosine_kernel(X, X)
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(kns, cmap=cmap)
    ax[0].set_axis_off()
    if logplot:
        ax[1].imshow(1+np.log(kns), cmap=cmap)
        ax[1].set_axis_off()
    plt.axis('off')

    metric_str = "NMD = %.3g" % nmd(X, y, lambda x, y: kernel_dict['cosine'](x, y)) \
        + "  ACC = %.3g" % acc(X, y, lambda x, y: kernel_dict['cosine'](x, y))

    if embed_title:
        plt.title(kernel_fullnames['cosine'] + '\n' + kernel_formulas['cosine']
                  + '\n' + metric_str + '\n' + comment)
    else:
        IPython.display.display(IPython.display.HTML('<h3>' + kernel_fullnames['cosine'] + '</h3>' + '<p>' +
                                                     kernel_formulas['cosine'].replace(
            '<', '&lt;').replace('>', '&gt;') + '</p>'
            + '<p>' + metric_str + '</p>' + '<p>' + comment + '</p>'))

    plt.show()


def time_cost_kernels(X, repeat=10, display=True):
    '''
    Time cost of kernels

    return
    ------
    dct_mu : a dict of mean time cost of each kernel
    '''
    dct = {}
    for key, f in kernel_dict.items():
        dct[key] = timeit.repeat(lambda: f(X, X), repeat=repeat, number=1)

    x, y = zip(*dct.items())
    y = np.array(y)

    if display:
        plt.figure(figsize=(15, 4))
        plt.scatter(x, y.mean(axis=1))
        plt.errorbar(x, y.mean(axis=1), y.std(axis=1) * 1,
                     color="dodgerblue", linewidth=1, elinewidth=10, ecolor='r',
                     alpha=0.5, label=' $\mu ± 1 \sigma$ (' + str(y.shape[1]) + ' runs)')  # X.std(axis = 0)

        plt.legend()
        # plt.bar(x, y)
        plt.xticks(rotation=45)
        plt.ylabel('second')
        plt.show()

    dct_mu = {}
    for k, v in zip(x, y.mean(axis=1)):
        dct_mu[k] = v

    return dct_mu


def plot_components_2d(X, y, labels=None, use_markers=False, ax=None, legends=None, tags=None):
    '''
    Copied from qsi.vis.plot_components_2d, to avoid package dependency.
    '''
    if X.shape[1] < 2:
        print(
            'ERROR: X MUST HAVE AT LEAST 2 FEATURES/COLUMNS! SKIPPING plot_components_2d().')
        return

    # Gray shades can be given as a string encoding a float in the 0-1 range
    colors = ['0.9', '0.1', 'red', 'blue', 'black',
              'orange', 'green', 'cyan', 'purple', 'gray']
    markers = ['o', 's', '^', 'D', 'H', 'o', 's', '^', 'D',
               'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H']

    if ax is None:
        _, ax = plt.subplots()

    if y is None or len(y) == 0:
        labels = [0]  # only one class
    if labels is None:
        labels = set(y)

    i = 0

    for label in labels:
        if y is None or len(y) == 0:
            cluster = X
        else:
            cluster = X[np.where(y == label)]
        # print(cluster.shape)

        if use_markers:
            ax.scatter([cluster[:, 0]], [cluster[:, 1]],
                       s=40,
                       marker=markers[i],
                       facecolors='none',
                       edgecolors=colors[i+3],
                       label=(str(legends[i]) if legends is not None else ("Y = " + str(label) + ' (' + str(len(cluster)) + ')')))
        else:
            ax.scatter([cluster[:, 0]], [cluster[:, 1]],
                       s=70,
                       facecolors=colors[i],
                       label=(str(legends[i]) if legends is not None else (
                           "Y = " + str(label) + ' (' + str(len(cluster)) + ')')),
                       edgecolors='black',
                       alpha=.4)  # cmap='tab20'
        i = i+1

    if tags is not None:
        for j, tag in enumerate(tags):
            ax.annotate(str(tag), (X[j, 0] + 0.1, X[j, 1] - 0.1))

    ax.legend()

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    return ax
