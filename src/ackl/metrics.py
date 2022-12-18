import math
import numpy as np
import numpy.testing as npt 
import matplotlib.pyplot as plt
from IPython.display import HTML, display

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from kernels import kernel_names, kernel_hparams, kernel_formulas, \
    kernel_fullnames,kernel_dict,kernel_hparas_divide_n, \
    cosine_kernel
from cla.metrics import get_metrics

'''
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, \
    linear_kernel, sigmoid_kernel, chi2_kernel, polynomial_kernel, \
    additive_chi2_kernel, laplacian_kernel
from kernels import anova_kernel, rq_kernel, rq_kernel_v2, exponential_kernel, imq_kernel, \
    cauchy_kernel, ts_kernel, spline_kernel, sorensen_kernel, min_kernel, minmax_kernel, \
    ghi_kernel, fourier_kernel, wavelet_kernel, log_kernel, power_kernel, matern_kernel, \
    ess_kernel, expmin_kernel, bessel_kernel, feijer_kernel, gaussian_kernel, tanimoto_kernel
'''

def acc(X, y, f, display=False):
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
    XK = f(X, X) # convert mxn matrix to mxm matrix in the kernel space
    XK = np.nan_to_num(XK, copy=False, nan=0)  # replace NaN with 0

    try:
        clf = LogisticRegressionCV().fit(XK, y)
        return clf.score(XK, y)
    except:
        return np.NaN

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
    X1 = X[y==labels[0]]
    X2 = X[y==labels[1]]

    LT = np.nan_to_num(f(X1,X1)) # left-top
    RB = np.nan_to_num(f(X2,X2)) # right-bottom
    RT = np.nan_to_num(f(X1,X2)) # right-top
    LB = np.nan_to_num(f(X2,X1)) # left-bottom

    # for a standard kernel, RT equals LB.    
    # print( npt.assert_almost_equal(RT,LB) )

    # we want RT LB significantly differ from LT RB.
    RT_LB = np.concatenate((RT.flatten(),LB.flatten()))
    LT_RB = np.concatenate((LT.flatten(),RB.flatten()))
    # print(RT_LB.shape, LT_RB.shape)

    mu1 = RT_LB.mean()
    mu2 = LT_RB.mean()
    std1 = RT_LB.std()
    std2 = LT_RB.std()
    m1 = len(RT_LB)
    m2 = len(LT_RB)
    pooled_std = math.sqrt ( ( (m1-1)*std1**2 + (m2-1)*std2**2 ) / (m1-1 + m2-1) )
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

def binary_response_pattern(cmap = 'gray'):
    '''
    Generates the simplest response pattern for just two points 0 and 1

           0      1 
    0  k(0,0) k(0,1)
    1  k(1,0) k(1,1)
    
    Parameter
    ---------
    cmap : color map scheme
    '''
    X = np.array([0,1]).reshape(-1,1)
    _ = preview_kernels(X, np.array([0,1]), cmap, False, True, False, False, False, False)

def linear_response_pattern(n = 10, dim = 1, cmap = 'gray'):
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

    X = np.array(range(1,n+1)).reshape(-1,1) # Generate a column vector of natural series, i.e., 1,2,3...
    y = [0]*round(n/2) + [1]*(n-round(n/2))
    
    if dim == 2:
        zeros = np.array([0] * n).reshape(-1,1)
        X = np.vstack( (np.hstack((X, zeros)), np.hstack((zeros,X)) ))
        y = [0] * n + [1] * n
    
    _ = preview_kernels(X, np.array(y), cmap, False, True, False, False, False, False)

def preview_kernels(X, y, cmap = None, optimize_hyper_params = True, \
    scale = True, metrics = True, logplot = False, scatterplot = True, embed_title = True):
    '''
    Try various kernel types to evaluate the pattern after kernel-ops.  
    Each kernel uses their own default/empirical paramters.    
    In binary classification, because the first and last half samples belong to two classes respectively. 
    We want to get a contrasive pattern, i.e., (LT、RB) 与 （LB、RT）have blocks of different colors. 

    Parameters
    ----------
    X : an m-by-n data matrix. 
        Should be already rescaled to non-negative ranges (required by chi2 family) 
        and re-ordered by y. 
    y : class labels
    cmap : color map scheme to use. set None to use default, or set another scheme, e.g., 'gray', 'viridis', 'jet', 'rainbow', etc.
        For small dataset, we recommend 'gray'. For complex dataset, we recommend 'viridis'.
    optimize_hyper_params : whether to optimize hyper parameters for each kernel. 
        For real-world dataset, use True. For toy dataset, usually use False.
    scale : whether do feature scaling
    metrics : whether calculate clam metrics.
    logplot : whether to output the log-scale plot in parallel.
    scatterplot : whether to ouptut the scatter plots after PCA / PLS, to check classifiability.
    embed_title : whether embed the title in the plots. If not, will generate the title in HTML.
    '''

    all_dic_metrics = {}

    if scale:
        X = MinMaxScaler().fit_transform(X)

    # to be safe, perform a re-order 
    if (y is not None and len(set(y)) == 2):
        labels = list(set(y)) # re-order X by y
        X = np.vstack( (X[y==labels[0]], X[y==labels[1]] ))
        y = [labels[0]] * np.sum(y==labels[0]) + [labels[1]] * np.sum(y == labels[1])
    
    for i, key in enumerate(kernel_names):
        best_hparam = None
        best_metric = -np.inf
        title = str(i+1) + '. ' + (kernel_fullnames[key] if key in kernel_fullnames else key) 

        if optimize_hyper_params and key in kernel_hparams:        
            for param in kernel_hparams[key]:
                if key in kernel_hparas_divide_n:
                    param = param/X.shape[1] # divide hparam by data dim
                new_metric = nmd(X, y, lambda x,y : kernel_dict[key](x,y, param))
                if (new_metric > best_metric):
                    best_metric = new_metric
                    best_hparam = param
            title = str(i+1) + '. ' + (kernel_fullnames[key] if key in kernel_fullnames else key) \
                + ('(' +format(best_hparam,'.2g')+ ')') if best_hparam is not None else ''

        if not optimize_hyper_params or key not in kernel_hparams:
            kns = kernel_dict[key](X,X)
            metric_nmd = nmd(X, y, lambda x,y : kernel_dict[key](x,y))
        else:
            kns = kernel_dict[key](X,X, best_hparam)
            metric_nmd = nmd(X, y, lambda x,y : kernel_dict[key](x,y,param))

        ######## plot after kernel transforms ########
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        ax[0].imshow(kns, cmap = cmap)
        ax[0].set_axis_off()
        if logplot:
            ax[1].imshow(1+np.log(kns), cmap = cmap)
            ax[1].set_axis_off()

        plt.axis('off')
        if embed_title:
            plt.title(title +  '\n' + kernel_formulas[key] + '\n' + "NMD = %.3g" % metric_nmd)
        else:
            # print(title)
            display(HTML('<h3>' + title + '</h3>' + '<p>' + kernel_formulas[key].replace('<','&lt;') \
                .replace('>','&gt;') + '</p><p>' + "NMD = %.3g" % metric_nmd + '</p>' ))
        plt.show()

        if scatterplot:

            ######## scatter plot after PCA ########
            kns = np.nan_to_num(kns)
            pca = PCA(n_components=2)  # keep the first 2 components
            X_pca = pca.fit_transform(kns)
            plotComponents2D(X_pca, y)
            plt.title('PCA')
            plt.show()

            ######## scatter plot after PLS ########
            
            try:
                kns = np.nan_to_num(kns)
                pls = PLSRegression(n_components=2, scale=False)
                X_pls = pls.fit(kns, y).transform(kns)
            
                pls.score(kns, y)
                plotComponents2D(X_pls, y)
                plt.title('PLS (R2 = ' + str(np.round(pls.score(kns, y),3)) + ')') # the coefficient of determination or R squared method is the proportion of the variance in the dependent variable that is predicted from the independent variable. 
                plt.show()
            except Exception as e:
                print('Exception : ', e)
                # print('X_pls = ', X_pls)
                # plt.title('PLS')
            
        ###### metrics ######
        if metrics:            
            kns = np.nan_to_num(np.hstack((kns,np.array(y).reshape(-1,1))),   # do nan filtering simultaneously for X and y
                nan=0, posinf = kns.max(), neginf = kns.min())
            _, dic_metrics = get_metrics(kns[:,:-1], kns[:,-1].flatten())
            dic_metrics['NMD'] = metric_nmd
            all_dic_metrics[key] = dic_metrics

    return all_dic_metrics

# metric_eval_dict={...}
metric_eval_dict={'classification.ACC': np.argmax,
 'classification.Kappa': np.argmax,
 'classification.F1_Score': np.argmax ,
 'classification.Jaccard': np.argmin,
 'classification.Precision': np.argmax,
 'classification.Recall': np.argmax,
 'classification.McNemar': np.argmin,
 'classification.McNemar.CHI2':np.argmax,
 'classification.CochranQ': np.argmin,
 'classification.CochranQ.T':np.argmax,
 'classification.CrossEntropy':np.argmin,
 'classification.Mean_KLD': np.argmin,
 'classification.AP': np.argmax,
 'classification.Brier':np.argmin ,
 'classification.ROC_AUC':np.argmax,
 'classification.PR_AUC':np.argmin,
 'classification.BER':np.argmin,
 'classification.SVM.Margin':np.argmax,
 'correlation.IG':np.argmax,
 'correlation.IG.max':np.argmax,
 'correlation.r':np.argmax,
 'correlation.r2':np.argmax,
 'correlation.r.p':np.argmin,
 'correlation.r.max':np.argmax,
 'correlation.r.p.min':np.argmin,
 'correlation.rho':np.argmax,
 'correlation.rho.p':np.argmin,
 'correlation.rho.max':np.argmax,
 'correlation.rho.p.min':np.argmin,
 'correlation.tau':np.argmin,
 'correlation.tau.p':np.argmin,
 'correlation.tau.max':np.argmax,
 'correlation.tau.p.min':np.argmin,
 'test.ES':np.argmax,
 'test.ES.max':np.argmax,
 'test.student':np.argmin,
 'test.student.min':np.argmin,
 'test.student.min.log10':np.argmin,
 'test.student.T':np.argmax,
 'test.student.T.max':np.argmin,
 'test.ANOVA':np.argmin,
 'test.ANOVA.min':np.argmin,
 'test.ANOVA.min.log10':np.argmin,
 'test.ANOVA.F':np.argmax,
 'test.ANOVA.F.max':np.argmax,
 'test.MANOVA':np.argmin,
 'test.MANOVA.log10':np.argmin,
 'test.MANOVA.F':np.argmax,
 'test.MWW':np.argmin,
 'test.MWW.min':np.argmin,
 'test.MWW.min.log10':np.argmin,
 'test.MWW.U':np.argmin,
 'test.MWW.U.min':np.argmin,
 'test.KS':np.argmin,
 'test.KS.min':np.argmin,
 'test.KS.min.log10':np.argmin,
 'test.KS.D':np.argmax,
 'test.KS.D.max':np.argmax,
 'test.CHISQ':np.argmin,
 'test.CHISQ.min':np.argmin,
 'test.CHISQ.min.log10':np.argmin,
 'test.CHISQ.CHI2':np.argmax,
 'test.CHISQ.CHI2.max':np.argmax,
 'test.KW':np.argmin,
 'test.KW.min':np.argmin,
 'test.KW.min.log10':np.argmin,
 'test.KW.H':np.argmax,
 'test.KW.H.max':np.argmax,
 'test.Median':np.argmin,
 'test.Median.min':np.argmin,
 'test.Median.min.log10':np.argmin,
 'test.Median.CH2':np.argmax,
 'test.Median.CH2.max':np.argmax,
 'overlapping.F1.mean':np.argmax,
 'overlapping.F1.sd':np.argmax,
 'overlapping.F1v.mean':np.argmax,
 'overlapping.F1v.sd':np.argmin,
 'overlapping.F2.mean':np.argmin,
 'overlapping.F2.sd':np.argmin,
 'overlapping.F3.mean':np.argmax,
 'overlapping.F3.sd':np.argmin,
 'overlapping.F4.mean':np.argmax,
 'overlapping.F4.sd':np.argmin,
 'neighborhood.N1':np.argmin,
 'neighborhood.N2.mean':np.argmin,
 'neighborhood.N2.sd':np.argmin,
 'neighborhood.N3.mean':np.argmin,
 'neighborhood.N3.sd':np.argmax,
 'neighborhood.N4.mean':np.argmin,
 'neighborhood.N4.sd':np.argmin,
 'neighborhood.T1.mean':np.argmax,
 'neighborhood.T1.sd':np.argmin,
 'neighborhood.LSC':np.argmax,
 'linearity.L1.mean':np.argmin,
 'linearity.L1.sd':np.argmin}

def visualize_metric_dicts(dics, plot = True):
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
    html_str += '</tr>'

    # use the 2nd loop to fill in data
    for row in row_names:
        html_str += '<tr><td>' + row + '</td>'
        metrics = []
        for col in column_names:
            metrics.append(dics[col][row] if row in dics[col] else np.nan)
            html_str += '<td>' + ( str(round(dics[col][row],3)) if row in dics[col] else '') + '</td>'
        
        # metric_eval_dict > find best
        # add a <td>best value</td>
        
        html_str += '</tr>'
        
        if plot:
            plt.figure(figsize = (20,3))
            plt.title(row)
            plt.bar(column_names, metrics, alpha = 0.7, width=0.6, edgecolor = 'black', color = 'white')
            plt.xticks(rotation = 40)
            plt.show()

    html_str += '</table>'
    # display(HTML( html_str ))

    return html_str

def optimize_kernel_hparam(X, y, key, hparams = [], cmap = None):
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
                + ('(' +format(h,'.2g')+ ')')
        plt.imshow(kernel_dict[key](X,X, h), cmap = cmap)
        metric_str = "NMD = %.3g" % nmd(X, y, lambda x,y : kernel_dict[key](x,y, h)) \
                + "\tACC = %.3g" % acc(X, y, lambda x,y : kernel_dict[key](x,y, h)) 
        plt.axis('off')
        plt.title(title +  '\n' + kernel_formulas[key] + '\n' + metric_str)
        plt.show()    

def cosine_kernel_response_pattern (n = 10, cmap = 'gray', logplot = False, embed_title = False):
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
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].imshow(kns, cmap = cmap)
    ax[0].set_axis_off()
    if logplot:
        ax[1].imshow(1+np.log(kns), cmap = cmap)
        ax[1].set_axis_off()
    plt.axis('off')

    metric_str = "NMD = %.3g" % nmd(X, y, lambda x,y : kernel_dict['cosine'](x,y)) \
                + "  ACC = %.3g" % acc(X, y, lambda x,y : kernel_dict['cosine'](x,y))

    if embed_title:
        plt.title(kernel_fullnames['cosine'] + '\n' + kernel_formulas['cosine'] \
            +'\n' + metric_str +'\n' + comment)
    else:
        display(HTML('<h3>' + kernel_fullnames['cosine'] + '</h3>' + '<p>' + \
            kernel_formulas['cosine'].replace('<','&lt;').replace('>','&gt;') + '</p>' \
            + '<p>' + metric_str + '</p>' + '<p>' + comment + '</p>'))

    plt.show()
 

def plotComponents2D(X, y, labels = None, use_markers = False, ax=None, legends = None, tags = None):
    '''
    Copied from qsi.vis.plotComponents2D, to avoid package dependency.
    '''
    if X.shape[1] < 2:
        print('ERROR: X MUST HAVE AT LEAST 2 FEATURES/COLUMNS! SKIPPING plotComponents2D().')
        return
    
    # Gray shades can be given as a string encoding a float in the 0-1 range
    colors = ['0.9', '0.1', 'red', 'blue', 'black','orange','green','cyan','purple','gray']
    markers = ['o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H']

    if (ax is None):
        fig, ax = plt.subplots()
        
    if (y is None or len(y) == 0):
        labels = [0] # only one class
    if (labels is None):
        labels = set(y)

    i=0        

    for label in labels:
        if y is None or len(y) == 0:
            cluster = X
        else:
            cluster = X[np.where(y == label)]
        # print(cluster.shape)

        if use_markers:
            ax.scatter([cluster[:,0]], [cluster[:,1]], 
                       s=40, 
                       marker=markers[i], 
                       facecolors='none', 
                       edgecolors=colors[i+3],
                       label= (str(legends[i]) if legends is not None else ("Y = " + str(label)  + ' (' + str(len(cluster)) + ')')) )
        else:
            ax.scatter([cluster[:,0]], [cluster[:,1]], 
                       s=70, 
                       facecolors=colors[i],  
                       label= (str(legends[i]) if legends is not None else ("Y = " + str(label) + ' (' + str(len(cluster)) + ')')), 
                       edgecolors = 'black', 
                       alpha = .4) # cmap='tab20'                
        i=i+1
    
    if (tags is not None):
        for j,tag in enumerate(tags):
            ax.annotate(str(tag), (X[j,0] + 0.1, X[j,1] - 0.1))
        
    ax.legend()

    ax.axes.xaxis.set_visible(False) 
    ax.axes.yaxis.set_visible(False)
    
    return ax