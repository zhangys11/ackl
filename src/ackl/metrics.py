from cmath import inf, nan
import math
import numpy as np
import numpy.testing as npt 
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, \
    linear_kernel, sigmoid_kernel, chi2_kernel, polynomial_kernel, \
    additive_chi2_kernel, laplacian_kernel
from sklearn.preprocessing import MinMaxScaler
from kernels import anova_kernel, rq_kernel, rq_kernel_v2, exponential_kernel, imq_kernel, \
    cauchy_kernel, ts_kernel, spline_kernel, sorensen_kernel, min_kernel, minmax_kernel, \
    ghi_kernel, fourier_kernel, wavelet_kernel, log_kernel, power_kernel, matern_kernel, \
    ess_kernel, expmin_kernel, bessel_kernel, feijer_kernel, gaussian_kernel
from kernels import kernel_names, kernel_hparams, kernel_formulas, kernel_fullnames,kernel_dict

def nmd(X1, X2, f, display=False):
    '''
    Computes the normalized mean difference between the between-class and within-class matrices.
    
    Parameter
    ---------
    X1 - samples of Class 1
    X2 - samples of Class 2
    f - a kernel function. Can use lambda to pass concrete kernel-specific parameters, e.g., 
        lambda x, y: laplacian_kernel(x, y, 2.0/N)
        lambda x, y: laplacian_kernel(x, y, 4.0/N)

    Return
    ------
    ri - ratio
    r2 - log of ratio
    '''
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
    nmd = abs(mu1-mu2) / pooled_std

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


    return nmd

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
    preview_kernels(X, X, cmap, False, True, False)

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

    X1 = np.array(range(1,n+1)).reshape(-1,1) # Generate a column vector of natural series, i.e., 1,2,3...
    X2 = X1.copy()

    if dim == 2:
        zeros = np.array([0] * n).reshape(-1,1)
        X1 = np.hstack((X1, zeros))
        X2 = np.hstack((zeros,X2))
    
    # X = np.vstack((X1,X2))
    y = [0]*round(n/2) + [1]*(n-round(n/2)) # we actually don't use y in this function
    preview_kernels(X1, X2, cmap, False, True, False)

def preview_kernels(X1, X2=None, cmap = None, optimize_hyper_params = True, \
    scale = True, embed_title = True):
    '''
    Try various kernel types to evaluate the pattern after kernel-ops.  
    Each kernel uses their own default/empirical paramters.    
    In binary classification, because the first and last half samples belong to two classes respectively. 
    We want to get a contrasive pattern, i.e., (LT、RB) 与 （LB、RT）have blocks of different colors. 

    Parameters
    ----------
    X1 : an m-by-n data matrix. Should be rescaled to non-negative ranges (required by chi2 family) and re-ordered by y. 
    X2 : the second m-by-n data matrix. If None, X2 = X1.
    cmap : color map scheme to use. set None to use default, or set another scheme, e.g., 'gray', 'viridis', 'jet', 'rainbow', etc.
        For small dataset, we recommend 'gray'. For complex dataset, we recommend 'viridis'.
    optimize_hyper_params : whether to optimize hyper parameters for each kernel. 
        For real-world dataset, use True. For toy dataset, usually use False.
    scale: whether do feature scaling
    embed_title : whether embed the title in the plots. If not, will generate the title in HTML.
    '''

    if X2 is None or X2 == []:
        X2 = X1.copy()

    if scale:
        X1 = MinMaxScaler().fit_transform(X1)
        X2 = MinMaxScaler().fit_transform(X2)
    
    for i, key in enumerate(kernel_names):
        best_hparam = None
        best_metric = -np.inf
        title = str(i+1) + '. ' + (kernel_fullnames[key] if key in kernel_fullnames else key) 

        if optimize_hyper_params and key in kernel_hparams:        
            for param in kernel_hparams[key]:                
                new_metric = nmd(X1, X2, lambda x,y : kernel_dict[key](x,y, param))
                if (new_metric > best_metric):
                    best_metric = new_metric
                    best_hparam = param
            title = (kernel_fullnames[key] if key in kernel_fullnames else key) \
                + ('(' +format(best_hparam,'.2g')+ ')') if best_hparam is not None else ''

        if not optimize_hyper_params or key not in kernel_hparams:
            plt.imshow(kernel_dict[key](X1,X2), cmap = cmap)
        else:
            plt.imshow(kernel_dict[key](X1,X2, best_hparam), cmap = cmap)

        plt.axis('off')
        if embed_title:
            plt.title(title +  '\n' + kernel_formulas[key])
        else:
            # print(title)
            display(HTML('<h3>' + title + '</h3>' + '<p>' + kernel_formulas[key].replace('<','&lt;').replace('>','&gt;') + '</p>'))
        plt.show()


def preview_kernels_on_dataset(X, y=None, cmap = None, optimize_hyper_params = True, \
    scale = True, embed_title = True):
    '''
    Parameters
    ----------
    X : an m-by-n data matrix. Should be rescaled to non-negative ranges (required by chi2 family) and re-ordered by y. 
    y : label. Only support 2 classes. If has more than 2 classes, split by ovr or ovo strategy first.
    '''

    if scale:
        X = MinMaxScaler().fit_transform(X)

    if y is None:
        X1 = X[:round(len(X)/2)]
        X2 = X[round(len(X)/2):]
    elif (len(set(y)) == 2):
        labels = list(set(y))
        X1 = X[y==labels[0]]
        X2 = X[y==labels[0]]
    else:
        print("Error: y must be binary labels. Exit.")
        return

    return preview_kernels(X1, X2, cmap, optimize_hyper_params, \
    scale = False, embed_title = embed_title)