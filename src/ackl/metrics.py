from cmath import inf, nan
import math
import numpy as np
import numpy.testing as npt 
import matplotlib.pyplot as plt
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
    Xns = np.array([0,1]).reshape(-1,1)
    preview_kernels(Xns, None, cmap, False)

def linear_response_pattern(n=20, cmap = 'gray'):
    '''
    Generates a response pattern of pairwise-point-distances.
    See how each kernel responds to / varies with linearly arranged points.
    
    Parameter
    ---------
    n : use n natural points as input, i.e., 0,1,2,3,4...
    cmap : color map scheme
    '''
    Xns = np.array(range(n)).reshape(-1,1) # Generate a natural series, i.e., 0,1,2,3...
    y = [0]*round(n/2) + [1]*(n-round(n/2)) # we actually don't use y in this function
    preview_kernels(Xns, None, cmap, False)

def preview_kernels(X, y=None, cmap = None, optimize_hyper_params = True, scale = True, show_title = True):
    '''
    Try various kernel types to evaluate the pattern after kernel-ops.  
    Each kernel uses their own default/empirical paramters.    
    In binary classification, because the first and last half samples belong to two classes respectively. 
    We want to get a contrasive pattern, i.e., (LT、RB) 与 （LB、RT）have blocks of different colors. 

    Parameters
    ----------
    X : an m-by-n data matrix. Should be rescaled to non-negative ranges (required by chi2 family) and re-ordered by y. 
    y : label. Only support 2 classes. If has more than 2 classes, split by ovr or ovo strategy first.
    cmap : color map scheme to use. set None to use default, or set another scheme, e.g., 'gray', 'viridis', 'jet', 'rainbow', etc.
        For small dataset, we recommend 'gray'. For complex dataset, we recommend 'viridis'.
    optimize_hyper_params : whether to optimize hyper parameters for each kernel. 
        For real-world dataset, use True. For toy dataset, usually use False.
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

    # TODO: replace with this 
    
    for key in kernel_names:
        best_hparam = None
        best_metric = -np.inf
        title = kernel_fullnames[key] + '\n' + kernel_formulas[key]
        if optimize_hyper_params and key in kernel_hparams:        
            for param in kernel_hparams[key]:
                new_metric = nmd(X1, X2, lambda x,y : kernel_dict[key](x,y, param))
                if (new_metric > best_metric):
                    best_metric = new_metric
                    best_hparam = param
            title = kernel_fullnames[key] + '(' +format(best_hparam,'.2g')+ '). \n' + kernel_formulas[key]

        plt.imshow(kernel_dict[key](X,X, best_hparam), cmap = cmap)
        plt.axis('off')
        if show_title:
            plt.title(title)
        else:
            print(title)
        plt.show()

    return
    
    

    ############# Linear kernel ############

    # linear_kernel返回Gram Matrix: n维欧式空间中任意k个向量之间两两的内积所组成的矩阵，称为这k个向量的格拉姆矩阵(Gram matrix)
    plt.imshow(linear_kernel(X,X), cmap=cmap) # return the Gram matrix, i.e. X @ Y.T.
    plt.axis('off')
    plt.title('linear kernel (Gram matrix).')
    plt.show()

    ############# Poly kernel ############

    best_hparam = None
    best_metric = -np.inf
    title = 'Poly kernel (default gamma = 1/n, degree = 3)\n'+r'$K(X, Y) = (\gamma <X, Y> + c)^d$'
    if optimize_hyper_params:        
        for gamma in [0.001 /X.shape[1] , 0.01 /X.shape[1] , 0.1 /X.shape[1], 1 /X.shape[1]]:
            new_metric = nmd(X1, X2, lambda x,y : polynomial_kernel(x,y, gamma = gamma))
            if (new_metric > best_metric):
                best_metric = new_metric
                best_hparam = gamma
        title='polynomial kernel (gamma = ' +format(best_hparam,'.2g')+ '). \n'+ \
            r'$K(X, Y) = (\gamma <X, Y> + c)^d$'

    plt.imshow(polynomial_kernel(X,X, gamma = best_hparam), cmap = cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()

    ############### Gausssian kernel #################

    best_hparam = None
    best_metric = -np.inf
    title = 'Gaussian kernel (default gamma = 1/n)\n'+r'$K(x, y) = exp(-\gamma ||x-y||^2)$'
    if optimize_hyper_params:        
        for gamma in kernel_hparams['gaussian']: # [0.1 /X.shape[1] , 0.33 /X.shape[1] , 1 /X.shape[1], 3.33 /X.shape[1], 10/X.shape[1]]:
            new_metric = nmd(X1, X2, lambda x,y : gaussian_kernel(x,y, gamma = gamma))
            if (new_metric > best_metric):
                best_metric = new_metric
                best_hparam = gamma
        title = 'Gaussian kernel (gamma = ' +format(best_hparam,'.2g')+ ')\n'+\
            r'$K(x, y) = exp(-\gamma ||x-y||^2)$'

    plt.imshow(gaussian_kernel(X,X, gamma = best_hparam), cmap = cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()

    ################## Exponential / Laplacian / Ornstein-Uhlenbeck kernel #####################

    best_hparam = None
    best_metric = -np.inf
    title = 'Exponential / Laplacian kernel (default gamma = 1/n)\n'+\
        r'$K(x, y) = exp(-\gamma ||x-y||_1)$'
    if optimize_hyper_params:  
        for gamma in kernel_hparams['laplace']: #[0.5 /X.shape[1] , 1 /X.shape[1], 2 /X.shape[1], 4/X.shape[1]]:
            new_metric = nmd(X1, X2, lambda x,y : laplacian_kernel(x,y, gamma = gamma))
            if (new_metric > best_metric):
                best_metric = new_metric
                best_hparam = gamma
        title = 'Exponential / Laplacian kernel (gamma = ' +\
            format(best_hparam,'.2g')+ '). \n'+r'$K(x, y) = exp(-\gamma ||x-y||_1)$'

    plt.imshow(laplacian_kernel(X,X, gamma = best_hparam), cmap = cmap) 
    plt.axis('off')
    plt.title(title)
    plt.show()
    
    ################### Sigmoid kernel ###############

    best_hparam = None
    best_metric = -np.inf
    title = 'Sigmoid kernel (default gamma = 1/n)\n'+\
        r'$K(X, Y) = tanh(\gamma <X, Y> + c)$'
    if optimize_hyper_params:  
        for gamma in [0.1 /X.shape[1] , 1 /X.shape[1], 10/X.shape[1], 100/X.shape[1]]:
            new_metric = nmd(X1, X2, lambda x,y : sigmoid_kernel(x,y, gamma = gamma))
            if (new_metric > best_metric):
                best_metric = new_metric
                best_hparam = gamma
        title = 'Sigmoid kernel (gamma = ' +format(best_hparam,'.2g')+ '). \n'+\
            r'$K(X, Y) = tanh(\gamma <X, Y> + c)$'
    plt.imshow(sigmoid_kernel(X,X, gamma = best_hparam), cmap = cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()
        
    ################### Chi2 kernel ##################

    best_hparam = None
    best_metric = -np.inf
    title = 'chi2 kernel (default gamma = 1.\n'+\
        r'$k(x, y) = exp(-\gamma \sum [(x - y)^2 / (x + y)])$'
    if optimize_hyper_params:  
        for gamma in kernel_hparams['chi2']:
            new_metric = nmd(X1, X2, lambda x,y : chi2_kernel(x,y, gamma = gamma))
            if (new_metric > best_metric):
                best_metric = new_metric
                best_hparam = gamma
        title = 'chi2 kernel (gamma = ' +format(best_hparam,'.2g')+ '). \n'+\
            r'$k(x, y) = exp(-\gamma \sum [(x - y)^2 / (x + y)])$'
    
    # chi2 requires non-negative input
    print(best_hparam)
    plt.imshow(chi2_kernel(X,X, gamma = best_hparam), cmap = cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()

    ######### Additive Chi2 Kernel ############

    # additive chi2 requires non-negative input
    plt.imshow(additive_chi2_kernel(X, X), cmap = cmap)
    plt.axis('off')
    plt.title('additive chi2 kernel. \n'+r'$k(x, y) = - \sum [(x - y)^2 / (x + y)]$')
    plt.show()

    ############# Cosine kernel ##################

    plt.imshow(cosine_similarity(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('cosine kernel. \n'+r'$K(X, Y) = <X, Y> / (||X||||Y||))$')
    plt.show()

    ############## Anova kernel ##############

    best_hparam = None
    best_metric = -np.inf
    title = 'ANOVA kernel. \n'+r'$K(x, y) = \sum_k exp( -sigma * (x_k - y_k)^2 )^d$'
    if optimize_hyper_params:  
        for gamma in kernel_hparams['anova']:
            new_metric = nmd(X1, X2, lambda x,y : anova_kernel(x,y, sigma = gamma))
            if (new_metric > best_metric):
                best_metric = new_metric
                best_hparam = gamma
        title = 'ANOVA kernel (sigma=.' + format(best_hparam,'.2g') + ')\n'+ \
            r'$K(x, y) = \sum_k exp( -sigma * (x_k - y_k)^2 )^d$'

    plt.imshow(anova_kernel(X,X,sigma=best_hparam), cmap = cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()

    ############## Rational Quadratic kernel ###################

    plt.imshow(rq_kernel_v2(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Rational Quadratic kernel V2.\n' + r'$K(x, y) = 1 - ||x-y||^2/(||x-y||^2+c)$')
    plt.show()   
    
    ############### Inverse Multi-Quadratic kernel ###############

    plt.imshow(imq_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Inverse Multi-Quadratic kernel.\n' + r'$K(x, y) = 1 / sqrt(||x-y||^2 + c^2)$')
    plt.show()

    ############### Cauchy kernel ##################

    best_hparam = None
    best_metric = -np.inf
    title = 'Cauchy kernel.\n' + r'$K(x, y) = 1 / (1 + ||x - y||^2 / s^2)$'
    if optimize_hyper_params:  
        for gamma in kernel_hparams['cauchy']:
            new_metric = nmd(X1, X2, lambda x,y : cauchy_kernel(x,y, sigma = gamma))
            if (new_metric > best_metric):
                best_metric = new_metric
                best_hparam = gamma
        title = 'Cauchy kernel (sigma=.' + format(best_hparam,'.2g') + ')\n'+ \
            r'$K(x, y) = 1 / (1 + ||x - y||^2 / s^2)$'

    plt.imshow(cauchy_kernel(X,X,sigma=best_hparam), cmap = cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()

    ############## T Student kernel #############

    plt.imshow(ts_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('T Student kernel.\n' + r'$K(x, y) = 1 / (1 + ||x - y||^d)$')
    plt.show()

    ############### Spline kernel ###############

    plt.imshow(spline_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Spline kernel.\n'+r'$K(x, y) = PROD_i 1 + x_iy_i + x_iy_i min(x_i,y_i) - (x_i+y_i)/2 * min(x_i,y_i)^2 + 1/3 * min(x_i, y_i)^3$')
    plt.show()

    ############## Sorensen kernel #############

    plt.imshow(sorensen_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Sorensen kernel.\n'+r'$K(x, y) = 2 <x, y> / (||x||^2 + ||y||^2)$')
    plt.show()

    ################ Min kernel ##############

    plt.imshow(min_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Min kernel.\n'+r'$K(x, y) = SUM_i min(x_i, y_i)$')
    plt.show()

    ############ MinMax kernel ###########

    plt.imshow(minmax_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('MinMax kernel.\n'+r'$K(x, y) = SUM_i min(x_i, y_i) / SUM_i max(x_i, y_i)$')
    plt.show()

    ############ Exponential min kernel ##########

    plt.imshow(expmin_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Exponential Min kernel.\n'+r'$K(x,y) = exp(-a min (|x-y|,|x+y|))^2$')
    plt.show()

    ############## Generalized Histogram Intersection kernel ##########

    plt.imshow(ghi_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Generalized Histogram Intersection kernel.\n' + r'$K(x, y) = \sum_i min(|x_i|^a, |y_i|^a)$')
    plt.show()

    ############## Fourier kernel #########

    plt.imshow(fourier_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Fourier kernel.\n'+r'$K(x, y) = PROD_i (1-q^2)/(2(1-2q cos(x_i-y_i)+q^2))$')
    plt.show()

    ############## wavelet kernel ###########

    plt.imshow(wavelet_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Wavelet kernel.\n'+r'K(x, y) = PROD_i h( (x_i-c)/a ) h( (y_i-c)/a )')
    plt.show()

    ############# Log kernel #############

    plt.imshow(log_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Log kernel.\n'+r'K(x, y) = -log(||x-y||^d + 1)')
    plt.show()

    ############### power kernel ############

    best_hparam = None
    best_metric = -np.inf
    title = 'Power kernel (default d = 2).'+r'$K(x, y) = -||x-y||^d$'
    if optimize_hyper_params:  
        for gamma in kernel_hparams['power']:
            new_metric = nmd(X1, X2, lambda x,y : power_kernel(x,y, d = gamma))
            if (new_metric > best_metric):
                best_metric = new_metric
                best_hparam = gamma
        title = 'Power kernel (d=.' + format(best_hparam,'.2g') + ')\n'+ \
            r'$K(x, y) = -||x-y||^d$'

    plt.imshow(power_kernel(X,X,d=best_hparam), cmap = cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()

    ############### Bessel kernel ############

    plt.imshow(bessel_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Bessel kernel.\n'+r'$JV_{v+1} ( -s* ||x-y|| )$')
    plt.show()

    ############ Matern kernel #############

    best_hparam = None
    best_metric = -np.inf
    title = 'Matern kernel (default v = 0.5).'
    if optimize_hyper_params:  
        for gamma in kernel_hparams['matern']:
            new_metric = nmd(X1, X2, lambda x,y : matern_kernel(x,y, v = gamma))
            if (new_metric > best_metric):
                best_metric = new_metric
                best_hparam = gamma
        title = 'Matern kernel (d=.' + format(best_hparam,'.2g') + ')'

    plt.imshow(matern_kernel(X,X,v=best_hparam), cmap = cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()

    ################## Exp-Sine-Sqaured kernel ################

    best_hparam = None
    best_metric = -np.inf
    title = 'Exp-Sine-Sqaured kernel (default p = 1).\n' + \
            r'$exp(-2* sin(\pi*||x-y||/p)/(s^2))$'

    if optimize_hyper_params:  
        for gamma in kernel_hparams['ess']:
            new_metric = nmd(X1, X2, lambda x,y : ess_kernel(x,y, p = gamma))
            if (new_metric > best_metric):
                best_metric = new_metric
                best_hparam = gamma
        title = 'Exp-Sine-Sqaured kernel (p=.' + format(best_hparam,'.2g') + ')\n' + \
            r'$exp(-2* sin(\pi*||x-y||/p)/(s^2))$'

    plt.imshow(ess_kernel(X,X, p = best_hparam), cmap = cmap)
    plt.axis('off')
    plt.title(title)
    plt.show()
    print('Check Implementation')

    ############## Feijer ##############    

    plt.imshow(feijer_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Feijer kernel.\n' + r'$( 1-cos(k*||x-y||) ) / ( 1-cos(||x-y||) / k$')
    plt.show()