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
    ess_kernel, expmin_kernel, bessel_kernel, feijer_kernel

def bw_ratio(X1, X2, f, display=False):
    '''
    Computes the ratio of the means of between-class and within-class matrices.
    In practice, this metric is not a perfect metric to optimize the kernel-specific params.

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

    r1 = ( RT.sum() + LB.sum()) / ( LT.sum() + RB.sum()) 
    r2 = ( np.log(RT).sum() + np.log(LB).sum()) / ( np.log(LT).sum() + np.log(RB).sum()) 

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

    return r1,r2

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
    preview_kernels(Xns, cmap)

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
    preview_kernels(Xns, cmap)

def preview_kernels(X, cmap = None):
    '''
    Try various kernel types to evaluate the pattern after kernel-ops.  
    In binary classification, because the first and last half samples belong to two classes respectively. 
    We want to get a contrasive pattern, i.e., (LT、RB) 与 （LB、RT）have blocks of different colors. 

    Parameters
    ----------
    X : an m-by-n data matrix. Should be rescaled to non-negative ranges (required by chi2 family) and re-ordered by y. 
    cmap : color map scheme to use. set None to use default, or set another scheme, e.g., 'gray', 'viridis', 'jet', 'rainbow', etc.
        For small dataset, we recommend 'gray'. For complex dataset, we recommend 'viridis'.
    '''

    X = MinMaxScaler().fit_transform(X)

    # linear_kernel返回Gram Matrix: n维欧式空间中任意k个向量之间两两的内积所组成的矩阵，称为这k个向量的格拉姆矩阵(Gram matrix)
    plt.imshow(linear_kernel(X,X), cmap=cmap) # return the Gram matrix, i.e. X @ Y.T.
    plt.axis('off')
    plt.title('linear kernel (Gram matrix). \nNo tunable params.')
    plt.show()


    for gamma in [0.1 /X.shape[1] , 0.33 /X.shape[1] , 
        1 /X.shape[1], 3.33 /X.shape[1], 10/X.shape[1]]:
    
        plt.imshow(rbf_kernel(X,X, gamma = gamma), cmap = cmap)
        plt.axis('off')
        plt.title('rbf kernel (gamma = ' +str(gamma)+ '). \nK(x, y) = exp(-gamma ||x-y||^2) \ndefault gamma = 1/n.')
        plt.show()


    plt.imshow(exponential_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Exponential kernel.') # laplacian = exponential
    plt.show()

    for gamma in [0.5 /X.shape[1] , 1 /X.shape[1], 2 /X.shape[1], 4/X.shape[1]]:

        plt.imshow(laplacian_kernel(X,X, gamma = gamma), cmap = cmap) 
        plt.axis('off')
        plt.title('laplacian kernel (gamma = ' +str(gamma)+ '). \nK(x, y) = exp(-gamma ||x-y||_1) \ndefault gamma = 1/n.')
        plt.show()
    

    for gamma in [0.1 /X.shape[1] , 1 /X.shape[1], 10/X.shape[1], 100/X.shape[1]]:
    
        plt.imshow(sigmoid_kernel(X,X, gamma = gamma), cmap = cmap)
        plt.axis('off')
        plt.title('sigmoid kernel (gamma = ' +str(gamma)+ '). \nK(X, Y) = tanh(gamma <X, Y> + coef0) \ndefault gamma = 1/n.')
        plt.show()
        

    for gamma in [0.001 /X.shape[1] , 0.01 /X.shape[1] , 0.1 /X.shape[1], 1 /X.shape[1]]:
 
        plt.imshow(polynomial_kernel(X,X, gamma = gamma), cmap = cmap)
        plt.axis('off')
        plt.title('polynomial kernel (gamma = ' +str(gamma)+ '). \nK(X, Y) = (gamma <X, Y> + coef0)^degree \ndefault gamma = 1/n, degree = 3')
        plt.show()


    for gamma in [0.001, 0.01 , 0.1 , 1]:
   
        # chi2 requires non-negative input
        plt.imshow(chi2_kernel(X,X, gamma = gamma), cmap = cmap)
        plt.axis('off')
        plt.title('chi2 kernel (gamma = ' +str(gamma)+ '). \nk(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)]) \ndefault gamma = 1')
        plt.show()


    # additive chi2 requires non-negative input
    plt.imshow(additive_chi2_kernel(X, X), cmap = cmap)
    plt.axis('off')
    plt.title('additive chi2 kernel. \nk(x, y) = -Sum [(x - y)^2 / (x + y)] \nNo tunable params.')
    plt.show()


    plt.imshow(cosine_similarity(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('cosine kernel. \nK(X, Y) = <X, Y> / (||X||*||Y||) \nNo tunable params.')
    plt.show()

    plt.imshow(anova_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('ANOVA kernel.')
    plt.show()

    plt.imshow(rq_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Rational Quadratic kernel.')
    plt.show()

    plt.imshow(rq_kernel_v2(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Rational Quadratic kernel V2.')
    plt.show()   
    
    plt.imshow(imq_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Inverse Multi-Quadratic kernel.')
    plt.show()

    plt.imshow(cauchy_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Cauchy kernel.')
    plt.show()

    plt.imshow(ts_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('T Student kernel.')
    plt.show()

    plt.imshow(spline_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Spline kernel.')
    plt.show()

    plt.imshow(sorensen_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Sorensen kernel.')
    plt.show()


    plt.imshow(min_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Min kernel.')
    plt.show()

    plt.imshow(minmax_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('MinMax kernel.')
    plt.show()

    plt.imshow(expmin_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Exponential Min kernel.')
    plt.show()

    plt.imshow(ghi_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Generalized Histogram Intersection kernel.')
    plt.show()

    plt.imshow(fourier_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Fourier kernel.')
    plt.show()

    plt.imshow(wavelet_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Wavelet kernel.')
    plt.show()

    plt.imshow(log_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Log kernel.')
    plt.show()

    plt.imshow(power_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Power kernel.')
    plt.show()

    plt.imshow(bessel_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Bessel kernel.')
    plt.show()

    L = round(len(X) / 2)

    plt.imshow(matern_kernel(X,X,v=0.5), cmap = cmap)
    plt.axis('off')
    plt.title('Matern kernel (1/2).')
    plt.show()
    print(bw_ratio(X[:L],X[L:], lambda x, y: matern_kernel(x, y, 0.5)))

    plt.imshow(matern_kernel(X,X,v=10), cmap = cmap)
    plt.axis('off')
    plt.title('Matern kernel (10).')
    plt.show()    
    print(bw_ratio(X[:L],X[L:], lambda x, y: matern_kernel(x, y, 10)))

    plt.imshow(matern_kernel(X,X,v=100), cmap = cmap)
    plt.axis('off')
    plt.title('Matern kernel (100).')
    plt.show()
    print(bw_ratio(X[:L],X[L:], lambda x, y: matern_kernel(x, y, 100)))


    plt.imshow(ess_kernel(X,X, p = 10), cmap = cmap)
    plt.axis('off')
    plt.title('Exp-Sine-Sqaured kernel (10).')
    plt.show()
    print(bw_ratio(X[:L],X[L:], lambda x, y: ess_kernel(x, y, 10)))

    plt.imshow(ess_kernel(X,X, p = 100), cmap = cmap)
    plt.axis('off')
    plt.title('Exp-Sine-Sqaured kernel (100).')
    plt.show()
    print(bw_ratio(X[:L],X[L:], lambda x, y: ess_kernel(x, y, 100)))

    plt.imshow(ess_kernel(X,X, p = 10000), cmap = cmap)
    plt.axis('off')
    plt.title('Exp-Sine-Sqaured kernel (10000).')
    plt.show()
    print(bw_ratio(X[:L],X[L:], lambda x, y: ess_kernel(x, y, 10000)))

    plt.imshow(feijer_kernel(X,X), cmap = cmap)
    plt.axis('off')
    plt.title('Feijer kernel.')
    plt.show()