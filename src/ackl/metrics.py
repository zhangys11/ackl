import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, \
    linear_kernel, sigmoid_kernel, chi2_kernel, polynomial_kernel, \
    additive_chi2_kernel, laplacian_kernel
from sklearn.preprocessing import MinMaxScaler
from kernels import anova_kernel, rq_kernel, rq_kernel_v2, exponential_kernel, imq_kernel, \
    cauchy_kernel, ts_kernel, spline_kernel, sorensen_kernel, min_kernel, minmax_kernel, \
    ghi_kernel, fourier_kernel, wavelet_kernel, log_kernel, power_kernel

def preview_kernels(X):
    '''
    Try various kernel types to evaluate the pattern after kernel-ops.  
    In binary classification, because the first and last half samples belong to two classes respectively. 
    We want to get a contrasive pattern, i.e., (LT、RB) 与 （LB、RT）have blocks of different colors. 

    Parameters
    ----------
    X : an m-by-n data matrix. Should be rescaled to non-negative ranges (required by chi2 family) and re-ordered by y. 
    '''

    X = MinMaxScaler().fit_transform(X)

    # linear_kernel返回Gram Matrix: n维欧式空间中任意k个向量之间两两的内积所组成的矩阵，称为这k个向量的格拉姆矩阵(Gram matrix)
    plt.imshow(linear_kernel(X,X)) # return the Gram matrix, i.e. X @ Y.T.
    plt.axis('off')
    plt.title('linear kernel (Gram matrix). \nNo tunable params.')
    plt.show()


    for gamma in [0.1 /X.shape[1] , 0.33 /X.shape[1] , 
        1 /X.shape[1], 3.33 /X.shape[1], 10/X.shape[1]]:
    
        plt.imshow(rbf_kernel(X,X, gamma = gamma))
        plt.axis('off')
        plt.title('rbf kernel (gamma = ' +str(gamma)+ '). \nK(x, y) = exp(-gamma ||x-y||^2) \ndefault gamma = 1/n.')
        plt.show()


    plt.imshow(exponential_kernel(X,X))
    plt.axis('off')
    plt.title('Exponential kernel.') # laplacian = exponential
    plt.show()

    for gamma in [0.5 /X.shape[1] , 1 /X.shape[1], 2 /X.shape[1], 4/X.shape[1]]:

        plt.imshow(laplacian_kernel(X,X, gamma = gamma)) 
        plt.axis('off')
        plt.title('laplacian kernel (gamma = ' +str(gamma)+ '). \nK(x, y) = exp(-gamma ||x-y||_1) \ndefault gamma = 1/n.')
        plt.show()
    

    for gamma in [0.1 /X.shape[1] , 1 /X.shape[1], 10/X.shape[1], 100/X.shape[1]]:
    
        plt.imshow(sigmoid_kernel(X,X, gamma = gamma))
        plt.axis('off')
        plt.title('sigmoid kernel (gamma = ' +str(gamma)+ '). \nK(X, Y) = tanh(gamma <X, Y> + coef0) \ndefault gamma = 1/n.')
        plt.show()
        

    for gamma in [0.001 /X.shape[1] , 0.01 /X.shape[1] , 0.1 /X.shape[1], 1 /X.shape[1]]:
 
        plt.imshow(polynomial_kernel(X,X, gamma = gamma))
        plt.axis('off')
        plt.title('polynomial kernel (gamma = ' +str(gamma)+ '). \nK(X, Y) = (gamma <X, Y> + coef0)^degree \ndefault gamma = 1/n, degree = 3')
        plt.show()


    for gamma in [0.001, 0.01 , 0.1 , 1]:
   
        # chi2 requires non-negative input
        plt.imshow(chi2_kernel(X,X, gamma = gamma))
        plt.axis('off')
        plt.title('chi2 kernel (gamma = ' +str(gamma)+ '). \nk(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)]) \ndefault gamma = 1')
        plt.show()


    # additive chi2 requires non-negative input
    plt.imshow(additive_chi2_kernel(X, X))
    plt.axis('off')
    plt.title('additive chi2 kernel. \nk(x, y) = -Sum [(x - y)^2 / (x + y)] \nNo tunable params.')
    plt.show()


    plt.imshow(cosine_similarity(X,X))
    plt.axis('off')
    plt.title('cosine kernel. \nK(X, Y) = <X, Y> / (||X||*||Y||) \nNo tunable params.')
    plt.show()

    plt.imshow(anova_kernel(X,X))
    plt.axis('off')
    plt.title('ANOVA kernel.')
    plt.show()

    plt.imshow(rq_kernel(X,X))
    plt.axis('off')
    plt.title('Rational Quadratic kernel.')
    plt.show()

    plt.imshow(rq_kernel_v2(X,X))
    plt.axis('off')
    plt.title('Rational Quadratic kernel V2.')
    plt.show()   
    
    plt.imshow(imq_kernel(X,X))
    plt.axis('off')
    plt.title('Inverse Multi-Quadratic kernel.')
    plt.show()

    plt.imshow(cauchy_kernel(X,X))
    plt.axis('off')
    plt.title('Cauchy kernel.')
    plt.show()

    plt.imshow(ts_kernel(X,X))
    plt.axis('off')
    plt.title('T Student kernel.')
    plt.show()

    plt.imshow(spline_kernel(X,X))
    plt.axis('off')
    plt.title('Spline kernel.')
    plt.show()

    plt.imshow(sorensen_kernel(X,X))
    plt.axis('off')
    plt.title('Sorensen kernel.')
    plt.show()


    plt.imshow(min_kernel(X,X))
    plt.axis('off')
    plt.title('Min kernel.')
    plt.show()

    plt.imshow(minmax_kernel(X,X))
    plt.axis('off')
    plt.title('MinMax kernel.')
    plt.show()

    plt.imshow(ghi_kernel(X,X))
    plt.axis('off')
    plt.title('Generalized Histogram Intersection kernel.')
    plt.show()

    plt.imshow(fourier_kernel(X,X))
    plt.axis('off')
    plt.title('Fourier kernel.')
    plt.show()

    plt.imshow(wavelet_kernel(X,X))
    plt.axis('off')
    plt.title('Wavelet kernel.')
    plt.show()

    plt.imshow(log_kernel(X,X))
    plt.axis('off')
    plt.title('Log kernel.')
    plt.show()

    plt.imshow(power_kernel(X,X))
    plt.axis('off')
    plt.title('Power kernel.')
    plt.show()