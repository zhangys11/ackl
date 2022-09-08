import ack
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, \
    linear_kernel, sigmoid_kernel, chi2_kernel, polynomial_kernel, \
    additive_chi2_kernel, laplacian_kernel

def KernelTuning(X):
    '''
    Try various kernel types to evaluate the pattern after kernel-ops.  
    In binary classification, because the first and last half samples belong to two classes respectively. 
    We want to get a contrasive pattern, i.e., (LT、RB) 与 （LB、RT）have blocks of different colors. 

    Parameters
    ----------
    X : an m-by-n data matrix. Should be rescaled to non-negative ranges (required by chi2 family) and re-ordered by y. 
    '''

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