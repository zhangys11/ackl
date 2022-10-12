'''
This file contains definitions for all kernels.
'''

from cmath import pi
import math
import numpy as np
from abc import abstractmethod, ABCMeta
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, \
    linear_kernel, sigmoid_kernel, chi2_kernel, polynomial_kernel, \
    additive_chi2_kernel, laplacian_kernel
from sklearn.gaussian_process.kernels import RationalQuadratic
from scipy import special

def gaussian_kernel(x,y,gamma=None):
    '''
    Gaussian kernel. An alias of sklearn.metrics.pairwise.rbf_kernel
    '''
    return rbf_kernel(x,y,gamma)

def exponential_kernel(x,y,gamma = None):
    '''
    We use exponental and laplacian kernels interchangably.
    '''
    return laplacian_kernel(x,y,gamma=gamma)

def anova_kernel(x,y,sigma=1,d=2):
    if sigma is None:
        sigma = 1
    if d is None:
        d = 2
    return ANOVA(sigma=sigma, d=d)(x,y)

def rq_kernel(x,y,l=1,a=1):
    '''
    The Rational Quadratic kernel. Using sklearn.gaussian_process.kernels.RationalQuadratic.
    k(x_i, x_j) = \\left(1 + \\frac{d(x_i, x_j)^2 }{ 2\\alpha  l^2}\\right)^{-\\alpha}

    Paramters
    ---------
    l - length scale
    a - scale mixture
    '''
    return RationalQuadratic(length_scale=l, alpha=a)(x,y)

def rq_kernel_v2(x,y,c=1):
    '''
    K(x, y) = 1 - ||x-y||^2/(||x-y||^2+c)    
    '''
    return RationalQuadratic2(c=c)(x,y)

def imq_kernel(x,y,c=1):
    return InverseMultiquadratic(c=c)(x,y)

def cauchy_kernel(x,y,sigma=None): # = img_kernel squared
    return Cauchy(sigma=sigma)(x,y)

def ts_kernel(x,y,d=3): # d = 2 becomes Cauchy
    return TStudent(degree = d)(x,y)

def spline_kernel(x,y):
    return Spline()(x,y)

def sorensen_kernel(x,y):
    return Sorensen()(x,y)

def min_kernel(x,y):
    return Min()(x,y)

def minmax_kernel(x,y):
    return MinMax()(x,y)

def expmin_kernel(x,y,a=1):
    '''
    exponential min kernel.
    K(x,y) = exp(-a min (|x-y|,|x+y|))^2
    '''
    M = np.zeros((len(x),len(y)))
    for idx1, x1 in enumerate(x):
        for idx2, x2 in enumerate(y):
            # print(np.linalg.norm(x1-x2) , np.linalg.norm(x1+x2))
            M[idx1,idx2] = math.exp(-a * (min(np.linalg.norm(x1-x2) , np.linalg.norm(x1+x2)) ** 2) )
            
    return M

def ghi_kernel(x,y,alpha=1):
    '''
    Generalized Histogram Intersection kernel
    '''
    return GeneralizedHistogramIntersection(alpha=alpha)(x,y)

def fourier_kernel(x,y,q=0.1):
    return Fourier(q=q)(x,y)

def wavelet_kernel(x,y):
    '''
    The wavelet is a family / series. This is just a commonly used specific implementation.
    '''
    return Wavelet()(x,y)

def log_kernel(x,y,d=2):
    '''
    On real data, we found the log kernel is insensative to param d. In most cases, use d's default value.
    '''
    return Log(d = d)(x,y)

def power_kernel(x,y,d=2):
    if d is None:
        d = 2
    return Power(d=d)(x,y)

def bessel_kernel(x,y,v=0, s=1):
    return special.jv(v+1, -s* euclidean_dist_matrix(x,y)) # / euclidean_dist_matrix(x,y)

def matern_kernel(x,y,v=0.5,s=1):
    '''
    The matern kernel. 
    Implemented by Zhang according to the math definition.

    Parameter
    ---------
    v : controls smoothness. when v = 1/2, it becomes into the laplacian/exp kernel. 
    s : controls scaling
    '''
    if v is None:
        v = 0.5
    z = math.sqrt(2*v) / s * euclidean_dist_matrix(x,y)
    return 1/(math.gamma(v) * 2**(v-1) ) * (z**v) * mod_bessel(z)

def mod_bessel(x):
    return np.sqrt (math.pi / (2*x)) * np.exp(-x)  

def ess_kernel(x,y,p=1,s=1):
    '''
    Parameter
    ---------
    p : periodical parameter
    s : scale parameter
    '''
    if p is None:
        p = 1
    if s is None:
        s = 1

    M = np.zeros((len(x),len(y)))
    for idx1, x1 in enumerate(x):
        for idx2, x2 in enumerate(y):
            M[idx1,idx2] = math.exp(-2*math.sin(pi*np.linalg.norm(x1-x2)/p)/(s**2))
            
    return M

def feijer_kernel(x,y,k=10):
    '''
    Parameter
    ---------
    k - order of feijer series. Usually we don't use k = 1 as it always equals 1.
    '''
    if k is None:
        k = 10

    M = np.zeros((len(x),len(y)))
    for idx1, x1 in enumerate(x):
        for idx2, x2 in enumerate(y):
            if math.cos( np.linalg.norm(x1-x2) ) == 1:
                M[idx1,idx2] = k # handle divided-by-0 error
            else:
                M[idx1,idx2] = ( 1-math.cos(k * np.linalg.norm(x1-x2)) ) / ( 1-math.cos( np.linalg.norm(x1-x2) ) ) / k
    return M
    # return ( 1-np.cos(k*euclidean_dist_matrix(x,y)) ) / ( 1-np.cos(euclidean_dist_matrix(x,y)) ) / k


kernel_fullnames = {"linear":"linear",
"poly":"polynomial",
"gaussian":"gaussian",
"rbf":"radial basis function",
"laplace":"laplacian",
"cosine":"cosine similarity",
"chi2":"chi-squared",
"achi2":"additive chi-squared",
"exp":"exponential",
"matern":"matern",
"ess":"exponential-Sine-Squared",
"rq":"rational quadratic",
"imp":"inverse Multi Quadric",
"cauchy":"cauchy",
"ts":"T-Student",
"anova":"anova",
"min":"min",
"minman":"minmax",
"expmin":"exponential-Min",
"ghi":"generalized histogram intersection",
"spline":"spline",
"sorensen":"sorense",
"fourier":"fourier",
"wavelet":"wavelet",
"log":"log",
"power":"power",
"bessel":"bessel",
"fejer":"fejer"
}

# kernel names and functions
# 按照论文次序
kernel_dict = {"linear": linear_kernel,
                    "poly": polynomial_kernel,
                    "sigmoid": sigmoid_kernel,                    
                    "rbf": rbf_kernel,
                    "gaussian":gaussian_kernel,
                    "exp": laplacian_kernel, #alias of laplacian
                    "laplace": laplacian_kernel,
                    "matern":matern_kernel,
                    "chi2": chi2_kernel,
                    "achi2": additive_chi2_kernel,
                    "cosine": cosine_similarity,
                    "ess": ess_kernel,
                    "rq": rq_kernel,  # rational quadratic
                    "imq": imq_kernel,  # inverse multi quadratic
                    "cauchy":cauchy_kernel,
                    "ts":ts_kernel,
                    "anova": anova_kernel,
                    "min":min_kernel,
                    "minmax":minmax_kernel,
                    "expmin":expmin_kernel,
                    "ghi":ghi_kernel,
                    "spine":spline_kernel,
                    "sorensen":sorensen_kernel,
                    "fourier":fourier_kernel,
                    "wavelet":wavelet_kernel,
                    "log":log_kernel,
                    "power":power_kernel,
                    "bessel":bessel_kernel,
                    "fejer":feijer_kernel
}
}

kernel_formulas = {"linear": r"$k(x,y)=<x,y>$", 
"poly":r"$k(x,y)=$K(X, Y) = (\gamma <X, Y> + c)^d$",
"gaussian": r"$K(x, y) = exp(-\gamma ||x-y||^2)$",
"sigmoid":r"$K(X, Y) = tanh(\gamma <X, Y> + c)$",
"exp":r"$K(x, y)=$e^(-||x - y||/(2*s^2))$",
"laplace":r"$K(x, y) = e^(-||x - y||/s)$",
"rq":r"$K(x, y) = 1 - ||x-y||^2/(||x-y||^2+c)$",
"imq":r"$K(x, y) = 1 / sqrt(||x-y||^2 + c^2)$",
"cauchy":r"$K(x, y) = 1 / (1 + ||x - y||^2 / s ^ 2)$",
"ts":r"$K(x, y) = 1 / (1 + ||x - y||^d)$",
"anova":r"$K(x, y) = SUM_k exp( -sigma * (x_k - y_k)^2 )^d$",
"wavelet":r"$K(x, y) = PROD_i h( (x_i-c)/a ) h( (y_i-c)/a )$",
"fourier":r"$K(x, y) = PROD_i (1-q^2)/(2(1-2q cos(x_i-y_i)+q^2))$",
"Tanimoto":r"$K(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)$",
"sorensen":r"$K(x, y) = 2 <x, y> / (||x||^2 + ||y||^2)$",
"achi2":r"$K(x, y) = SUM_i 2 x_i y_i / (x_i + y_i)$",
"chi2":r"$K(x, y) = exp( -gamma * SUM_i (x_i - y_i)^2 / (x_i + y_i) )$",
"min":r"$K(x, y) = SUM_i min(x_i, y_i)$",
"ghi":r"$K(x, y) = SUM_i min(|x_i|^alpha, |y_i|^alpha)$",
"minmax":r"$K(x, y) = SUM_i min(x_i, y_i) / SUM_i max(x_i, y_i)$",
"expmin":r"$K(x,y) = exp(-a min (|x-y|,|x+y|))^2$",
"spine":r"$K(x, y) = PROD_i 1 + x_iy_i + x_iy_i min(x_i,y_i)- (x_i+y_i)/2 * min(x_i,y_i)^2+ 1/3 * min(x_i, y_i)^3$",
"log":r"$K(x, y) = -log(||x-y||^d + 1)$",
"power":r"$K(x, y) = -||x-y||^d$",
"bessel":r"$JV_{v+1} ( -s* ||x-y|| )$",
"ess":r"$k(x,y)=exp(-2* sin(\pi*||x-y||/p)/(s^2))$",
"fejer":r"$( 1-cos(k*||x-y||) ) / ( 1-cos(||x-y||) / k$"

}


# Stores the hyper parameter search range for each kernel. 
# Some hparams are dynamic (based on data dim).
# Not all kernels have tunable hyper-parameters.
kernel_hparams = {
    "gaussian": [0.1, 0.33, 1, 3.33],
    "laplace": [0.01,0.05,0.1],
    "chi2":  [0.001, 0.01 , 0.1 , 1.0], 
    "anova": [0.0001, 0.001, 0.01, 0.1, 1],
    "cauchy": [100,1000,10000],
    "power": [0.1, 0.5, 1],
    "matern": [0.5, 1, 10],
    "ess": [1,2,3,4],
    "feijer":[2,3,4,5]
 }

kernel_names = list(kernel_dict.keys())


"""
The following codes are based on https://github.com/gmum/pykernels. (MIT Licence)
original authors: Wojciech Marian Czarnecki and Katarzyna Janocha
"""

class Kernel(object):
    """
    Base, abstract kernel class
    """
    __metaclass__ = ABCMeta

    def __call__(self, data_1, data_2):
        return self._compute(data_1, data_2)

    @abstractmethod
    def _compute(self, data_1, data_2):
        """
        Main method which given two lists data_1 and data_2, with
        N and M elements respectively should return a kernel matrix
        of size N x M where K_{ij} = K(data_1_i, data_2_j)
        """
        raise NotImplementedError('This is an abstract class')

    def gram(self, data):
        """
        Returns a Gramian, kernel matrix of matrix and itself
        """
        return self._compute(data, data)

    @abstractmethod
    def dim(self):
        """
        Returns dimension of the feature space
        """
        raise NotImplementedError('This is an abstract class')

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def __add__(self, kernel):
        return KernelSum(self, kernel)

    def __mul__(self, value):
        if isinstance(value, Kernel):
            return KernelProduct(self, value)
        else:
            if isinstance(self, ScaledKernel):
                return ScaledKernel(self._kernel, self._scale * value)
            else:
                return ScaledKernel(self, value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __div__(self, scale):
        return ScaledKernel(self, 1./scale)

    def __pow__(self, value):
        return KernelPower(self, value)

class KernelSum(Kernel):
    """
    Represents sum of a pair of kernels
    """

    def __init__(self, kernel_1, kernel_2):
        self._kernel_1 = kernel_1
        self._kernel_2 = kernel_2

    def _compute(self, data_1, data_2):
        return self._kernel_1._compute(data_1, data_2) + \
               self._kernel_2._compute(data_1, data_2)

    def dim(self):
        # It is too complex to analyze combined dimensionality, so we give a lower bound
        return max(self._kernel_1.dim(), self._kernel_2.dim())

    def __str__(self):
        return '(' + str(self._kernel_1) + ' + ' + str(self._kernel_2) + ')'


class KernelProduct(Kernel):
    """
    Represents product of a pair of kernels
    """

    def __init__(self, kernel_1, kernel_2):
        self._kernel_1 = kernel_1
        self._kernel_2 = kernel_2

    def _compute(self, data_1, data_2):
        return self._kernel_1._compute(data_1, data_2) * \
               self._kernel_2._compute(data_1, data_2)

    def dim(self):
        # It is too complex to analyze combined dimensionality, so we give a lower bound
        return max(self._kernel_1.dim(), self._kernel_2.dim())

    def __str__(self):
        return '(' + str(self._kernel_1) + ' * ' + str(self._kernel_2) + ')'


class KernelPower(Kernel):
    """
    Represents natural power of a kernel
    """

    def __init__(self, kernel, d):
        self._kernel = kernel
        self._d = d
        if not isinstance(d, int) or d<0:
            raise Exception('Kernel power is only defined for non-negative integer degrees')

    def _compute(self, data_1, data_2):
        return self._kernel._compute(data_1, data_2) ** self._d

    def dim(self):
        # It is too complex to analyze combined dimensionality, so we give a lower bound
        return self._kernel.dim()

    def __str__(self):
        return str(self._kernel) + '^' + str(self._d)


class ScaledKernel(Kernel):
    """
    Represents kernel scaled by a float
    """

    def __init__(self, kernel, scale):
        self._kernel = kernel
        self._scale = scale
        if scale < 0:
            raise Exception('Negation of the kernel is not a kernel!')

    def _compute(self, data_1, data_2):
        return self._scale * self._kernel._compute(data_1, data_2)

    def dim(self):
        return self._kernel.dim()

    def __str__(self):
        if self._scale == 1.0:
            return str(self._kernel)      
        else:
            return str(self._scale) + ' ' + str(self._kernel)


class GraphKernel(Kernel):
    """
    Base, abstract GraphKernel kernel class
    """
    pass

import warnings

def euclidean_dist_matrix(data_1, data_2):
    """
    Returns matrix of pairwise, squared Euclidean distances
    """
    norms_1 = (data_1 ** 2).sum(axis=1)
    norms_2 = (data_2 ** 2).sum(axis=1)
    return np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))

"""
class Exponential(Kernel):
    
    #Exponential kernel, 

    #    K(x, y) = e^(-||x - y||/(2*s^2))

    #where:
    #    s = sigma
    
    def __init__(self, sigma=None):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = 2 * sigma**2

    def _compute(self, data_1, data_2):
        if self._sigma is None:
            # modification of libSVM heuristics
            self._sigma = float(data_1.shape[1])

        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return np.exp(-np.sqrt(dists_sq) / self._sigma)

    def dim(self):
        return np.inf


# So Laplacian is the same as Exponential
class Laplacian(Exponential):
    
    #Laplacian kernel, 

    #    K(x, y) = e^(-||x - y||/s)

    #where:
    #    s = sigma
    
    def __init__(self, sigma=None):
        self._sigma = sigma
"""

class RationalQuadratic2(Kernel):
    """
    Rational quadratic kernel, V2. This implementation is differnt from sklearn. By default, we use sklearn.

        K(x, y) = 1 - ||x-y||^2/(||x-y||^2+c)

    where:
        c > 0
    """

    def __init__(self, c=1):
        self._c = c

    def _compute(self, data_1, data_2):
        
        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return 1. - (dists_sq / (dists_sq + self._c))

    def dim(self):
        return None #unknown?


class InverseMultiquadratic(Kernel):
    """
    Inverse multiquadratic kernel, 

        K(x, y) = 1 / sqrt(||x-y||^2 + c^2)

    where:
        c > 0

    as defined in:
    "Interpolation of scattered data: Distance matrices and conditionally positive definite functions"
    Charles Micchelli
    Constructive Approximation
    """

    def __init__(self, c=1):
        self._c = c ** 2

    def _compute(self, data_1, data_2):
        
        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return 1. / np.sqrt(dists_sq + self._c)

    def dim(self):
        return np.inf


class Cauchy(Kernel):
    """
    Cauchy kernel, 

        K(x, y) = 1 / (1 + ||x - y||^2 / s ^ 2)

    where:
        s = sigma

    as defined in:
    "A least square kernel machine with box constraints"
    Jayanta Basak
    International Conference on Pattern Recognition 2008
    """

    def __init__(self, sigma=None):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = sigma**2

    def _compute(self, data_1, data_2):
        if self._sigma is None:
            # modification of libSVM heuristics
            self._sigma = float(data_1.shape[1])

        dists_sq = euclidean_dist_matrix(data_1, data_2)

        return 1 / (1 + dists_sq / self._sigma)

    def dim(self):
        return np.inf



class TStudent(Kernel):
    """
    T-Student kernel, 

        K(x, y) = 1 / (1 + ||x - y||^d)

    where:
        d = degree

    as defined in:
    "Alternative Kernels for Image Recognition"
    Sabri Boughorbel, Jean-Philippe Tarel, Nozha Boujemaa
    INRIA - INRIA Activity Reports - RalyX
    http://ralyx.inria.fr/2004/Raweb/imedia/uid84.html
    """

    def __init__(self, degree=2):
        self._d = degree

    def _compute(self, data_1, data_2):

        dists = np.sqrt(euclidean_dist_matrix(data_1, data_2))
        return 1 / (1 + dists ** self._d)

    def dim(self):
        return None


class ANOVA(Kernel):
    """
    ANOVA kernel, 
        K(x, y) = SUM_k exp( -sigma * (x_k - y_k)^2 )^d

    as defined in

    "Kernel methods in machine learning"
    Thomas Hofmann, Bernhard Scholkopf and Alexander J. Smola
    The Annals of Statistics
    http://www.kernel-machines.org/publications/pdfs/0701907.pdf
    """

    def __init__(self, sigma=1., d=2):
        self._sigma = sigma
        self._d = d

    def _compute(self, data_1, data_2):

        kernel = np.zeros((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            kernel += np.exp( -self._sigma * (column_1 - column_2.T)**2 ) ** self._d

        return kernel

    def dim(self):
        return None


def default_wavelet(x):
    return np.cos(1.75*x)*np.exp(-x**2/2)

class Wavelet(Kernel):
    """
    Wavelet kernel,

        K(x, y) = PROD_i h( (x_i-c)/a ) h( (y_i-c)/a )

    or for c = None

        K(x, y) = PROD_i h( (x_i - y_i)/a )

    as defined in
    "Wavelet Support Vector Machine"
    Li Zhang, Weida Zhou, Licheng Jiao
    IEEE Transactions on System, Man, and Cybernetics
    """

    def __init__(self, h=default_wavelet, c=None, a=1):
        self._c = c
        self._a = a
        self._h = h

    def _compute(self, data_1, data_2):

        kernel = np.ones((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            if self._c is None:
                kernel *= self._h( (column_1 - column_2.T) / self._a )
            else:
                kernel *= self._h( (column_1 - self._c) / self._a ) * self._h( (column_2.T - self._c) / self._a )

        return kernel

    def dim(self):
        return None


class Fourier(Kernel):
    """
    Fourier kernel,

        K(x, y) = PROD_i (1-q^2)/(2(1-2q cos(x_i-y_i)+q^2))
    """

    def __init__(self, q=0.1):
        self._q = q

    def _compute(self, data_1, data_2):

        kernel = np.ones((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            kernel *= (1-self._q ** 2) / \
                      (2.*(1. - 2.*self._q *np.cos(column_1 - column_2.T) + self._q ** 2))

        return kernel

    def dim(self):
        return None

class Tanimoto(Kernel):
    """
    Tanimoto kernel
        K(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)

    as defined in:

    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.483&rep=rep1&type=pdf
    """
    def _compute(self, data_1, data_2):

        norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
        norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
        prod = data_1.dot(data_2.T)
        return prod / (norm_1 + norm_2.T - prod)

    def dim(self):
        return None


class Sorensen(Kernel):
    """
    Sorensen kernel
        K(x, y) = 2 <x, y> / (||x||^2 + ||y||^2)

    as defined in:

    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.483&rep=rep1&type=pdf
    """
    def _compute(self, data_1, data_2):

        norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
        norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
        prod = data_1.dot(data_2.T)
        return 2 * prod / (norm_1 + norm_2.T)

    def dim(self):
        return None

from abc import ABCMeta

class PositiveKernel(Kernel):
    """
    Defines kernels which can be only used with positive values
    """
    __metaclass__ = ABCMeta

class AdditiveChi2(PositiveKernel):
    """
    Additive Chi^2 kernel, 
        K(x, y) = SUM_i 2 x_i y_i / (x_i + y_i)

    as defined in

    "Efficient Additive Kernels via Explicit Feature Maps"
    Andrea Vedaldi, Andrew Zisserman
    IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
    http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf
    """

    def _compute(self, data_1, data_2):

        if np.any(data_1 < 0) or np.any(data_2 < 0):
            warnings.warn('Additive Chi^2 kernel requires data to be strictly positive!')

        kernel = np.zeros((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            kernel += 2 * (column_1 * column_2.T) / (column_1 + column_2.T)

        return kernel

    def dim(self):
        return None

class Chi2(PositiveKernel):
    """
    Chi^2 kernel, 
        K(x, y) = exp( -gamma * SUM_i (x_i - y_i)^2 / (x_i + y_i) )

    as defined in:

    "Local features and kernels for classification 
     of texture and object categories: A comprehensive study"
    Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C. 
    International Journal of Computer Vision 2007 
    http://eprints.pascal-network.org/archive/00002309/01/Zhang06-IJCV.pdf
    """

    def __init__(self, gamma=1.):
        self._gamma = gamma

    def _compute(self, data_1, data_2):

        if np.any(data_1 < 0) or np.any(data_2 < 0):
            warnings.warn('Chi^2 kernel requires data to be strictly positive!')

        kernel = np.zeros((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            kernel += (column_1 - column_2.T)**2 / (column_1 + column_2.T)

        return np.exp(-self._gamma * kernel)

    def dim(self):
        return None

class Min(PositiveKernel):
    """
    Min kernel (also known as Histogram intersection kernel)
        K(x, y) = SUM_i min(x_i, y_i)

    """

    def _compute(self, data_1, data_2):

        if np.any(data_1 < 0) or np.any(data_2 < 0):
            warnings.warn('Min kernel requires data to be strictly positive!')

        kernel = np.zeros((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            kernel += np.minimum(column_1, column_2.T)

        return kernel

    def dim(self):
        return None


class GeneralizedHistogramIntersection(Kernel):
    """
    Generalized histogram intersection kernel
        K(x, y) = SUM_i min(|x_i|^alpha, |y_i|^alpha)

    as defined in
    "Generalized histogram intersection kernel for image recognition"
    Sabri Boughorbel, Jean-Philippe Tarel, Nozha Boujemaa
    International Conference on Image Processing (ICIP-2005)
    http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icip05.pdf
    """

    def __init__(self, alpha=1.):
        self._alpha = alpha

    def _compute(self, data_1, data_2):

        return Min()._compute(np.abs(data_1)**self._alpha,
                              np.abs(data_2)**self._alpha)

    def dim(self):
        return None

class MinMax(PositiveKernel):
    """
    MinMax kernel
        K(x, y) = SUM_i min(x_i, y_i) / SUM_i max(x_i, y_i)

    bounded by [0,1] as defined in:

    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.483&rep=rep1&type=pdf
    """

    def _compute(self, data_1, data_2):

        if np.any(data_1 < 0) or np.any(data_2 < 0):
            warnings.warn('MinMax kernel requires data to be strictly positive!')

        minkernel = np.zeros((data_1.shape[0], data_2.shape[0]))
        maxkernel = np.zeros((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            minkernel += np.minimum(column_1, column_2.T) 
            maxkernel += np.maximum(column_1, column_2.T)

        return minkernel/maxkernel

    def dim(self):
        return None


class Spline(PositiveKernel):
    """
    Spline kernel, 
        K(x, y) = PROD_i 1 + x_iy_i + x_iy_i min(x_i,y_i)
                           - (x_i+y_i)/2 * min(x_i,y_i)^2
                           + 1/3 * min(x_i, y_i)^3

    as defined in

    "Support Vector Machines for Classification and Regression"
    Steve Gunn
    ISIS Technical Report
    http://www.svms.org/tutorials/Gunn1998.pdf
    """

    def _compute(self, data_1, data_2):

        if np.any(data_1 < 0) or np.any(data_2 < 0):
            warnings.warn('Spline kernel requires data to be strictly positive!')

        kernel = np.ones((data_1.shape[0], data_2.shape[0]))

        for d in range(data_1.shape[1]):
            column_1 = data_1[:, d].reshape(-1, 1)
            column_2 = data_2[:, d].reshape(-1, 1)
            c_prod = column_1 * column_2.T
            c_sum = column_1 + column_2.T
            c_min = np.minimum(column_1, column_2.T)
            kernel *= 1. + c_prod + c_prod * c_min \
                         - c_sum/2. * c_min ** 2. \
                         + 1./3. * c_min ** 3.
        return kernel

    def dim(self):
        return None



class ConditionalyPositiveDefiniteKernel(Kernel):
    """
    Defines kernels which are only CPD
    """
    __metaclass__ = ABCMeta

class Log(ConditionalyPositiveDefiniteKernel):
    """
    Log kernel
        K(x, y) = -log(||x-y||^d + 1)

    """

    def __init__(self, d=2.):
        self._d = d

    def _compute(self, data_1, data_2):
        return -np.log(euclidean_dist_matrix(data_1, data_2) ** self._d / 2. + 1)

    def dim(self):
        return None


class Power(ConditionalyPositiveDefiniteKernel):
    """
    Power kernel
        K(x, y) = -||x-y||^d

    as defined in:
    "Scale-Invariance of Support Vector Machines based on the Triangular Kernel"
    Hichem Sahbi, Francois Fleuret
    Research report
    https://hal.inria.fr/inria-00071984
    """

    def __init__(self, d=2.):
        self._d = d

    def _compute(self, data_1, data_2):
        return - euclidean_dist_matrix(data_1, data_2) ** self._d / 2.

    def dim(self):
        return None



""" --------------------- graph kernels ---------------------- 

A module containing Shortest Path Kernel.
__author__ = 'kasiajanocha'
"""


class Graph(object):
    """Basic Graph class.
    Can be labeled by edges or nodes."""
    def __init__(self, adjacency_matix, node_labels=None, edge_labels=None):
        self.adjacency_matix = adjacency_matix
        self.node_labels = node_labels
        self.edge_labels = edge_labels

def graphs_to_adjacency_lists(data):
    """
    Given a list of graphs, output a numpy.array
    containing their adjacency matices.
    """
    try:
        if data.ndim == 3:
            return np.array(data)
    except:
        try:
            return np.array([G.adjacency_matix for G in data])
        except:
            return np.array(data)

def relabel(data, data_2):
    """
    Given list of labels for each graph in the dataset,
    rename them so they belong to set {1, ..., num_labels},
    where num_labels is number of the distinct labels.
    Return tuple consisting of new labels and maximal label.
    """
    len_first = len(data)
    for d in data_2:
        data.append(d)
    data = np.array(data)
    label_set = dict()
    for node_labels in data:
        for label in node_labels:
            if label not in label_set.keys():
                llen = len(label_set)
                label_set[label] = llen
    res = []
    for i, node_labels in enumerate(data):
        res.append([])
        for j, label in enumerate(node_labels):
            res[i].append(label_set[label] + 1)
    return res[:len_first], res[len_first:], len(label_set)


import numpy as np
import numpy.matlib as matlib
from scipy.sparse import lil_matrix

def floyd_warshall(adj_mat, weights):
    """
    Returns matrix of shortest path weights.
    """
    N = adj_mat.shape[0]
    res = np.zeros((N, N))
    res = res + ((adj_mat != 0) * weights)
    res[res == 0] = np.inf
    np.fill_diagonal(res, 0)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if res[i, j] + res[j, k] < res[i, k]:
                    res[i, k] = res[i, j] + res[j, k]
    return res

def _apply_floyd_warshall(data):
    """
    Applies Floyd-Warshall algorithm on a dataset.
    Returns a tuple containing dataset of FW transformates and max path length
    """
    res = []
    maximal = 0
    for graph in data:
        floyd = floyd_warshall(graph, graph)
        maximal = max(maximal, (floyd[~np.isinf(floyd)]).max())
        res.append(floyd)
    return res, maximal

class ShortestPath(GraphKernel):
    """
    Shortest Path kernel [3]
    """
    def __init__(self, labeled=False):
        self.labeled = labeled

    def _create_accum_list_labeled(self, shortest_paths, maxpath,
                                   labels_t, numlabels):
        """
        Construct accumulation array matrix for one dataset
        containing labaled graph data.
        """
        res = lil_matrix(
            np.zeros((len(shortest_paths),
                      (maxpath + 1) * numlabels * (numlabels + 1) / 2)))
        for i, s in enumerate(shortest_paths):
            labels = labels_t[i]
            labels_aux = matlib.repmat(labels, 1, len(labels))
            min_lab = np.minimum(labels_aux.T, labels_aux)
            max_lab = np.maximum(labels_aux.T, labels_aux)
            subsetter = np.triu(~(np.isinf(s)))
            min_lab = min_lab[subsetter]
            max_lab = max_lab[subsetter]
            ind = s[subsetter] * numlabels * (numlabels + 1) / 2 + \
                    (min_lab - 1) * (2*numlabels + 2 - min_lab) / 2 + \
                    max_lab - min_lab
            accum = np.zeros((maxpath + 1) * numlabels * (numlabels + 1) / 2)
            accum[:ind.max() + 1] += np.bincount(ind.astype(int))
            res[i] = lil_matrix(accum)
        return res

    def _create_accum_list(self, shortest_paths, maxpath):
        """
        Construct accumulation array matrix for one dataset
        containing unlabaled graph data.
        """
        res = lil_matrix(np.zeros((len(shortest_paths), maxpath+1)))
        for i, s in enumerate(shortest_paths):
            subsetter = np.triu(~(np.isinf(s)))
            ind = s[subsetter]
            accum = np.zeros(maxpath + 1)
            accum[:ind.max() + 1] += np.bincount(ind.astype(int))
            res[i] = lil_matrix(accum)
        return res

    def _compute(self, data_1, data_2):
        ams_1 = graphs_to_adjacency_lists(data_1)
        ams_2 = graphs_to_adjacency_lists(data_2)
        sp_1, max1 = _apply_floyd_warshall(np.array(ams_1))
        sp_2, max2 = _apply_floyd_warshall(np.array(ams_2))
        maxpath = max(max1, max2)
        if not self.labeled:
            accum_list_1 = self._create_accum_list(sp_1, maxpath)
            accum_list_2 = self._create_accum_list(sp_2, maxpath)
        else:
            labels_1, labels_2, numlabels = relabel(
                [G.node_labels for G in data_1], [G.node_labels for G in data_2])
            accum_list_1 = self._create_accum_list_labeled(sp_1, maxpath,
                                                           labels_1, numlabels)
            accum_list_2 = self._create_accum_list_labeled(sp_2, maxpath,
                                                           labels_2, numlabels)
        return np.asarray(accum_list_1.dot(accum_list_2.T).todense())

    def dim(self):
        return None


