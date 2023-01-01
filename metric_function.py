import numpy as np
from scipy import *
def asymmetricKL(P,Q):
    return sum(P * log(P / Q)) #calculate the kl divergence between P and Q
 
def symmetricalKL(P,Q):
    return (asymmetricKL(P,Q)+asymmetricKL(Q,P))/2.00

def cal_pccs(x, y):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    n = len(x)
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc    


def evaluate_cluster(X, labels):
    #ss = -1
    from sklearn.metrics import silhouette_score
    ss = silhouette_score(X, labels, metric='precomputed', sample_size=None, random_state=None)
    return ss