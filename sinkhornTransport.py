import random
import numpy as np
from scipy.sparse import rand
import matplotlib.pyplot as plt


def sinkhornTransport(a, b, K, U, lamb, stoppingCriterion='marginalDifference', p_norm=np.inf, tolerance=.5e-2,
                      max_iter=5000, verbose=0):
    '''
    This Code is Python translation of Cuturi's Optimal Transport Algorithm, original MATLAB code can be found at:
    https://marcocuturi.net/SI.html

    Original Paper discussing the Algorithm can be found at:
    
    https://papers.nips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf
    

    This Code is Translated by Hau Phan.

    inputs: 

    a is either a n x 1 column vector in the probability simplex (nonnegative, summing to one). This is the [1-vs-N mode]
    - a n x N matrix, where each column vector is in the probability simplex. This is the [N x 1-vs-1 mode]
    
    b is a m x N matrix of N vectors in the probability simplex.
    
    K is a n x m matrix, equal to exp(-lambda M), where M is the n x m matrix of pairwise distances between bins 
    described in a and bins in the b_1,...b_N histograms. In the most simple case n = m and M is simply a distance matrix (zero
    on the diagonal and such that m_ij < m_ik + m_kj

    U = K.*M is a n x m matrix, pre-stored to speed up the computation of the distances.

    Optional Inputs:

    stoppingCriterion in {'marginalDifference','distanceRelativeDecrease'}
    - marginalDifference (Default) : checks whether the difference between 
    the marginals of the current optimal transport and the theoretical marginals set by a b_1,...,b_N are satisfied.
    - distanceRelativeDecrease : only focus on convergence of the vector of distances

    p_norm: parameter in {(1,+infty]} used to compute a stoppingCriterion statistic
    rom N numbers (these N numbers might be the 1-norm of marginal
    differences or the vector of distances.

    tolerance : > 0 number to test the stoppingCriterion.

    maxIter: maximal number of Sinkhorn fixed point iterations.
    
    verbose: verbose level. 0 by default.
    
    Output

    D : vector of N dual-sinkhorn divergences, or upper bounds to the Eearth Movers Disatnce.

    L : vector of N lower bounds to the original OT problem, a.k.a EMD. This is computed by using
    the dual variables of the smoothed problem, which, when modified
    adequately, are feasible for the original (non-smoothed) OT dual problem

    u : n x N matrix of left scalings
    v : m x N matrix of right scalings

    The smoothed optimal transport between (a_i,b_i) can be recovered as
    T_i = np.diag(u[:,i]) @ K @ diag(v[:,i]);

    '''

    if a.shape[1] == 1:
        one_vs_n = True
    elif a.shape[1] == b.shape[1]:
        one_vs_n = False
    else:
        print(
            "The first parameter a is either a column vector in the probability simplex, or N column vectors in the probability simplex where N is size(b,2)")
        return

    if b.shape[1] > b.shape[0]:
        bign = True
    else:
        bign = False

    if one_vs_n:
        I = np.array(a > 0)
        some_zero_values = False
        if not (np.sum(I) == len(I)):
            some_zero_values = True
            K = K[I.squeeze()]
            U = U[I.squeeze()]
            a = a[I.squeeze()]
        ainvK = K / a
    # fixed point counter
    compt = 0
    # initialization of left scaling factors, N column vectors
    u = np.ones((a.shape[0], b.shape[1])) / a.shape[0]

    if stoppingCriterion == 'distanceRelativeDecrease':
        Dold = np.ones((1, b.shape[1]))

    while compt < max_iter:
        if one_vs_n:
            if bign:
                u = 1 / (ainvK @ (b / (K.T @ u)))
            else:
                u = 1 / (ainvK @ (b / (u.T @ K).T))
        else:
            if bign:
                u = a / (K @ (b / (u.T @ K).T))
            else:
                u = a / (K @ (b / (K.T @ u)))
        compt += 1

        if compt % 20 == 1 or compt == max_iter:
            if bign:
                v = b / (K.T @ u)
            else:
                v = b / (u.T @ K).T

            if one_vs_n:
                u = 1 / (ainvK @ v)
            else:
                u = a / (K @ v)

            if stoppingCriterion == 'distanceRelativeDecrease':
                D = np.sum(u * (U @ v), axis=0)
                Criterion = np.linalg.norm(D / Dold - 1, p_norm)
                if Criterion < tolerance or np.isnan(Criterion):
                    break
                Dold = D
            elif stoppingCriterion == 'marginalDifference':
                temp = np.sum(np.abs(v * (K.T @ u) - b), axis=0)
                Criterion = np.linalg.norm(temp, p_norm)
                if Criterion < tolerance or np.isnan(Criterion):
                    break
            else:
                print("Stopping Criterion not recognized")
                return

            compt += 1
            if verbose > 0:
                print('Iteration :', str(compt), ' Criterion: ', str(Criterion))
            if np.sum(np.isnan(Criterion)) > 0:
                print(
                    'NaN values have appeared during the fixed point iteration. This problem appears because of insufficient machine precision when processing computations with a regularization value of lambda that is too high. Try again with a reduced regularization parameter lambda or with a thresholded metric matrix M')

    if stoppingCriterion == "marginalDifference":
        D = np.sum(u * (U @ v), axis=0)

    alpha = np.log(u)
    beta = np.log(v)
    beta[beta == -np.inf] = 0
    if one_vs_n:
        L = (a.T @ alpha + np.sum(b * beta, axis=0)) / lamb
    else:
        alpha[alpha == -np.inf] = 0
        print(a.shape)
        print(alpha.shape)
        L = (np.sum(a * alpha, axis=0) + np.sum(b * beta, axis=0)) / lamb

    if one_vs_n and some_zero_values:
        uu = u
        u = np.zeros((len(I), b.shape[1]))
        u[I.squeeze()] = uu

    return D, L, u, v
