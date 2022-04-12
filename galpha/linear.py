# -*- coding: utf-8 -*-
import numpy as np

__all__ = ["generalized_alpha_algorithm", "generalized_alpha_method"]

def generalized_alpha_algorithm(M, C, K, p, h, d0, v0, af=.5, am=.5, b=0.25, g=0.5):
    """Solve a second order system with the generalized-alpha algorithm

    The generalized alpha algorithm implements a family of time
    integration methods commonly used in solving structural dynamics
    problems, i.e. systems on the form:

        Ma(t) + Cv(t) + Kd(t) = p(t)

    where v and a are the first and second derivative of d wrt time.
    Included is the well known HHT-alpha (alpha_m=0),
    WBZ-alpha (alpha_f=0)  and Newmark method (alpha_f=alpha_m=0).

    Stability
    ---------
    The algorithm is unconditionally stable for linear systems provided
    that

        alpha_m <= alpha_f <= 1/2

    and

        beta > 1/4 + 1/2*(alpha_f-alpha_m)

    Accuracy
    --------
    The algorithm is second-order accurate provided that

        gamma = 1/2 - alpha_m + alpha_f

    Arguments
    ---------
    M, C, K : float or 2darray
        Mass, Damping and Stiffness matrices defining the structural
        system to be solved.
    p : 1darray or 2darray
        External load matrix, where each column is the load vector at
        time a time instance.
    h : float
        Time step
    d0, v0 : float or 1darray
        Initial displacement and velocity vectors, respectively.
    af, am, g, b : float
        Algorithmic parameters which determine the characteristics of
        the algorithm. The default values corresponds to constant
        acceleration integrator with no algorithmic damping.

    Returns
    -------
    A, V, D : 2darray
        Acceleration, velocity and displacement matrices

    References
    ----------
        J. Chung, G. M. Hulbert. A time integration Algorithm for
        Structural Dynamics With Improved Numerical Dissipation:
        The Generalized-alpha method.
        Journal of Applied Mechanics (1993) vol. 60, pg 371-375.
    """
    M = np.atleast_2d(M)
    C = np.atleast_2d(C)
    K = np.atleast_2d(K)
    p = np.atleast_2d(p)

    N = p.shape[0]
    I = np.eye(N)

    X = np.zeros((3*N, p.shape[1]))
    X[:1*N, 0] = d0
    X[1*N:2*N, 0] = v0
    X[2*N:3*N, 0] = np.linalg.solve(M, p[:, 0] - C@X[N:2*N, 0]-K@X[:N, 0])

    Q = (1-am)*M+(1-af)*g*h*C+(1-af)*b*h**2*K

    Mq = np.linalg.solve(Q, (am*M + (1-af)*(1-g)*h*C + (1-af)*(0.5-b)*h**2*K))
    Cq = np.linalg.solve(Q, (C + (1-af)*h*K))
    Kq = np.linalg.solve(Q, K)

    A = np.zeros((3*N, 3*N))

    A[:N, :N] = I-b*h**2*Kq
    A[:N, N:2*N] = h*I-b*h**2*Cq
    A[:N, 2*N:] = (0.5-b)*h**2*I-b*h**2*Mq

    A[N:2*N, :N] = -g*h*Kq
    A[N:2*N, N:2*N] = I-g*h*Cq
    A[N:2*N, 2*N:] = (1-g)*h*I-g*h*Mq

    A[2*N:3*N, :N] = -Kq
    A[2*N:3*N, N:2*N] = -Cq
    A[2*N:3*N, 2*N:] = -Mq

    B = np.zeros((3*N, N))
    B[:N, :] = b*h**2*I
    B[N:2*N, :] = g*h*I
    B[2*N:, :] = I

    BU = B@np.linalg.solve(Q, (1-af)*p[:,1:] + af*p[:, :-1])
    for n in range(p.shape[1]-1):
        X[:, n+1] = A@X[:, n] + BU[:, n]
    return X[2*N:], X[N:2*N], X[:N]


def generalized_alpha_method(M, C, K, p, h, d0, v0, rho=1.0):
    """Implementation of the generalized-alpha method

    The generalized alpha method is an implicit integration algorithm
    controlled by a single parameter, `rho`. `rho` governs the amount
    of numerical damping introduced by the method.

    Arguments
    ---------
    M, C, K : float or 2darray
        Mass, Damping and Stiffness matrices defining the structural
        system to be solved.
    p : 1darray or 2darray
        External load matrix, where each column is the load vector at
        time a time instance.
    h : float
        Time step
    d0, v0 : float or 1darray
        Initial displacement and velocity vectors, respectively.
    rho : float
        Determines the numerical damping by the algorithm, a value of 1
        introduces no damping while 0 introduces maximum damping.

    Returns
    -------
    A, V, D : 2darray
        Acceleration, velocity and displacement matrices
    """
    alpha_f = rho / (rho + 1)
    alpha_m = (2*rho - 1) / (rho + 1)
    beta = 0.25 * (1-alpha_m+alpha_f)**2
    gamma = 0.5 - alpha_m + alpha_f
    return generalized_alpha_algorithm(M, C, K, p, h, d0, v0,
                                        alpha_f, alpha_m, beta, gamma)
