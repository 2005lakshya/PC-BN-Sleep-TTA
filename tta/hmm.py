"""Decoders that turn per-epoch probabilities into a stage sequence, without gradients.

The CNN gives p_s(y_t | x_t), learned under the source prior pi_s. Dividing by pi_s gives a
scaled likelihood p(x_t | y_t), which an HMM can combine with a prior over stage sequences:

    argmax        no temporal model
    em_prior      EM re-estimate of the class prior for this night (Saerens et al. 2002)
    hmm_src       HMM with the source transition matrix, fixed (OATTA-style temporal prior)
    hmm_full      HMM whose whole transition matrix is re-fit on the night by Baum-Welch
    hmm_dwell     ours: only the self-transitions (how long each stage lasts) are re-fit;
                  where a stage goes when it ends (the jump chain) stays at the source value

A transition matrix A is split as  A = diag(s) + diag(1 - s) J,  with s_i = P(stay in i)
and J the jump chain (zero diagonal, rows sum to 1). A population with a different stage
mix, such as elderly sleepers with little N3, mostly changes s; hmm_dwell bets on that.
"""
import numpy as np


def split_dwell(A):
    s = np.diag(A).copy()
    J = A * (1 - np.eye(len(A)))
    return s, J / J.sum(1, keepdims=True)


def join_dwell(s, J):
    return np.diag(s) + (1 - s)[:, None] * J


def stationary(A):
    w, v = np.linalg.eig(A.T)
    p = np.abs(np.real(v[:, np.argmin(np.abs(w - 1))]))
    return p / p.sum()


def forward_backward(lik, A, p0, segments):
    """lik (n, k) scaled likelihoods. Returns posteriors (n, k), expected transition
    counts (k, k) and the log-likelihood. Each segment (run of consecutive epochs) is an
    independent chain."""
    n, k = lik.shape
    post, xi, ll = np.zeros((n, k)), np.zeros((k, k)), 0.0
    for seg in segments:
        L = lik[seg]
        T = len(seg)
        alpha, beta, c = np.zeros((T, k)), np.ones((T, k)), np.zeros(T)
        a = p0 * L[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * L[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (L[t + 1] * beta[t + 1])) / c[t + 1]
        g = alpha * beta
        post[seg] = g / g.sum(1, keepdims=True)
        if T > 1:
            xi += np.einsum("ti,ij,tj->ij", alpha[:-1], A, L[1:] * beta[1:] / c[1:, None])
        ll += np.log(c).sum()
    return post, xi, ll


def em_prior(probs, prior_src, iters=50):
    """Saerens-Latinne-Decaestecker EM: re-estimate the target class prior, re-weight."""
    pi = prior_src.copy()
    for _ in range(iters):
        q = probs * (pi / prior_src)
        q /= q.sum(1, keepdims=True)
        pi = q.mean(0)
    return q


def decode(probs, prior_src, A_src, segments, mode, iters=10, kappa=20.0):
    """probs (n, k) numpy. kappa = pseudo-count pulling re-fitted transitions toward the
    source values (a MAP prior), so a short or odd night cannot drive them anywhere."""
    if mode == "argmax":
        return probs
    if mode == "em_prior":
        return em_prior(probs, prior_src)
    lik = probs / prior_src
    lik /= lik.max(1, keepdims=True)
    A = A_src.copy()
    s_src, J_src = split_dwell(A_src)
    for _ in range(iters if mode != "hmm_src" else 1):
        post, xi, _ = forward_backward(lik, A, stationary(A), segments)
        if mode == "hmm_full":
            A = (xi + kappa * A_src) / (xi.sum(1, keepdims=True) + kappa)
        elif mode == "hmm_dwell":
            s = (np.diag(xi) + kappa * s_src) / (xi.sum(1) + kappa)
            A = join_dwell(s, J_src)
    if mode != "hmm_src":
        post, _, _ = forward_backward(lik, A, stationary(A), segments)
    return post


DECODERS = ["argmax", "em_prior", "hmm_src", "hmm_full", "hmm_dwell"]
