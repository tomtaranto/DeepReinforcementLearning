from line_world_mdp_definition import *

import time

import numpy as np
from numba import jit

MAX_CELLS = 100

pi = np.zeros((len(S), len(A)))

# pi[:, 1] = 1.0  # Aller toujours à droite ! :)
# pi[:, 0] = 1.0  # Aller toujours à gauche ! :)
pi[:, :] = 0.5  # Aller 50% du temps à gauche et 50% du temps à droite ! :)


@jit(nopython=True)
def iterative_policy_evaluation(pi, S, A, R, p):
    theta = 0.0001
    V = np.zeros((len(S),))
    gamma = 1.0

    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = 0
            for a in A:
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        V[s] += pi[s, a] * p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break
    return V


t = time.time()
V = iterative_policy_evaluation(pi, S, A, R, p)
print(time.time() - t)
print(V)
