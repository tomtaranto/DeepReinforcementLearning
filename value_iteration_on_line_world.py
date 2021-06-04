import numpy as np

from line_world_mdp_definition import *

import random
import time

gamma = 1 - 1e-5


def value_iteration(theta):
    V = np.zeros((len(S),))

    while True:
        delta = 0
        for s in S:
            v = V[s]
            a_score_max = 0
            for a in A:
                a_score = 0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        a_score += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                if a_score > a_score_max:
                    a_score_max = a_score
            V[s] = a_score_max
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    # Output :

    pi = np.zeros((len(S), len(A)))
    for s in S:
        best_a = -1
        best_a_score = None
        for a in A:
            a_score = 0.0
            for s_p in S:
                for r_idx, r in enumerate(R):
                    a_score += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
            if best_a_score is None or best_a_score < a_score:
                best_a = a
                best_a_score = a_score
        pi[s, :] = 0.0
        pi[s, best_a] = 1.0

    return V, pi


def main():
    theta = 1e-2
    start = time.time()
    V, pi = value_iteration(theta)
    print(time.time() - start)
    print((V, pi))


if __name__ == '__main__':
    main()
