import time

from numba import njit

# from grid_world_mdp_definition import *
from grid_world_mdp_definition import *


@njit
def value_iteration(theta, gamma):
    V = np.zeros((len(S),))

    while True:
        delta = 0
        for s in S:
            v = V[s]
            a_score_max = 0
            for aa in A:
                a_score = 0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        a_score += p[s, aa, s_p, r_idx] * (r + gamma * V[s_p])
                if a_score > a_score_max:
                    a_score_max = a_score
            V[s] = a_score_max
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
        print(np.reshape(V, (LIGNES, COLONNES)))
    # Output :

    pi = np.zeros((len(S), len(A)))
    for s in S:
        best_a = -1
        best_a_score = None
        for aa in A:
            a_score = 0.0
            for s_p in S:
                for r_idx, r in enumerate(R):
                    a_score += p[s, aa, s_p, r_idx] * (r + gamma * V[s_p])
            if best_a_score is None or best_a_score < a_score:
                best_a = aa
                best_a_score = a_score
        pi[s, :] = 0.0
        pi[s, best_a] = 1.0
    print(V)
    return V, pi


def main():
    theta = 1e-2
    gamma = 1 - 1e-5
    start = time.time()
    V, pi = value_iteration(theta, gamma)
    print(time.time() - start)
    print((V, pi))


if __name__ == '__main__':
    main()
