import numpy as np
import time
from numba import jit


def work():
    MAX_CELLS = 200

    S = np.arange(MAX_CELLS)
    A = np.array([0, 1])  # 0 : gauche, 1:droite
    R = np.array([-1, 0, 1])

    p = np.zeros((len(S), len(A), len(S), len(R)))

    for i in range(1, MAX_CELLS - 2):
        p[i, 1, i + 1, 1] = 1.0

    for i in range(1, MAX_CELLS - 1):
        p[i, 0, i - 1, 1] = 1.0

    p[MAX_CELLS - 2, 1, MAX_CELLS - 1, 2] = 1.0
    p[1, 0, 0, 0] = 1.0

    # Stratégie de jeu
    pi = np.zeros((len(S), len(A)))

    pi[:, :] = 0.5  # 50% d' aller à gauche ou à droite
    # pi[:,1] = 1.0 Aller toujours à droite
    # pi[:,0] = 1.0 Aller toujours à gauche

    # Evaluation de la stratégie
    gamma = 0.999999
    theta = 1e-4
    V = np.zeros((len(S),))

    t1 = time.time()

    @jit(nopython=True)
    def iterative_policy(pi, S, A, R, p, V):
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

    V = iterative_policy(pi, S, A, R, p, V)
    t2 = time.time()

    print(V)
    print(t2 - t1)


def main():
    work()
    print("yo")


if __name__ == '__main__':
    main()
