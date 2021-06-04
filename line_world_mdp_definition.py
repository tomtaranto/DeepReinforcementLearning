import numpy as np

MAX_CELLS = 7

S = np.arange(MAX_CELLS)
A = np.array([0, 1])  # 0: Gauche, 1: Droite
R = np.array([-1, 0, 1])
p = np.zeros((len(S), len(A), len(S), len(R)))

for i in range(1, MAX_CELLS - 2):
    p[i, 1, i + 1, 1] = 1.0

for i in range(2, MAX_CELLS - 1):
    p[i, 0, i - 1, 1] = 1.0

p[MAX_CELLS - 2, 1, MAX_CELLS - 1, 2] = 1.0
p[1, 0, 0, 0] = 1.0

for s in S:
    for a in A:
        for s1 in S:
            for r in R:
                if p[s, a, s1, r] != 0:
                    print((s, a, s1, r, p[s, a, s1, r]))
