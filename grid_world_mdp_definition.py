import numpy as np

LIGNES = 40
COLONNES = 40
print("****** Creating GridWorld ({:d} x {:d}) ******".format(LIGNES, COLONNES))
S = np.arange(LIGNES * COLONNES)
A = np.array([0, 1, 2, 3])  # 0: Gauche, 1: Droite, 2: Haut, 3 : Bas
R = np.array([-1, 0, 1])
p = np.zeros((len(S), len(A), len(S), len(R)))
N = len(S)

for i in range(1, N - 2):
    # Action a gauche
    a = 0
    # Si on est pas en debut de ligne
    if i % LIGNES != 0 and i != 1:
        p[i, a, i - 1, 1] = 1.0
    elif i == 1:
        p[i, a, 0, 0] = 1.0
    else:
        p[i, a, i, 1] = 1.0

    # Action a droite
    a = 1
    if i % LIGNES != LIGNES - 1:
        p[i, a, i + 1, 1] = 1.0
    else:
        p[i, a, i, 1] = 1.0

    # Action en haut
    a = 2
    # Si on est a la deuxième colonne ou plus
    if i > COLONNES:
        p[i, a, i - COLONNES, 1] = 1.0
    else:
        p[i, a, i, 1] = 1.0

    # Action en bas
    a = 3
    # Si on est pas sur la dernier ligne
    if i < (N - COLONNES):
        p[i, a, i + COLONNES, 1] = 1.0
    else:
        p[i, a, i, 1] = 1.0

p[N - 2, 1, N - 1, 2] = 1.0
if LIGNES > 1:
    p[N - COLONNES - 1, 3, N - 1, 2] = 1.0

p[1, 0, 0, 0] = 1.0
if LIGNES > 1:
    p[COLONNES + 1, 2, 0, 0] = 1.0

print('****** Grid World Fully Created ******')

'''Check
for s in S:
    for a in A:
        for s1 in S:
            for r in R:
                if p[s, a, s1, r] > 0:
                    print((s, a, s1, r, p[s, a, s1, r]))
'''
