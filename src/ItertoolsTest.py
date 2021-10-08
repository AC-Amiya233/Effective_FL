import itertools

import numpy as np

num_usr = 5

comb_list = []
for i in range(num_usr):
    comb_list += list(itertools.combinations(np.arange(num_usr), i))
print('comb list:', comb_list)
print(len(comb_list))

perm_list = []
for i in range(num_usr):
    perm_list += list(itertools.permutations(np.arange(num_usr)))
print('perm list:', perm_list)
print(len(perm_list))

