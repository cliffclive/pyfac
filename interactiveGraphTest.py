__author__ = 'ccclive'

from graph import Graph
import numpy as np

""" Graphs for testing sum product implementation
"""


def check_eq(a, b):
    epsilon = 10**-6
    return abs(a-b) < epsilon


toy_graph = Graph()

a = toy_graph.add_var_node('a', 2)
b = toy_graph.add_var_node('b', 3)

p12 = np.array([[0.3], [0.7]])
toy_graph.add_fac_node(p12, a)

p23 = np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
toy_graph.add_fac_node(p23, b, a)

toy_marginals = toy_graph.marginals()
toy_sumprod = toy_graph.sum_product()

# check the results
# want to verify incoming messages
# if vars are correct then factors must be as well
a = toy_graph.var['a'].incoming
assert check_eq(a[0][0], 0.3)
assert check_eq(a[0][1], 0.7)
assert check_eq(a[1][0], 0.23333333)
assert check_eq(a[1][1], 0.76666667)

b = toy_graph.var['b'].incoming
assert check_eq(b[0][0], 0.34065934)
assert check_eq(b[0][1], 0.2967033)
assert check_eq(b[0][2], 0.36263736)

# check the marginals
toy_am = toy_marginals['a']
assert check_eq(toy_am[0], 0.11538462)
assert check_eq(toy_am[1], 0.88461538)

toy_bm = toy_marginals['b']
assert check_eq(toy_bm[0], 0.34065934)
assert check_eq(toy_bm[1], 0.2967033)
assert check_eq(toy_bm[2], 0.36263736)
"""
# check brute force against sum-product
toy_amb = toy_graph.marginalize_brute(toy_sumprod, 'a')
toy_bmb = toy_graph.marginalize_brute(toy_sumprod, 'b')
assert check_eq(toy_am[0], toy_amb[0])
assert check_eq(toy_am[1], toy_amb[1])
assert check_eq(toy_bm[0], toy_bmb[0])
assert check_eq(toy_bm[1], toy_bmb[1])
assert check_eq(toy_bm[2], toy_bmb[2])
"""
print "All tests passed!"


test_graph = Graph()

a = test_graph.add_var_node('a', 2)
b = test_graph.add_var_node('b', 3)
c = test_graph.add_var_node('c', 4)
d = test_graph.add_var_node('d', 5)

test_graph.add_fac_node(p12, a)

test_graph.add_fac_node(p23, b, a)

p245 = np.array([[[3., 1.], [1.2, 0.4], [0.1, 0.9], [0.1, 0.9]],
              [[11., 9.], [8.8, 9.4], [6.4, 0.1], [8.8, 9.4]],
              [[3., 2.], [2., 2.], [2., 2.], [3., 2.]],
              [[0.3, 0.7], [0.44, 0.56], [0.37, 0.63], [0.44, 0.56]],
              [[0.2, 0.1], [0.64, 0.44], [0.37, 0.63], [0.2, 0.1]]])
test_graph.add_fac_node(p245, d, c, a)

test_marginals = test_graph.marginals()
test_sumprod = test_graph.sum_product()

# check the marginals
test_am = test_marginals['a']
assert check_eq(test_am[0], 0.13755539)
assert check_eq(test_am[1], 0.86244461)

test_bm = test_marginals['b']
assert check_eq(test_bm[0], 0.33928227)
assert check_eq(test_bm[1], 0.30358863)
assert check_eq(test_bm[2], 0.3571291)

test_cm = test_marginals['c']
assert check_eq(test_cm[0], 0.30378128)
assert check_eq(test_cm[1], 0.29216947)
assert check_eq(test_cm[2], 0.11007584)
assert check_eq(test_cm[3], 0.29397341)

test_dm = test_marginals['d']
assert check_eq(test_dm[0], 0.076011)
assert check_eq(test_dm[1], 0.65388724)
assert check_eq(test_dm[2], 0.18740039)
assert check_eq(test_dm[3], 0.05341787)
assert check_eq(test_dm[4], 0.0292835)

"""
# check brute force against sum-product
test_amb = test_graph.marginalize_brute(test_sumprod, 'a')
test_bmb = test_graph.marginalize_brute(test_sumprod, 'b')
test_cmb = test_graph.marginalize_brute(test_sumprod, 'c')
test_dmb = test_graph.marginalize_brute(test_sumprod, 'd')

assert check_eq(test_am[0], test_amb[0])
assert check_eq(test_am[1], test_amb[1])

assert check_eq(test_bm[0], test_bmb[0])
assert check_eq(test_bm[1], test_bmb[1])
assert check_eq(test_bm[2], test_bmb[2])

assert check_eq(test_cm[0], test_cmb[0])
assert check_eq(test_cm[1], test_cmb[1])
assert check_eq(test_cm[2], test_cmb[2])
assert check_eq(test_cm[3], test_cmb[3])

assert check_eq(test_dm[0], test_dmb[0])
assert check_eq(test_dm[1], test_dmb[1])
assert check_eq(test_dm[2], test_dmb[2])
assert check_eq(test_dm[3], test_dmb[3])
assert check_eq(test_dm[4], test_dmb[4])
"""

print "All tests passed!"

