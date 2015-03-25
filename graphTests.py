from graph import Graph
import numpy as np

""" Graphs for testing sum product implementation
"""


def check_eq(a, b):
    epsilon = 10**-6
    return abs(a-b) < epsilon


def make_toy_graph():
    """ Simple graph encoding, basic testing
        2 vars, 2 facs
        f_a, f_ba - p(a)p(a|b)
        factors functions are a little funny but it works
    """
    toy_graph = Graph()
    
    a = toy_graph.add_var_node('a', 2)
    b = toy_graph.add_var_node('b', 3)
    
    p = np.array([[0.3], [0.7]])
    toy_graph.add_fac_node(p, a)
    
    p = np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    toy_graph.add_fac_node(p, b, a)
    
    return toy_graph


def test_toy_graph():
    """ Actual test case
    """
    
    toy_graph = make_toy_graph()
    marginals = toy_graph.marginals()
    brute = toy_graph.brute_force()
    
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
    am = marginals['a']
    assert check_eq(am[0], 0.11538462)
    assert check_eq(am[1], 0.88461538)
    
    bm = marginals['b']
    assert check_eq(bm[0], 0.34065934)
    assert check_eq(bm[1], 0.2967033)
    assert check_eq(bm[2], 0.36263736)
    
    # check brute force against sum-product
    amm = toy_graph.marginalize_brute(brute, 'a')
    bmm = toy_graph.marginalize_brute(brute, 'b')
    assert check_eq(am[0], amm[0])
    assert check_eq(am[1], amm[1])
    assert check_eq(bm[0], bmm[0])
    assert check_eq(bm[1], bmm[1])
    assert check_eq(bm[2], bmm[2])
    
    print "All tests passed!"


def make_test_graph():
    """ Graph for HW problem 1.c.
        4 vars, 3 facs
        f_a, f_ba, f_dca
    """
    test_graph = Graph()
    
    a = test_graph.add_var_node('a', 2)
    b = test_graph.add_var_node('b', 3)
    c = test_graph.add_var_node('c', 4)
    d = test_graph.add_var_node('d', 5)
    
    p = np.array([[0.3], [0.7]])
    test_graph.add_fac_node(p, a)
    
    p = np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
    test_graph.add_fac_node(p, b, a)
    
    p = np.array([[[3., 1.], [1.2, 0.4], [0.1, 0.9], [0.1, 0.9]],
                  [[11., 9.], [8.8, 9.4], [6.4, 0.1], [8.8, 9.4]],
                  [[3., 2.], [2., 2.], [2., 2.], [3., 2.]],
                  [[0.3, 0.7], [0.44, 0.56], [0.37, 0.63], [0.44, 0.56]],
                  [[0.2, 0.1], [0.64, 0.44], [0.37, 0.63], [0.2, 0.1]]])
    test_graph.add_fac_node(p, d, c, a)
    
    return test_graph


def test_test_graph():
    """ Automated test case
    """
    test_graph = make_test_graph()
    marginals = test_graph.marginals()
    brute = test_graph.brute_force()
    
    # check the marginals
    am = marginals['a']
    assert check_eq(am[0], 0.13755539)
    assert check_eq(am[1], 0.86244461)
    
    bm = marginals['b']
    assert check_eq(bm[0], 0.33928227)
    assert check_eq(bm[1], 0.30358863)
    assert check_eq(bm[2], 0.3571291)
    
    cm = marginals['c']
    assert check_eq(cm[0], 0.30378128)
    assert check_eq(cm[1], 0.29216947)
    assert check_eq(cm[2], 0.11007584)
    assert check_eq(cm[3], 0.29397341)
    
    dm = marginals['d']
    assert check_eq(dm[0], 0.076011)
    assert check_eq(dm[1], 0.65388724)
    assert check_eq(dm[2], 0.18740039)
    assert check_eq(dm[3], 0.05341787)
    assert check_eq(dm[4], 0.0292835)
    
    # check brute force against sum-product
    amm = test_graph.marginalize_brute(brute, 'a')
    bmm = test_graph.marginalize_brute(brute, 'b')
    cmm = test_graph.marginalize_brute(brute, 'c')
    dmm = test_graph.marginalize_brute(brute, 'd')
    
    assert check_eq(am[0], amm[0])
    assert check_eq(am[1], amm[1])
    
    assert check_eq(bm[0], bmm[0])
    assert check_eq(bm[1], bmm[1])
    assert check_eq(bm[2], bmm[2])
    
    assert check_eq(cm[0], cmm[0])
    assert check_eq(cm[1], cmm[1])
    assert check_eq(cm[2], cmm[2])
    assert check_eq(cm[3], cmm[3])
    
    assert check_eq(dm[0], dmm[0])
    assert check_eq(dm[1], dmm[1])
    assert check_eq(dm[2], dmm[2])
    assert check_eq(dm[3], dmm[3])
    assert check_eq(dm[4], dmm[4])
    
    print "All tests passed!"
    
# standard run of test cases
test_toy_graph()
test_test_graph()