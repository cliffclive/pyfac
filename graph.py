# Graph class
import numpy as np
from node import FacNode, VarNode
import pdb

""" Factor Graph classes forming structure for PGMs
    Basic structure is port of MATLAB code by J. Pacheco
    Central difference: nbrs stored as references, not ids
        (makes message propagation easier)
"""


class Graph:
    """ Putting everything together
    """
    
    def __init__(self):
        self.var = {}
        self.fac = []
        self.dims = []
        self.converged = False
        
    def add_var_node(self, name, dim):
        new_id = len(self.var)
        new_var = VarNode(name, dim, new_id)
        self.var[name] = new_var
        self.dims.append(dim)
        
        return new_var
    
    def add_fac_node(self, P, *args):
        new_id = len(self.fac)
        new_fac = FacNode(P, new_id, *args)
        self.fac.append(new_fac)
        
        return new_fac
    
    def disable_all(self):
        """ Disable all nodes in graph
            Useful for switching on small subnetworks
            of bayesian nets
        """
        for k, v in self.var.iteritems():
            v.disable()
        for f in self.fac:
            f.disable()
    
    def reset(self):
        """ Reset messages to original state
        """
        for k, v in self.var.iteritems():
            v.reset()
        for f in self.fac:
            f.reset()
        self.converged = False
    
    def sum_product(self, max_steps=500):
        """ This is the algorithm!
            Each time_step:
            take incoming messages and multiply together to produce outgoing for all nodes
            then push outgoing to neighbors' incoming
            check outgoing v. previous outgoing to check for convergence
        """
        # loop to convergence
        time_step = 0
        while time_step < max_steps and not self.converged:  # run for max_steps cycles
            time_step += 1
            print time_step
            
            for f in self.fac:
                # start with factor-to-variable
                # can send immediately since not sending to any other factors
                f.prep_messages()
                f.send_messages()
            
            for k, v in self.var.iteritems():
                # variable-to-factor
                v.prep_messages()
                v.send_messages()
            
            # check for convergence
            t = True
            for k, v in self.var.iteritems():
                t = t and v.check_convergence()
                if not t:
                    break
            if t:        
                for f in self.fac:
                    t = t and f.check_convergence()
                    if not t:
                        break
            
            if t:  # we have convergence!
                self.converged = True
        
        # if run for 500 steps and still no convergence:impor
        if not self.converged:
            print "No convergence!"
        
    def marginals(self, max_steps=500):
        """ Return dictionary of all marginal distributions
            indexed by corresponding variable name
        """
        # Message pass
        self.sum_product(max_steps)
        
        marginals = {}
        # for each var
        for k, v in self.var.iteritems():
            if v.enabled:  # only include enabled variables
                # multiply together messages
                v_marginal = 1
                for i in xrange(0, len(v.incoming)):
                    v_marginal = v_marginal * v.incoming[i]
            
                # normalize
                n = np.sum(v_marginal)
                v_marginal = v_marginal / n
            
                marginals[k] = v_marginal
        
        return marginals
    
    def brute_force(self):
        """ Brute force method. Only here for completeness.
            Don't use unless you want your code to take forever to produce results.
            Note: index corresponding to var determined by order added
            Problem: max number of dims in numpy is 32???
            Limit to enabled vars as work-around
        """
        # Figure out what is enabled and save dimensionality
        enabled_dims = []
        enabled_nids = []
        enabled_names = []
        enabled_observed = []
        for k, v in self.var.iteritems():
            if v.enabled:
                enabled_nids.append(v.nid)
                enabled_names.append(k)
                enabled_observed.append(v.observed)
                if v.observed < 0:
                    enabled_dims.append(v.dim)
                else:
                    enabled_dims.append(1)
        
        # initialize matrix over all joint configurations
        joint = np.zeros(enabled_dims)
        
        # loop over all configurations
        self.configuration_loop(joint, enabled_nids, enabled_observed, [])
        
        # normalize
        joint = joint / np.sum(joint)
        return {'joint': joint, 'names': enabled_names}
    
    def configuration_loop(self, joint, enabled_nids, enabled_observed, current_state):
        """ Recursive loop over all configurations
            Used for brute force computation
            joint - matrix storing joint probabilities
            enabled_nids - nids of enabled variables
            enabled_observed - observed variables (if observed!)
            current_state - list storing current configuration of vars up to this point
        """
        current_var = len(current_state)
        if current_var != len(enabled_nids):
            # need to continue assembling current configuration
            if enabled_observed[current_var] < 0:
                for i in xrange(0, joint.shape[current_var]):
                    # add new variable value to state
                    current_state.append(i)
                    self.configuration_loop(joint, enabled_nids, enabled_observed, current_state)
                    # remove it for next value
                    current_state.pop()
            else:
                # do the same thing but only once w/ observed value!
                current_state.append(enabled_observed[current_var])
                self.configuration_loop(joint, enabled_nids, enabled_observed, current_state)
                current_state.pop()
                
        else:
            # compute value for current configuration
            potential = 1.
            for f in self.fac:
                if f.enabled and False not in [x.enabled for x in f.neighbors]:
                    # figure out which vars are part of factor
                    # then get current values of those vars in correct order
                    args = [current_state[enabled_nids.index(x.nid)] for x in f.neighbors]
                
                    # get value and multiply in
                    potential = potential * f.P[tuple(args)]
            
            # now add it to joint after correcting state for observed nodes
            ind = [current_state[i] if enabled_observed[i] < 0 else 0 for i in range(0, current_var)]
            joint[tuple(ind)] = potential

    @staticmethod
    def marginalize_brute(brute, var):
        """ Util for marginalizing over joint configuration arrays produced by brute_force
        """
        sum_out = range(0, len(brute['names']))
        del sum_out[brute['names'].index(var)]
        marg = np.sum(brute['joint'], tuple(sum_out))
        return marg / np.sum(marg)  # normalize to sum to one
