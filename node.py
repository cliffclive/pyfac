import numpy as np
import pdb

""" Factor Graph classes forming structure for PGMs
    Basic structure is port of MATLAB code by J. Pacheco
    Central difference: nbrs stored as references, not ids
        (makes message propagation easier)
    
    Note to self: use %pdb and %load_ext autoreload followed by %autoreload 2
"""


class Node(object):
    """ Superclass for graph nodes
    """
    epsilon = 10**(-4)
    
    def __init__(self, nid):
        self.enabled = True
        self.nid = nid
        self.neighbors = []
        self.incoming = []
        self.outgoing = []
        self.old_outgoing = []
    
    def reset(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def enable(self):
        self.enabled = True
        for n in self.neighbors:
            # don't call enable() as it will recursively enable entire graph
            n.enabled = True
    
    def next_step(self):
        """ Used to have this line in prep_messages
            but it didn't work?
        """
        self.old_outgoing = self.outgoing[:]
    
    def normalize_messages(self):
        """ Normalize to sum to 1
        """
        self.outgoing = [x / np.sum(x) for x in self.outgoing]
    
    def receive_message(self, node, message):
        """ Places new message into correct location in new message list
        """
        if self.enabled:
            i = self.neighbors.index(node)
            self.incoming[i] = message
    
    def send_messages(self):
        """ Sends all outgoing messages
        """
        for i in xrange(0, len(self.outgoing)):
            self.neighbors[i].receive_message(self, self.outgoing[i])
    
    def check_convergence(self):
        """ Check if any messages have changed
        """
        if self.enabled:
            for i in xrange(0, len(self.outgoing)):
                # check messages have same shape
                self.old_outgoing[i].shape = self.outgoing[i].shape
                delta = np.absolute(self.outgoing[i] - self.old_outgoing[i])
                if (delta > Node.epsilon).any():  # if there has been change
                    return False
            return True
        else:
            # Always return True if disabled to avoid interrupting check
            return True


class VarNode(Node):
    """ Variable node in factor graph
    """
    def __init__(self, name, dim, nid):
        super(VarNode, self).__init__(nid)
        self.name = name
        self.dim = dim
        self.observed = -1  # only >= 0 if variable is observed
    
    def reset(self):
        super(VarNode, self).reset()
        size = range(0, len(self.incoming))
        self.incoming = [np.ones((self.dim, 1)) for i in size]
        self.outgoing = [np.ones((self.dim, 1)) for i in size]
        self.old_outgoing = [np.ones((self.dim, 1)) for i in size]
        self.observed = -1
    
    def condition(self, observation):
        """ Condition on observing certain value
        """
        self.enable()
        self.observed = observation
        # set messages (won't change)
        for i in xrange(0, len(self.outgoing)):
            self.outgoing[i] = np.zeros((self.dim, 1))
            self.outgoing[i][self.observed] = 1.
        self.next_step()  # copy into old_outgoing
    
    def prep_messages(self):
        """ Multiplies together incoming messages to make new outgoing
        """
        # compute new messages if no observation has been made
        if self.enabled and self.observed < 0 and len(self.neighbors) > 1:
            # switch reference for old messages
            self.next_step()
            for i in xrange(0, len(self.incoming)):
                # multiply together all excluding message at current index
                curr = self.incoming[:]
                del curr[i]
                self.outgoing[i] = reduce(np.multiply, curr)
        
            # normalize once finished with all messages
            self.normalize_messages()


class FacNode(Node):
    """ Factor node in factor graph
    """
    def __init__(self, P, nid, *args):
        super(FacNode, self).__init__(nid)
        self.P = P
        self.neighbors = list(args)  # list storing refs to variable nodes
        
        # num of edges
        n_neighbors = len(self.neighbors)
        n_dependencies = self.P.squeeze().ndim
        
        # init messages
        for i in xrange(0, n_neighbors):
            v = self.neighbors[i]
            vdim = v.dim
            
            # init for factor
            self.incoming.append(np.ones((vdim, 1)))
            self.outgoing.append(np.ones((vdim, 1)))
            self.old_outgoing.append(np.ones((vdim, 1)))

            # TODO: do this in an add_neighbor function in the VarNode class!
            # init for variable
            v.neighbors.append(self)
            v.incoming.append(np.ones((vdim, 1)))
            v.outgoing.append(np.ones((vdim, 1)))
            v.old_outgoing.append(np.ones((vdim, 1)))
        
        # error check
        assert (n_neighbors == n_dependencies), "Factor dimensions does not match size of domain."
    
    def reset(self):
        super(FacNode, self).reset()
        for i in xrange(0, len(self.incoming)):
            self.incoming[i] = np.ones((self.neighbors[i].dim, 1))
            self.outgoing[i] = np.ones((self.neighbors[i].dim, 1))
            self.old_outgoing[i] = np.ones((self.neighbors[i].dim, 1))
    
    def prep_messages(self):
        """ Multiplies incoming messages w/ P to make new outgoing
        """
        if self.enabled:
            # switch references for old messages
            self.next_step()
        
            n_messages = len(self.incoming)
        
            # do tiling in advance
            # roll axes to match shape of newMessage after
            for i in xrange(0, n_messages):
                # find tiling size
                next_shape = list(self.P.shape)
                del next_shape[i]
                next_shape.insert(0, 1)
                # need to expand incoming message to correct num of dims to tile properly
                prep_shape = [1 for x in next_shape]
                prep_shape[0] = self.incoming[i].shape[0]
                self.incoming[i].shape = prep_shape
                # tile and roll
                self.incoming[i] = np.tile(self.incoming[i], next_shape)
                self.incoming[i] = np.rollaxis(self.incoming[i], 0, i+1)
            
            # loop over subsets
            for i in xrange(0, n_messages):
                curr = self.incoming[:]
                del curr[i]
                new_message = reduce(np.multiply, curr, self.P)
                    
                # sum over all vars except i!
                # roll axis i to front then sum over all other axes
                new_message = np.rollaxis(new_message, i, 0)
                new_message = np.sum(new_message, tuple(range(1, n_messages)))
                new_message.shape = (new_message.shape[0], 1)
                    
                #store new message
                self.outgoing[i] = new_message
        
            # normalize once finished with all messages
            self.normalize_messages()
