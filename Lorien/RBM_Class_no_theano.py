import time
import numpy
import os
from numpy import exp, sqrt, zeros, dot, log, sum, mean, round
from scipy.linalg import sqrtm, inv

class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, input=None, n_visible=256, n_hidden=30, W=None, hbias=None, vbias=None, numpy_rng=None):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.bit = 0
        
        if numpy_rng is None:
            self.numpy_rng = numpy.random.RandomState(1234)
            
        #Need to initialize these to a reasonable value
        if W is None:
            W = numpy.asarray(self.numpy_rng.uniform(
                    low=-4 * sqrt(6. / (n_hidden + n_visible)),
                    high=4 * sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)))
            W = self.sym(W)

        if hbias is None:
            hbias = zeros(n_hidden)
            
        if vbias is None:
            vbias = zeros(n_visible)
            
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.params = [self.W, self.hbias, self.vbias]
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        self.persistent_value = ph_sample
    
    def sym(self, w):
        return w.dot(inv(sqrtm(w.T.dot(w))))   
    
    def sigmoid(self,x):
        """sigmoid function"""
        y = 1/(1+exp(-x))
        return y

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = dot(v_sample, self.W) + self.hbias
        vbias_term = dot(v_sample, self.vbias)
        hidden_term = sum(log(1 + exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units
        '''
        pre_sigmoid_activation = vis.dot(self.W) + self.hbias
        return [pre_sigmoid_activation, self.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.numpy_rng.binomial(size=h1_mean.shape,n=1, p=h1_mean)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units
        '''
        pre_sigmoid_activation = hid.dot(self.W.T) + self.vbias
        return [pre_sigmoid_activation, self.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.numpy_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]
    
    def get_cost_updates(self, n_visible=256, n_hidden=30, lr=0.1, persistent=False, k=1):
        """This functions implements one step of CD-k or PCD-k
        :param lr: learning rate used to train the RBM
        :param persistent: None for CD. For PCD, variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).
        :param k: number of Gibbs steps to do in CD-k/PCD-k
        """
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        
        if persistent:
            chain_start = self.persistent_value
        else:
            chain_start = ph_sample
        
        n_samples = self.input.shape[0]
        pre_sigmoid_nvs = zeros((k, n_samples, n_visible))
        nv_means = zeros((k, n_samples, n_visible))
        nv_samples = zeros((k, n_samples, n_visible))
        pre_sigmoid_nhs = zeros((k, n_samples, n_hidden))
        nh_means = zeros((k, n_samples, n_hidden))
        nh_samples =zeros((k, n_samples, n_hidden))
        for i in xrange(k):
            if i == 0:
                pre_sigmoid_nvs[i],nv_means[i],nv_samples[i],pre_sigmoid_nhs[i],nh_means[i],nh_samples[i] = self.gibbs_hvh(chain_start)
            else:
                pre_sigmoid_nvs[i],nv_means[i],nv_samples[i],pre_sigmoid_nhs[i],nh_means[i],nh_samples[i] = self.gibbs_hvh(nh_samples[k-1])
        chain_end = nv_samples[-1]
     
        cost = mean(self.free_energy(self.input)) - mean(self.free_energy(chain_end))
        gparams = self.grad(chain_end)
        self.W += lr*gparams[0]
        self.hbias += lr*gparams[1]
        self.vbias += lr*gparams[2]
        self.params = [self.W, self.hbias, self.vbias]

        if persistent:
            self.persistent_value = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost()
        else:
            monitoring_cost = self.get_reconstruction_cost(pre_sigmoid_nvs[-1])

        return monitoring_cost

    def grad(self, chain_end):
        """ Calculate the gradient of the cost wrt to parameters"""
        g_W = mean(self.useful_func(chain_end), axis=0) - mean(self.useful_func(self.input), axis=0)
        g_hbias = mean(self.useful_func2(chain_end), axis=0) - mean(self.useful_func2(self.input), axis=0)
        g_vbias = mean(chain_end, axis=0) - mean(self.input, axis=0)
        return [g_W, g_hbias, g_vbias]
    

    def useful_func(self,x):
        nv = self.n_visible
        nh = self.n_hidden
        piece = self.sigmoid(x.dot(self.W)+self.hbias)# n_samples x n_hidden
        part = numpy.einsum('il,im->ilm', x, piece)
        return part
    
  
    def useful_func2(self, vis):
        return self.sigmoid(dot(vis, self.W) + self.hbias)
    
      
    def get_pseudo_likelihood_cost(self):
        """Stochastic approximation to the pseudo-likelihood"""
        bit_i_idx = self.bit
        xi = round(self.input)
        fe_xi = self.free_energy(xi)
        xi_flip = xi.copy()
        xi_flip[:,bit_i_idx] =  1 - xi[:, bit_i_idx]
        fe_xi_flip = self.free_energy(xi_flip)
        cost = numpy.mean(self.n_visible * numpy.log(self.sigmoid(fe_xi_flip - fe_xi)))
        self.bit = (self.bit + 1) % self.n_visible
        return cost

    def get_reconstruction_cost(self,pre_sigmoid_nv):
        """Approximation to the reconstruction error"""
        cross_entropy = mean(sum(self.input*log(self.sigmoid(pre_sigmoid_nv))+(1 - self.input)*log(1-self.sigmoid(pre_sigmoid_nv)), axis=1))
        return cross_entropy
    
