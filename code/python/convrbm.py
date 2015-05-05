import numpy as np
from scipy.linalg import inv, sqrtm
from copy import copy

def sigmoid(x):
    return 1./(1.+np.exp(-x))

class ConvRBM(object):
    def __init__(self, s, b, width = None, eps = 0.1, k=1, batchsize = 100, 
                 sparsity = 1E-4, mom = 0.7, verbose=False):
        """
        s : array of +/-1 visible units of shape (N_samples, N)
        b : scale factor to reduce L_v by
        width : convolutional filter width
        eps : float, "learning rate" or approx. gradient scale
        k : int, number of gibbs samples to use to estimate gradients
        batchsize : No. of samples to use in each gradient step
        sparsity : constant supressing W matrix
        mom : momentum for carrying previous steps forward
        verbose : bool for printing training stats
        """
        self.s = s
        self.N_samples, self.N_v = s.shape
        self.L_v = int(np.sqrt(self.N_v))
        self.b = b
        self.L_h = self.L_v/self.b
        self.width = width or self.L_v
        self.N_h = self.L_h * self.L_h
        self.eps = eps
        self.k = k
        self.batchsize = batchsize
        self.sparsity = sparsity
        self.mom = mom
        self.verbose = verbose
        self.h_state = np.ones(self.N_h)
        self.s_state = np.ones(self.N_v)
        self.rng = np.random.RandomState()
        self.rng.seed(92089)
        self.filter = np.random.normal(0,1./np.sqrt(self.N_h), 
                                       (self.L_v, self.L_v))
        mask = np.arange(self.L_v) < self.width
        self.mask = np.outer(mask, mask)
        self.W = np.zeros((self.L_v,self.L_v,self.L_h,self.L_h))
        self._makeW(self.filter*self.mask)
        #Start with orthogonal columns
        #self.W = self.W.dot(inv(sqrtm(self.W.T.dot(self.W))))
        self.E_list = []
        self.dWs = []
        if self.verbose:
            self.p_check = []
            self.histcheck = []
  
    def _makeW(self, convFilter):
        W = self.W.reshape(self.L_v, self.L_v, self.L_h, self.L_h)
        cFilter = (convFilter[::-1] + convFilter[:,::-1] + convFilter[::-1,::-1] +
                   convFilter)/4.
        for i in range(self.L_h):
            for j in range(self.L_h):
                shift = np.roll(np.roll(cFilter, self.b*i,0), self.b*j, 1)
                W[:,:,i,j] = shift
        self.W = W.reshape(self.N_v, self.N_h)

    def _getFilterUpdate(self, Wupdate):
        W = Wupdate.reshape(self.L_v, self.L_v, self.L_h, self.L_h)
        filterUpdate = np.zeros_like(W[:,:,0,0])
        for i in range(self.L_h):
            for j in range(self.L_h):
                convFilter = W[:,:,i,j]
                shiftback = np.roll(np.roll(convFilter, -self.b*i,0), -self.b*j, 1)
                filterUpdate += shiftback
        #Wupdate.reshape(self.N_v, self.N_h)
        return filterUpdate/(self.L_h*self.L_h+0.0)

    def energy(self, s, h):
        return -np.einsum('j,jk,k->',s, self.W, h)
    
    def _prob_hi_is_1_given_s(self, s):
        #return 0.5*(np.tanh(np.einsum('i,ik->k', s, self.W))+1.)
        return sigmoid(s.dot(self.W))

    def _prob_si_is_1_given_h(self, h):
        #return 0.5*(np.tanh(np.einsum('ik,k->i',self.W, h))+1.)
        return sigmoid(self.W.dot(h))

    def _sample_h_given_s(self, s):
        rand_h = self.rng.rand(self.N_h)
        p_h_given_s = self._prob_hi_is_1_given_s(s)
        return 2*(rand_h < p_h_given_s)-1, p_h_given_s
        
    def _sample_s_given_h(self, h):
        rand_s = self.rng.rand(self.N_v)
        p_s_given_h = self._prob_si_is_1_given_h(h)
        return 2*(rand_s < p_s_given_h)-1, p_s_given_h
        
    def _gibbs_sample_hsh(self, s = None):
        if s is None:
            rand_s = self.rng.randint(0, high=self.N_samples)
            h_sample, p_h = self._sample_h_given_s(self.s[rand_s])
        else:
            h_sample, p_h = self._sample_h_given_s(s)
        s_sample, p_s = self._sample_s_given_h(h_sample)
        return s_sample, h_sample, p_s, p_h
        
    def _gibbs_sample_shs_given_h(self, h):
        s_sample, p_s = self._sample_s_given_h(h)
        h_sample, p_h = self._sample_h_given_s(s_sample)
        return s_sample, h_sample, p_s, p_h
    
    def compute_model_si_hj(self):
        s_samples, h_samples, p_h_list = [], [], []
        s, h, p_s, p_h = self._gibbs_sample_hsh()
        for i in range(self.k):
            s, h, p_s, p_h = self._gibbs_sample_hsh(s=s)
            s_samples += [copy(s)]
            h_samples += [copy(h)]
            p_h_list += [p_h]
        h_samples = np.array(h_samples)
        s_samples = np.array(s_samples)
        self.persist = h_samples[-1]
        p_h = np.array(p_h)
        return np.einsum('ni,nj->ij',s_samples, 
                         np.r_[h_samples[:-1,:], [p_h_list[-1]]])/float(self.k)
    
    def compute_data_si_hj(self, batchint):
        low = (batchint*self.batchsize)%self.N_samples
        high = min(((batchint+1)*self.batchsize, self.N_samples))
        s_list = self.s[low:high,:]
        h_list, p_list = [], []
        for n, s in enumerate(s_list):
            h, p_h = self._sample_h_given_s(s)
            h_list += [h]
            if n%10==0:
                p_list += [p_h]
        #if self.verbose:
        #    meanenergy = np.mean([self.energy(s,h) for s,h in zip(s_list, h_list)])
        #    self.E_list += [meanenergy]
        #    #print "At batch {} E = {}".format(batchint, meanenergy)
        #    #self.p_check += [np.array(p_list)]
        return np.einsum('ni,nj->ij', np.array(s_list),
                         np.array(h_list))/float(self.batchsize)
        
    def weight_update(self, batchint):
        si_hj_m = self.compute_model_si_hj()
        si_hj_d = self.compute_data_si_hj(batchint)
        return (si_hj_d-si_hj_m)/float(self.batchsize)
        #Divide by batchsize so eps doesn't change with batchsize
        
    def one_epoch(self):
        N_sweeps = int(np.ceil(self.N_samples/float(self.batchsize)))
        step = np.zeros_like(self.W)
        for i in range(N_sweeps):
            step = self.eps*((1-self.mom)*self.weight_update(i) + self.mom*step 
                    - self.sparsity*(np.sign(self.W)))
            filterStep = self._getFilterUpdate(step)
            self.filter += filterStep * self.mask 
            self.dWs += [np.abs(filterStep).sum()]
            self._makeW(self.filter)
        #if self.verbose:
            #self.histcheck += [copy(self.W)]



