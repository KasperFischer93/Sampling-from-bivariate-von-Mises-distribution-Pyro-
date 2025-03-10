import math
import torch
import pyro
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from pyro.distributions import *
from torch.autograd import Variable


def _log_im(order, x):
    # Based on '_log_modified_bessel_fn'
    # Tanabe, A., Fukumizu, K., Oba, S., Takenouchi, T., & Ishii, S. (2007). 
    # Parameter estimation for von Mises–Fisher distributions. Computational Statistics, 22(1), 145-157.
    """ terms to sum over, 10 by 'shape of x' and sums over the first dimension """
    """ vectorized logarithmic Im """
    s = torch.arange(0 , int(5*50+1)).reshape(int(5*50+1), 1).float()
    fs = 2 * s * (x.log() - math.log(2)) - torch.lgamma(s + 1) - torch.lgamma(order + s + 1)
    return order * (x.log() - math.log(2)) + fs.logsumexp(-2)

def _log_binom(n, k):
    """ Returns of the log of n choose k. """
    return torch.lgamma(n+1) - ( torch.lgamma(k+1) + torch.lgamma((n-k)+1))

def _logCinv(k1, k2, lam, terms):
    # Harshinder Singh, Vladimir Hnizdo, and Eugene Demchuk
    # Probabilistic model for twodependent circular variables.
    # Biometrika, 89(3):719–723, 2002.
    """
    Closed form expression of the normalizing constant
    Vectorized and in log-space
    
    k1, k2 & lam is the parameters from the bivariate von Mises
    
    Since the closed expression is an infinite sum, 'terms' is the number
    of terms, over which the expression is summed over. Estimation by convergence.
    """
    m = torch.arange(0, terms).float()
    logC = _log_binom(2*m, m) + m*((2*lam.log()) - (4*k1*k2).log()) + _log_im(m, k1) + _log_im(m, k2)
    return math.log(4) + 2* math.log(math.pi) + logC.logsumexp(-1)

def _bfind(eig):
    # John  T  Kent,  Asaad  M  Ganeiber,  and  Kanti  V  Mardia.  
    # A new  unified  approach  forthe simulation of a wide class of directional distributions.
    # Journal of Computational andGraphical Statistics, 27(2):291–301, 2018.
    """
    Estimates b0, as being the solution to equation 3.6 in the article mentioned above.
    """
    q = eig.shape[0]
    if (eig**2).sum() == 0.:
        return torch.tensor(q).float()
    else:
        lr = 1e-4
        b = torch.tensor(1., requires_grad=True)
        for _ in range(1000):
            F = torch.abs( 1 - torch.sum(1 / (b + 2 * eig)) )
            F.backward()
            b.data -= lr * b.grad
            b.grad.zero_()
        return b.data
    
def _acg_bound(nsim, k1, k2, lam, mtop = 1000):
    # John  T  Kent,  Asaad  M  Ganeiber,  and  Kanti  V  Mardia.  
    # A new  unified  approach  forthe simulation of a wide class of directional distributions.
    # Journal of Computational andGraphical Statistics, 27(2):291–301, 2018.
    
    """
    Sampling approach used in Kent et al. (2018)
    Samples the cartesian coordinates from bivariate ACG: x, y
    Acceptance criterion:
        - Sample values v, from uniform between 0 and 1
        - If v < fg, accept x, y
    Convert x, y to angles phi using atan2,
    we have now simulated the bessel density.
    """

    ntry = 0; nleft = nsim; mloop = 0
    eig = torch.tensor([0., 0.5 * (k1 - lam**2/k2)]); eigmin = 0
    if eig[1] < 0:
        eigmin = eig[1]; eig = eig - eigmin

    q = 2; b0 = _bfind(eig)
    phi = 1 + 2*eig/b0; den = _log_im(0, k2)
    values = torch.empty(nsim, 2); accepted = 0

    while nleft > 0 and mloop < mtop:
        x = Normal(0., 1.).sample((nleft*q,)).reshape(nleft, q) *  torch.ones(nleft, 1) * torch.tensor( (1/phi).sqrt()).reshape(1, q)
        r = (x*x).sum(-1).sqrt()
        # Dividing a vector by its norm, gives the unit vector
        # So the ACG samples unit vectors?
        x = x / (r.reshape(nleft, 1) * torch.ones(1, q))
        u = ((x*x) * torch.ones(nleft, 1) * torch.tensor(eig).reshape(1, q)).sum(-1)
        v = Uniform(0, 1).sample((nleft, ))
        # eq 7.3 + eq 4.2
        logf = (k1*(x[:,0] - 1) + eigmin) + (_log_im(0, torch.sqrt(k2**2 + lam**2 * x[:,1]**2 )) - den )
        # eq 3.4
        loggi = 0.5 * (q - b0) + q/2 * ((1+2*u/b0).log() + (b0/q).log())
        logfg = logf + loggi

        ind = (v < logfg.exp())
        nacc = ind.sum(); nleft = nleft - nacc; mloop = mloop + 1; ntry=ntry+nleft
        if nacc > 0:
            start = accepted
            accepted += x[ind].shape[0] 
            values[start:accepted,:] = x[ind,:]

    print("Sampling efficiency:", (nsim - nleft.item())/ntry.item())

    return torch.atan2(values[:,1], values[:,0])

class BivariateVonMises(TorchDistribution):
    """
    Bivariate von Mises distribution on the torus
    
    Modality:
        If lam^2 / (k1*k2) > 1, the distribution is bimodal, otherwise unimodal.
            - This distribution is only defined for some 'slightly' bimodal cases (alpha < -7)
    
    :param torch.Tensor mu, nu: an angle in radians.
        - mu & nu can be any real number but are interpreted as 2*pi
    :param torch.Tensor k1, k2: concentration parameter
        - This distribution is only defined for k1, k2 > 0
    :param torch.Tensor lam: correlation parameter
        - Can be any real number, but is not defined for very bimodal cases
        - See 'Modality' above
        
    :param torch.Tensor w: reparameterization parameter
        - Has to be between -1 and 1
    """
    
    arg_constraints = {'mu': constraints.real, 'nu': constraints.real,
                       'k1': constraints.positive, 'k2': constraints.positive, 
                       'lam': constraints.real}
    support = constraints.real
    has_rsample = False
    
    def __init__(self, mu, nu, k1, k2, lam=None, w=None, validate_args=None):
        if (lam is None) == (w is None):
            raise ValueError("Either `lam` or `w` must be specified, but not both.")
        elif w is None:
            self.mu, self.nu, self.k1, self.k2, self.lam = broadcast_all(mu, nu, k1, k2, lam)
            alpha = (self.k1 - self.lam**2 / self.k2) / 2
            if alpha < torch.ones(1) * -7:
                raise ValueError("Distribution is too bimodal or has too high concentration while being bimodal.")
        elif lam is None:
            self.mu, self.nu, self.k1, self.k2, self.w = broadcast_all(mu, nu, k1, k2, w)
            self.lam = torch.sqrt(self.k1 * self.k2) * self.w
                
        batch_shape = self.mu.shape
        event_shape = torch.Size([2])

        self.logC = _logCinv(self.k1, self.k2, self.lam, 50)
        
        super(BivariateVonMises, self).__init__(batch_shape, event_shape, validate_args)
        
    def log_prob(self, phi, psi):
        # Actual likelihood function
        """ log Joint distribution of phi and psi """
        return (self.k1 * torch.cos(phi - self.mu) + self.k2 * torch.cos(psi - self.nu) + 
                self.lam * torch.sin(phi - self.mu) * torch.sin(psi - self.nu)) - self.logC
    
    
    
    def sample(self, sample_shape=torch.Size()):
        # Harshinder Singh, Vladimir Hnizdo, and Eugene Demchuk
        # Probabilistic model for twodependent circular variables.
        # Biometrika, 89(3):719–723, 2002.
        """
        marg: marginal distribution (using _acg_bound())
        cond: conditional distribution using a modified univariate von Mises
            - as described in Singh et al. (2002)
        """
        
        # Sampling from the marginal distribution
        marg = _acg_bound(sample_shape, self.k1, self.k2, self.lam)
        # Applying the mean angle
        marg = (marg + self.mu + math.pi) % (2 * math.pi) - math.pi

        # Sampling from the conditional distribution
        alpha = torch.sqrt( self.k2**2 + self.lam**2 * torch.sin(marg - self.mu)**2 )
        beta = torch.atan( self.lam / self.k2 * torch.sin(marg - self.mu) )
        cond = pyro.sample("psi", VonMises(self.nu + beta, alpha))
            
        return torch.stack( (marg, cond) )
    
    
    def expand(self, batch_shape):
        
        try:
            return super(BivariateVonMises, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            mu = self.mu.expand(batch_shape)
            nu = self.nu.expand(batch_shape)
            k1 = self.k1.expand(batch_shape)
            k2 = self.k2.expand(batch_shape)
            lam = self.lam.expand(batch_shape)
            
            return type(self)(mu, nu, k1, k2, lam, validate_args=validate_args)
    

