import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torchmetrics import ExplainedVariance
from pyro.infer import Predictive

class BayesianLassoRegression(PyroModule):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.linear = PyroModule[nn.Linear](n_features, 1)
        self.linear.bias = PyroSample(dist.Normal(0, 10).expand([1]).to_event(1))
        
        self.linear.weight = PyroSample(dist.Laplace(0, 1).expand([1,n_features]).to_event(2))
        
    def forward(self, X, y=None):
        n_samples, n_features = X.shape
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        y_hat = self.linear(X).squeeze(-1)
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(y_hat, sigma), obs=y)
        #l1_penalty = self.beta * torch.abs(self.coefs).sum()
        
        #print((-dist.Laplace(0, 1).log_prob(self.coefs).sum() + l1_penalty).shape)
        return y_hat 
