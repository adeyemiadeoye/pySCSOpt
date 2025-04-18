import numpy as np
import pyscsopt as scs
from pyscsopt.regularizers import PHuberSmootherL1L2, LogExpSmootherIndBox, ExponentialSmootherIndBox
from pyscsopt.algorithms import ProxGradient, ProxLQNSCORE

import numpy as np
import jax.numpy as jnp
import jax.random as random

np.random.seed(1234)

nvar = 10
Q = jnp.tril(random.uniform(random.PRNGKey(42), (nvar, nvar)))
Q = Q + Q.T - jnp.diag(Q.diagonal())
Q = Q + nvar * jnp.eye(nvar)
c = jnp.ones(nvar)

def f(x):
    return 0.5 * jnp.dot(x, jnp.dot(Q, x)) + jnp.dot(c, x)

x0 = random.uniform(random.PRNGKey(42), (nvar,))
lbda = 0.5
reg_name = "indbox"
mu = 0.1
C_set = (-5.0, 5.0)
hmu = LogExpSmootherIndBox(C_set, mu)
problem = scs.Problem(x0, f, lbda, C_set=C_set)

method_pg = ProxGradient(use_prox=True, ss_type=1)
sol_pg = scs.iterate(method_pg, problem, reg_name, hmu, verbose=1, max_epoch=200)

method_lqn = ProxLQNSCORE(use_prox=True, ss_type=2, m=10)
sol_lqn = scs.iterate(method_lqn, problem, reg_name, hmu, verbose=1, max_epoch=200)

print("=" * 50)
print("ProxGradient:")
print("N. Iterations:", sol_pg.epochs)
print("Solution x:", sol_pg.x)
print("Objective values:", sol_pg.obj[-1])
print("Function values:", sol_pg.fval[-1])
print("gradient norms:", sol_pg.grad_norms[-1])
print("=" * 50)
print("ProxLQNSCORE:")
print("N. Iterations:", sol_lqn.epochs)
print("Solution x:", sol_lqn.x)
print("Objective values:", sol_lqn.obj[-1])
print("Function values:", sol_lqn.fval[-1])
print("gradient norms:", sol_lqn.grad_norms[-1])