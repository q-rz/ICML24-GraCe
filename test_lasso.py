from inc.benchmarks import *
from inc.evaluator import *

args = Dict(
    seed = 123456,
    save_dir = 'output',
    benchmark = 'attack',
    max_steps = 100,
    max_queries = 20000,
    
    lr = 0.2,
    eps = 1e-6,
    num_samples = 199,
    lasso_max_iter = 100,
)

from sklearn.linear_model import Lasso

class Optimizer(BaseOptimizer):
    NAME = 'LASSO'
    def init(self):
        self.d = self.bm.func.d
        self.s = self.bm.func.s
    def step(self):
        x = self.x
        Z = 2 * (np.random.rand(args.num_samples, self.d) > 0.5) - 1
        yZ = self.bm.query(x + args.eps * Z) / args.eps
        lasso = Lasso(alpha = self.s / (2 * args.eps), max_iter = args.lasso_max_iter)
        lasso.fit(Z, yZ)
        mu = lasso.intercept_
        g_hat = lasso.coef_
        g_tld = g_hat + (Z * (yZ - (Z * g_hat).sum(axis = 1) - mu)[:, None]).mean(axis = 0)
        return self.update(x - args.lr * g_tld)

bm_list = get_benchmark(args.benchmark)
for bm in bm_list:
    evaluator = Evaluator(bm = bm, opt_cls = Optimizer, max_steps = args.max_steps, max_queries = args.max_queries, seed = args.seed, save_dir = args.save_dir)
    evaluator.main()