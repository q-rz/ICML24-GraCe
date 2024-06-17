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
    kappa = 1.,
)

class Optimizer(BaseOptimizer):
    NAME = 'SZOHT'
    def init(self):
        self.d = self.bm.func.d
        self.s = self.bm.func.s
    def step(self):
        x0 = self.x
        y0 = self.y
        keep = int(self.s * args.kappa)
        Z = np.zeros((args.num_samples, self.d))
        for i in range(args.num_samples):
            Z[i, np.random.choice(self.d, size = keep, replace = False)] = np.random.normal(size = keep)
        U = Z / np.sqrt(np.square(Z).sum(axis = 1, keepdims = True))
        yU = self.bm.query(x0 + args.eps * U)
        grad = (((yU - y0)[:, None] / args.eps) * U).mean(axis = 0) * self.d
        x1 = x0 - args.lr * grad
        x1[np.argsort(np.abs(x1))[: -keep]] = 0.
        return self.update(x1)

bm_list = get_benchmark(args.benchmark)
for bm in bm_list:
    evaluator = Evaluator(bm = bm, opt_cls = Optimizer, max_steps = args.max_steps, max_queries = args.max_queries, seed = args.seed, save_dir = args.save_dir)
    evaluator.main()