from inc.benchmarks import *
from inc.evaluator import *

args = Dict(
    seed = 123456,
    save_dir = 'output',
    benchmark = 'attack',
    max_steps = 10000,
    max_queries = 20000,
    
    lr = 0.002,
    eps = 1e-6,
    beta1 = 0.9,
    beta2 = 0.5,
)

class Optimizer(BaseOptimizer):
    NAME = 'ZO-AdaMM'
    def init(self):
        self.d = self.bm.func.d
        self.s = self.bm.func.s
        self._m = np.zeros(self.d, dtype = np.float64)
        self._v = np.zeros(self.d, dtype = np.float64)
        self._V = np.zeros(self.d, dtype = np.float64)
    def step(self):
        x0 = self.x
        y0 = self.y
        Z = np.random.normal(size = (1, self.d))
        U = Z / np.sqrt(np.square(Z).sum(axis = 1, keepdims = True))
        yU = self.bm.query(x0 + args.eps * U)
        g = (((yU - y0)[:, None] / args.eps) * U).mean(axis = 0) * self.d
        self._m = args.beta1 * self._m + (1. - args.beta1) * g
        self._v = args.beta2 * self._v + (1. - args.beta2) * np.square(g)
        self._V = np.maximum(self._V, self._v)
        grad = np.where(self._V == 0, 0., self._m / np.sqrt(self._V))
        return self.update(x0 - args.lr * grad)

bm_list = get_benchmark(args.benchmark)
for bm in bm_list:
    evaluator = Evaluator(bm = bm, opt_cls = Optimizer, max_steps = args.max_steps, max_queries = args.max_queries, seed = args.seed, save_dir = args.save_dir)
    evaluator.main()