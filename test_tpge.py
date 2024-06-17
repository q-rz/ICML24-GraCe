from inc.benchmarks import *
from inc.evaluator import *

args = Dict(
    seed = 123456,
    save_dir = 'output',
    benchmark = 'attack',
    max_steps = 6667,
    max_queries = 20000,
    
    lr = 0.002,
    eps = 1e-6,
)

class Optimizer(BaseOptimizer):
    NAME = 'TPGE'
    def init(self):
        self.d = self.bm.func.d
        self.s = self.bm.func.s
    def step(self):
        x0 = self.x
        y0 = self.y
        Z1 = np.random.normal(size = (1, self.d))
        Z2 = np.random.normal(size = (1, self.d))
        y1 = self.bm.query(x0 + args.eps * Z1)
        y2 = self.bm.query(x0 + args.eps * Z1 + args.eps * Z2)
        grad = (((y2 - y1)[:, None] / args.eps) * Z2).mean(axis = 0)
        return self.update(x0 - args.lr * grad)

bm_list = get_benchmark(args.benchmark)
for bm in bm_list:
    evaluator = Evaluator(bm = bm, opt_cls = Optimizer, max_steps = args.max_steps, max_queries = args.max_queries, seed = args.seed, save_dir = args.save_dir)
    evaluator.main()