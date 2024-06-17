from inc.benchmarks import *
from inc.evaluator import *

args = Dict(
    seed = 123456,
    save_dir = 'output',
    benchmark = 'attack',
    max_steps = 4000,
    max_queries = 20000,
    
    lr = 0.01,
    num_samples = 4,
)

class Optimizer(BaseOptimizer):
    NAME = 'GLD'
    def init(self):
        self.d = self.bm.func.d
        self.s = self.bm.func.s
    def step(self):
        x0 = self.x
        y0 = self.y
        x_list = [x0]
        y_list = [y0]
        for k in range(args.num_samples):
            x1 = x0 + np.random.normal(scale = args.lr / (2 ** k), size = x0.shape)
            y1 = self.bm.query1(x1)
            x_list.append(x1)
            y_list.append(y1)
        return self.update(x_list[np.argmin(y_list)])

bm_list = get_benchmark(args.benchmark)
for bm in bm_list:
    evaluator = Evaluator(bm = bm, opt_cls = Optimizer, max_steps = args.max_steps, max_queries = args.max_queries, seed = args.seed, save_dir = args.save_dir)
    evaluator.main()