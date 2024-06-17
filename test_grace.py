from inc.benchmarks import *
from inc.evaluator import *

args = Dict(
    seed = 123456,
    save_dir = 'output',
    benchmark = 'attack',
    max_steps = 100,
    max_queries = 20000,
    
    lr = 0.5,
    eps = 1e-6,
    reps = 1,
    gamma = 0.7,
    div = 10,
)

class Optimizer(BaseOptimizer):
    NAME = 'GraCe'
    def init(self):
        self.d = self.bm.func.d
        self.s = self.bm.func.s
    def step(self):
        x0, y0 = self.x, self.y
        cands = []
        for rep in range(args.reps):
            perm = np.random.permutation(self.d)
            group = int(args.gamma * self.d / self.s)
            for i in range(0, self.d, group):
                j = min(i + group, self.d)
                coos = perm[i : j]
                div = args.div
                while len(coos) > 2:
                    siz = len(coos)
                    block = (siz + div - 1) // div # i.e., ceil(siz / div)
                    sig = np.random.randint(0, 2, size = coos.shape) * 2 - 1
                    pos = np.arange(block + 1, block + siz + 1, dtype = np.int32) // block
                    u = x0.copy()
                    v = x0.copy()
                    for k in range(siz):
                        u[coos[k]] += args.eps * sig[k]
                        v[coos[k]] += args.eps * sig[k] * pos[k]
                    yu = self.bm.query1(u)
                    if yu == y0:
                        coos = coos[: 0] # failed
                    else:
                        yv = self.bm.query1(v)
                        targ = max(0, min(pos[-1], round((yv - y0) / (yu - y0))))
                        coos = coos[pos == targ]
                    div = int(div ** 1.5)
                cands.extend(coos.tolist())
        cands = np.unique(cands)
        y1 = []
        for i in cands:
            x1 = x0.copy()
            x1[i] += args.eps
            y1.append(self.bm.query1(x1))
        y1 = np.array(y1)
        grad = np.zeros_like(x0)
        if len(cands):
            grad[cands] = (y1 - y0) / args.eps
            return self.update(x0 - args.lr * grad)
        else:
            return self.x, self.y

bm_list = get_benchmark(args.benchmark)
for bm in bm_list:
    evaluator = Evaluator(bm = bm, opt_cls = Optimizer, max_steps = args.max_steps, max_queries = args.max_queries, seed = args.seed, save_dir = args.save_dir)
    evaluator.main()