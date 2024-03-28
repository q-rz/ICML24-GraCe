from inc.header import *

class Benchmark:
    def __init__(self, func):
        self.func = func
        self.q = 0
        self.x0 = self.func.x0
        self.y0 = self.query1(self.x0)
    def query1(self, x):
        self.q += 1
        if not isinstance(x, np.ndarray):
            xi = np.array(x)
        return self.func(x)
    def query(self, x):
        return np.stack([self.query1(xi) for xi in x], axis = 0)
    def save(self, save_dir):
        return self.func.save(save_dir = save_dir)
    @property
    def n_queries(self):
        return self.q
    @property
    def name(self):
        return self.func.name

class Func:
    BASE_SEED = 998244353
    NAME = None
    x0 = None
    def __init__(self, tid: int):
        self.tid = tid
        set_seed(self.BASE_SEED ^ self.tid)
    def __call__(self, x: np.ndarray):
        raise NotImplementedError
    def save(self, save_dir):
        raise NotImplementedError
    @property
    def name(self):
        return f'{self.NAME}{self.tid}'

class Distance(Func):
    NAME = 'Distance'
    def __init__(self, tid, d = 10000, s = 10):
        super().__init__(tid = tid)
        self.d = d
        self.s = s
        self.coef = np.random.rand(self.d).astype(np.float64)
        self.coo = np.random.choice(self.d, size = self.s, replace = False)
        self.v = np.zeros(shape = self.d, dtype = np.float64)
        self.v[self.coo] = np.random.rand(self.s).astype(np.float64)
        self.x0 = np.zeros(shape = self.d, dtype = np.float64)
    def __call__(self, x):
        return (self.coef * np.square(x - self.v)).sum()
    def save(self, save_dir):
        save_fname = osp.join(save_dir, f'{self.name}.pkl')
        with open(save_fname, 'wb') as fo:
            pkl.dump(dict(tid = self.tid, d = self.d, s = self.s, coef = self.coef.tolist(), v = self.v.tolist(), x0 = self.x0.tolist()), fo)
        return save_fname

class Magnitude(Func):
    NAME = 'Magnitude'
    def __init__(self, tid, d = 10000, s = 5, coef = 0.1):
        super().__init__(tid = tid)
        self.d = d
        self.s = s
        self.coef = coef
        self.coo = np.random.choice(self.d, size = self.s, replace = False)
        self.x0 = np.zeros(self.d, dtype = np.float64)
        self.x0[self.coo] = 0.2 * ((np.random.rand(self.s) < 0.5).astype(np.float64) * 2 - 1)
        #self.x0 = 0.002 * ((np.random.rand(self.d) < 0.5).astype(np.float64) * 2 - 1)
        #self.x0[self.coo] *= 100
    def __call__(self, x):
        z = np.partition(-np.square(x), kth = self.s)
        return self.s + np.tanh(z[: self.s]).sum() - self.coef * np.tanh(z[self.s :]).sum()
    def save(self, save_dir):
        save_fname = osp.join(save_dir, f'{self.name}.pkl')
        with open(save_fname, 'wb') as fo:
            pkl.dump(dict(tid = self.tid, d = self.d, s = self.s, coef = self.coef, coo = self.coo.tolist(), x0 = self.x0.tolist()), fo)
        return save_fname

class Attack(Func):
    NAME = 'Attack'
    def __init__(self, tid, s = 30, u = 0, v = 1, hops = 4, eps = 1e-8, reg = 100., fpath = 'data/football.gml'):
        super().__init__(tid = tid)
        with open(fpath) as fi:
            self.G = nx.parse_gml(fi.read().split('\n'))
        self.n_nodes = self.G.number_of_nodes()
        self.d = self.n_nodes ** 2
        self.s = s
        self.A = nx.adjacency_matrix(self.G).todense().astype(np.float64)
        self.u = u
        self.v = v
        self.hops = hops
        self.eps = eps
        self.reg = reg
        self.x0 = np.zeros(self.d, dtype = np.float64)
    def __call__(self, x):
        dA = np.abs(x).reshape(self.A.shape)
        A = np.maximum(self.A * (1. - dA) + (1. - self.A) * dA, 0.)
        d = np.sqrt(A.sum(axis = 1))
        d = np.maximum(d, self.eps)
        Asym = A / (d[:, None] * d[None, :])
        y = z = Asym[self.u]
        for k in range(1, self.hops):
            z = Asym @ z
            y = y + z
        return y[self.v] + self.reg * np.square(dA).mean()
    def save(self, save_dir):
        save_fname = osp.join(save_dir, f'{self.name}.pkl')
        with open(save_fname, 'wb') as fo:
            pkl.dump(dict(tid = self.tid, d = self.d, s = self.s, u = self.u, v = self.v, hops = self.hops, eps = self.eps, reg = self.reg), fo)
        return save_fname

FUNCS_CLS = Dict(
    distance = Distance,
    magnitude = Magnitude,
    attack = Attack,
)

def get_benchmark(name, tests = 10):
    cls = FUNCS_CLS[name]
    return [Benchmark(func = cls(tid = tid)) for tid in range(tests)]
