from inc.header import *

class BaseOptimizer:
    NAME = None
    def __init__(self, bm):
        self.bm = bm
        self.x = self.bm.x0.copy()
        self.y = self.bm.y0
        self.init()
    def update(self, x):
        self.x = x
        self.y = self.bm.query1(self.x)
        return self.x, self.y
    def init(self):
        pass
    def step(self):
        raise NotImplementedError

class Evaluator:
    def __init__(self, bm, opt_cls, max_steps, max_queries, seed, save_dir):
        self.save_dir = save_dir
        self.bm = bm
        self.opt_cls = opt_cls
        self.max_steps = int(max_steps)
        self.max_queries = int(max_queries)
        self.seed = seed
        self.x = self.bm.x0.copy()
        self.y_list = [self.bm.y0]
        self.q_list = [self.bm.n_queries]
    def save(self):
        save_fname = osp.join(self.save_dir, f'{self.bm.name}~{self.opt_cls.NAME}.pkl')
        with open(save_fname, 'wb') as fo:
            pkl.dump(dict(y = self.y_list, q = self.q_list), fo)
        print('Saved to', save_fname, flush = True)
        tt = np.arange(len(self.y_list))
        plt.figure(figsize = (12, 3))
        plt.subplot(1, 3, 1)
        sns.lineplot(x = tt, y = self.y_list)
        plt.xlabel('Step')
        plt.title('Objective Value')
        plt.subplot(1, 3, 2)
        sns.lineplot(x = tt, y = self.q_list)
        plt.title('Number of Queries')
        plt.xlabel('Step')
        plt.subplot(1, 3, 3)
        sns.scatterplot(x = self.q_list, y = self.y_list)
        plt.title('Objective Value')
        plt.xlabel('Number of Queries')
        plt.show()
        return save_fname
    def main(self):
        set_seed(self.seed ^ self.bm.func.tid)
        opt = self.opt_cls(bm = self.bm)
        tbar = trange(1, self.max_steps + 1)
        for step in tbar:
            x, y = opt.step()
            self.x = x.copy()
            self.y_list.append(y)
            self.q_list.append(self.bm.n_queries)
            tbar.set_description(f'[step={step}] q={self.q_list[-1]}, y={self.y_list[-1]}')
            if self.bm.n_queries >= self.max_queries:
                break
        return self.save()
