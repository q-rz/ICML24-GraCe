# GraCe (ICML 2024)

Gradient compressed sensing: A query-efficient gradient estimator for high-dimensional zeroth-order optimization

```bibtex
@inproceedings{qiu2024gradient,
  title={Gradient compressed sensing: A query-efficient gradient estimator for high-dimensional zeroth-order optimization},
  author={Ruizhong Qiu and Hanghang Tong},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024},
  number={235}
}
```

<center><img src="https://raw.githubusercontent.com/q-rz/ICML24-GraCe/main/fig-landscape.svg" alt="Illustration of sparse and dense gradients" /></center>

## Dependencies

Our code was tested under the following dependencies:

- Python 3.10.12
- Tqdm 4.66.1
- NumPy 1.24.3
- SkLearn 1.2.2
- NetworkX 3.1
- Matplotlib 3.7.4
- Seaborn 0.12.2

## Usage

We provide our implementation for our method GraCe as well as baseline methods, where `test_{method}.py` is the code for the corresponding `{method}` (e.g., `grace`). 

Before running the code, please configure via the `args` variable in the code. Every method has the following arguments:
- `benchmark`: the benchmark function (options: `distance`/`magnitude`/`attack`);
- `max_steps`: maximum number $T$ of steps for the optimization process;
- `max_queries`: maximum number of queries that can be used by the optimization method;
- `lr`: step size $\eta$ (a.k.a. learning rate) for each optimization step;
- `eps`: finite difference $\epsilon$ for zeroth-order gradient estimation.

For our method GraCe, there are a few additional arguments available:
- `reps`: number $m$ of repeats in GraCe;
- `gamma`: group size parameter $\gamma$ (the group size is $n:=\big\lfloor\frac{\gamma d}s\big\rfloor$);
- `div`: initial division parameter $D\_1$ (the division schedule $\\{D\_r\\}\_{r\ge1}$ is given by $D\_{r+1}:=\lfloor D\_r^{3/2}\rfloor$).
