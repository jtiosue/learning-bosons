# LearningBosons

Implementation of our bosonic learning algorithms from our paper [arXiv:2510.01610](https://arxiv.org/abs/2510.01610) (citation [below](#citation)).

## How to run

```bash
git clone https://github.com/jtiosue/learning-bosons
cd learning-bosons/
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Now you can run the analysis with `python analysis.py` (see under the `if __name___ ...` for different things to run).

## Description

$r$ refers to the $(x,p)$ basis. $q$ refers to the $(a, a^\dagger)$ basis. For $n$ modes, `methods.q_from_r_unitary(n)` is the unitary that takes you between these two bases. When sampling heterodyne, you naturally can compute the expectation value of anti-normal ordered operators (see our paper). Then, you can build up the expectation values of all operators made out of $a$'s and $a^\dagger$'s. Then finally you can use the aforementioned unitary to rotate back to the $(x, p)$ basis to compute 
```math
\Lambda^{(t)}_{i_1,\dots,i_t,j_1,\dots,j_t} = \langle r_{i_1}\dots r_{i_t}r_{j_1} \dots r_{j_t}\rangle.
```
This whole process of taking heterodyne samples and creating $\Lambda^{(1)}$ and $\Lambda^{(2)}$ is done in `methods.estimate_Lambda_from_samples`. To go from $\Lambda^{(t)}$ to 
```math
\sigma^{(t)}_{i_1,\dots,i_t,j_1,\dots,j_t} = \langle a_{i_1}\dots a_{i_t} a_{j_1}^\dagger \dots a_{j_t}^\dagger \rangle,
```
use `methods.sigma_from_Lambda`, where we simply rotate back to the $q$ basis and then take a submatrix.

Given $\Lambda^{(t)}$ and $\sigma^{(t)}$, the three `algorithm#.py` files implement our learning algorithms. Namely:
- `algorithm1.py` implements Algorithm 1 from the paper, which is the function `findV`. Given the fourth moment matrix $\sigma^{(2)}$ of a state $\mathcal U_W \ket{b,\dots, b}$ for a unitary $W$, `findV(sigma2, b)` returns a matrix $V$ such that $V$ equals $W$ up to unimportant permutations and phases.
- `algorithm2.py` implements Algorithm 2 from the paper, which is the function `findVFock`. Given the second and fourth moment matrices $\sigma^{(1)}, \sigma^{(2)}$ of a state $\mathcal U_W \ket{f_1,\dots,f_n}$ for a unitary $W$, `findVFock(sigma1, sigma2)` returns a matrix $V$ and a vector $\boldsymbol g = (g_1,\dots, g_n)$ such that $V$ equals $W$ up to unimportant permutations and phases, and $\boldsymbol g$ is a permutation of $\boldsymbol f$. 
- `algorithm3.py` implements Algorithm 3 from the paper, which is the function `findQ`. Given the second and fourth moment matrices $\Lambda^{(1)}, \Lambda^{(2)}$ of a state $\mathcal U_S \ket{f_1,\dots,f_n}$ for a symplectic $S$, `findQ(Lambda1, Lambda2)` returns a matrix $Q$ and a vector $\boldsymbol g = (g_1,\dots, g_n)$ such that $Q$ is close to $S$, up to unimportant permutations and phases, and $\boldsymbol g$ is a permutation of $\boldsymbol f$. 


Finally, what remains is to understand how we simulate heterodyne measurements from our states $\mathcal U_S \ket{\boldsymbol f}$. This is done in `sample.py`. 
I actually did not implement this generally. Instead, I only performed numerical simulations for algorithm 1 for learning the state $\mathcal U_W \ket{1\dots 1}$.
Simulating heterodyne sampling from this state is done in `sample.sample_heterodyne_passive_fock1`. Simulating $\mathcal U_W \ket{\boldsymbol f}$ is a straightforward extension, but I didn't implement it.
Simulating heterodyne sampling from $\mathcal U_S \ket{\boldsymbol f}$ is not clear to me how to do it efficiently (I think it cannot be done efficiently actually).

In the `analysis.py` file, you can see how to run/test algorithms 1, 2, and 3. However, because only `sample.sample_heterodyne_passive_fock1` is implemented and not `sample.sample_heterodyne_passive_fock` or `sample.sample_heterodyne_Gaussian_fock`, currently you cannot test algorithm 2 or 3 (even though algorithm 2 and 3 are themselves implemented in `algorithm2.py` and `algorithm3.py`).
`sample.sample_heterodyne_passive_fock` would be very easy to implement to allow for testing of algorithm 2, but I did not do it.

# Citation

If you use this code in your research, or any of the ideas from our paper, please cite our paper:

```bibtex
@misc{iosue2025higher-moment-t,
	archiveprefix = {arXiv},
	author = {Joseph T. Iosue and Yu-Xin Wang and Ishaun Datta and Soumik Ghosh and Changhun Oh and Bill Fefferman and Alexey V. Gorshkov},
	doi = {10.48550/arXiv.2510.01610},
	eprint = {2510.01610},
	month = {10},
	primaryclass = {quant-ph},
	title = {Higher moment theory and learnability of bosonic states},
	year = {2025}}
```
