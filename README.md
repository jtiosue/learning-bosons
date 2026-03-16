# LearningBosons

Implementation of our bosonic learning algorithms from our paper [arXiv:2510.01610](https://arxiv.org/abs/2510.01610) (citation [below](#citation)).

## How to run

```bash
git clone https://github.com/jtiosue/learning-bosons
cd learning-bosons/heterodyne
gcc -std=c11 -O3 -shared -fPIC anneal.c overlap.c random.c pcg_basic.c -lm -o libanneal.so
cd ..
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Now you can run basic tests, `python basic_tests.py`, or the analysis, `python analysis.py`.

## Description

$r$ refers to the $(x,p)$ basis. $q$ refers to the $(a, a^\dag)$ basis. For $n$ modes, `methods.q_from_r_unitary(n)` is the unitary that takes you between these two bases. When sampling heterodyne, you naturally can compute the expectation value of anti-normal ordered operators. Then, you can build up the expectation values of all operators made out of $a$'s and $a^\dag$'s. Then finally you can use the aforementioned unitary to rotate back to the $(x, p)$ basis to compute $\Lambda^{(t)}_{i_1,\dots,i_t,j_1,\dots,j_t} = \langle r_{i_1}\dots r_{i_t}r_{j_1} \dots r_{i_t}\rangle$. This whole process of taking heterodyne samples and creating $\Lambda^{(1)}$ and $\Lambda^{(2)}$ is done in `methods.estimate_Lambda_from_samples`. To go from $\Lambda^{(t)}$ to $\sigma^{(t)}_{i_1,\dots,i_t,j_1,\dots,j_t} = \langle a_{i_1}\dots a_{i_t} a_{j_1}^\dag \dots a_{j_t}^\dag \rangle$, we use `methods.sigma_from_Lambda`, where we simply rotate back to the $q$ basis and then take a submatrix.

Given $\Lambda^{(t)}$ and $\sigma^{(t)}$, the three `algorithm#.py` files implement our learning algorithms. Namely:
- `algorithm1.py` implements Algorithm 1 from the paper, which is the function `findV`. Given the fourth moment matrix $\sigma^{(2)}$ of a state $\mathcal U_W \ket{b,\dots, b}$ for a unitary $W$, `findV(sigma2, b)` returns a matrix $V$ such that $V$ equals $W$ up to unimportant permutations and phases.
- `algorithm2.py` implements Algorithm 2 from the paper, which is the function `findVFock`. Given the second and fourth moment matrices $\sigma^{(1)}, \sigma^{(2)}$ of a state $\mathcal U_W \ket{f_1,\dots,f_n}$ for a unitary $W$, `findVFock(sigma1, sigma2)` returns a matrix $V$ and a vector $\bm g = (g_1,\dots, g_n)$ such that $V$ equals $W$ up to unimportant permutations and phases, and $\bm g$ is a permutation of $\bm f$. 
- `algorithm3.py` implements Algorithm 3 from the paper, which is the function `findQ`. Given the second and fourth moment matrices $\Lambda^{(1)}, \Lambda^{(2)}$ of a state $\mathcal U_S \ket{f_1,\dots,f_n}$ for a symplectic $S$, `findQ(Lambda1, Lambda2)` returns a matrix $Q$ and a vector $\bm g = (g_1,\dots, g_n)$ such that $Q$ is close to $S$, up to unimportant permutations and phases, and $\bm g$ is a permutation of $\bm f$. 


Finally, what remains is to understand how we simulate heterodyne measurements from our states $\mathcal U_S \ket{\bm f}$. This is done in `heterodyne/`. Using the ability to compute heterodyne probability $p(\alpha) = |\bra{\alpha} \mathcal U_S \ket{\bm f}|^2$, we run a Metropolis-Hastings algorithm in order to simulate sampling from $p(\alpha)$. This is done in `heterodyne/anneal.c`. So, given an $\alpha$, how do we actually compute $p(\alpha)$? This is done in `heterodyne/overlap.c`, which I will describe below. Finally, `heterodyne/anneal_wrapper.py` then wraps the C functionality so that it is usable in Python.


### Computing the overlap

To compute the heterodyne probability distribution, we use the overlap calculation written [here](https://github.com/XanaduAI/fockgaussian) and published [here](https://arxiv.org/pdf/1811.09597) by N. Quesada. However, the overall simulated annealing algorithm was quite slow so I wanted to implement it in C. I wrote the `anneal.c` file, and *GPT 5.4 wrote the `overlap.c` file*, as well as the `anneal_wrapper.py` file. Everything not in the `heterodyne/` folder was written entirely by me.

In order to get GPT 5.4 to code a C implementation of the overlap calculation, I gave it [this file by Quesada](https://github.com/XanaduAI/fockgaussian/blob/master/fockgaussian.py) and simply asked it to be coded in C. Once it was complete, I tested the C and Quesada's implementation against each other to ensure that they gave the same result, and they do.


# Citation

If you use this code in your research, please cite our paper:

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

Additionally, you should cite [Quesada's paper](https://pubs.aip.org/aip/jcp/article-abstract/150/16/164113/198316/Franck-Condon-factors-by-counting-perfect?redirectedFrom=fulltext), since I use his implementation to compute the heterodyne probabilities.