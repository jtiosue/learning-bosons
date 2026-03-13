# LearningBosons

- $r$ refers to (x,p). q refers to (a, adag).

# To do: implement heterodyne sampling of our states
StrawberryFields does not do it.

# To do: explain heterodyne.estimate_Lambda_from_samples
When I coded the generation of L2, I did things for normal order, but we need anti-normal order!! Need to go through and fix all of that.
The reason we need anti-normal order is because: heterodyne allows us to sample from $p(\alpha) = \frac{1}{\pi^n}|\bra\alpha\ket\psi|^2$. Let $O$ be an anti-normal ordered operator, so that $O = AB$, where $A$ consists of only $a$'s and $B$ of only $a^\dag$'s. Then, using $I = \frac{1}{\pi^n}\int d\alpha \ket\alpha\bra\alpha$, we have
$$
\begin{aligned}
\bra\psi O \ket\psi
%
&= Tr[B\ket\psi\bra\psi A] \\
%
&= \frac{1}{\pi^n} \int d\alpha \bra\alpha B \ket\psi\bra\psi A \ket\alpha  \\
%
&= \frac{1}{\pi^n} \int d\alpha B(\bar\alpha)A(\alpha) \bra\alpha \ket\psi\bra\psi \ket\alpha  \\
%
&= \int d\alpha B(\bar\alpha)A(\alpha) p(\alpha).
\end{aligned}
$$






## File descriptions
- `algorithm1.py` implements Algorithm 1 from the paper, which is the function `findV`. Given the fourth moment matrix $\sigma^{(2)}$ of a state $\mathcal U_W \ket{b,\dots, b}$ for a unitary $W$, `findV(sigma2, b)` returns a matrix $V$ such that $V$ equals $W$ up to unimportant permutations and phases.
- `algorithm2.py` implements Algorithm 2 from the paper, which is the function `findVFock`. Given the second and fourth moment matrices $\sigma^{(1)}, \sigma^{(2)}$ of a state $\mathcal U_W \ket{f_1,\dots,f_n}$ for a unitary $W$, `findVFock(sigma1, sigma2)` returns a matrix $V$ and a vector $\bm g = (g_1,\dots, g_n)$ such that $V$ equals $W$ up to unimportant permutations and phases, and $\bm g$ is a permutation of $\bm f$. 
- `algorithm3.py` implements Algorithm 3 from the paper, which is the function `findQ`. Given the second and fourth moment matrices $\Lambda^{(1)}, \Lambda^{(2)}$ of a state $\mathcal U_S \ket{f_1,\dots,f_n}$ for a symplectic $S$, `findQ(Lambda1, Lambda2)` returns a matrix $Q$ and a vector $\bm g = (g_1,\dots, g_n)$ such that $Q$ is close to $S$, up to unimportant permutations and phases, and $\bm g$ is a permutation of $\bm f$. 
-  FINISH