# Prompt: `heterodyne.py`

Write a complete Python program named `heterodyne.py`. Let $x_1,\dots, x_n$ be the position operators for a bosonic system of $n$ modes, and $p_1,\dots,p_n$ the momentum operators. Let $U_S$ denote an arbitrary Gaussian unitary specified by an $2n\times 2n$ symplectic matrix $S$. Let $f = (f_1,\dots, f_n)$ denote a Fock state.

Create the public function `sample_heterodyne(S, f, nsamples, **params)` that, given $S$, $f$, an integer specifying the number of samples, and keyword parameters (described below), outputs `nsamples` samples from the probability distribution coming from heterodyne measurements on the state $U_S \ket f$. Thus, the output of `sample_heterodyne(S, f, nsamples, **params)` should be an `nsamples` by `n` complex array.

The keyword arguments `**params` specifies whatever other parameters you think is necessary. In particular, exactly simulating heterodyne measurements is not possible, and therefore there should be some cutoff parameter.

Finally, *avoid using StrawberryFields*. StrawberryFields is far too slow and it lacks the functionality natively. Feel free the write the entire program using `thewalrus`, `scipy`, and `numpy`, and Python and/or C can be used.

This prompt is intentionally concrete. Follow it exactly unless doing so would introduce a bug; if you must deviate, document the deviation in comments in the code.
