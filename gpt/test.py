from heterodyne import sample_heterodyne
import numpy as np
from scipy.stats import unitary_group
random_unitary = lambda n: unitary_group.rvs(n)


f = 1, 2, 2, 1
n = len(f)
W = random_unitary(n)
S = np.zeros((2*n,2*n))
S[:n,:n] = W.real
S[n:,n:] = W.real
S[:n,n:] = W.imag
S[n:,:n] = -W.imag
nsamples = 10000
samples = sample_heterodyne(S, f, nsamples, cutoff=sum(f)+1,max_tensor_entries=6_000_000)
print((samples.T @ samples.conj() / nsamples - W @ np.diag(1+np.array(f))  @ W.conj().T).round(3))