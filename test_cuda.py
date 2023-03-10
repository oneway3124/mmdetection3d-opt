import numpy as np
from numba import cuda

@cuda.jit
def vectorAdd(a, b, c):
    i = cuda.threadIdx.x
    if i < a.size:
        c[i] = a[i] + b[i]

if __name__ == '__main__':
    N = 1000
    a = np.arange(N).astype(np.int32)
    b = (np.arange(N) ** 2).astype(np.int32)
    c = np.zeros_like(a)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    threads_per_block = 32
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    vectorAdd[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    d_c.copy_to_host(c)

    print(c)
