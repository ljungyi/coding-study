import numpy as np


rng = np.random.default_rng(0)

x = (
    1e5 +
    rng.normal(
        0,
        1,
        size=100_000
    )
).astype(np.float32)


def naive_variance(x):

    mean = np.mean(
        x,
        dtype=np.float32
    )

    mean_sq = np.mean(
        x * x,
        dtype=np.float32
    )

    return mean_sq - mean * mean


def welford_variance(x):

    mean = np.float32(0.0)
    M2 = np.float32(0.0)

    n = 0

    for value in x:

       n=n+1
       sigma1=value-mean
       mean=mean+sigma1/n
       sigma2=value-mean
       M2=M2+sigma1*sigma2

    return M2 / np.float32(n)


print(
    "naive:",
    naive_variance(x)
)

print(
    "Welford:",
    welford_variance(x)
)

print(
    "float64 reference:",
    np.var(
        x.astype(np.float64)
    )
)