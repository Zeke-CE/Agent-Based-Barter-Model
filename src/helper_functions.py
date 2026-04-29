import scipy as sp
import numpy as np
import quantecon as qe




def get_truncated_normal(mean=0.5, sd=0.1, low=0, high=1, rng=None):
    '''
    
    Get truncated normal distribution samples
    -- Mean: mean of distribution
    -- SD: standard deviation of distribution
    -- Low: lower bound of distribution
    -- High: upper bound of distribution
    -- RNG: random state for reproducibility
    
    '''

    a = (low - mean) / sd
    b = (high - mean) / sd


    return sp.stats.truncnorm(a, b, loc=mean, scale=sd).rvs(random_state=rng)


def initialize_bundle(n_goods, mode="uniform", rng=None, low=5.0, high=25.0):
    """
    Generate initial bundles of goods for agents
    Params:
    -- n_goods : integer number of goods
    -- mode : String specifying how to initialize bundles
        - "single_random": bundles contain a random amount of one good -- creates a lot of initial death, especially when the resource peaks don't overlap
        - "uniform":       bundles contain random amounts of each good -- less initial death
        - "empty":         bundles contain no goods -- creates a lot of initial death, especially when the resource peaks don't overlap
    rng : np.random.Generator -- optional random number generator for reproducibility
    low : Lower bound for truncated normal draw
    high : Upper bound for truncated normal draw

    Returns:
    -- np.ndarray: Shape (n_goods,).

    """
    if rng is None: #Set up random number generator if not provided
        rng = np.random.default_rng()


    if mode == "single_random": #Initialize agents with a random amount of one good. Creates a lot of initial death, especially when the reasource peaks don't overlap

        bundle = np.zeros(n_goods)

        g = rng.integers(n_goods)

        bundle[g] = rng.uniform(low, high)

        return bundle
    
    elif mode == "uniform": #Initialize agents with random amount of each good. Less initial death

        return rng.uniform(low, high, size=n_goods)
    
    elif mode == "empty": #No amount of either good

        return np.zeros(n_goods)
    
    raise ValueError(f"Unknown mode: {mode}, chose one of 'single_random', 'uniform', or 'empty'")


# def gini(x):
#     """Standard Gini coefficient for a 1-D array of non-negative wealth values.

#     Parameters
#     ----------
#     x : array-like
#         Non-negative wealth values.

#     Returns
#     -------
#     float
#         Gini coefficient in [0, 1].
#     """
#     x = np.asarray(x, dtype=float)
#     if x.size == 0 or np.all(x == 0):
#         return 0.0
    
#     # Source - https://stackoverflow.com/a/61154922
#     # Posted by Ulf Aslak, modified by community. See post 'Timeline' for change history
#     # Retrieved 2026-04-29, License - CC BY-SA 4.0

#     diffsum = 0
#     for i, xi in enumerate(x[:-1], 1):
#         diffsum += np.sum(np.abs(xi - x[i:]))
#     return diffsum / (len(x)**2 * np.mean(x))


def generate_lorenz_and_gini(weath_series):
    """
    source: https://python.quantecon.org/wealth_dynamics.html
    Generate the Lorenz curve data and gini coefficient
    Params:
    -- weath_series : array-like of wealth values for agents

    Returns:
    -- gini_coefficient : float in [0, 1]
    -- lorenz_curve : array of cumulative wealth shares corresponding to cumulative population shares

    """

    if len(weath_series) == 0 or weath_series.sum() == 0:
        return 0, np.zeros(len(weath_series))

    w = np.array(weath_series)
    w = np.sort(w)  # Sort wealth values in ascending order for Lorenz curve

    return qe.gini_coefficient(w)#, qe.lorenz_curve(w)