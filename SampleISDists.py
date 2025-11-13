import numpy as np
from scipy.stats import norm

from cholcov import cholcov


def SampleISDists(
    nsamps: int,
    sigmas: float,
    mus: float,
    offsets: float,
    islastdecade: float,
    corris: float,
    seed: int = 1234,
) -> float:
    """Produces samples from the fitted ice sheet distributions

    Parameters:
        nsamps (int): Number of samples to produce
        sigmas (float): 1D array of sigmas for each ice sheet component
        mus (float): 1D array of mus for each ice sheet component
        offsets (float): 1D array of offsets for each ice sheet component
        islastdecade (float): Ice sheet accelerations(?) for the last decade
        corris (float): Correlation structure across ice sheets
        seed (int): Seed for the random number generator

    Return:
        An array of dimensions [nsamps, length(sigmas)] of samples of ice sheet melts
        accelerations.

    Note: The lengths of 'sigmas', 'mus', 'offsets', and 'islastdecade' must be the same.
    Also, 'corris' must be a square matrix of the same size as the length of the previous
    parameters.
    """

    # Evenly sample an inverse normal distribution
    np.random.seed(seed)
    x = np.linspace(0, 1, nsamps + 2)[1 : (nsamps + 1)]
    norm_inv = norm.ppf(x)

    # Find the Cholesky Covariance
    covis = np.dot(np.dot(np.diag(sigmas), corris), np.diag(sigmas))
    T = cholcov(covis)

    # Build a matrix of permutated norm_inv values
    norm_inv_perm = np.full((nsamps, sigmas.shape[0]), np.nan)
    for i in range(sigmas.shape[0]):
        norm_inv_perm[:, i] = np.random.permutation(norm_inv)

    # Create the correlated samples for the ice sheet accelerations
    sampeps = np.dot(norm_inv_perm, T)
    sampisrates = offsets + np.exp((sampeps + mus))
    sampisaccel = (sampisrates - islastdecade) / (2100 - 2011)

    # Return the sampled accelerations
    return sampisaccel
