from typing import Literal

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.stats import truncnorm


def SF(
    inj_id: int,
    dPo: float,
    gammaW: float,
    dP: NDArray[float64],
    z: NDArray[float64],
    alpha: NDArray[float64],
    Ko: NDArray[float64],
    Ka: NDArray[float64],
    theta: NDArray[float64],
    phi: NDArray[float64],
    c: NDArray[float64],
    gamma: list[NDArray[float64]],
    thickness: list[float],
) -> NDArray[float64]:
    SvEff0 = np.zeros(gamma[0].shape)
    for g, h in zip(gamma[:inj_id], thickness[:inj_id]):
        SvEff0 += g * h
    SvEff0 += gamma[-1] * (z - np.array(thickness[:inj_id]).sum()) - alpha * (
        gammaW * z + dPo
    )
    SvEff = SvEff0 - alpha * dP
    ShEff = Ko * SvEff0 - Ka * alpha * dP
    return (
        c + (ShEff * (np.cos(theta) ** 2) + SvEff * (np.sin(theta) ** 2)) * np.tan(phi)
    ) / ((SvEff - ShEff) * np.sin(theta) * np.cos(theta))


def dist(
    number_of_values: int,
    distribution_choice: Literal["Uniform", "Triangular", "Normal"],
    mean: float,
    std_dev: float,
    lower_value: float,
    upper_value: float,
):
    """Return a distribution based on statistics"""
    a = (lower_value - mean) / std_dev
    b = (upper_value - mean) / std_dev
    if distribution_choice == "Uniform":
        return np.random.uniform(lower_value, upper_value, number_of_values)
    elif distribution_choice == "Triangular":
        return np.random.triangular(lower_value, mean, upper_value, number_of_values)
    elif distribution_choice == "Normal":
        return np.array(truncnorm.rvs(a, b, mean, std_dev, number_of_values))
