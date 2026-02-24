from typing import Literal

import numpy as np
from scipy.stats import truncnorm


def FS(
    inj_id: int,
    dPo: float,
    gammaW: float,
    dP: float,
    z: float,
    alpha: float,
    Ko: float,
    Ka: float,
    theta: float,
    phi: float,
    c: float,
    *gamma_thickness: tuple[float, float],
):
    gammas, thickness = zip(*gamma_thickness)
    gammas = np.array(gammas)[:inj_id]
    thickness = np.broadcast_to(
        np.array(thickness)[:inj_id][:, None, None, None], gammas.shape
    )
    SvEff0 = (
        (gammas * thickness).sum()
        + gammas[-1] * (z - thickness.sum())
        - alpha * (gammaW * z + dPo)
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
