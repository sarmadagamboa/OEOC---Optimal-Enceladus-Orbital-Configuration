"""Crossover-count and radial orbit-error calculations."""

from __future__ import annotations

import math

from mission_config import MissionConfig


def crossover_count_scaling(number_of_orbits: float):
    """Estimates crossover count using the rough N squared scaling."""
    return number_of_orbits**2 #this essentially shows that the more the spacecraft orbits, the more crosspoints it generates. 



def crossover_uncertainty_m(radial_error_m: float, altimeter_noise_m: float):
    """Calculates a simplified version of the crossover height difference uncertainty.
    You have 2 big errors; SSpacecraft orbit determination error (radial_error_m) and the altimeter noise error (altimeter_noise_m)
    """
    return math.sqrt(2.0 * radial_error_m**2 + 2.0 * altimeter_noise_m**2)


def self_calibration_estimate(
    raw_radial_error_m: float,
    altimeter_noise_m: float,
    number_of_orbits: float,
    crossover_count: float,
    systematic_floor_m: tuple[float, float],
):
    """Estimates crossover least-squares redundancy and radial-error reduction.
   It calculates how much your spacecraft's radial orbit error will shrink when 
   you run a least-squares adjustment using millions of overlapping altimeter measurements 
    """
    raw_xover_sigma = crossover_uncertainty_m(raw_radial_error_m, altimeter_noise_m)
    unknowns = number_of_orbits + 1.0
    redundancy = crossover_count / unknowns
    random_only_arc_sigma = raw_xover_sigma / math.sqrt(redundancy)

    floor_low, floor_high = systematic_floor_m
    effective_low = max(random_only_arc_sigma, floor_low)
    effective_high = max(random_only_arc_sigma, floor_high)

    return {
        "raw_xover_sigma_m": raw_xover_sigma,
        "unknown_count": unknowns,
        "redundancy": redundancy,
        "random_only_arc_sigma_m": random_only_arc_sigma,
        "effective_radial_low_m": effective_low,
        "effective_radial_high_m": effective_high,
    }


def compute_crossover_outputs(
    config: MissionConfig,
    number_of_orbits: float,
):
    """This is the main core simulation of the altimeter error budget and performance. 
    Evaluates whether or not it is capable of detecting Enceladus' tides. 

    Computes crossover count, raw uncertainty, and calibrated radial-error cases."""
    crossover_count = crossover_count_scaling(number_of_orbits)

    raw_error_rows = []
    for radial_error in config.raw_radial_error_cases_m:
        raw_error_rows.append(
            {
                "radial_error_m": radial_error,
                "xover_sigma_m": crossover_uncertainty_m(
                    radial_error,
                    config.altimeter_height_noise_m,
                ),
            }
        )

    calibrated_error_rows = []
    for radial_error in config.calibrated_radial_error_cases_m:
        xover_sigma = crossover_uncertainty_m(radial_error, config.altimeter_height_noise_m)
        calibrated_error_rows.append(
            {
                "radial_error_m": radial_error,
                "xover_sigma_m": xover_sigma,
                "snr_1m_tide": 1.0 / xover_sigma,
                "snr_2m_tide": 2.0 / xover_sigma,
            }
        )

    calibration = self_calibration_estimate(
        raw_radial_error_m=2.0,
        altimeter_noise_m=config.altimeter_height_noise_m,
        number_of_orbits=number_of_orbits,
        crossover_count=crossover_count,
        systematic_floor_m=config.calibrated_radial_error_cases_m,
    )

    return {
        "crossover_count": crossover_count,
        "raw_error_rows": raw_error_rows,
        "calibrated_error_rows": calibrated_error_rows,
        "calibration": calibration,
    }
