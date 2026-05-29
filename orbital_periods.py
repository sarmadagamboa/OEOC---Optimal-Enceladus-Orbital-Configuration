"""Orbital period and tidal phase sampling calculations."""

from __future__ import annotations

import math

from mission_config import MissionConfig, SECONDS_PER_HOUR


def orbital_period_s(semimajor_axis_m: float, mu_m3_s2: float):
    """Computes Keplerian orbital period for a circular or near-circular orbit."""
    return 2.0 * math.pi * math.sqrt(semimajor_axis_m**3 / mu_m3_s2)


def compute_orbital_period_outputs(config: MissionConfig):
    """Computes spacecraft period, tidal phase stepping, and yearly orbit count."""
    stable_period_s = orbital_period_s(config.stable_semimajor_m, config.gm_enceladus_m3_s2)
    stable_period_h = stable_period_s / SECONDS_PER_HOUR
    exact_200km_period_h = (
        orbital_period_s(config.exact_200km_semimajor_m, config.gm_enceladus_m3_s2)
        / SECONDS_PER_HOUR
    )

    return {
        "stable_period_h": stable_period_h,
        "exact_200km_period_h": exact_200km_period_h,
        "orbits_per_tidal_cycle": config.tidal_period_h / stable_period_h,
        "tidal_cycles_per_spacecraft_orbit": stable_period_h / config.tidal_period_h,
        "tidal_phase_step_deg": 360.0 * stable_period_h / config.tidal_period_h,
        "number_of_orbits": config.primary_science_days * 24.0 / stable_period_h,
    }
