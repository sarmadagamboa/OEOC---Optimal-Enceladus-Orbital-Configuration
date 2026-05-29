"""Ground velocity and along-track sampling calculations."""

from __future__ import annotations

import math


def circular_orbital_speed_m_s(radius_m, mu_m3_s2):
    """Computes circular orbital speed at a given distance from Enceladus center."""
    return math.sqrt(mu_m3_s2 / radius_m)


def ground_speed_m_s(radius_m, body_radius_m, mu_m3_s2):
    """Projects orbital speed down to the surface footprint speed."""
    orbital_speed = circular_orbital_speed_m_s(radius_m, mu_m3_s2)
    return orbital_speed * body_radius_m / radius_m


def compute_ground_velocity_outputs(config):
    """Computes spacecraft speed, surface footprint speed, and sample spacing."""
    nrho_periapsis_radius_m = config.radius_enceladus_m + config.nrho_periapsis_altitude_m
    stable_orbital_speed = circular_orbital_speed_m_s(
        config.stable_semimajor_m, config.gm_enceladus_m3_s2
    )
    stable_ground_speed = ground_speed_m_s(
        config.stable_semimajor_m,
        config.radius_enceladus_m,
        config.gm_enceladus_m3_s2,
    )
    nrho_periapsis_orbital_speed = circular_orbital_speed_m_s(
        nrho_periapsis_radius_m,
        config.gm_enceladus_m3_s2,
    )
    nrho_periapsis_ground_speed = ground_speed_m_s(
        nrho_periapsis_radius_m,
        config.radius_enceladus_m,
        config.gm_enceladus_m3_s2,
    )

    sample_spacing_rows = []
    for rate_hz in (1.0, 10.0, 100.0):
        sample_spacing_rows.append(
            {
                "rate_hz": rate_hz,
                "nrho_spacing_m": nrho_periapsis_ground_speed / rate_hz,
                "stable_spacing_m": stable_ground_speed / rate_hz,
            }
        )

    return {
        "stable_orbital_speed_m_s": stable_orbital_speed,
        "stable_ground_speed_m_s": stable_ground_speed,
        "nrho_periapsis_orbital_speed_m_s": nrho_periapsis_orbital_speed,
        "nrho_periapsis_ground_speed_m_s": nrho_periapsis_ground_speed,
        "sample_spacing_rows": sample_spacing_rows,
    }
