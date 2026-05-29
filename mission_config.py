"""Shared mission constants for the Enceladus orbit configuration study."""

from __future__ import annotations

from dataclasses import dataclass


SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class MissionConfig:
    """Stores the constants and assumptions for the architecture analysis."""

    gm_enceladus_m3_s2: float = 7.21037e9 #from Cassini data
    radius_enceladus_m: float = 252.1e3
    tidal_period_h: float = 32.88
    primary_science_days: float = 365.0
    stable_semimajor_m: float = 427.0e3
    stable_inclination_deg: float = 60.0
    nrho_period_h: float = 12.0
    nrho_periapsis_altitude_m: float = 100.0e3
    raw_radial_error_cases_m: tuple[float, ...] = (1.0, 2.0, 3.0)
    calibrated_radial_error_cases_m: tuple[float, ...] = (0.5, 1.0)
    altimeter_height_noise_m: float = 0.30 #check with Amir/Floris

    @property
    def exact_200km_semimajor_m(self):
        """Returns the radius of a literal 200 km altitude circular orbit."""
        return self.radius_enceladus_m + 200.0e3
