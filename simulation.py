"""CR3BP Phase 1 propagation and two-phase orbit visualization."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from mission_config import SECONDS_PER_HOUR
from orbital_periods import orbital_period_s


_NRHO_SELECTION_CACHE = {}
_TRANSFER_SELECTION_CACHE = {}


def low_inclination_ground_track(
    config,
    orbit_count = 42,
    points_per_orbit = 160,
):
    """Generates a simplified body-fixed ground track for the stable two-body orbit."""
    period_s = orbital_period_s(config.stable_semimajor_m, config.gm_enceladus_m3_s2)
    total_time_s = orbit_count * period_s
    t = np.linspace(0.0, total_time_s, orbit_count * points_per_orbit)
    inclination_rad = math.radians(config.stable_inclination_deg)
    mean_motion = 2.0 * np.pi / period_s
    body_spin = 2.0 * np.pi / (config.tidal_period_h * SECONDS_PER_HOUR)

    argument_of_latitude = mean_motion * t
    x = np.cos(argument_of_latitude)
    y = np.sin(argument_of_latitude) * math.cos(inclination_rad)
    z = np.sin(argument_of_latitude) * math.sin(inclination_rad)

    latitude_deg = np.degrees(np.arcsin(z))
    longitude_inertial_rad = np.arctan2(y, x)
    longitude_body_rad = longitude_inertial_rad - body_spin * t
    longitude_deg = (np.degrees(longitude_body_rad) + 180.0) % 360.0 - 180.0
    return t, longitude_deg, latitude_deg


def stable_orbit_xyz(
    config,
    samples = 600,
):
    """Builds a circular low-inclination two-body orbit curve in 3D."""
    theta = np.linspace(0.0, 2.0 * np.pi, samples)
    inclination_rad = math.radians(config.stable_inclination_deg)
    r = config.stable_semimajor_m / 1000.0
    x = r * np.cos(theta)
    y = r * np.sin(theta) * math.cos(inclination_rad)
    z = r * np.sin(theta) * math.sin(inclination_rad)
    return x, y, z


def cr3bp_mass_parameter(config):
    """Computes Saturn-Enceladus CR3BP mass parameter."""
    return config.gm_enceladus_m3_s2 / (
        config.gm_saturn_m3_s2 + config.gm_enceladus_m3_s2
    )


def cr3bp_mean_motion(config):
    """Computes the Saturn-Enceladus mean motion in rad/s."""
    return math.sqrt(
        (config.gm_saturn_m3_s2 + config.gm_enceladus_m3_s2)
        / config.saturn_enceladus_distance_m**3
    )


def cr3bp_potential_gradient(x, y, z, mu):
    """Computes CR3BP pseudo-potential gradient in the rotating frame."""
    r1 = math.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = math.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)

    d_omega_dx = (
        x
        - (1.0 - mu) * (x + mu) / r1**3
        - mu * (x - 1.0 + mu) / r2**3
    )
    d_omega_dy = y - (1.0 - mu) * y / r1**3 - mu * y / r2**3
    d_omega_dz = - (1.0 - mu) * z / r1**3 - mu * z / r2**3
    return np.array([d_omega_dx, d_omega_dy, d_omega_dz])


def cr3bp_equations(t, state, mu):
    """Propagates nondimensional CR3BP rotating-frame equations of motion."""
    x, y, z, xdot, ydot, zdot = state
    grad = cr3bp_potential_gradient(x, y, z, mu)
    return np.array(
        [
            xdot,
            ydot,
            zdot,
            2.0 * ydot + grad[0],
            -2.0 * xdot + grad[1],
            grad[2],
        ]
    )


def cr3bp_jacobian(state, mu):
    """Builds the CR3BP variational-equation Jacobian."""
    x, y, z, _, _, _ = state
    x1 = x + mu
    x2 = x - 1.0 + mu
    r1 = math.sqrt(x1**2 + y**2 + z**2)
    r2 = math.sqrt(x2**2 + y**2 + z**2)
    m1 = 1.0 - mu
    m2 = mu

    omega_xx = (
        1.0
        - m1 * (1.0 / r1**3 - 3.0 * x1**2 / r1**5)
        - m2 * (1.0 / r2**3 - 3.0 * x2**2 / r2**5)
    )
    omega_yy = (
        1.0
        - m1 * (1.0 / r1**3 - 3.0 * y**2 / r1**5)
        - m2 * (1.0 / r2**3 - 3.0 * y**2 / r2**5)
    )
    omega_zz = (
        - m1 * (1.0 / r1**3 - 3.0 * z**2 / r1**5)
        - m2 * (1.0 / r2**3 - 3.0 * z**2 / r2**5)
    )
    omega_xy = 3.0 * m1 * x1 * y / r1**5 + 3.0 * m2 * x2 * y / r2**5
    omega_xz = 3.0 * m1 * x1 * z / r1**5 + 3.0 * m2 * x2 * z / r2**5
    omega_yz = 3.0 * m1 * y * z / r1**5 + 3.0 * m2 * y * z / r2**5

    return np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [omega_xx, omega_xy, omega_xz, 0.0, 2.0, 0.0],
            [omega_xy, omega_yy, omega_yz, -2.0, 0.0, 0.0],
            [omega_xz, omega_yz, omega_zz, 0.0, 0.0, 0.0],
        ]
    )


def cr3bp_variational_equations(t, state_phi, mu):
    """Propagates the CR3BP state and state-transition matrix together."""
    state = state_phi[:6]
    phi = state_phi[6:].reshape(6, 6)
    state_dot = cr3bp_equations(t, state, mu)
    phi_dot = cr3bp_jacobian(state, mu) @ phi
    return np.concatenate([state_dot, phi_dot.ravel()])


def compute_lagrange_points(config):
    """Computes collinear L1, L2, and L3 points for Saturn-Enceladus CR3BP."""
    mu = cr3bp_mass_parameter(config)
    x_enceladus = 1.0 - mu
    gamma = (mu / 3.0) ** (1.0 / 3.0)

    def equilibrium_x(x):
        return cr3bp_potential_gradient(x, 0.0, 0.0, mu)[0]

    x_l1 = brentq(equilibrium_x, x_enceladus - 2.0 * gamma, x_enceladus - 0.1 * gamma)
    x_l2 = brentq(equilibrium_x, x_enceladus + 0.1 * gamma, x_enceladus + 2.0 * gamma)
    x_l3 = brentq(equilibrium_x, -1.0 - 2.0 * mu, -1.0 + 2.0 * mu)
    distance_km = config.saturn_enceladus_distance_m / 1000.0

    return {
        "mu": mu,
        "x_enceladus": x_enceladus,
        "L1": x_l1,
        "L2": x_l2,
        "L3": x_l3,
        "L1_distance_from_enceladus_km": abs(x_l1 - x_enceladus) * distance_km,
        "L2_distance_from_enceladus_km": abs(x_l2 - x_enceladus) * distance_km,
    }


def y_crossing_event(t, state_phi):
    """Stops propagation at a y=0 symmetry-plane crossing."""
    if t < 1.0e-6:
        return 1.0
    return state_phi[1]


y_crossing_event.terminal = True
y_crossing_event.direction = 0


def differential_correct_halo(config, initial_state, mu):
    """Corrects x0 and ydot0 for a symmetric CR3BP halo-style orbit."""
    state = np.array(initial_state, dtype=float)
    for iteration in range(30):
        state_phi_0 = np.concatenate([state, np.eye(6).ravel()])
        solution = solve_ivp(
            lambda t, y: cr3bp_variational_equations(t, y, mu),
            (0.0, 10.0),
            state_phi_0,
            rtol=config.cr3bp_rtol,
            atol=config.cr3bp_atol,
            events=y_crossing_event,
            max_step=0.01,
        )

        if solution.t_events[0].size == 0:
            return None

        half_period = solution.t_events[0][0]
        final = solution.y_events[0][0]
        final_state = final[:6]
        phi = final[6:].reshape(6, 6)
        dynamics = cr3bp_equations(half_period, final_state, mu)
        residual = np.array([final_state[3], final_state[5]])

        correction_matrix = np.zeros((2, 2))
        for column_index, state_index in enumerate([0, 4]):
            dt_dp = -phi[1, state_index] / final_state[4]
            correction_matrix[0, column_index] = phi[3, state_index] + dynamics[3] * dt_dp
            correction_matrix[1, column_index] = phi[5, state_index] + dynamics[5] * dt_dp

        try:
            correction = np.linalg.solve(correction_matrix, -residual)
        except np.linalg.LinAlgError:
            return None

        state[0] += correction[0]
        state[4] += correction[1]

        if np.linalg.norm(residual) < 1.0e-10 and np.linalg.norm(correction) < 1.0e-10:
            period = 2.0 * half_period
            return {
                "initial_state_nd": state,
                "period_nd": period,
                "correction_residual": float(np.linalg.norm(residual)),
                "iterations": iteration + 1,
                "converged": True,
            }

    period = 2.0 * half_period
    return {
        "initial_state_nd": state,
        "period_nd": period,
        "correction_residual": float(np.linalg.norm(residual)),
        "iterations": 30,
        "converged": False,
    }


def propagate_cr3bp_state(config, corrected, samples):
    """Propagates a corrected CR3BP orbit and converts it to Enceladus-relative km."""
    mu = cr3bp_mass_parameter(config)
    lagrange = compute_lagrange_points(config)
    mean_motion = cr3bp_mean_motion(config)
    t_eval = np.linspace(0.0, corrected["period_nd"], samples)
    solution = solve_ivp(
        lambda t, y: cr3bp_equations(t, y, mu),
        (0.0, corrected["period_nd"]),
        corrected["initial_state_nd"],
        t_eval=t_eval,
        rtol=config.cr3bp_rtol,
        atol=config.cr3bp_atol,
        max_step=corrected["period_nd"] / max(samples, 200),
    )

    if not solution.success:
        raise RuntimeError("CR3BP propagation failed for the corrected Phase 1 orbit.")

    distance_km = config.saturn_enceladus_distance_m / 1000.0
    relative_nd = solution.y[:3].T - np.array([lagrange["x_enceladus"], 0.0, 0.0])
    rotating_velocity_nd = solution.y[3:6].T
    inertial_velocity_nd = rotating_velocity_nd + np.cross(
        np.array([0.0, 0.0, 1.0]),
        relative_nd,
    )
    relative_km = relative_nd * distance_km
    relative_velocity_km_s = inertial_velocity_nd * distance_km * mean_motion
    radius_km = np.linalg.norm(relative_km, axis=1)
    altitude_km = radius_km - config.radius_enceladus_m / 1000.0
    latitude_deg = np.degrees(np.arcsin(relative_km[:, 2] / radius_km))
    longitude_deg = (np.degrees(np.arctan2(relative_km[:, 1], relative_km[:, 0])) + 180.0) % 360.0 - 180.0
    period_h = corrected["period_nd"] / cr3bp_mean_motion(config) / SECONDS_PER_HOUR

    return {
        "time_nd": solution.t,
        "relative_km": relative_km,
        "relative_velocity_km_s": relative_velocity_km_s,
        "longitude_deg": longitude_deg,
        "latitude_deg": latitude_deg,
        "altitude_km": altitude_km,
        "period_h": period_h,
    }


def validation_from_propagation(config, corrected, propagation, z_amplitude_km):
    """Creates the Phase 1 CR3BP validation dictionary."""
    lagrange = compute_lagrange_points(config)
    min_altitude = float(np.min(propagation["altitude_km"]))
    max_altitude = float(np.max(propagation["altitude_km"]))
    min_latitude = float(np.min(propagation["latitude_deg"]))
    max_latitude = float(np.max(propagation["latitude_deg"]))
    spt_boundary_access = bool(
        min_latitude <= config.spt_boundary_latitude_deg and min_altitude > 0.0
    )
    spt_full_access = bool(
        min_latitude <= config.spt_full_access_latitude_deg and min_altitude > 0.0
    )

    return {
        "phase1_cr3bp_propagated": True,
        "phase1_libration_point": config.nrho_libration_point,
        "cr3bp_mu": lagrange["mu"],
        "L1_distance_from_enceladus_km": lagrange["L1_distance_from_enceladus_km"],
        "L2_distance_from_enceladus_km": lagrange["L2_distance_from_enceladus_km"],
        "selected_z_amplitude_km": float(z_amplitude_km),
        "min_altitude_km": min_altitude,
        "max_altitude_km": max_altitude,
        "period_h": float(propagation["period_h"]),
        "groundtrack_min_latitude_deg": min_latitude,
        "groundtrack_max_latitude_deg": max_latitude,
        "spt_access": spt_boundary_access,
        "spt_boundary_access": spt_boundary_access,
        "spt_full_access": spt_full_access,
        "surface_intersection": bool(min_altitude <= 0.0),
        "correction_residual": corrected["correction_residual"],
        "correction_iterations": corrected["iterations"],
        "differential_correction_converged": corrected["converged"],
    }


def select_cr3bp_nrho_candidate(config):
    """Searches a simple L2 halo branch and selects the closest safe SPT pass."""
    cache_key = "selected"
    if cache_key in _NRHO_SELECTION_CACHE:
        return _NRHO_SELECTION_CACHE[cache_key]

    mu = cr3bp_mass_parameter(config)
    lagrange = compute_lagrange_points(config)
    x_enceladus = lagrange["x_enceladus"]
    x_libration = lagrange[config.nrho_libration_point]
    libration_distance = abs(x_libration - x_enceladus)
    distance_km = config.saturn_enceladus_distance_m / 1000.0
    target_altitude_km = config.nrho_periapsis_altitude_m / 1000.0

    z_values = np.arange(
        config.nrho_halo_z_min_km,
        config.nrho_halo_z_max_km + 0.5 * config.nrho_halo_z_step_km,
        config.nrho_halo_z_step_km,
    )

    candidates = []
    current_state = np.array(
        [
            x_libration - 0.20 * libration_distance,
            0.0,
            -z_values[0] / distance_km,
            0.0,
            0.0,
            0.0,
        ]
    )

    for z_amplitude_km in z_values:
        current_state[2] = -z_amplitude_km / distance_km
        corrected = differential_correct_halo(config, current_state, mu)
        if corrected is None:
            continue
        if not corrected["converged"]:
            continue

        propagation = propagate_cr3bp_state(config, corrected, 1200)
        validation = validation_from_propagation(config, corrected, propagation, z_amplitude_km)
        if validation["period_h"] < 1.0:
            continue
        if validation["surface_intersection"]:
            continue

        candidates.append((corrected, validation))
        current_state = corrected["initial_state_nd"].copy()

    if not candidates:
        raise RuntimeError("No safe differentially corrected CR3BP Phase 1 candidate was found.")

    safe_above_target = [
        item for item in candidates
        if item[1]["min_altitude_km"] >= target_altitude_km
    ]
    candidate_pool = safe_above_target if safe_above_target else candidates
    selected = min(
        candidate_pool,
        key=lambda item: abs(item[1]["min_altitude_km"] - target_altitude_km),
    )

    _NRHO_SELECTION_CACHE[cache_key] = {
        "corrected": selected[0],
        "validation": selected[1],
    }
    return _NRHO_SELECTION_CACHE[cache_key]


def compute_cr3bp_nrho_validation(config):
    """Returns validation values for the selected CR3BP Phase 1 orbit."""
    return select_cr3bp_nrho_candidate(config)["validation"]


def cr3bp_nrho_xyz(config, samples = 600):
    """Returns the selected propagated CR3BP Phase 1 orbit in km relative to Enceladus."""
    selected = select_cr3bp_nrho_candidate(config)
    propagation = propagate_cr3bp_state(config, selected["corrected"], samples)
    relative = propagation["relative_km"]
    return relative[:, 0], relative[:, 1], relative[:, 2]


def cr3bp_nrho_ground_track(config, samples = 900):
    """Returns the selected CR3BP Phase 1 ground track."""
    selected = select_cr3bp_nrho_candidate(config)
    propagation = propagate_cr3bp_state(config, selected["corrected"], samples)
    return propagation["time_nd"], propagation["longitude_deg"], propagation["latitude_deg"]


def stable_orbit_state(config, theta):
    """Returns stable circular-orbit position and velocity at one true anomaly."""
    radius_km = config.stable_semimajor_m / 1000.0
    mu_km3_s2 = config.gm_enceladus_m3_s2 / 1.0e9
    inclination_rad = math.radians(config.stable_inclination_deg)
    speed_km_s = math.sqrt(mu_km3_s2 / radius_km)
    position = np.array(
        [
            radius_km * math.cos(theta),
            radius_km * math.sin(theta) * math.cos(inclination_rad),
            radius_km * math.sin(theta) * math.sin(inclination_rad),
        ]
    )
    velocity = np.array(
        [
            -speed_km_s * math.sin(theta),
            speed_km_s * math.cos(theta) * math.cos(inclination_rad),
            speed_km_s * math.cos(theta) * math.sin(inclination_rad),
        ]
    )
    return position, velocity


def stumpff_c(z):
    """Computes the Lambert Stumpff C function."""
    if z > 1.0e-8:
        root_z = math.sqrt(z)
        return (1.0 - math.cos(root_z)) / z
    if z < -1.0e-8:
        root_z = math.sqrt(-z)
        return (math.cosh(root_z) - 1.0) / -z
    return 0.5


def stumpff_s(z):
    """Computes the Lambert Stumpff S function."""
    if z > 1.0e-8:
        root_z = math.sqrt(z)
        return (root_z - math.sin(root_z)) / root_z**3
    if z < -1.0e-8:
        root_z = math.sqrt(-z)
        return (math.sinh(root_z) - root_z) / root_z**3
    return 1.0 / 6.0


def lambert_universal(r1, r2, time_of_flight_s, mu_km3_s2):
    """Solves a short-way two-body Lambert transfer yeahh. """
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    cos_dtheta = np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0)
    cross = np.cross(r1, r2)
    sin_dtheta = np.linalg.norm(cross) / (r1_norm * r2_norm)
    if cross[2] < 0.0:
        sin_dtheta *= -1.0

    if abs(1.0 - cos_dtheta) < 1.0e-10:
        return None

    a_lambert = sin_dtheta * math.sqrt(r1_norm * r2_norm / (1.0 - cos_dtheta))
    if abs(a_lambert) < 1.0e-10:
        return None

    def y_value(z):
        c_value = stumpff_c(z)
        s_value = stumpff_s(z)
        if c_value <= 0.0:
            return None
        return r1_norm + r2_norm + a_lambert * (z * s_value - 1.0) / math.sqrt(c_value)

    def time_residual(z):
        c_value = stumpff_c(z)
        s_value = stumpff_s(z)
        y = y_value(z)
        if y is None or y <= 0.0:
            return float("nan")
        x = math.sqrt(y / c_value)
        computed_time = (x**3 * s_value + a_lambert * math.sqrt(y)) / math.sqrt(mu_km3_s2)
        return computed_time - time_of_flight_s

    z_grid = np.linspace(-4.0 * np.pi**2, 4.0 * np.pi**2, 240)
    previous_z = None
    previous_value = None
    bracket = None
    for z in z_grid:
        value = time_residual(float(z))
        if not np.isfinite(value):
            continue
        if previous_value is not None and previous_value * value <= 0.0:
            bracket = (previous_z, float(z))
            break
        previous_z = float(z)
        previous_value = value

    if bracket is None:
        return None

    try:
        z_root = brentq(time_residual, bracket[0], bracket[1], xtol=1.0e-10, rtol=1.0e-10)
    except ValueError:
        return None

    c_value = stumpff_c(z_root)
    s_value = stumpff_s(z_root)
    y = y_value(z_root)
    if y is None or y <= 0.0:
        return None

    f_value = 1.0 - y / r1_norm
    g_value = a_lambert * math.sqrt(y / mu_km3_s2)
    gdot_value = 1.0 - y / r2_norm
    if abs(g_value) < 1.0e-10:
        return None

    v1 = (r2 - f_value * r1) / g_value
    v2 = (gdot_value * r2 - r1) / g_value
    return v1, v2


def two_body_equations(t, state, mu_km3_s2):
    """Propagates an Enceladus-centered two-body transfer arc."""
    position = state[:3]
    velocity = state[3:]
    radius = np.linalg.norm(position)
    acceleration = -mu_km3_s2 * position / radius**3
    return np.concatenate([velocity, acceleration])


def compute_optimal_transfer(config, samples = 160):
    """Finds the minimum-delta-v Lambert transfer from Phase 1 to Phase 2."""
    search_key = "search"
    path_key = ("path", samples)
    if path_key in _TRANSFER_SELECTION_CACHE:
        return _TRANSFER_SELECTION_CACHE[path_key]

    if search_key in _TRANSFER_SELECTION_CACHE:
        best = dict(_TRANSFER_SELECTION_CACHE[search_key])
    else:
        selected = select_cr3bp_nrho_candidate(config)
        departure_propagation = propagate_cr3bp_state(
            config,
            selected["corrected"],
            config.transfer_departure_samples,
        )
        departure_positions = departure_propagation["relative_km"]
        departure_velocities = departure_propagation["relative_velocity_km_s"]
        departure_times_h = (
            departure_propagation["time_nd"]
            / cr3bp_mean_motion(config)
            / SECONDS_PER_HOUR
        )
        departure_altitudes = departure_propagation["altitude_km"]
        mu_km3_s2 = config.gm_enceladus_m3_s2 / 1.0e9
        stable_radius_km = config.stable_semimajor_m / 1000.0
        arrival_angles = np.linspace(0.0, 2.0 * np.pi, config.transfer_arrival_samples, endpoint=False)

        best = None
        for departure_index, r1 in enumerate(departure_positions):
            v_phase1 = departure_velocities[departure_index]
            r1_norm = np.linalg.norm(r1)
            nominal_transfer_time_s = math.pi * math.sqrt(
                ((r1_norm + stable_radius_km) / 2.0) ** 3 / mu_km3_s2
            )
            time_candidates_s = np.array(
                [
                    0.65,
                    0.85,
                    1.00,
                    1.20,
                    1.45,
                ]
            ) * nominal_transfer_time_s

            for arrival_theta in arrival_angles:
                r2, v_stable = stable_orbit_state(config, arrival_theta)
                for time_of_flight_s in time_candidates_s:
                    lambert = lambert_universal(r1, r2, time_of_flight_s, mu_km3_s2)
                    if lambert is None:
                        continue
                    v_transfer_departure, v_transfer_arrival = lambert
                    departure_delta_v = np.linalg.norm(v_transfer_departure - v_phase1)
                    arrival_delta_v = np.linalg.norm(v_stable - v_transfer_arrival)
                    total_delta_v = departure_delta_v + arrival_delta_v
                    if best is None or total_delta_v < best["total_delta_v_km_s"]:
                        best = {
                            "departure_index": departure_index,
                            "departure_time_h": float(departure_times_h[departure_index]),
                            "departure_altitude_km": float(departure_altitudes[departure_index]),
                            "departure_position_km": r1,
                            "departure_velocity_km_s": v_phase1,
                            "arrival_theta_rad": float(arrival_theta),
                            "arrival_position_km": r2,
                            "arrival_velocity_km_s": v_stable,
                            "transfer_departure_velocity_km_s": v_transfer_departure,
                            "transfer_arrival_velocity_km_s": v_transfer_arrival,
                            "time_of_flight_s": float(time_of_flight_s),
                            "departure_delta_v_km_s": float(departure_delta_v),
                            "arrival_delta_v_km_s": float(arrival_delta_v),
                            "total_delta_v_km_s": float(total_delta_v),
                        }

        if best is None:
            raise RuntimeError("No valid Lambert transfer was found between Phase 1 and Phase 2.")

        best["model"] = "minimum-delta-v two-body Lambert grid search"
        best["departure_is_phase1_periapsis"] = bool(
            abs(best["departure_altitude_km"] - np.min(departure_altitudes)) < 2.0
        )
        _TRANSFER_SELECTION_CACHE[search_key] = dict(best)

    mu_km3_s2 = config.gm_enceladus_m3_s2 / 1.0e9

    initial_state = np.concatenate(
        [
            best["departure_position_km"],
            best["transfer_departure_velocity_km_s"],
        ]
    )
    t_eval = np.linspace(0.0, best["time_of_flight_s"], samples)
    solution = solve_ivp(
        lambda t, y: two_body_equations(t, y, mu_km3_s2),
        (0.0, best["time_of_flight_s"]),
        initial_state,
        t_eval=t_eval,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not solution.success:
        raise RuntimeError("Optimized two-body transfer propagation failed.")

    transfer_positions = solution.y[:3].T
    transfer_altitudes = np.linalg.norm(transfer_positions, axis=1) - config.radius_enceladus_m / 1000.0
    if np.min(transfer_altitudes) <= 0.0:
        raise RuntimeError("Optimized transfer intersects Enceladus.")

    best["positions_km"] = transfer_positions
    best["minimum_altitude_km"] = float(np.min(transfer_altitudes))
    _TRANSFER_SELECTION_CACHE[path_key] = best
    return best


def compute_transfer_validation(config):
    """Returns validation values for the optimized Phase 1 to Phase 2 transfer."""
    transfer = compute_optimal_transfer(config)
    return {
        "model": transfer["model"],
        "departure_time_h": transfer["departure_time_h"],
        "departure_altitude_km": transfer["departure_altitude_km"],
        "departure_is_phase1_periapsis": transfer["departure_is_phase1_periapsis"],
        "time_of_flight_h": transfer["time_of_flight_s"] / SECONDS_PER_HOUR,
        "departure_delta_v_m_s": transfer["departure_delta_v_km_s"] * 1000.0,
        "arrival_delta_v_m_s": transfer["arrival_delta_v_km_s"] * 1000.0,
        "total_delta_v_m_s": transfer["total_delta_v_km_s"] * 1000.0,
        "minimum_altitude_km": transfer["minimum_altitude_km"],
    }


def transition_xyz(
    config,
    samples = 180,
):
    """Returns the optimized transfer curve from Phase 1 to Phase 2."""
    transfer = compute_optimal_transfer(config, samples)
    curve = transfer["positions_km"]
    return curve[:, 0], curve[:, 1], curve[:, 2]


def sphere_xyz(radius_km, samples = 50):
    """Creates an Enceladus sphere mesh for 3D plotting."""
    u = np.linspace(0.0, 2.0 * np.pi, samples)
    v = np.linspace(0.0, np.pi, samples)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def make_overview_figure(config, outputs, output_dir):
    """Creates the main static visualization of the two-phase architecture."""
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    radius_km = config.radius_enceladus_m / 1000.0
    sx, sy, sz = sphere_xyz(radius_km)
    ax3d.plot_surface(sx, sy, sz, color="#d8d6cc", linewidth=0, alpha=0.88, shade=True)

    nrho_x, nrho_y, nrho_z = cr3bp_nrho_xyz(config)
    stable_x, stable_y, stable_z = stable_orbit_xyz(config)
    trans_x, trans_y, trans_z = transition_xyz(config)
    ax3d.plot(nrho_x, nrho_y, nrho_z, color="#b82e2e", lw=2.2, label="Phase 1 CR3BP NRHO")
    ax3d.plot(trans_x, trans_y, trans_z, color="#333333", lw=1.8, ls="--", label="Optimized transfer")
    ax3d.plot(stable_x, stable_y, stable_z, color="#2468b2", lw=2.2, label="Phase 2 two-body orbit")
    ax3d.scatter([0.0], [0.0], [-radius_km], color="#b82e2e", s=42, label="SPT")
    ax3d.text(0.0, 0.0, -radius_km - 85.0, "SPT", color="#8d1f1f", ha="center")

    limit = 1750.0
    ax3d.set_xlim(-limit, limit)
    ax3d.set_ylim(-limit, limit)
    ax3d.set_zlim(-1900, 1000)
    ax3d.set_xlabel("x, km")
    ax3d.set_ylabel("y, km")
    ax3d.set_zlabel("z, km")
    ax3d.set_title("Two-phase orbit architecture transition")
    ax3d.legend(loc="upper left")
    ax3d.view_init(elev=20.0, azim=38.0)

    ax_ground = fig.add_subplot(gs[0, 1])
    _, nrho_lon, nrho_lat = cr3bp_nrho_ground_track(config)
    _, stable_lon, stable_lat = low_inclination_ground_track(config)
    ax_ground.axhspan(-90.0, -80.0, color="#ffe1dd", label="SPT: 80-90 deg S")
    ax_ground.scatter(stable_lon, stable_lat, s=2, color="#2468b2", alpha=0.35, label="Phase 2")
    ax_ground.plot(nrho_lon, nrho_lat, color="#b82e2e", lw=1.6, label="Phase 1 CR3BP")
    ax_ground.set_xlim(-180.0, 180.0)
    ax_ground.set_ylim(-90.0, 90.0)
    ax_ground.set_xlabel("Body-fixed longitude, deg")
    ax_ground.set_ylabel("Latitude, deg")
    ax_ground.set_title("Ground-track consequence")
    ax_ground.grid(True, alpha=0.25)
    ax_ground.legend(loc="upper right", markerscale=4)

    ax_phase = fig.add_subplot(gs[1, 1])
    orbit_numbers = np.arange(0, 40)
    tidal_phase = (orbit_numbers * outputs["tidal_phase_step_deg"]) % 360.0
    ax_phase.scatter(orbit_numbers, tidal_phase, color="#2468b2", s=34)
    ax_phase.plot(orbit_numbers, tidal_phase, color="#2468b2", lw=1.0, alpha=0.5)
    ax_phase.set_xlabel("Stable-orbit revolution number")
    ax_phase.set_ylabel("Tidal phase, deg")
    ax_phase.set_title("Tidal phase sampled by successive stable orbits")
    ax_phase.set_ylim(-10.0, 370.0)
    ax_phase.grid(True, alpha=0.25)

    fig.suptitle(
        "Enceladus Mission Orbit Configuration: CR3BP SPT Access + Stable Low-Inclination Crossover Network",
        fontsize=15,
    )
    output_path = output_dir / "orbit_transition_overview.png"
    fig.savefig(output_path, dpi=180)
    return output_path


def make_transition_animation(config, output_dir):
    """Saves a compact GIF showing the spacecraft moving from Phase 1 to Phase 2."""
    fig = plt.figure(figsize=(7.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    radius_km = config.radius_enceladus_m / 1000.0
    sx, sy, sz = sphere_xyz(radius_km, 34)
    nrho_x, nrho_y, nrho_z = cr3bp_nrho_xyz(config, 240)
    trans_x, trans_y, trans_z = transition_xyz(config, 90)
    stable_x, stable_y, stable_z = stable_orbit_xyz(config, 240)

    path_x = np.concatenate([nrho_x, trans_x, stable_x[:180]])
    path_y = np.concatenate([nrho_y, trans_y, stable_y[:180]])
    path_z = np.concatenate([nrho_z, trans_z, stable_z[:180]])
    phase_labels = (
        ["Phase 1 CR3BP NRHO: SPT access"] * len(nrho_x)
        + ["Optimized transfer"] * 90
        + ["Phase 2 stable two-body orbit"] * 180
    )

    ax.plot_surface(sx, sy, sz, color="#d8d6cc", linewidth=0, alpha=0.88, shade=True)
    ax.plot(nrho_x, nrho_y, nrho_z, color="#b82e2e", lw=1.7, alpha=0.70)
    ax.plot(trans_x, trans_y, trans_z, color="#333333", lw=1.4, ls="--", alpha=0.70)
    ax.plot(stable_x, stable_y, stable_z, color="#2468b2", lw=1.7, alpha=0.70)
    ax.scatter([0.0], [0.0], [-radius_km], color="#b82e2e", s=34)
    spacecraft = ax.scatter([path_x[0]], [path_y[0]], [path_z[0]], color="#111111", s=42)
    title = ax.set_title(phase_labels[0])

    limit = 1750.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-1900, 1000)
    ax.set_xlabel("x, km")
    ax.set_ylabel("y, km")
    ax.set_zlabel("z, km")
    ax.view_init(elev=20.0, azim=38.0)

    frame_indices = np.linspace(0, len(path_x) - 1, 96).astype(int)

    def update(frame_number):
        """Moves the spacecraft marker along the combined architecture path."""
        idx = frame_indices[frame_number]
        spacecraft._offsets3d = ([path_x[idx]], [path_y[idx]], [path_z[idx]])
        title.set_text(phase_labels[idx])
        ax.view_init(elev=20.0, azim=38.0 + 0.18 * frame_number)
        return spacecraft, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=90,
        blit=False,
    )
    output_path = output_dir / "orbit_transition_animation.gif"
    ani.save(output_path, writer=animation.PillowWriter(fps=12), dpi=110)
    plt.close(fig)
    return output_path
