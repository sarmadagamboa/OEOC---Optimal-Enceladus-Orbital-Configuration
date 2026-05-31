"""
Main runner for the Enceladus two-phase orbit configuration study.

Run:
    python OEOC\\enceladus_orbit_architecture.py

All generated outputs are written into this same folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.dont_write_bytecode = True

from crossovers_radial_error import compute_crossover_outputs
from ground_velocity_sampling import compute_ground_velocity_outputs
from mission_config import MissionConfig
from orbital_periods import compute_orbital_period_outputs
from simulation import (
    compute_cr3bp_nrho_validation,
    compute_transfer_validation,
    make_overview_figure,
    make_transition_animation,
)


def compute_mission_outputs(config):
    """This runs all topic-specific computations and merges their results."""
    period_outputs = compute_orbital_period_outputs(config)
    phase1_validation = compute_cr3bp_nrho_validation(config)
    transfer_validation = compute_transfer_validation(config)
    crossover_outputs = compute_crossover_outputs(
        config,
        number_of_orbits=period_outputs["number_of_orbits"],
    )
    ground_velocity_outputs = compute_ground_velocity_outputs(
        config,
        phase1_altitude_m=phase1_validation["min_altitude_km"] * 1000.0,
    )

    return {
        **period_outputs,
        **crossover_outputs,
        **ground_velocity_outputs,
        "phase1_validation": phase1_validation,
        "transfer_validation": transfer_validation,
    }


def build_summary_text(config, outputs):
    """Formats the numerical results into a readable mission summary."""
    stable_altitude_km = (
        config.stable_semimajor_m - config.radius_enceladus_m
    ) / 1000.0
    exact_200km_a_km = config.exact_200km_semimajor_m / 1000.0

    lines: list[str] = []
    lines.append("ENCELADUS ORBIT CONFIGURATION OUTPUTS")
    lines.append("=" * 45)
    lines.append("")
    lines.append("Baseline assumptions")
    lines.append(f"- Enceladus GM: {config.gm_enceladus_m3_s2:.5e} m^3/s^2")
    lines.append(f"- Saturn GM: {config.gm_saturn_m3_s2:.5e} m^3/s^2")
    lines.append(
        f"- Saturn-Enceladus distance used for CR3BP: {config.saturn_enceladus_distance_m/1000.0:.1f} km"
    )
    lines.append(f"- Enceladus mean radius: {config.radius_enceladus_m/1000.0:.1f} km")
    lines.append(f"- Enceladus tidal/orbital period around Saturn: {config.tidal_period_h:.2f} h")
    lines.append(f"- Nominal Phase 1 period target used for architecture: {config.nrho_period_h:.2f} h")
    lines.append(
        f"- Phase 1 closest-approach altitude target over SPT: {config.nrho_periapsis_altitude_m/1000.0:.1f} km"
    )
    lines.append(
        f"- Stable phase semi-major axis used: {config.stable_semimajor_m/1000.0:.1f} km"
    )
    lines.append(
        f"- This equals {stable_altitude_km:.1f} km over the mean radius and reproduces the prompt's ~5.74 h period."
    )
    lines.append(
        f"- A literal 200 km altitude circular orbit would use a = {exact_200km_a_km:.1f} km."
    )
    lines.append("")

    validation = outputs["phase1_validation"]
    lines.append("Phase 1 CR3BP NRHO validation table")
    lines.append("| Quantity | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| CR3BP mass parameter mu | {validation['cr3bp_mu']:.6e} |")
    lines.append(
        f"| L1 distance from Enceladus | {validation['L1_distance_from_enceladus_km']:.1f} km |"
    )
    lines.append(
        f"| L2 distance from Enceladus | {validation['L2_distance_from_enceladus_km']:.1f} km |"
    )
    lines.append(
        f"| Selected family | {validation['phase1_libration_point']} halo-style CR3BP orbit |"
    )
    lines.append(
        f"| Selected z-amplitude seed | {validation['selected_z_amplitude_km']:.1f} km |"
    )
    lines.append(f"| Phase 1 CR3BP propagated | {validation['phase1_cr3bp_propagated']} |")
    lines.append(
        f"| Differential correction | {validation['differential_correction_converged']} after {validation['correction_iterations']} iterations |"
    )
    lines.append(f"| Correction residual | {validation['correction_residual']:.2e} |")
    lines.append(f"| Period | {validation['period_h']:.3f} h |")
    lines.append(f"| Minimum altitude | {validation['min_altitude_km']:.1f} km |")
    lines.append(f"| Maximum altitude | {validation['max_altitude_km']:.1f} km |")
    lines.append(
        f"| Ground-track latitude range | {validation['groundtrack_min_latitude_deg']:.1f} to {validation['groundtrack_max_latitude_deg']:.1f} deg |"
    )
    lines.append(f"| SPT access | {validation['spt_access']} |")
    lines.append(f"| Surface intersection | {validation['surface_intersection']} |")
    lines.append("")

    transfer = outputs["transfer_validation"]
    lines.append("Phase 1 to Phase 2 transfer optimization table")
    lines.append("| Quantity | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Transfer model | {transfer['model']} |")
    lines.append(f"| Departure time on Phase 1 orbit | {transfer['departure_time_h']:.3f} h |")
    lines.append(f"| Departure altitude | {transfer['departure_altitude_km']:.1f} km |")
    lines.append(f"| Departure at Phase 1 periapsis | {transfer['departure_is_phase1_periapsis']} |")
    lines.append(f"| Time of flight | {transfer['time_of_flight_h']:.3f} h |")
    lines.append(f"| Departure delta-v | {transfer['departure_delta_v_m_s']:.1f} m/s |")
    lines.append(f"| Arrival insertion delta-v | {transfer['arrival_delta_v_m_s']:.1f} m/s |")
    lines.append(f"| Total transfer delta-v | {transfer['total_delta_v_m_s']:.1f} m/s |")
    lines.append(f"| Minimum transfer altitude | {transfer['minimum_altitude_km']:.1f} km |")
    lines.append("")

    lines.append("A. Spacecraft orbital period and tidal phase sampling")
    lines.append(f"- Stable orbit period: {outputs['stable_period_h']:.3f} h")
    lines.append(
        f"- Literal 200 km-altitude period check: {outputs['exact_200km_period_h']:.3f} h"
    )
    lines.append(
        f"- Orbits per 32.88 h tidal cycle: {outputs['orbits_per_tidal_cycle']:.3f}"
    )
    lines.append(
        f"- Fraction of tidal cycle covered per spacecraft orbit: {outputs['tidal_cycle_fraction_per_spacecraft_orbit']:.3f}"
    )
    lines.append(
        f"- Tidal phase advance per spacecraft orbit: {outputs['tidal_phase_step_deg']:.2f} deg"
    )
    lines.append(
        "- Interpretation: this gives favorable tidal-phase diversity. Actual tide recovery still depends on repeat-track/crossover geometry, not on consecutive-orbit phase spacing alone."
    )
    lines.append("")

    lines.append("B. Crossover count scaling")
    lines.append(
        f"- Primary science duration: {config.primary_science_days:.1f} days"
    )
    lines.append(f"- Number of stable-orbit revolutions: {outputs['number_of_orbits']:.1f}")
    lines.append(
        f"- Order-of-magnitude N^2 crossover opportunity estimate: {outputs['crossover_count']:.3e}"
    )
    lines.append(
        f"- Rounded planning estimate, not exact count: {outputs['crossover_count']/1.0e6:.2f} million possible crossovers"
    )
    lines.append(
        "- Caveat: actual accepted crossovers depend on inclination, longitude drift, spacing, orbit precision, rejected near-parallel crossings, and coverage gaps."
    )
    lines.append("")

    lines.append("C. Radial orbit error and crossover self-calibration")
    lines.append(
        f"- Assumed single-height measurement noise: {config.altimeter_height_noise_m:.2f} m"
    )
    lines.append("- Raw radial-error cases:")
    for row in outputs["raw_error_rows"]:
        lines.append(
            f"  radial {row['radial_error_m']:.1f} m -> crossover sigma {row['xover_sigma_m']:.2f} m"
        )
    lines.append("- Target radial-error cases after crossover self-calibration:")
    for row in outputs["calibrated_error_rows"]:
        lines.append(
            f"  radial {row['radial_error_m']:.1f} m -> crossover sigma {row['xover_sigma_m']:.2f} m, "
            f"SNR for 1 m tide {row['snr_1m_tide']:.2f}, SNR for 2 m tide {row['snr_2m_tide']:.2f}"
        )
    calibration = outputs["calibration"]
    lines.append("- Optimistic least-squares sizing using a 2 m raw radial case:")
    lines.append(f"  unknown arc offsets plus h2: {calibration['unknown_count']:.0f}")
    lines.append(f"  observations per unknown, approx.: {calibration['redundancy']:.0f}")
    lines.append(
        f"  random-only arc precision from independent-crossing assumption: {calibration['random_only_arc_sigma_m']:.3f} m"
    )
    lines.append(
        f"  assumed practical effective radial-error target with systematics: "
        f"{calibration['effective_radial_low_m']:.2f} to {calibration['effective_radial_high_m']:.2f} m"
    )
    lines.append(
        "- Interpretation: the redundancy calculation is optimistic because crossovers are correlated. The 0.5-1.0 m range is a target assumption, not a proven result from this simplified model."
    )
    lines.append("")

    lines.append("D. Ground velocity and sampling")
    lines.append(
        f"- Phase 1 closest-approach altitude used: {outputs['phase1_closest_altitude_km']:.1f} km"
    )
    lines.append(
        f"- Phase 1 closest-approach orbital speed: {outputs['phase1_closest_orbital_speed_m_s']:.1f} m/s"
    )
    lines.append(
        f"- Phase 1 closest-approach ground speed: {outputs['phase1_closest_ground_speed_m_s']:.1f} m/s"
    )
    lines.append(
        f"- Stable orbit orbital speed: {outputs['stable_orbital_speed_m_s']:.1f} m/s"
    )
    lines.append(
        f"- Stable orbit ground speed: {outputs['stable_ground_speed_m_s']:.1f} m/s"
    )
    lines.append(
        f"- Enceladus equatorial surface rotation speed: {outputs['body_equatorial_rotation_speed_m_s']:.1f} m/s"
    )
    lines.append(
        "- Caveat: ground speeds are approximate surface-projected speeds for sampling estimates; exact ground-track speed depends on body rotation and pass direction and can differ by roughly 10-20%."
    )
    lines.append("- Along-track sample spacing examples:")
    for row in outputs["sample_spacing_rows"]:
        lines.append(
            f"  {row['rate_hz']:.0f} Hz -> Phase 1 {row['phase1_spacing_m']:.2f} m, "
            f"stable {row['stable_spacing_m']:.2f} m"
        )
    lines.append("")

    lines.append("Architecture conclusion")
    lines.append(
        "- Phase 1 is now a propagated Saturn-Enceladus CR3BP halo-style orbit near Enceladus, selected for about 100 km closest approach and SPT latitude access."
    )
    lines.append(
        "- Phase 2 stable low-inclination orbit gives long-duration coverage and an order-of-magnitude million-scale crossover opportunity estimate, but cannot directly cover 80-90 deg S."
    )
    lines.append(
        "- The Phase 2 stable orbit remains a two-body low-inclination approximation. The baseline calculation uses 60 deg inclination for geodetic/tidal coverage; an ESA-compatible <=48 deg case should be re-run if that constraint is adopted."
    )
    lines.append(
        "- The mission architecture is therefore justified as Phase 1 for targeted SPT passes and Phase 2 for stable crossover-rich gravity/topography/tide analysis."
    )
    lines.append("")
    lines.append(
        "Note: the Phase 1 curve is generated from propagated CR3BP states. The dashed transfer is now the minimum-delta-v Lambert transfer found by the Enceladus-centered two-body grid search, not a full CR3BP manifold optimization."
    )
    return "\n".join(lines)


def save_summary(output_dir, summary_text):
    """Writes the text summary to disk."""
    output_path = output_dir / "mission_outputs.txt"
    output_path.write_text(summary_text, encoding="utf-8")
    return output_path


def parse_args():
    """Parses small run-control options for the script."""
    parser = argparse.ArgumentParser(description="Run Enceladus orbit configuration analysis.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save files and print outputs without opening the matplotlib window.",
    )
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip GIF generation if you only need the text and PNG.",
    )
    return parser.parse_args()


def main():
    """Runs the full analysis, writes files, and optionally displays the figure."""
    args = parse_args()
    output_dir = Path(__file__).resolve().parent
    config = MissionConfig()
    outputs = compute_mission_outputs(config)
    summary_text = build_summary_text(config, outputs)

    summary_path = save_summary(output_dir, summary_text)
    overview_path = make_overview_figure(config, outputs, output_dir)

    gif_path = None
    if not args.skip_gif:
        gif_path = make_transition_animation(config, output_dir)

    print(summary_text)
    print("")
    print("Generated files")
    print(f"- {summary_path}")
    print(f"- {overview_path}")
    if gif_path is not None:
        print(f"- {gif_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
