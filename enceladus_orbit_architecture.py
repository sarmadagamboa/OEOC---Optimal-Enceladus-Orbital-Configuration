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
    """Formats the numerical results into an interpretation-led mission summary."""
    stable_altitude_km = (
        config.stable_semimajor_m - config.radius_enceladus_m
    ) / 1000.0

    validation = outputs["phase1_validation"]
    transfer = outputs["transfer_validation"]
    calibration = outputs["calibration"]
    sample_10hz = next(
        row for row in outputs["sample_spacing_rows"]
        if abs(row["rate_hz"] - 10.0) < 1.0e-9
    )

    min_lat = validation["groundtrack_min_latitude_deg"]
    boundary_lat = abs(config.spt_boundary_latitude_deg)
    full_access_lat = abs(config.spt_full_access_latitude_deg)
    full_access_shortfall = max(
        0.0,
        min_lat - config.spt_full_access_latitude_deg,
    )

    lines: list[str] = []
    lines.append("ENCELADUS ORBIT CONFIGURATION INTERPRETATION")
    lines.append("=" * 48)
    lines.append("")
    lines.append("Bottom line")
    lines.append(
        f"- The selected Phase 1 orbit reaches {abs(min_lat):.1f} deg S, so it only clips the outer SPT boundary."
    )
    lines.append(
        f"- It does not meet the stronger full-SPT target of {full_access_lat:.0f} deg S; it misses that target by about {full_access_shortfall:.1f} deg."
    )
    lines.append(
        f"- The Phase 2 stable orbit is {config.stable_inclination_deg:.0f} deg inclination, so it is for repeat altimetry and crossovers, not direct south-pole coverage."
    )
    lines.append("")

    lines.append("Key checks")
    lines.append("| Quantity | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Phase 1 period | {validation['period_h']:.3f} h |")
    lines.append(
        f"| Phase 1 altitude range | {validation['min_altitude_km']:.1f} to {validation['max_altitude_km']:.1f} km |"
    )
    lines.append(
        f"| Phase 1 latitude range | {validation['groundtrack_min_latitude_deg']:.1f} to {validation['groundtrack_max_latitude_deg']:.1f} deg |"
    )
    lines.append(
        f"| SPT boundary access, >= {boundary_lat:.0f} deg S | {validation['spt_boundary_access']} |"
    )
    lines.append(
        f"| Full SPT access target, >= {full_access_lat:.0f} deg S | {validation['spt_full_access']} |"
    )
    lines.append(f"| Stable-orbit altitude | {stable_altitude_km:.1f} km |")
    lines.append(f"| Stable-orbit period | {outputs['stable_period_h']:.3f} h |")
    lines.append(f"| Stable-orbit inclination | {config.stable_inclination_deg:.1f} deg |")
    lines.append(f"| Stable-orbit revolutions in primary science | {outputs['number_of_orbits']:.0f} |")
    lines.append(
        f"| Planning crossover opportunity | {outputs['crossover_count']/1.0e6:.2f} million |"
    )
    lines.append(f"| Transfer delta-v | {transfer['total_delta_v_m_s']:.1f} m/s |")
    lines.append(
        f"| Effective radial-error target after calibration | {calibration['effective_radial_low_m']:.2f} to {calibration['effective_radial_high_m']:.2f} m |"
    )
    lines.append("")

    lines.append("Why the SPT flag was true")
    lines.append(
        f"The previous output used SPT access to mean crossing the outer South Polar Terrain definition, taken here as reaching at least {boundary_lat:.0f} deg S. Because the propagated Phase 1 ground track reaches {abs(min_lat):.1f} deg S and stays above the surface, that boundary-access test is true. That is not the same as full polar access. If the science requirement is coverage close to the pole, around {full_access_lat:.0f}-87 deg S, then the current selected orbit should be treated as insufficient and the Phase 1 design needs to be retuned."
    )
    lines.append("")

    lines.append("Phase 1: SPT reconnaissance altimetry")
    lines.append(
        f"Phase 1 gives the altimeter its only direct high-southern-latitude opportunity in this architecture. The closest approach is {validation['min_altitude_km']:.1f} km, which is good for strong returned signal and dense along-track sampling, but the {abs(min_lat):.1f} deg S latitude limit means the pass samples the edge of the SPT rather than the full polar cap. For the overall altimeter mission, this phase is useful for targeted fracture/topography observations and for tying the polar region into the rest of the height model, but it should not be sold as complete south-pole coverage unless the orbit is redesigned toward the {full_access_lat:.0f}-87 deg S band."
    )
    lines.append("")

    lines.append("Transfer phase: science continuity and risk")
    lines.append(
        f"The transfer is mainly an architecture enabler, not the core mapping phase. It costs about {transfer['total_delta_v_m_s']:.1f} m/s and remains above {transfer['minimum_altitude_km']:.1f} km, so it does not drive the altimeter error budget directly. Its effect on the altimeter mission is indirect: it preserves enough propellant and altitude margin to move from targeted SPT passes into the stable mapping orbit without making the low-orbit campaign too expensive or risky. Any altimeter data during transfer should be treated as opportunistic, with less controlled geometry than the two planned science phases."
    )
    lines.append("")

    lines.append("Phase 2: stable repeat-altimetry campaign")
    lines.append(
        f"Phase 2 is the workhorse for geodetic and tidal altimetry. The {stable_altitude_km:.1f} km altitude gives a {outputs['stable_period_h']:.3f} h period, producing about {outputs['number_of_orbits']:.0f} revolutions over the primary science year and roughly {outputs['crossover_count']/1.0e6:.2f} million potential crossover opportunities. This is what lets the mission reduce radial orbit error through self-calibration and compare surface heights at different tidal phases. The limitation is latitude: a {config.stable_inclination_deg:.0f} deg stable orbit cannot directly observe the 80-90 deg S polar terrain, so it complements Phase 1 rather than replacing it."
    )
    lines.append("")

    lines.append("Sampling speed")
    lines.append(
        f"The altimeter footprint moves at about {outputs['phase1_closest_ground_speed_m_s']:.1f} m/s during the Phase 1 closest pass and {outputs['stable_ground_speed_m_s']:.1f} m/s in the stable Phase 2 orbit. At a representative 10 Hz measurement rate, this gives along-track samples every {sample_10hz['phase1_spacing_m']:.1f} m in Phase 1 and {sample_10hz['stable_spacing_m']:.1f} m in Phase 2, so sampling density is not the main limitation; orbit knowledge and crossover geometry are."
    )
    lines.append("")

    lines.append("Altimeter performance interpretation")
    lines.append(
        f"With {config.altimeter_height_noise_m:.2f} m single-shot height noise, the limiting issue is not the altimeter sample precision alone; it is the radial orbit knowledge and how well crossover adjustment can remove it. The current model suggests a practical calibrated radial-error target of {calibration['effective_radial_low_m']:.2f}-{calibration['effective_radial_high_m']:.2f} m, which is enough to make meter-scale tides plausible but not guaranteed. The conclusion should therefore be cautious: Phase 2 supplies the redundancy needed for calibration and tide recovery, while Phase 1 supplies the polar context. If full SPT access is mandatory, the Phase 1 orbit must be retuned before the altimeter science case is fully closed."
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
