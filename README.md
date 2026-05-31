# Enceladus Orbit Configuration
This is an orbital configuration optimized for an Enceladus mission to comprehend and study its habitability.
Run the analysis from the main Enceladus mission folder:

```powershell
python OEOC\enceladus_orbit_architecture.py
```

The code is split by mission-analysis topic:

- `enceladus_orbit_architecture.py`: main runner that combines all computations and writes outputs.
- `mission_config.py`: shared Enceladus and mission constants.
- `orbital_periods.py`: orbital period, tidal phase sampling, and yearly orbit count.
- `crossovers_radial_error.py`: crossover count, crossover uncertainty, and radial-error self-calibration.
- `ground_velocity_sampling.py`: orbital speed, ground speed, and along-track sample spacing.
- `simulation.py`: Saturn-Enceladus CR3BP Phase 1 propagation, two-body Phase 2 visualization, ground-track plot, 3D architecture view, and transition GIF.
- `environment.py`: suggested `EOC-configuration` package environment, including SciPy for CR3BP integration and correction.

This prints the important mission outputs and writes:

- `mission_outputs.txt`: computed period, tidal sampling, crossover count, radial-error budget, and ground velocity values.
- `orbit_transition_overview.png`: static architecture summary figure.
- `orbit_transition_animation.gif`: visual simulation of the propagated Phase 1 -> transition connector -> Phase 2 architecture.

For a non-interactive run:

```powershell
python OEOC\enceladus_orbit_architecture.py --no-show
```
