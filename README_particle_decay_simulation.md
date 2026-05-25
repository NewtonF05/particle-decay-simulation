# Particle Decay Beam Simulation

> Vectorised Monte Carlo simulation of unstable particles decaying in flight through a four-station tracking detector, with realistic measurement smearing and acceptance cuts.

## Overview

A 10,000-event Monte Carlo simulation of an experimental physics setup: a beam of unstable particles travels along the z-axis, decays exponentially in flight, and emits daughter particles isotropically. The daughters are then tracked through four downstream detector stations, where hit positions are recorded with Gaussian measurement resolution and finite detector acceptance.

The simulation is fully vectorised — every event is generated and propagated in parallel via NumPy array operations rather than Python loops, making it fast and scalable.

## Key Features

- **Sampling from non-trivial physical distributions** — Gaussian beam velocities, exponential decay times, isotropic emission directions (uniform in cos θ and φ)
- **Geometric ray-tracing** to compute (x, y) hits at each tracking station from the decay vertex and emission direction
- **Detector-effects modelling** — Gaussian position smearing and rectangular active-area cuts
- **Forward / backward decay analysis** — quantifies how often daughter particles fire detectors despite emission into the backward hemisphere
- **Statistical validation** — every sampled distribution is compared against its analytic form (Gaussian PDF, exponential decay length, sin θ angular distribution)

## Tech Stack

`Python` · `NumPy` · `SciPy` · `Matplotlib`

## Approach

The full simulation is structured as a sequence of pure NumPy operations: sample velocities, sample decay times, compute decay vertices, sample isotropic directions, project rays onto detector planes, apply smearing, apply acceptance. No Python loops over events anywhere in the pipeline — 10,000 events run essentially instantly.

The isotropic sampling is worth highlighting: a naive uniform sample over (θ, φ) produces a non-isotropic distribution biased toward the poles. The correct approach samples cos θ uniformly on [-1, 1] and φ uniformly on [0, 2π], which the script implements and then validates against the analytic sin θ distribution in θ-space.

The validation section at the end runs three independent cross-checks:
1. Sampled beam velocities vs Gaussian PDF
2. Decay distance distribution vs theoretical exponential
3. Isotropy of emission directions in φ, cos θ, and θ — including a 3D scatter on the unit sphere

These aren't decoration — they're the kind of sanity checks any real experimental simulation needs before its results are trusted.

## Results

The simulation produces:
- **Hit maps** at each of the four tracking stations (with shared colour scale for cross-station comparison)
- **Hit-multiplicity distribution** — how many of the four stations each event fires
- **Backward-detection rate** — the fraction of detected events where the daughter was emitted with negative z-velocity
- **Angular spectra** at each station, separated into forward and backward emission

[Add a screenshot of the four-station hit map here.]

## How to Run

```bash
git clone https://github.com/<your-username>/particle-decay-simulation.git
cd particle-decay-simulation
pip install -r requirements.txt
python particle_decay_simulation.py
```

No external data required. The number of events, detector geometry, and physical parameters (lifetime, beam velocity) are all defined as module-level constants at the top of the script — change them and re-run.
