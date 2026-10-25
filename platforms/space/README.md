# CORTEX-Space: Orbital Mission Control & Telemetry Anomaly Console

<p align="center">
  <img src="../../assets/ui_previews/cortex_space_mission.png" alt="CORTEX-Space Mission Control" width="100%" />
</p>

## Overview
**CORTEX-Space** provides ground-based orbital telemetry parsing, spacecraft subsystem health monitoring, and radiation single-event-upset (SEU) anomaly detection. Designed around NASA software engineering qualification practices (**NASA NPR 7150.2 / SP-8000**), it ensures full requirements-to-test traceability for flight mission simulations.

---

## Key Capabilities
- **3D Orbital Ground Track**: Real-time globe visualization plotting satellite ephemeris, orbit ribbons, day/night solar terminators, and ground station visibility cones (Goldstone, Madrid, Canberra).
- **Subsystem Health Matrix**: High-reliability telemetry feeds for Electrical Power Systems (EPS), Attitude Determination & Control (ADCS), and Thermal Control.
- **Radiation Anomaly Timeline**: Causal event scrubber correlating thermal excursions and radiation latchup anomalies with spacecraft telemetry passes.
- **High-Fidelity Telemetry Decoder**: Monospaced CCSDS frame sync parser with automated fault injection verification.
