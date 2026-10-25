# CORTEX-Robotics: Industrial Cyber-Physical Console

<p align="center">
  <img src="../../assets/ui_previews/cortex_robotics_console.png" alt="CORTEX-Robotics Mission Console" width="100%" />
</p>

## Overview
**CORTEX-Robotics** is a supervisory control and simulation console for 6-DOF industrial robotic manipulators and autonomous mobile robots (AMRs). It couples continuous attractor trajectory prediction with independent, deterministic hardware safety interlocks compliant with **ISO 10218-1:2025** and **ISO 10218-2:2025**.

---

## Key Capabilities
- **Kinematic Digital Twin**: Real-time 3D viewport of robotic workcells displaying live joint states, tool center points (TCP), and payload forces.
- **6-DOF Telemetry Gauges**: Microsecond latency monitoring of joint torque, angular velocity, and encoder positions.
- **Deterministic E-Stop Separation**: Neural inference acts solely as an advisory supervisory layer; the hardware Emergency Stop and boundary interlocks run on hard real-time deterministic PLCs/RTOS.
- **Dynamic Safety Envelopes**: Translucent 3D exclusion zones adapting to detected human operators and obstacles in shared workspaces.
