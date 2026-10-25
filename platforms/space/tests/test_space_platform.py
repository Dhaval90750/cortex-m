"""Unit tests for CORTEX-Space Platform."""

import math
import struct
import unittest

from platforms.space.src.ccsds_telemetry import (
    ASM_MARKER,
    CCSDSTelemetryProcessor,
    DecommutatedTelemetry,
    SyncState,
    crc16_ccitt,
)
from platforms.space.src.orbit_propagator import (
    DEFAULT_GROUND_STATIONS,
    GroundStation,
    OrbitalElements,
    OrbitalStatePoint,
    OrbitPropagator,
)
from platforms.space.src.safety_supervisor import (
    AnomalySeverity,
    SpaceSafetySupervisor,
    TelemetryAlert,
)


class TestCortexSpacePlatform(unittest.TestCase):
    def setUp(self):
        self.processor = CCSDSTelemetryProcessor()
        # LEO Orbit: ~500 km circular, 51.6 deg inclination (ISS-like)
        self.elements = OrbitalElements(
            semi_major_axis_km=6878.137,
            eccentricity=0.001,
            inclination_deg=51.6,
            raan_deg=45.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=0.0,
            epoch_timestamp_s=1700000000.0,
        )
        self.propagator = OrbitPropagator(self.elements)
        self.supervisor = SpaceSafetySupervisor(beta=6.0, seu_energy_threshold=-0.75)

    def test_ccsds_frame_synchronization_and_decommutation(self):
        # 1. Build valid packet
        packet_bytes = self.processor.build_telemetry_packet(
            apid=101,
            seq_count=42,
            timestamp_s=1700000010.5,
            bus_voltage=28.1,
            solar_current=6.2,
            battery_temp=21.5,
            gyro_x=0.01,
            gyro_y=-0.02,
            gyro_z=0.005,
            heater_on=True,
            radiator_temp=14.8,
        )

        # Prepend random noise bytes to test bit-stream search
        noisy_stream = b"\x00\xff\xaa\x55" + packet_bytes

        telemetry_list, remainder = self.processor.process_stream(noisy_stream)
        self.assertEqual(len(telemetry_list), 1)
        telem = telemetry_list[0]

        self.assertEqual(telem.apid, 101)
        self.assertEqual(telem.sequence_count, 42)
        self.assertAlmostEqual(telem.bus_voltage_v, 28.1, places=2)
        self.assertAlmostEqual(telem.solar_current_a, 6.2, places=2)
        self.assertTrue(telem.heater_on)
        self.assertTrue(telem.crc_valid)
        self.assertIn(self.processor.state, [SyncState.CHECK, SyncState.LOCK])

    def test_ccsds_corrupted_crc_detection(self):
        valid_packet = self.processor.build_telemetry_packet(
            apid=102,
            seq_count=1,
            timestamp_s=1700000000.0,
            bus_voltage=28.0,
            solar_current=5.0,
            battery_temp=20.0,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            heater_on=False,
            radiator_temp=10.0,
        )

        # Corrupt one byte in the payload
        corrupted = bytearray(valid_packet)
        corrupted[15] ^= 0xFF
        corrupted_bytes = bytes(corrupted)

        initial_fails = self.processor.crc_failures
        telemetry_list, _ = self.processor.process_stream(corrupted_bytes)
        self.assertEqual(len(telemetry_list), 0)
        self.assertGreater(self.processor.crc_failures, initial_fails)

    def test_orbital_propagation_and_ground_track(self):
        # Orbital period for 6878 km is ~5677 seconds (~94.6 minutes)
        self.assertAlmostEqual(self.propagator.period_s, 5677.0, delta=20.0)

        # Propagate 1 quarter of orbit (~1400s)
        p1 = self.propagator.propagate_state(1400.0)
        self.assertIsInstance(p1, OrbitalStatePoint)
        # Latitude cannot exceed inclination (51.6 deg)
        self.assertLessEqual(abs(p1.latitude_deg), 52.0)
        self.assertGreaterEqual(p1.altitude_km, 480.0)
        self.assertLessEqual(p1.altitude_km, 520.0)

        # Generate ground track over 1 full orbit (6000s) in 300s steps
        track = self.propagator.generate_ground_track(duration_s=6000.0, step_s=300.0)
        self.assertGreaterEqual(len(track), 20)

    def test_seu_radiation_anomaly_detection_and_npr7150_traceability(self):
        # 1. Nominal telemetry in sunlight
        telem_nominal = DecommutatedTelemetry(
            apid=100,
            sequence_count=10,
            timestamp_seconds=1700000050.0,
            bus_voltage_v=28.0,
            solar_current_a=6.5,
            battery_temp_c=22.0,
            gyro_rate_x=0.0,
            gyro_rate_y=0.0,
            gyro_rate_z=0.0,
            heater_on=False,
            radiator_temp_c=15.0,
            crc_valid=True,
        )
        alert_nom = self.supervisor.evaluate_telemetry(telem_nominal, in_eclipse=False)
        self.assertIsNone(alert_nom)

        # 2. Critical radiation SEU latchup (bus voltage collapse, current surge)
        telem_seu = DecommutatedTelemetry(
            apid=100,
            sequence_count=11,
            timestamp_seconds=1700000051.0,
            bus_voltage_v=14.2,  # Severe voltage drop
            solar_current_a=18.5,  # Latchup surge
            battery_temp_c=45.0,
            gyro_rate_x=6.5,
            gyro_rate_y=-7.2,
            gyro_rate_z=1.0,
            heater_on=False,
            radiator_temp_c=15.0,
            crc_valid=True,
        )
        alert_seu = self.supervisor.evaluate_telemetry(telem_seu, in_eclipse=False)
        self.assertIsNotNone(alert_seu)
        self.assertIsInstance(alert_seu, TelemetryAlert)
        self.assertEqual(alert_seu.severity, AnomalySeverity.CRITICAL_SEU)
        self.assertEqual(alert_seu.subsystem, "EPS")
        self.assertIn("SEU latchup", alert_seu.message)
        self.assertEqual(alert_seu.mitigation_action, "AUTONOMOUS_PAYLOAD_POWER_CYCLE")
        self.assertEqual(alert_seu.traceability_req_id, "NPR-7150.2-FR-SPC-03")
        self.assertTrue(len(alert_seu.verification_hash) > 8)


if __name__ == "__main__":
    unittest.main()
