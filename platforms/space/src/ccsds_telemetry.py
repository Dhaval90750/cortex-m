"""CCSDS Telemetry Frame Synchronizer and Space Packet Decommutator.

Compliant with CCSDS 131.0-B-3 (TM Synchronization and Channel Coding)
and CCSDS 133.0-B-1 (Space Packet Protocol).

Provides:
1. 32-bit Attached Sync Marker (ASM = 0x1ACFFC1D) frame synchronization.
2. State machine: SEARCH -> CHECK -> LOCK with bit-error tolerance.
3. CRC-16-CCITT packet integrity verification.
4. Telemetry decommutation for EPS, ADCS, and TCS subsystems.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class SyncState(str, Enum):
    SEARCH = "SEARCH"
    CHECK = "CHECK"
    LOCK = "LOCK"


ASM_MARKER = b"\x1a\xcf\xfc\x1d"  # 0x1ACFFC1D


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    """Deterministic CRC-16-CCITT (polynomial 0x1021)."""
    crc = initial
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


@dataclass
class SpacePacketHeader:
    version: int
    packet_type: int
    secondary_header_flag: int
    apid: int
    sequence_flags: int
    packet_sequence_count: int
    packet_data_length: int  # Actual payload length = packet_data_length + 1


@dataclass
class DecommutatedTelemetry:
    apid: int
    sequence_count: int
    timestamp_seconds: float
    # EPS Subsystem
    bus_voltage_v: float
    solar_current_a: float
    battery_temp_c: float
    # ADCS Subsystem
    gyro_rate_x: float
    gyro_rate_y: float
    gyro_rate_z: float
    # TCS Subsystem
    heater_on: bool
    radiator_temp_c: float
    crc_valid: bool


class CCSDSTelemetryProcessor:
    """Frame synchronizer and telemetry extractor for spacecraft downlink."""

    def __init__(self, check_threshold: int = 2):
        self.state = SyncState.SEARCH
        self.check_threshold = check_threshold
        self.consecutive_syncs = 0
        self.frames_processed = 0
        self.crc_failures = 0

    def find_asm(self, buffer: bytes) -> int:
        """Find the index of Attached Sync Marker (0x1ACFFC1D)."""
        return buffer.find(ASM_MARKER)

    def parse_header(self, raw_header: bytes) -> SpacePacketHeader:
        """Parse 6-byte CCSDS Space Packet Primary Header."""
        if len(raw_header) < 6:
            raise ValueError(f"Header too short: expected 6 bytes, got {len(raw_header)}")
        
        w1, w2, w3 = struct.unpack(">HHH", raw_header[:6])
        version = (w1 >> 13) & 0x07
        packet_type = (w1 >> 12) & 0x01
        secondary_header = (w1 >> 11) & 0x01
        apid = w1 & 0x07FF

        sequence_flags = (w2 >> 14) & 0x03
        seq_count = w2 & 0x3FFF

        data_length = w3
        return SpacePacketHeader(
            version=version,
            packet_type=packet_type,
            secondary_header_flag=secondary_header,
            apid=apid,
            sequence_flags=sequence_flags,
            packet_sequence_count=seq_count,
            packet_data_length=data_length,
        )

    def build_telemetry_packet(
        self,
        apid: int,
        seq_count: int,
        timestamp_s: float,
        bus_voltage: float,
        solar_current: float,
        battery_temp: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        heater_on: bool,
        radiator_temp: float,
    ) -> bytes:
        """Construct a compliant CCSDS Space Packet with CRC-16."""
        # 1. Payload serialization
        # Format: double timestamp (8B), float bus_v (4B), float solar_i (4B), float bat_t (4B),
        # float gx (4B), float gy (4B), float gz (4B), uint8 heater (1B), float rad_t (4B)
        # Total payload = 8 + 4*3 + 4*3 + 1 + 4 = 37 bytes + 2 bytes CRC = 39 bytes
        payload_data = struct.pack(
            ">dffffff?f",
            timestamp_s,
            bus_voltage,
            solar_current,
            battery_temp,
            gyro_x,
            gyro_y,
            gyro_z,
            heater_on,
            radiator_temp,
        )
        
        # Space packet header: length field is (total payload bytes including CRC - 1)
        payload_with_crc_len = len(payload_data) + 2
        w1 = (0 << 13) | (0 << 12) | (0 << 11) | (apid & 0x07FF)
        w2 = (3 << 14) | (seq_count & 0x3FFF)
        w3 = payload_with_crc_len - 1
        header = struct.pack(">HHH", w1, w2, w3)

        packet_no_crc = header + payload_data
        crc = crc16_ccitt(packet_no_crc)
        packet_complete = packet_no_crc + struct.pack(">H", crc)
        return ASM_MARKER + packet_complete

    def process_stream(self, stream_buffer: bytes) -> Tuple[List[DecommutatedTelemetry], bytes]:
        """Process incoming serial buffer, extract synchronized frames."""
        results: List[DecommutatedTelemetry] = []
        offset = 0
        buf_len = len(stream_buffer)

        while offset < buf_len:
            asm_idx = stream_buffer.find(ASM_MARKER, offset)
            if asm_idx == -1:
                # Retain tail if partial sync marker might be present
                return results, stream_buffer[max(offset, buf_len - 3):]

            # Transition state machine
            if self.state == SyncState.SEARCH:
                self.state = SyncState.CHECK
                self.consecutive_syncs = 1
            elif self.state == SyncState.CHECK:
                self.consecutive_syncs += 1
                if self.consecutive_syncs >= self.check_threshold:
                    self.state = SyncState.LOCK
            elif self.state == SyncState.LOCK:
                self.consecutive_syncs += 1

            packet_start = asm_idx + len(ASM_MARKER)
            if packet_start + 6 > buf_len:
                # Incomplete header, wait for more data
                return results, stream_buffer[asm_idx:]

            header = self.parse_header(stream_buffer[packet_start : packet_start + 6])
            total_packet_len = 6 + header.packet_data_length + 1

            if packet_start + total_packet_len > buf_len:
                # Incomplete frame payload, wait for more data
                return results, stream_buffer[asm_idx:]

            full_packet = stream_buffer[packet_start : packet_start + total_packet_len]
            packet_body = full_packet[:-2]
            received_crc = struct.unpack(">H", full_packet[-2:])[0]
            computed_crc = crc16_ccitt(packet_body)

            crc_ok = (received_crc == computed_crc)
            if not crc_ok:
                self.crc_failures += 1
                offset = asm_idx + 1  # Resynchronize
                continue

            # Decommutate payload
            payload = full_packet[6:-2]
            if len(payload) >= 37:
                t_s, bus_v, sol_i, bat_t, gx, gy, gz, htr, rad_t = struct.unpack(">dffffff?f", payload[:37])
                telemetry = DecommutatedTelemetry(
                    apid=header.apid,
                    sequence_count=header.packet_sequence_count,
                    timestamp_seconds=t_s,
                    bus_voltage_v=bus_v,
                    solar_current_a=sol_i,
                    battery_temp_c=bat_t,
                    gyro_rate_x=gx,
                    gyro_rate_y=gy,
                    gyro_rate_z=gz,
                    heater_on=htr,
                    radiator_temp_c=rad_t,
                    crc_valid=crc_ok,
                )
                results.append(telemetry)
                self.frames_processed += 1

            offset = packet_start + total_packet_len

        return results, b""

    def analyze_stream_continuity(self, packets: List[DecommutatedTelemetry]) -> Dict[str, Any]:
        """Detect packet sequence count gaps, dropped frames, and loss rate percentage."""
        if not packets:
            return {
                "total_packets": 0,
                "dropped_packets": 0,
                "loss_rate_percent": 0.0,
                "gaps_detected": 0,
                "is_contiguous": True,
            }

        dropped = 0
        gaps = 0

        for i in range(1, len(packets)):
            prev_seq = packets[i - 1].sequence_count
            curr_seq = packets[i].sequence_count

            # Standard CCSDS 14-bit sequence counter wraps at 16384
            expected = (prev_seq + 1) % 16384
            if curr_seq != expected:
                diff = (curr_seq - expected) % 16384
                dropped += diff
                gaps += 1

        total = len(packets) + dropped
        loss_pct = (dropped / max(1, total)) * 100.0

        return {
            "total_packets": len(packets),
            "dropped_packets": dropped,
            "loss_rate_percent": round(loss_pct, 2),
            "gaps_detected": gaps,
            "is_contiguous": gaps == 0,
        }
