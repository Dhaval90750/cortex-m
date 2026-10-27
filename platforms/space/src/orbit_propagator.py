"""Orbital Ephemeris & Ground Track Propagator for CORTEX-Space.

Implements Keplerian two-body dynamics with J2 oblateness perturbations:
1. Kepler's equation solver (Newton-Raphson iteration).
2. Earth-Centered Inertial (ECI) state vector propagation (r, v).
3. Earth-Centered Earth-Fixed (ECEF) transform via Greenwich Mean Sidereal Time (GMST).
4. Sub-satellite ground track computation (latitude, longitude, altitude).
5. Ground station line-of-sight elevation and pass schedule calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Earth Physical & Gravitational Constants (WGS-84)
EARTH_RADIUS_KM = 6378.137
EARTH_MU = 398600.4418  # km^3 / s^2
EARTH_J2 = 1.08262668e-3
EARTH_J3 = -2.532656485e-6
EARTH_J4 = -1.61962159e-6
EARTH_ROTATION_RATE_RAD_S = 7.2921159e-5  # rad/s
SPEED_OF_LIGHT_M_S = 299792458.0  # m/s
BOLTZMANN_K = 1.380649e-23  # J/K


@dataclass
class GroundStation:
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    min_elevation_deg: float = 10.0


# Standard Deep Space Network / Earth Ground Stations
DEFAULT_GROUND_STATIONS = [
    GroundStation("Goldstone", 35.426, -116.890, 1036.0, 10.0),
    GroundStation("Madrid", 40.427, -4.248, 789.0, 10.0),
    GroundStation("Canberra", -35.401, 148.981, 655.0, 10.0),
]


@dataclass
class OrbitalElements:
    semi_major_axis_km: float  # a
    eccentricity: float        # e (0 <= e < 1)
    inclination_deg: float     # i (degrees)
    raan_deg: float            # Right Ascension of Ascending Node (degrees)
    arg_perigee_deg: float     # Argument of perigee (degrees)
    mean_anomaly_deg: float    # Mean anomaly at epoch (degrees)
    epoch_timestamp_s: float   # Epoch timestamp in seconds


@dataclass
class OrbitalStatePoint:
    timestamp_s: float
    true_anomaly_deg: float
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    velocity_km_s: float
    in_eclipse: bool
    station_contacts: List[str]


class OrbitPropagator:
    """Propagates satellite orbital state with Earth J2 perturbations."""

    def __init__(
        self,
        elements: OrbitalElements,
        ground_stations: Optional[List[GroundStation]] = None,
    ):
        self.elements = elements
        self.ground_stations = ground_stations or list(DEFAULT_GROUND_STATIONS)

        a = elements.semi_major_axis_km
        e = elements.eccentricity
        i_rad = math.radians(elements.inclination_deg)

        # Mean motion n (rad/s)
        self.n0 = math.sqrt(EARTH_MU / (a ** 3))
        self.period_s = 2.0 * math.pi / self.n0

        # Semi-latus rectum p
        p = a * (1.0 - e ** 2)

        # J2 Secular Perturbations (rad/s)
        j2_factor = 1.5 * EARTH_J2 * ((EARTH_RADIUS_KM / p) ** 2) * self.n0
        self.raan_dot = -j2_factor * math.cos(i_rad)
        self.arg_perigee_dot = 0.5 * j2_factor * (5.0 * (math.cos(i_rad) ** 2) - 1.0)

    def _solve_kepler(self, mean_anomaly_rad: float, ecc: float) -> float:
        """Solve Kepler's equation M = E - e*sin(E) for Eccentric Anomaly E."""
        m = mean_anomaly_rad % (2.0 * math.pi)
        e_curr = m if ecc < 0.8 else math.pi
        for _ in range(15):
            f = e_curr - ecc * math.sin(e_curr) - m
            f_prime = 1.0 - ecc * math.cos(e_curr)
            delta = f / f_prime
            e_curr -= delta
            if abs(delta) < 1e-10:
                break
        return e_curr

    def propagate_state(self, delta_t_s: float) -> OrbitalStatePoint:
        """Propagate orbit state by delta_t seconds from epoch."""
        el = self.elements
        a = el.semi_major_axis_km
        e = el.eccentricity
        i_rad = math.radians(el.inclination_deg)

        # Updated angles under J2 secular drifts
        raan = math.radians(el.raan_deg) + self.raan_dot * delta_t_s
        arg_perigee = math.radians(el.arg_perigee_deg) + self.arg_perigee_dot * delta_t_s
        mean_anomaly = math.radians(el.mean_anomaly_deg) + self.n0 * delta_t_s

        # Solve for Eccentric Anomaly E
        e_anom = self._solve_kepler(mean_anomaly, e)

        # True Anomaly nu
        sin_nu = (math.sqrt(1.0 - e ** 2) * math.sin(e_anom)) / (1.0 - e * math.cos(e_anom))
        cos_nu = (math.cos(e_anom) - e) / (1.0 - e * math.cos(e_anom))
        nu = math.atan2(sin_nu, cos_nu)

        # Orbital radius r
        r = a * (1.0 - e * math.cos(e_anom))

        # Position in perifocal orbital plane (P-Q coordinates)
        x_p = r * math.cos(nu)
        y_p = r * math.sin(nu)

        # Transform perifocal to ECI coordinates via Euler 3-1-3 rotation (raan, inc, arg_perigee)
        px = (math.cos(raan) * math.cos(arg_perigee) - math.sin(raan) * math.sin(arg_perigee) * math.cos(i_rad)) * x_p + \
             (-math.cos(raan) * math.sin(arg_perigee) - math.sin(raan) * math.cos(arg_perigee) * math.cos(i_rad)) * y_p
        py = (math.sin(raan) * math.cos(arg_perigee) + math.cos(raan) * math.sin(arg_perigee) * math.cos(i_rad)) * x_p + \
             (-math.sin(raan) * math.sin(arg_perigee) + math.cos(raan) * math.cos(arg_perigee) * math.cos(i_rad)) * y_p
        pz = (math.sin(arg_perigee) * math.sin(i_rad)) * x_p + (math.cos(arg_perigee) * math.sin(i_rad)) * y_p

        # Vis-Viva orbital speed v = sqrt(mu * (2/r - 1/a))
        v = math.sqrt(max(0.0, EARTH_MU * (2.0 / r - 1.0 / a)))

        # ECEF transformation via Greenwich Mean Sidereal Time (GMST)
        gmst = (EARTH_ROTATION_RATE_RAD_S * delta_t_s) % (2.0 * math.pi)
        ecef_x = math.cos(gmst) * px + math.sin(gmst) * py
        ecef_y = -math.sin(gmst) * px + math.cos(gmst) * py
        ecef_z = pz

        # Geographic Latitude & Longitude
        lat_rad = math.asin(min(1.0, max(-1.0, ecef_z / r)))
        lon_rad = math.atan2(ecef_y, ecef_x)
        alt_km = r - EARTH_RADIUS_KM

        lat_deg = math.degrees(lat_rad)
        lon_deg = math.degrees(lon_rad)

        # Eclipse check (cylindrical Earth shadow approximation, Sun along +X ECI)
        # In eclipse when behind Earth: px < 0 and sqrt(py^2 + pz^2) < EARTH_RADIUS_KM
        in_eclipse = (px < 0.0) and (math.sqrt(py ** 2 + pz ** 2) < EARTH_RADIUS_KM)

        # Ground station line of sight check
        contacts = []
        sat_pos = (ecef_x, ecef_y, ecef_z)
        for stn in self.ground_stations:
            if self._station_visible(stn, sat_pos):
                contacts.append(stn.name)

        return OrbitalStatePoint(
            timestamp_s=el.epoch_timestamp_s + delta_t_s,
            true_anomaly_deg=math.degrees(nu) % 360.0,
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            altitude_km=alt_km,
            velocity_km_s=v,
            in_eclipse=in_eclipse,
            station_contacts=contacts,
        )

    def _station_visible(self, stn: GroundStation, sat_ecef: Tuple[float, float, float]) -> bool:
        """Calculate if satellite is above station elevation cutoff."""
        stn_lat = math.radians(stn.latitude_deg)
        stn_lon = math.radians(stn.longitude_deg)
        r_stn = EARTH_RADIUS_KM + (stn.elevation_m / 1000.0)

        # Station ECEF position
        sx = r_stn * math.cos(stn_lat) * math.cos(stn_lon)
        sy = r_stn * math.cos(stn_lat) * math.sin(stn_lon)
        sz = r_stn * math.sin(stn_lat)

        # Range vector from station to satellite
        rx = sat_ecef[0] - sx
        ry = sat_ecef[1] - sy
        rz = sat_ecef[2] - sz
        range_dist = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

        # Up normal vector at station
        ux = math.cos(stn_lat) * math.cos(stn_lon)
        uy = math.cos(stn_lat) * math.sin(stn_lon)
        uz = math.sin(stn_lat)

        # Dot product with zenith normal
        cos_zenith = (rx * ux + ry * uy + rz * uz) / max(1e-6, range_dist)
        elevation_rad = 0.5 * math.pi - math.acos(min(1.0, max(-1.0, cos_zenith)))
        elevation_deg = math.degrees(elevation_rad)

        return elevation_deg >= stn.min_elevation_deg

    def compute_azimuth_elevation_range(
        self,
        stn: GroundStation,
        delta_t_s: float,
    ) -> Tuple[float, float, float]:
        """Compute topocentric Azimuth (deg), Elevation (deg), and Range (km) to ground station."""
        state = self.propagate_state(delta_t_s)
        alt_km = state.altitude_km
        r_mag = EARTH_RADIUS_KM + alt_km
        lat_rad = math.radians(state.latitude_deg)
        lon_rad = math.radians(state.longitude_deg)
        sat_x = r_mag * math.cos(lat_rad) * math.cos(lon_rad)
        sat_y = r_mag * math.cos(lat_rad) * math.sin(lon_rad)
        sat_z = r_mag * math.sin(lat_rad)

        stn_lat = math.radians(stn.latitude_deg)
        stn_lon = math.radians(stn.longitude_deg)
        r_stn = EARTH_RADIUS_KM + (stn.elevation_m / 1000.0)

        # Station ECEF position
        sx = r_stn * math.cos(stn_lat) * math.cos(stn_lon)
        sy = r_stn * math.cos(stn_lat) * math.sin(stn_lon)
        sz = r_stn * math.sin(stn_lat)

        # Vector from station to satellite in ECEF
        rx = sat_x - sx
        ry = sat_y - sy
        rz = sat_z - sz
        range_dist = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

        # Topocentric ENU basis vectors:
        e_east = -math.sin(stn_lon) * rx + math.cos(stn_lon) * ry
        e_north = -math.sin(stn_lat) * math.cos(stn_lon) * rx - math.sin(stn_lat) * math.sin(stn_lon) * ry + math.cos(stn_lat) * rz
        e_up = math.cos(stn_lat) * math.cos(stn_lon) * rx + math.cos(stn_lat) * math.sin(stn_lon) * ry + math.sin(stn_lat) * rz

        elevation_deg = math.degrees(math.asin(max(-1.0, min(1.0, e_up / max(1e-6, range_dist)))))
        azimuth_deg = math.degrees(math.atan2(e_east, e_north)) % 360.0

        return round(azimuth_deg, 2), round(elevation_deg, 2), round(range_dist, 3)

    def compute_atmospheric_drag(
        self,
        altitude_km: float,
        velocity_km_s: float,
        area_m2: float = 1.0,
        mass_kg: float = 100.0,
        cd: float = 2.2,
    ) -> Dict[str, float]:
        """Compute thermospheric atmospheric drag acceleration and orbital decay rate.
        
        Uses layered exponential scale height density model:
            rho(h) = rho_0 * exp(-(h - h_0) / H)
        """
        table = [
            (100.0, 150.0, 5.297e-10, 5.877),
            (150.0, 200.0, 2.076e-11, 26.74),
            (200.0, 300.0, 2.541e-12, 43.49),
            (300.0, 500.0, 1.916e-13, 53.628),
            (500.0, 800.0, 6.967e-15, 76.4),
            (800.0, 2000.0, 1.170e-16, 120.0),
        ]
        
        alt = max(100.0, min(2000.0, altitude_km))
        rho = 1.170e-16
        for h_min, h_max, rho0, H in table:
            if h_min <= alt <= h_max:
                rho = rho0 * math.exp(-(alt - h_min) / H)
                break
        
        v_m_s = velocity_km_s * 1000.0
        f_drag_n = 0.5 * rho * (v_m_s ** 2) * cd * area_m2
        a_drag_m_s2 = f_drag_n / max(0.1, mass_kg)
        
        a_km = EARTH_RADIUS_KM + alt
        da_dt_km_s = -2.0 * math.sqrt((a_km ** 3) / EARTH_MU) * (a_drag_m_s2 / 1000.0)
        da_dt_km_day = da_dt_km_s * 86400.0

        return {
            "altitude_km": altitude_km,
            "density_kg_m3": rho,
            "drag_force_n": f_drag_n,
            "drag_acceleration_m_s2": a_drag_m_s2,
            "decay_rate_km_day": da_dt_km_day,
        }

    def compute_higher_harmonics_perturbations(self, delta_t_s: float) -> Dict[str, float]:
        """Compute secular rates and long-period oscillations under J2, J3, and J4 geopotential harmonics."""
        a = self.elements.semi_major_axis_km
        e = max(1e-5, self.elements.eccentricity)
        i_rad = math.radians(self.elements.inclination_deg)
        p = a * (1.0 - e ** 2)
        n = self.n0
        re_p = EARTH_RADIUS_KM / p

        cos_i = math.cos(i_rad)
        sin_i = math.sin(i_rad)
        raan_dot_j4 = (15.0 / 16.0) * EARTH_J4 * n * (re_p ** 4) * cos_i * (7.0 * (cos_i ** 2) - 3.0)
        arg_perigee_dot_j4 = -(15.0 / 32.0) * EARTH_J4 * n * (re_p ** 4) * (21.0 * (cos_i ** 4) - 14.0 * (cos_i ** 2) + 1.0)

        delta_e_j3_amp = -(EARTH_J3 / (2.0 * EARTH_J2)) * re_p * sin_i
        omega_curr = math.radians(self.elements.arg_perigee_deg) + self.arg_perigee_dot * delta_t_s
        ecc_perturbed = e + delta_e_j3_amp * math.sin(omega_curr)

        total_raan_dot = self.raan_dot + raan_dot_j4
        total_arg_perigee_dot = self.arg_perigee_dot + arg_perigee_dot_j4

        return {
            "raan_dot_j2_rad_s": self.raan_dot,
            "raan_dot_j4_rad_s": raan_dot_j4,
            "total_raan_dot_deg_day": math.degrees(total_raan_dot) * 86400.0,
            "arg_perigee_dot_j2_rad_s": self.arg_perigee_dot,
            "arg_perigee_dot_j4_rad_s": arg_perigee_dot_j4,
            "total_arg_perigee_dot_deg_day": math.degrees(total_arg_perigee_dot) * 86400.0,
            "j3_eccentricity_amplitude": delta_e_j3_amp,
            "instantaneous_perturbed_eccentricity": max(0.0, ecc_perturbed),
        }

    @staticmethod
    def compute_rf_link_budget(
        tx_power_dbw: float,
        tx_gain_dbi: float,
        rx_gain_dbi: float,
        frequency_hz: float,
        range_km: float,
        bandwidth_hz: float = 1e6,
        noise_temp_k: float = 290.0,
        required_eb_n0_db: float = 6.0,
    ) -> Dict[str, float]:
        """Compute Friis space RF link budget and digital transmission margin."""
        d_m = max(1.0, range_km * 1000.0)
        fspl_db = 20.0 * math.log10(4.0 * math.pi * d_m * frequency_hz / SPEED_OF_LIGHT_M_S)
        rx_power_dbw = tx_power_dbw + tx_gain_dbi + rx_gain_dbi - fspl_db
        noise_power_watts = BOLTZMANN_K * noise_temp_k * bandwidth_hz
        noise_power_dbw = 10.0 * math.log10(max(1e-30, noise_power_watts))
        snr_db = rx_power_dbw - noise_power_dbw
        link_margin_db = snr_db - required_eb_n0_db

        return {
            "range_km": range_km,
            "fspl_db": round(fspl_db, 2),
            "rx_power_dbw": round(rx_power_dbw, 2),
            "noise_power_dbw": round(noise_power_dbw, 2),
            "snr_db": round(snr_db, 2),
            "link_margin_db": round(link_margin_db, 2),
            "link_closed": link_margin_db >= 0.0,
        }

    def generate_ground_track(
        self,
        duration_s: float,
        step_s: float = 60.0,
    ) -> List[OrbitalStatePoint]:
        """Generate high-density sub-satellite ground track over duration."""
        points: List[OrbitalStatePoint] = []
        t = 0.0
        while t <= duration_s:
            points.append(self.propagate_state(t))
            t += step_s
        return points

    def predict_station_passes(
        self,
        duration_s: float,
        step_s: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """Compute upcoming ground station contact passes with AOS/LOS and max elevation."""
        passes: List[Dict[str, Any]] = []
        active_passes: Dict[str, Dict[str, Any]] = {}

        t = 0.0
        while t <= duration_s:
            state = self.propagate_state(t)
            current_contacts = set(state.station_contacts)

            # Check newly established contacts (AOS)
            for stn_name in current_contacts:
                if stn_name not in active_passes:
                    active_passes[stn_name] = {
                        "station": stn_name,
                        "aos_time_s": t,
                        "los_time_s": t,
                        "max_altitude_km": state.altitude_km,
                        "sample_count": 1,
                    }
                else:
                    active_passes[stn_name]["los_time_s"] = t
                    active_passes[stn_name]["sample_count"] += 1

            # Check terminated contacts (LOS)
            finished_stations = [s for s in active_passes if s not in current_contacts]
            for stn_name in finished_stations:
                p = active_passes.pop(stn_name)
                p["duration_seconds"] = round(p["los_time_s"] - p["aos_time_s"] + step_s, 1)
                passes.append(p)

            t += step_s

        # Flush any remaining active passes
        for stn_name, p in active_passes.items():
            p["duration_seconds"] = round(p["los_time_s"] - p["aos_time_s"] + step_s, 1)
            passes.append(p)

        return passes

    @classmethod
    def from_tle(cls, name: str, line1: str, line2: str, ground_stations: Optional[List[GroundStation]] = None) -> OrbitPropagator:
        """Parse standard NORAD Two-Line Element (TLE) set into an OrbitPropagator."""
        # Line 2 columns:
        # Inclination: 09-16
        # RAAN: 18-25
        # Eccentricity: 27-33 (assumed decimal point)
        # Arg of Perigee: 35-42
        # Mean Anomaly: 44-51
        # Mean Motion: 53-63 (revs/day)
        inc = float(line2[8:16].strip())
        raan = float(line2[17:25].strip())
        ecc = float("0." + line2[26:33].strip())
        arg_p = float(line2[34:42].strip())
        ma = float(line2[43:51].strip())
        mean_motion_rev_day = float(line2[52:63].strip())

        # n in rad/s: rev/day * 2pi / 86400
        n_rad_s = mean_motion_rev_day * 2.0 * math.pi / 86400.0
        # a = (mu / n^2)^(1/3)
        semi_major_axis = (EARTH_MU / (n_rad_s ** 2)) ** (1.0 / 3.0)

        elements = OrbitalElements(
            semi_major_axis_km=round(semi_major_axis, 3),
            eccentricity=ecc,
            inclination_deg=inc,
            raan_deg=raan,
            arg_perigee_deg=arg_p,
            mean_anomaly_deg=ma,
            epoch_timestamp_s=1700000000.0,
        )
        return cls(elements=elements, ground_stations=ground_stations)
