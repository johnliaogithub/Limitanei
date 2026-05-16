"""Drone-mountable weapons: recoil physics + a small catalog of models.

Each gun is a rigid attachment to the drone body. Per shot the gun produces
an impulse equal to the momentum carried away by the bullet plus the
propellant gases:

    J  =  m_bullet * v_muzzle  +  m_propellant * (k_gas * v_muzzle)

with a textbook k_gas ~ 1.4 (gas leaves the muzzle ~40 % faster than the
bullet on average). Per-shot impulses for common cartridges are around:

    9 x 19      Parabellum   ~  3.0  N*s
    5.56 x 45   NATO         ~  4.7  N*s
    7.62 x 39   AK           ~  6.4  N*s
    7.62 x 51   NATO         ~ 11.7  N*s

Average recoil thrust during sustained automatic fire is

    F_avg  =  J  *  cyclic_rate     [N]

For an M249 SAW (J ~ 4.7, cyclic 13 Hz)  ->  ~60 N continuous backward push,
which on a 2.5 kg drone is about 2.5 g of horizontal acceleration: easy for
the controller to compensate, but you can SEE it in flight.

Each shot is applied to the drone body as an impulsive wrench (force +
torque about the CoM, since the muzzle is offset from the CoM):

    F_recoil_body =  -fire_dir_body * J / dt        (over one sim step)
    M_recoil_body =  mount_offset x F_recoil_body   (moment from off-center mount)

Then transformed into the world frame and added to data.xfrc_applied.

To add a new gun: drop another entry in GUNS at the bottom.
"""
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


PROPELLANT_GAS_FACTOR = 1.4   # v_gas_avg / v_bullet, typical for small arms


@dataclass
class Gun:
    """Specification + state for one mounted weapon.

    All vectors are in the drone's BODY frame (x = forward, y = left, z = up)."""
    name:                  str
    mass_kg:               float           # weapon mass alone (no ammo)
    mount_offset_m:        Tuple[float, float, float]   # muzzle position rel. drone CoM
    fire_dir_body:         Tuple[float, float, float]   # unit-vector the bullet leaves along
    bullet_mass_kg:        float
    muzzle_vel_mps:        float
    propellant_mass_kg:    float
    cyclic_rate_rpm:       float           # rounds per minute on full auto
    capacity_rounds:       int             # rounds in the magazine/belt at startup
    round_total_mass_kg:   float           # mass of one cartridge (bullet+case+powder)

    # ---------- spent-casing geometry & ejection (used by --projectiles) ----------
    case_mass_kg:          float = 0.005          # brass case alone (no powder, no bullet)
    case_diameter_m:       float = 0.0096
    case_length_m:         float = 0.045
    eject_port_offset_m:   Tuple[float, float, float] = (0.05, -0.06, 0.0)
    eject_dir_body:        Tuple[float, float, float] = (0.2, -1.0, 0.4)  # mostly to the right (-y), tilted slightly fwd & up
    eject_speed_mps:       float = 4.0            # mean ejection speed, rifle-typical

    # ---------- per-shot ballistic pellets (1 for rifles/SMGs, 9 for 00-buck) ----------
    pellets_per_shot:      int = 1
    pellet_spread_deg:     float = 0.1            # angular std-dev of each pellet from boresight

    # ---------- runtime state ----------
    ammo:                  int = field(init=False)
    _fire_accum:           float = field(init=False, default=0.0)

    def __post_init__(self):
        self.fire_dir_body = tuple(np.asarray(self.fire_dir_body, float)
                                   / np.linalg.norm(self.fire_dir_body))
        self.ammo = self.capacity_rounds

    # ---------- physics quantities ----------
    @property
    def impulse_per_shot_Ns(self) -> float:
        """Total recoil impulse (momentum) per round, in N*s."""
        return (self.bullet_mass_kg * self.muzzle_vel_mps
                + self.propellant_mass_kg * PROPELLANT_GAS_FACTOR * self.muzzle_vel_mps)

    @property
    def avg_recoil_force_N(self) -> float:
        """Time-averaged recoil force during full-auto sustained fire."""
        return self.impulse_per_shot_Ns * (self.cyclic_rate_rpm / 60.0)

    @property
    def loaded_mass_kg(self) -> float:
        """Mass of weapon plus a full load of ammunition."""
        return self.mass_kg + self.capacity_rounds * self.round_total_mass_kg

    # ---------- per-step update ----------
    def step(self, dt: float, firing: bool) -> int:
        """Advance the firing cycle and return the number of shots fired in this dt."""
        if not firing or self.ammo <= 0:
            self._fire_accum = 0.0          # reset cadence when trigger released
            return 0
        period = 60.0 / self.cyclic_rate_rpm
        self._fire_accum += dt
        n = 0
        while self._fire_accum >= period and self.ammo > 0:
            self._fire_accum -= period
            self.ammo -= 1
            n += 1
        return n

    def reload(self):
        self.ammo = self.capacity_rounds
        self._fire_accum = 0.0


# ---------------------------------------------------------------------------
#  Catalog
#
#  Numbers come from open-source small-arms reference data (Wikipedia, the
#  Modern Firearms encyclopedia, and standard ammo manufacturer datasheets).
#  Realistic to within ~5 %; tweak freely.
#
#  Mount offset note: muzzle is forward of CoM by ~0.2-0.3 m on a heavy quad,
#  and slightly below to keep the rotors clear of the muzzle blast. The
#  off-center vertical position means recoil also produces a small
#  PITCH-UP torque on the drone, which the controller has to fight.
# ---------------------------------------------------------------------------
GUNS = {
    "glock18": dict(
        name="Glock 18 (9x19mm Parabellum, full auto)",
        mass_kg=0.66,
        mount_offset_m=(0.20, 0.0, -0.05),
        fire_dir_body=(1.0, 0.0, 0.0),
        bullet_mass_kg=0.0080,
        muzzle_vel_mps=360.0,
        propellant_mass_kg=0.00040,
        cyclic_rate_rpm=1200.0,
        capacity_rounds=33,
        round_total_mass_kg=0.012,
        case_mass_kg=0.0040,
        case_diameter_m=0.0098,
        case_length_m=0.0190,
        eject_speed_mps=3.5,
    ),
    "mp5": dict(
        name="H&K MP5 (9x19mm SMG)",
        mass_kg=2.5,
        mount_offset_m=(0.22, 0.0, -0.05),
        fire_dir_body=(1.0, 0.0, 0.0),
        bullet_mass_kg=0.0080,
        muzzle_vel_mps=400.0,
        propellant_mass_kg=0.00045,
        cyclic_rate_rpm=800.0,
        capacity_rounds=30,
        round_total_mass_kg=0.012,
        case_mass_kg=0.0040,
        case_diameter_m=0.0098,
        case_length_m=0.0190,
        eject_speed_mps=3.5,
    ),
    "m4_carbine": dict(
        name="M4 Carbine (5.56x45 NATO)",
        mass_kg=3.4,
        mount_offset_m=(0.25, 0.0, -0.05),
        fire_dir_body=(1.0, 0.0, 0.0),
        bullet_mass_kg=0.0040,
        muzzle_vel_mps=940.0,
        propellant_mass_kg=0.00170,
        cyclic_rate_rpm=800.0,
        capacity_rounds=30,
        round_total_mass_kg=0.0118,
        case_mass_kg=0.0050,
        case_diameter_m=0.0096,
        case_length_m=0.045,
        eject_speed_mps=4.0,
    ),
    "akm": dict(
        name="AKM (7.62x39mm)",
        mass_kg=3.3,
        mount_offset_m=(0.25, 0.0, -0.05),
        fire_dir_body=(1.0, 0.0, 0.0),
        bullet_mass_kg=0.0079,
        muzzle_vel_mps=715.0,
        propellant_mass_kg=0.00160,
        cyclic_rate_rpm=600.0,
        capacity_rounds=30,
        round_total_mass_kg=0.0163,
        case_mass_kg=0.0073,
        case_diameter_m=0.0115,
        case_length_m=0.039,
        eject_speed_mps=4.5,
    ),
    "aa12_shotgun": dict(
        # Atchisson AA-12: gas-operated 12-gauge full-auto shotgun. Each shot
        # in this catalog assumes 00 buckshot (9 pellets, ~32 g shot total)
        # at ~400 m/s. Per-shot impulse is enormous (~16 N*s); cyclic rate is
        # only 5 Hz so it kicks like a sledgehammer between shots rather than
        # producing a steady push like a high-RPM rifle.
        name="AA-12 Auto Shotgun (12 ga, 00 buckshot)",
        mass_kg=5.5,
        mount_offset_m=(0.30, 0.0, -0.06),
        fire_dir_body=(1.0, 0.0, 0.0),
        bullet_mass_kg=0.036,         # 32 g shot + 4 g wad treated as one mass
        muzzle_vel_mps=400.0,
        propellant_mass_kg=0.00250,
        cyclic_rate_rpm=300.0,
        capacity_rounds=20,
        round_total_mass_kg=0.054,
        case_mass_kg=0.008,
        case_diameter_m=0.0185,
        case_length_m=0.070,
        eject_speed_mps=3.0,
        pellets_per_shot=9,           # 00 buckshot: nine 8.4 mm lead spheres
        pellet_spread_deg=1.5,        # ~30 cm pattern at 25 m
    ),
    "hk416": dict(
        # HK416 A5 with 10.4" barrel — compact CQB variant, 5.56x45 NATO.
        # Best recoil-to-weight ratio in the catalog for a sustained-fire role:
        # avg recoil ~79 N against ~57 N drone+gun weight → ~57% ratio.
        # Belt-fed M249 reaches ~49%; heavier guns fare worse.
        # 850 RPM cyclic with a 30-round STANAG mag (compatible with 100-round
        # Beta-C drum for extended operations). Shorter OAL than a 14.5" barrel
        # improves centre-of-gravity placement under the drone.
        name="HK416 A5 Carbine (5.56x45 NATO, 10.4\" barrel)",
        mass_kg=3.0,
        mount_offset_m=(0.25, 0.0, -0.05),
        fire_dir_body=(1.0, 0.0, 0.0),
        bullet_mass_kg=0.0040,
        muzzle_vel_mps=870.0,
        propellant_mass_kg=0.00170,
        cyclic_rate_rpm=850.0,
        capacity_rounds=30,
        round_total_mass_kg=0.0118,
        case_mass_kg=0.0050,
        case_diameter_m=0.0096,
        case_length_m=0.045,
        eject_speed_mps=4.0,
    ),
    "m249_saw": dict(
        name="M249 SAW (5.56x45 NATO, belt fed)",
        mass_kg=7.5,
        mount_offset_m=(0.30, 0.0, -0.06),
        fire_dir_body=(1.0, 0.0, 0.0),
        bullet_mass_kg=0.0040,
        muzzle_vel_mps=915.0,
        propellant_mass_kg=0.00170,
        cyclic_rate_rpm=800.0,
        capacity_rounds=200,
        round_total_mass_kg=0.0118,
        case_mass_kg=0.0050,
        case_diameter_m=0.0096,
        case_length_m=0.045,
        eject_speed_mps=4.5,
    ),
    "pkm": dict(
        name="PKM (7.62x54R, belt fed)",
        mass_kg=7.5,
        mount_offset_m=(0.32, 0.0, -0.06),
        fire_dir_body=(1.0, 0.0, 0.0),
        bullet_mass_kg=0.0096,
        muzzle_vel_mps=825.0,
        propellant_mass_kg=0.00310,
        cyclic_rate_rpm=650.0,
        capacity_rounds=100,
        round_total_mass_kg=0.0215,
        case_mass_kg=0.011,
        case_diameter_m=0.0124,
        case_length_m=0.054,
        eject_speed_mps=5.0,
    ),
    "barrett_m82": dict(
        # Semi-automatic .50 BMG anti-materiel rifle. Each shot's impulse is
        # ~42 N*s - by far the largest in the catalog per shot. Realistic
        # field sustained rate is ~30 RPM (slower than the mechanical limit).
        name="Barrett M82 (.50 BMG, semi-auto)",
        mass_kg=14.0,
        mount_offset_m=(0.40, 0.0, -0.06),
        fire_dir_body=(1.0, 0.0, 0.0),
        bullet_mass_kg=0.042,
        muzzle_vel_mps=853.0,
        propellant_mass_kg=0.0162,
        cyclic_rate_rpm=30.0,
        capacity_rounds=10,
        round_total_mass_kg=0.114,
        case_mass_kg=0.030,
        case_diameter_m=0.0203,
        case_length_m=0.099,
        eject_speed_mps=4.0,
    ),
    "m134_minigun": dict(
        # Included as a stress test - this gun's recoil EXCEEDS the drone's
        # weight several times over, so the drone will be pushed backwards
        # while firing no matter how good the controller is.
        name="M134 Minigun (7.62x51 NATO, electric Gatling)",
        mass_kg=18.0,
        mount_offset_m=(0.32, 0.0, -0.06),
        fire_dir_body=(1.0, 0.0, 0.0),
        bullet_mass_kg=0.0095,
        muzzle_vel_mps=850.0,
        propellant_mass_kg=0.00300,
        cyclic_rate_rpm=4000.0,
        capacity_rounds=2000,
        round_total_mass_kg=0.0234,
        case_mass_kg=0.011,
        case_diameter_m=0.0124,
        case_length_m=0.051,
        eject_speed_mps=8.0,                  # mechanically ejected, much faster
    ),
}


def make_gun(name: str) -> Gun:
    if name not in GUNS:
        raise ValueError(
            f"Unknown gun '{name}'. Available: {list(GUNS)}"
        )
    return Gun(**GUNS[name])
