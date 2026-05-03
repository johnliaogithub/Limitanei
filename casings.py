"""Spent-casing physics + propeller-impact detection.

Casings are slow enough (3-8 m/s ejection) to be honest MuJoCo free-joint
rigid bodies. We pre-allocate a fixed pool of casing bodies in the XML
because MuJoCo doesn't support runtime body creation. To "spawn" a casing
we teleport one of the parked bodies to the ejection port and write its
linear+angular velocity. To "despawn" we move it far below the floor and
zero its state.

Collision flags (contype/conaffinity bitmasks):

       bit  used by
       ---  ----------------------------------
       1    floor                  (default)
       2    drone airframe         (so drone collides with floor only)
       4    propeller-disc geoms   (only collide with casings)
       8    casings                (collide with floor + props + airframe)

A casing-vs-prop contact is reported in real time so you can see in the
console whether the ejected brass is hitting the rotors.
"""
from typing import List
import numpy as np
import mujoco


# ---------------------------------------------------------------------------
#  XML snippets injected into build_xml() when casings are enabled.
# ---------------------------------------------------------------------------

def casing_pool_xml(n: int, weapon, contype: int = 8, conaffinity: int = 7,
                    park_z: float = -50.0) -> str:
    """N free-joint cylinder bodies sitting parked far below the floor.

    Each casing has the dimensions of one spent cartridge case for the
    chosen weapon. The contype/conaffinity defaults: bit 8 = casings,
    contact with bits 1 (floor) + 2 (drone airframe) + 4 (props) = 7.
    """
    r = weapon.case_diameter_m / 2.0
    half_l = weapon.case_length_m / 2.0
    m = weapon.case_mass_kg
    # Solid-cylinder inertia (cylinder axis along z, half-length half_l):
    Iz = 0.5 * m * r * r
    Ixy = (m / 12.0) * (3.0 * r * r + (2.0 * half_l) ** 2)
    chunks = []
    for i in range(n):
        # park positions spread out so they don't pile up if collisions
        # somehow get evaluated against each other while parked.
        px = 80.0 + 0.3 * i
        chunks.append(f"""
    <body name="casing_{i}" pos="{px} 80 {park_z}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{m}" diaginertia="{Ixy} {Ixy} {Iz}"/>
      <geom type="cylinder" size="{r} {half_l}" rgba="0.85 0.65 0.20 1"
            contype="{contype}" conaffinity="{conaffinity}"
            friction="0.3 0.005 0.0001"/>
    </body>""")
    return "".join(chunks)


def prop_disc_geom_xml(L: float, drone_top_z: float = 0.04) -> str:
    """Four thin disc geoms at the rotor positions, used purely for
    collision with casings (contype bit 4, conaffinity bit 8 = casings only).
    They don't collide with the floor or anything else, so they don't
    interfere with the drone's flight dynamics."""
    # Slightly larger radius than the rotor "site" markers so a near miss
    # is correctly logged as a hit by a real spinning prop.
    r = 0.18
    h = 0.003   # half-height
    return f"""
      <geom name="prop_FR" type="cylinder" size="{r} {h}" pos=" {L} {-L} {drone_top_z}"
            rgba="0.4 0.4 0.4 0.18" contype="4" conaffinity="8"/>
      <geom name="prop_FL" type="cylinder" size="{r} {h}" pos=" {L}  {L} {drone_top_z}"
            rgba="0.4 0.4 0.4 0.18" contype="4" conaffinity="8"/>
      <geom name="prop_BL" type="cylinder" size="{r} {h}" pos="{-L}  {L} {drone_top_z}"
            rgba="0.4 0.4 0.4 0.18" contype="4" conaffinity="8"/>
      <geom name="prop_BR" type="cylinder" size="{r} {h}" pos="{-L} {-L} {drone_top_z}"
            rgba="0.4 0.4 0.4 0.18" contype="4" conaffinity="8"/>"""


# ---------------------------------------------------------------------------
#  Casing pool
# ---------------------------------------------------------------------------

class CasingPool:
    """Manages the ring buffer of pre-allocated casing bodies."""

    def __init__(self, model, data, weapon, lifetime: float, rng,
                 prop_collision_warn: bool = True):
        self.model = model
        self.data = data
        self.weapon = weapon
        self.lifetime = lifetime
        self.rng = rng
        self.warn = prop_collision_warn

        # discover how many "casing_*" bodies exist in the model
        self.body_ids: List[int] = []
        idx = 0
        while True:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"casing_{idx}")
            if bid < 0:
                break
            self.body_ids.append(bid)
            idx += 1
        self.n = len(self.body_ids)
        if self.n == 0:
            return

        # qpos / qvel address for each casing's free joint
        self.qpos_adr = []
        self.dof_adr = []
        for bid in self.body_ids:
            j = model.body_jntadr[bid]
            self.qpos_adr.append(int(model.jnt_qposadr[j]))
            self.dof_adr.append(int(model.jnt_dofadr[j]))

        # geom_id -> slot index, used for contact filtering and to despawn the
        # right slot when a casing-prop hit is detected.
        self._geom_to_slot = {}
        for slot, bid in enumerate(self.body_ids):
            g0 = int(model.body_geomadr[bid])
            ng = int(model.body_geomnum[bid])
            for k in range(ng):
                self._geom_to_slot[g0 + k] = slot

        # prop disc geom ids (used for casing-prop contact warnings)
        self.prop_geom_ids = set()
        for nm in ("prop_FR", "prop_FL", "prop_BL", "prop_BR"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if gid >= 0:
                self.prop_geom_ids.add(gid)

        self.lifetimes = np.zeros(self.n)
        self.next_slot = 0
        for i in range(self.n):
            self._park(i)

    # ------------------------------------------------------------------
    def _park(self, i: int):
        """Send slot i below the floor with zero velocity."""
        adr = self.qpos_adr[i]
        dof = self.dof_adr[i]
        self.data.qpos[adr:adr + 3] = (80.0 + 0.3 * i, 80.0, -50.0)
        self.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dof:dof + 6] = 0.0
        self.lifetimes[i] = 0.0

    def _spawn_one(self, port_world_pos: np.ndarray,
                   eject_vel_world: np.ndarray,
                   drone_lin_vel: np.ndarray):
        """Activate one casing slot at the ejection port."""
        if self.n == 0:
            return
        i = self.next_slot
        self.next_slot = (self.next_slot + 1) % self.n
        adr = self.qpos_adr[i]
        dof = self.dof_adr[i]
        self.data.qpos[adr:adr + 3] = port_world_pos
        # random initial orientation - real casings tumble as they leave
        q = self.rng.standard_normal(4)
        q /= np.linalg.norm(q)
        self.data.qpos[adr + 3:adr + 7] = q
        # linear vel = drone vel + ejection vel + random jitter
        self.data.qvel[dof:dof + 3] = drone_lin_vel + eject_vel_world
        # angular vel: real ejected casings spin at tens of rad/s
        self.data.qvel[dof + 3:dof + 6] = self.rng.standard_normal(3) * 20.0
        self.lifetimes[i] = self.lifetime

    # ------------------------------------------------------------------
    def eject(self, n_shots: int, drone_pos: np.ndarray, drone_R: np.ndarray,
              drone_lin_vel: np.ndarray,
              speed_sigma: float, angle_sigma_deg: float):
        """Spawn `n_shots` casings from the gun's ejection port."""
        if self.n == 0:
            return
        port_body = np.asarray(self.weapon.eject_port_offset_m, float)
        eject_dir_body = np.asarray(self.weapon.eject_dir_body, float)
        eject_dir_body = eject_dir_body / np.linalg.norm(eject_dir_body)
        port_world = drone_pos + drone_R @ port_body
        for _ in range(n_shots):
            speed = max(0.0, self.weapon.eject_speed_mps
                        + speed_sigma * float(self.rng.standard_normal()))
            # angle jitter: build perpendicular axes and tilt
            sigma = np.radians(angle_sigma_deg)
            helper = np.array([0.0, 0.0, 1.0]) if abs(eject_dir_body[2]) < 0.9 \
                     else np.array([1.0, 0.0, 0.0])
            e1 = np.cross(eject_dir_body, helper); e1 /= np.linalg.norm(e1)
            e2 = np.cross(eject_dir_body, e1)
            d = eject_dir_body \
                + sigma * float(self.rng.standard_normal()) * e1 \
                + sigma * float(self.rng.standard_normal()) * e2
            d = d / np.linalg.norm(d)
            v_world = drone_R @ (speed * d)
            self._spawn_one(port_world, v_world, drone_lin_vel)

    # ------------------------------------------------------------------
    def step(self, dt: float):
        """Decrement lifetimes, park anything that timed out."""
        if self.n == 0:
            return
        active = self.lifetimes > 0
        self.lifetimes[active] -= dt
        for i in range(self.n):
            if active[i] and self.lifetimes[i] <= 0:
                self._park(i)

    # ------------------------------------------------------------------
    def check_prop_hits(self) -> int:
        """Scan the contact list for casing-vs-prop contacts.

        Each detected impact:
          * is logged once (not on every subsequent timestep the casing rests
            on the static disc),
          * destroys the casing (parks the slot) - simulating a real spinning
            prop pulverizing the brass on first contact.

        Returns the number of fresh hits this step.
        """
        if self.n == 0 or not self.prop_geom_ids:
            return 0
        slots_to_kill = []
        prop_for_slot = {}
        contact_pos_for_slot = {}
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            casing_g = prop_g = None
            if g1 in self._geom_to_slot and g2 in self.prop_geom_ids:
                casing_g, prop_g = g1, g2
            elif g2 in self._geom_to_slot and g1 in self.prop_geom_ids:
                casing_g, prop_g = g2, g1
            if casing_g is None:
                continue
            slot = self._geom_to_slot[casing_g]
            if slot not in prop_for_slot:
                prop_for_slot[slot] = prop_g
                contact_pos_for_slot[slot] = np.array(con.pos)
                slots_to_kill.append(slot)
        for slot in slots_to_kill:
            if self.warn:
                pname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                          prop_for_slot[slot])
                p = contact_pos_for_slot[slot]
                print(f"[!! prop hit] casing struck {pname} at "
                      f"({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}) - "
                      f"real prop would chip or break")
            self._park(slot)
        return len(slots_to_kill)
