# Drone + Machine-Gun Recoil Simulation

A 6-DOF MuJoCo simulation of an X-frame quadrotor flown by a cascaded PID
controller, carrying one of several mountable machine guns. The point of
the experiment is to study **how recoil disturbs flight** and how well the
flight controller can compensate.

```
   .                       (rotor)               .
    \                                           /
     \  rotor             [drone body]    rotor /
      \____________  ------|----|------  _____/
                     |     |    |     |
                     |     +----+     |
                     |       v        |   <-- gun (mounted under nose)
                     +================+
                                       >>>>>>  bullet
                                <----  recoil F
```

## Run it

```
# Autonomous waypoint demo (drone fires while it dwells at each point):
env/bin/python main.py --mode auto

# Pilot it yourself, with a different weapon:
env/bin/python main.py --mode keyboard --gun pkm

# See all weapons and their stats:
env/bin/python main.py --list-guns
```

Keyboard mode uses **the numeric keypad** (NumLock ON) because every letter
A–Z is bound to a render-flag toggle inside the MuJoCo viewer (W toggles
wireframe, D toggles SDF, etc.) and there's no way to suppress those. The
numpad keys are unbound, so they don't fight the viewer.

```
        7 yaw-L     8 fwd      9 yaw-R
        4 strafe-L  5 RESET    6 strafe-R
                    2 back
        +  climb               -  descend
        0  FIRE
```

## Files

| File             | Role                                                              |
|------------------|-------------------------------------------------------------------|
| [config.py](config.py)            | All adjustable constants (drone, controller, sim, gun, disturbances, projectiles). |
| [gun.py](gun.py)                  | `Gun` class + a catalog of weapon models. Drop in a new one to add a gun. |
| [controller.py](controller.py)    | Cascaded position+attitude PID + the rotor mixer.            |
| [modes.py](modes.py)              | Setpoint generators: keyboard pilot, autonomous waypoint loop.|
| [disturbances.py](disturbances.py)| Wind (Ornstein-Uhlenbeck gusts), aero drag, per-shot recoil noise.|
| [projectiles.py](projectiles.py)  | Optional bullet + casing physics. Casing pool, bullet ballistics, prop-hit detection.|
| [main.py](main.py)                | Builds the MuJoCo XML from config, runs the sim+control+gun+projectile loop.|

---

# Physics

## 1. Quadrotor as a 6-DOF rigid body

The drone has six degrees of freedom: position **p** = (x, y, z) and orientation,
expressed internally as a unit quaternion **q** = (q_w, q_x, q_y, q_z) to avoid
gimbal lock. Newton's and Euler's equations:

```
m * a    =  Σ F                          (translation)
I * dω/dt + ω × (I·ω)  =  Σ τ            (rotation, body frame)
```

where **m** is total mass, **I** is the inertia tensor, **a** is linear
acceleration, **ω** is body-frame angular velocity, and the sums are over all
forces and torques. MuJoCo integrates these for us; our code just supplies
forces.

The only forces on the drone are:
1. Four propeller thrusts (along body +z, applied at the rotor positions)
2. Gravity (along world −z)
3. Recoil from the mounted gun (along body −fire-direction, at the muzzle)

## 2. Propeller thrust and torque

Each rotor obeys the standard quadratic momentum-theory laws:

```
T_i  =  k_T * ω_i²        (thrust force, N)
Q_i  =  k_Q * ω_i²        (drag torque on body, N·m, opposite to rotor spin)
```

where ω_i is the rotor's angular velocity in rad/s. The drag torque is what
lets a quadrotor yaw — by spinning two rotors faster than the other two, the
**net** drag torque around z becomes nonzero. With diagonally paired
spin directions (CW–CCW–CW–CCW), the net yaw torque is zero at hover,
which is why the layout works.

For the default loadout (~15-inch prop):

| Constant   | Value     | Units      |
|------------|-----------|------------|
| k_T        | 2.5 × 10⁻⁴ | N / (rad/s)² |
| k_Q        | 5.0 × 10⁻⁶ | N·m / (rad/s)² |
| ω_max      | 600       | rad/s (~5700 RPM) |
| T_max/rotor| 90        | N |
| τ_motor    | 0.05      | s (motor first-order lag) |

The controller in this sim commands **thrust** directly (in newtons) and the
implicit ω just falls out of the rotor law. Each motor in the MuJoCo XML
applies `gear="0 0 1 0 0 ±k_Q/k_T"`, meaning: for control input *T*, apply
force *(0,0,T)* at the rotor site (giving correct thrust + roll/pitch
moment-arm) plus a yaw torque ±(k_Q/k_T)·T.

## 3. Motor first-order lag

Real motors can't change RPM instantly. We model that as a discrete first-
order filter at simulator rate:

```
T_actual  +=  α * (T_command − T_actual)        with  α = dt / (τ_motor + dt)
```

Set `motor_tau = 0` in `config.py` to make the drone unrealistically
responsive (useful for tuning).

## 4. The X-frame mixer

Going from a desired wrench `(T_total, τ_x, τ_y, τ_z)` to per-rotor thrust
commands is just a 4×4 matrix inversion. With rotors at body-frame positions
(±L, ±L, 0) where `L = arm_length / √2`, and alternating spin directions
`s_i ∈ {+1, −1}` such that diagonally opposite rotors share s:

```
    [ 1   1   1   1 ]   [T_1]     [ T_total ]
    [-L   L   L  -L ] · [T_2]  =  [  τ_x    ]
    [-L  -L   L   L ]   [T_3]     [  τ_y    ]
    [ c  -c   c  -c ]   [T_4]     [  τ_z    ]    (c = k_Q / k_T)
```

The controller assembles the right-hand side and the mixer multiplies by the
precomputed inverse matrix.

## 5. Cascaded PID

A quadrotor is **underactuated**: 4 rotors → 4 controllable DOFs, but you
might want to control 6. The trick is that the four chosen DOFs
`(thrust, τ_x, τ_y, τ_z)` give you direct control of altitude and attitude,
and *attitude can be used to point thrust horizontally* — so horizontal
position is reachable but only by tilting first. That's why the controller
is cascaded:

### Outer loop — position controller

Runs at 250 Hz. PID on world-frame position error:

```
a_des  =  K_p_pos * (p_des − p)                      (proportional)
       +  K_i_pos * ∫(p_des − p) dt                  (integral)
       +  K_d_pos * (0 − v)                          (derivative — desired velocity is 0)
       +  g * ẑ                                      (gravity-cancel feedforward)
```

The integral term has horizontal components in this build so that a **constant
disturbance** like sustained recoil is driven out instead of producing a
permanent offset. The integrator has anti-windup clamping.

Then it converts the desired world-frame acceleration into a
`(roll_des, pitch_des, T_total)` triplet using a small-angle decomposition
in the yaw-aligned frame:

```
a_b_x  =   cos(ψ)·a_des_x + sin(ψ)·a_des_y          (forward in yaw frame)
a_b_y  =  -sin(ψ)·a_des_x + cos(ψ)·a_des_y          (left    in yaw frame)
pitch_des  =  arctan2(a_b_x, a_des_z)               (lean forward = +pitch)
roll_des   =  arctan2(-a_b_y, a_des_z)              (lean right  = -roll)
T_total    =  m * a_des_z / (cos(roll) · cos(pitch))
```

Tilt angles are clipped to `max_tilt_deg`. This caps the maximum horizontal
acceleration the drone is willing to demand — important when fighting a big
recoil force.

### Inner loop — attitude controller

Runs at the same 250 Hz. PID on Euler-angle error:

```
α_des  =  K_p_att * angle_error  +  K_i_att * ∫angle_error  −  K_d_att * ω_body
τ      =  diag(I_x, I_y, I_z) · α_des
```

Two small but important details:
* **D-term reads angular velocity directly** instead of differentiating an
  angle signal, which would amplify quaternion conversion noise.
* **Output is angular acceleration**, multiplied by the inertia tensor to
  produce torque. This makes the closed-loop bandwidth depend on K_p_att
  alone — *not* on inertia — so the gains transfer between drone sizes.
  (Motor lag and mixer saturation still impose a practical limit.)

## 6. Recoil — what the gun actually does to the drone

Each shot, the gun expels a bullet (mass `m_b`, velocity `v_b`) and a slug of
hot propellant gas (mass `m_p`, average velocity ≈ 1.4·v_b). By momentum
conservation, the gun (and through it, the drone) absorbs an equal and
opposite impulse:

```
J_per_shot  =  m_b * v_b  +  m_p * (1.4 * v_b)         [N·s]
```

For full-auto fire at cyclic rate **R** (rounds per second), the average
recoil force on the drone is

```
F_recoil_avg  =  J_per_shot * R                         [N]
```

For each cartridge in the catalog:

| Cartridge       | m_b (g) | v_b (m/s) | m_p (g) | J/shot (N·s) |
|-----------------|---------|-----------|---------|--------------|
| 9×19 Parabellum | 8.0     | 360       | 0.40    | 3.08         |
| 5.56×45 NATO    | 4.0     | 940       | 1.70    | 5.99         |
| 7.62×39 (AK)    | 7.9     | 715       | 1.60    | 7.25         |
| 7.62×51 NATO    | 9.5     | 850       | 3.00    | 11.65        |
| 7.62×54R (PKM)  | 9.6     | 825       | 3.10    | 11.50        |

And the catalog of weapons (mass / cyclic rate / capacity / avg recoil):

| Gun        | Cartridge   | Mass | Rate    | Capacity | F_avg  |
|------------|-------------|------|---------|----------|--------|
| Glock 18   | 9×19        | 0.66 kg | 1200 RPM | 33   | 62 N   |
| M4         | 5.56 NATO   | 3.4 kg  | 800 RPM  | 30   | 80 N   |
| AKM        | 7.62×39     | 3.3 kg  | 600 RPM  | 30   | 73 N   |
| M249 SAW   | 5.56 NATO   | 7.5 kg  | 800 RPM  | 200  | 80 N   |
| PKM        | 7.62×54R    | 7.5 kg  | 650 RPM  | 100  | 125 N  |
| M134       | 7.62 NATO   | 18 kg   | 4000 RPM | 2000 | 776 N  |

### How recoil enters the simulation

We don't average it — we apply each shot as a discrete impulse over one sim
step (`dt = 2 ms`). Per shot:

```
F_body  =  -fire_direction * J / dt            (huge force for one timestep)
M_body  =  mount_offset × F_body               (moment about drone CoM)
```

Both vectors are then rotated into the world frame (because MuJoCo's
`xfrc_applied` expects world-frame wrenches) and added to the drone body for
that step:

```python
data.xfrc_applied[drone, 0:3] = R · F_body
data.xfrc_applied[drone, 3:6] = R · M_body
```

The integrated impulse `F·dt = J` is correct, *and* the discrete shot-to-shot
dynamics are preserved — for low-RoF guns you can actually see each round
shake the drone, while for the M134 the fire rate is so high it acts as a
steady push.

The **moment arm** matters. The gun is mounted forward of and below the CoM
(typical real-world geometry — clear of the rotors, balanced ammo box). A
backward force at a below-CoM mount produces a **pitch-up torque**, which
the controller has to fight on top of the linear push.

## 7. Why the drone has to be big

A bare 0.5 kg racing drone with hover thrust ≈ 1.2 N per rotor would be
tossed into orbit by an M4's recoil (80 N average, 65 × the drone's weight).
Real military weaponized drones are **5–25 kg** for exactly this reason.
The defaults here:

| Quantity        | Value           |
|-----------------|-----------------|
| Airframe mass   | 2.5 kg          |
| Arm length      | 0.35 m          |
| Iₓₓ, I_yy       | 0.05 kg·m²      |
| I_zz            | 0.09 kg·m²      |
| Max thrust/rotor| 90 N            |
| Total max thrust| 360 N           |

Loaded with an M249 (7.5 kg gun + 200 × 11.8 g rounds = 9.86 kg payload), the
drone weighs ~12.4 kg ≈ 121 N. Hover throttle is 34 % of max — plenty of
headroom for tilt-compensation maneuvering. Loaded with a Glock 18 (0.66 +
0.40 = 1.06 kg), the drone weighs ~3.6 kg, and full-auto recoil at 62 N is
**178 % of its weight** — physically uncompensable, the drone tumbles
backwards. That's not a bug, it's the experimental answer: full-auto handgun
fire on a small drone is unphysical.

## 8. Where the inertia tensor comes from

The MuJoCo body's inertia is the airframe's plus a parallel-axis transfer for
the rigidly mounted gun + ammo treated as a point mass at the mount offset
**r** = (m_x, m_y, m_z):

```
I_xx_total  =  I_xx_drone  +  m_gun · (m_y² + m_z²)
I_yy_total  =  I_yy_drone  +  m_gun · (m_x² + m_z²)
I_zz_total  =  I_zz_drone  +  m_gun · (m_x² + m_y²)
```

For a 9.86 kg M249 mounted at (0.30, 0, −0.06), I_yy goes from 0.05 to
**0.92** kg·m² — pitch becomes an order of magnitude harder to swing. The
controller compensates because its gains are inertia-scaled (see §5).

## 9. Tuning hints

| Symptom                                | Adjust                              |
|----------------------------------------|-------------------------------------|
| Drone wobbles / overshoots in attitude | ↓ `kp_att`, ↑ `kd_att`             |
| Drone has steady horizontal offset under fire | ↑ `ki_pos[0:2]`              |
| Drone sags below altitude setpoint     | ↑ `ki_pos[2]`                       |
| Drone responds too aggressively        | ↓ `kp_pos`, or ↓ `max_tilt_deg`     |
| More realistic sluggish drone          | ↑ `motor_tau` (try 0.08 – 0.12)     |
| Drone tumbles every time it fires      | gun is too big — try a lighter one or scale up `mass`, `omega_max`, `arm_length` |

## 10. Stochastic disturbances

Realistic flight is never deterministic. Two sources of randomness are
modeled, both driven by a single `numpy.random.Generator` seeded by `--seed`
so any run can be reproduced exactly.

### Wind

Mean wind plus an **Ornstein-Uhlenbeck** turbulent gust process — a
band-limited Gaussian noise with a sensible correlation time (1-3 s for
near-ground turbulence). Each component obeys

```
dW/dt  =  -W/τ  +  σ · √(2/τ) · η(t)
```

where η is unit white noise. The stationary distribution is N(0, σ²); τ
controls how quickly gusts evolve. Apparent wind on the drone is
`v_apparent = wind − v_drone`, and we apply quadratic drag

```
F_drag  =  ½ · ρ · Cd·A · |v_apparent| · v_apparent
```

with the airframe's `Cd·A` product (default 0.025 m², typical of a 0.35 m
quadrotor presenting itself broadside). At 5 m/s headwind that's about
0.4 N — small relative to a 100 N drone weight, but visible in flight as a
slow lean and integrator-driven drift.

### Per-shot recoil noise

Real-gun recoil varies shot to shot because of:

* powder charge tolerance (~1-2 % std dev for milspec ammo)
* bullet weight tolerance (~0.5 %)
* friction in the action (gas-operated systems are noisier than direct blowback)
* barrel temperature affecting peak chamber pressure
* muzzle whip (the barrel oscillates as the bullet exits, giving each shot a slightly different exit angle)

Two knobs implement this:

* `--recoil-noise σ` multiplies each shot's impulse by `1 + σ·z` where `z ~ N(0, 1)`. Typical: 0.02-0.05.
* `--recoil-angle-noise σ_deg` adds a Gaussian wobble to the firing direction. Typical: 0.5-1.5°.

Over many shots both average out to the nominal value, so steady-state
behaviour is unchanged — but the controller has to fight a more chaotic
disturbance, and the bullets visibly scatter.

### Reproducibility

```
python main.py --seed 42 --gun m4_carbine --recoil-noise 0.04 --wind 5 0 0 --gust 1.5
```

Same seed + same flags = bit-for-bit identical trajectory. Useful for
A/B testing controller changes against a fixed disturbance profile.

---

## 11. Bullets and casings (`--projectiles`)

Off by default; enable with `--projectiles`.

### Bullets

Bullets travel at 360-940 m/s — too fast for honest MuJoCo collision
detection at 2 ms timesteps (a 5.56 NATO bullet covers 1.9 m per step,
tunneling through any floor or target thinner than that). So bullets are
simulated entirely in **Python** with explicit ballistics:

```
F_drag  =  -½ · ρ · Cd · A · |v| · v        (per kg)
acc     =  g + F_drag / m
```

The frontal area `A` is approximated from the bullet's mass assuming lead
density (11.34 g/cm³) and a sphere — close enough for visualization; for
actual external-ballistics accuracy you'd plug in cartridge-specific G7
coefficients. Each active bullet renders as a short line segment between
its previous and current position in `viewer.user_scn`, giving a tracer
effect.

### Casings

Casings are slow (3-8 m/s ejection), so they're full MuJoCo free-joint
rigid bodies. We **pre-allocate a pool** in the XML (default 60 slots)
because MuJoCo doesn't support runtime body creation. To "spawn" a casing
we teleport one of the parked bodies to the gun's ejection port and write
its linear and angular velocity:

```
v_world  =  v_drone  +  R_body→world · (eject_speed · eject_dir_body + jitter)
ω_world  ~  N(0, 20 rad/s)         (real casings tumble fast leaving the port)
```

Once a casing's lifetime expires (default 5 s), it's parked back below the
floor with zero velocity. Because the pool is a ring buffer, sustained
firing eventually overwrites old casings — fine for a visualization layer.

Each gun in the catalog declares its own case mass + dimensions, so the
in-flight rendering uses the right brass for the chosen weapon (a 50 mm
.50 BMG case looks distinctly larger than a 19 mm 9 mm case).

### Casing-vs-propeller collision

Each rotor gets a thin **disc geom** placed at the rotor location, set up
with collision bitmasks so it ONLY collides with casings — never the floor,
never the airframe, never bullets:

```
contype/conaffinity bitmasks
       bit 1 = floor
       bit 2 = drone airframe       (collides with floor only)
       bit 4 = propeller discs      (collides with casings only)
       bit 8 = casings              (collides with floor + airframe + props)
```

After every physics step we walk MuJoCo's contact list, and any
casing-vs-prop pair is logged once and the casing is **immediately
despawned** — this models the real prop pulverizing the brass on first
contact (without it, the static disc would let the casing sit there
generating duplicate contacts every step).

### Would a casing damage a real propeller?

**Yes — and badly.** Order of magnitude:

* Spent 5.56 NATO brass case: 5 g, ejected at 3-6 m/s
* 38 cm carbon prop spinning at 5000 RPM: tip speed ≈ 100 m/s
* Relative impact velocity: 50-100 m/s
* Kinetic energy at impact: ½ · 0.005 · 100² ≈ **25 J**

That's far more than a thin carbon-fiber blade can absorb without chipping
or fracturing. Real military weaponized drones use:

* **Brass deflectors** that route ejected cases away from the rotor disc.
* **Downward-ejecting** weapons (some M4 variants are converted to bottom-
  eject for vehicle/aircraft mounts).
* **Sacrificial / replaceable** props on cheap loitering munitions where
  per-mission damage is acceptable.

Run `python main.py --gun m249_saw --projectiles --mode auto` and watch the
console — with the default M4-pattern right-side ejection, ~70 % of spent
cases land on the front-right rotor in the first few seconds.

---

## 12. Sources

* Mellinger & Kumar, *Minimum snap trajectory generation and control for
  quadrotors*, ICRA 2011 — the canonical cascaded controller derivation.
* Bouabdallah, *Design and control of quadrotors with application to
  autonomous flying*, EPFL 2007 — rotor model and inertia values.
* Modern Firearms encyclopedia (modernfirearms.net) and Wikipedia for cartridge
  ballistics and weapon cyclic rates.
* MuJoCo documentation (mujoco.readthedocs.io) for the `<motor>` actuator and
  free-joint state conventions.
