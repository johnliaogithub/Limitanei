# Physics Reference

Detailed derivations for the quadrotor model, controller, recoil, and disturbances used in the simulation.

---

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
forces and torques. MuJoCo integrates these; our code supplies forces.

The only forces on the drone are:
1. Four propeller thrusts (along body +z, applied at the rotor positions)
2. Gravity (along world −z)
3. Recoil from the mounted payload (along body −fire-direction, at the muzzle)

---

## 2. Propeller thrust and torque

Each rotor obeys the standard quadratic momentum-theory laws:

```
T_i  =  k_T * ω_i²        (thrust force, N)
Q_i  =  k_Q * ω_i²        (drag torque on body, N·m, opposite to rotor spin)
```

where ω_i is the rotor's angular velocity in rad/s. The drag torque lets a
quadrotor yaw — by spinning two rotors faster than the other two, the net drag
torque around z becomes nonzero. With diagonally paired spin directions
(CW–CCW–CW–CCW), net yaw torque is zero at hover.

For the default loadout (~15-inch prop):

| Constant   | Value      | Units         |
|------------|------------|---------------|
| k_T        | 2.5 × 10⁻⁴ | N / (rad/s)²  |
| k_Q        | 5.0 × 10⁻⁶ | N·m / (rad/s)²|
| ω_max      | 600        | rad/s (~5700 RPM) |
| T_max/rotor| 90         | N             |
| τ_motor    | 0.05       | s (motor first-order lag) |

---

## 3. Motor first-order lag

Real motors cannot change RPM instantly. We model that as a discrete first-order
filter at simulator rate:

```
T_actual  +=  α * (T_command − T_actual)        with  α = dt / (τ_motor + dt)
```

Set `motor_tau = 0` in `config.py` to make the drone unrealistically
responsive (useful for gain tuning).

---

## 4. The X-frame mixer

Going from a desired wrench `(T_total, τ_x, τ_y, τ_z)` to per-rotor thrust
commands is a 4×4 matrix inversion. With rotors at body-frame positions
(±L, ±L, 0) where `L = arm_length / √2`, and alternating spin directions
`s_i ∈ {+1, −1}`:

```
    [ 1   1   1   1 ]   [T_1]     [ T_total ]
    [-L   L   L  -L ] · [T_2]  =  [  τ_x    ]
    [-L  -L   L   L ]   [T_3]     [  τ_y    ]
    [ c  -c   c  -c ]   [T_4]     [  τ_z    ]    (c = k_Q / k_T)
```

The controller assembles the right-hand side and the mixer multiplies by the
precomputed inverse matrix.

---

## 5. Cascaded PID controller

A quadrotor is **underactuated**: 4 rotors → 4 controllable DOFs. The trick
is that `(thrust, τ_x, τ_y, τ_z)` give direct control of altitude and
attitude, and *attitude can be used to point thrust horizontally* — so
horizontal position is reachable by tilting. That is why the controller is
cascaded:

### Outer loop — position controller

Runs at 250 Hz. PID on world-frame position error:

```
a_des  =  K_p_pos * (p_des − p)
       +  K_i_pos * ∫(p_des − p) dt
       +  K_d_pos * (0 − v)
       +  g * ẑ                          (gravity-cancel feedforward)
```

The integral term has horizontal components so that a **constant disturbance**
like sustained recoil is driven out instead of producing a permanent offset.
The integrator has anti-windup clamping.

Then it converts the desired world-frame acceleration into
`(roll_des, pitch_des, T_total)` using a small-angle decomposition:

```
a_b_x  =   cos(ψ)·a_des_x + sin(ψ)·a_des_y
a_b_y  =  -sin(ψ)·a_des_x + cos(ψ)·a_des_y
pitch_des  =  arctan2(a_b_x, a_des_z)
roll_des   =  arctan2(-a_b_y, a_des_z)
T_total    =  m * a_des_z / (cos(roll) · cos(pitch))
```

Tilt angles are clipped to `max_tilt_deg`.

### Inner loop — attitude controller

Runs at 250 Hz. PID on Euler-angle error:

```
α_des  =  K_p_att * angle_error  +  K_i_att * ∫angle_error  −  K_d_att * ω_body
τ      =  diag(I_x, I_y, I_z) · α_des
```

Two details:
* **D-term reads angular velocity directly** instead of differentiating an
  angle signal, which would amplify quaternion conversion noise.
* **Output is angular acceleration**, multiplied by the inertia tensor to
  produce torque. This makes closed-loop bandwidth depend on K_p_att alone —
  not on inertia — so gains transfer between drone sizes.

### Tuning hints

| Symptom                                | Adjust                              |
|----------------------------------------|-------------------------------------|
| Drone wobbles / overshoots in attitude | ↓ `kp_att`, ↑ `kd_att`             |
| Drone has steady horizontal offset under recoil | ↑ `ki_pos[0:2]`         |
| Drone sags below altitude setpoint     | ↑ `ki_pos[2]`                       |
| Drone responds too aggressively        | ↓ `kp_pos`, or ↓ `max_tilt_deg`     |
| More realistic sluggish drone          | ↑ `motor_tau` (try 0.08 – 0.12)     |
| Drone destabilises every engagement    | payload is too heavy — try lighter one or scale up `mass`, `omega_max`, `arm_length` |

---

## 6. Recoil model

Each shot expels a projectile (mass `m_b`, velocity `v_b`) and propellant gas
(mass `m_p`, average velocity ≈ 1.4·v_b). By momentum conservation the drone
absorbs an equal and opposite impulse:

```
J_per_shot  =  m_b * v_b  +  m_p * (1.4 * v_b)         [N·s]
```

For full-auto fire at cyclic rate **R** (rounds per second):

```
F_recoil_avg  =  J_per_shot * R                         [N]
```

### How recoil enters the simulation

Each shot is applied as a discrete impulse over one sim step (`dt = 2 ms`):

```
F_body  =  -fire_direction * J / dt            (large force for one timestep)
M_body  =  mount_offset × F_body               (moment about drone CoM)
```

Both vectors are rotated into the world frame and added to the drone body via
`data.xfrc_applied`. The **moment arm** matters: the payload is mounted forward
of and below the CoM, so a backward force produces a **pitch-up torque** the
controller must fight in addition to the linear push.

### Payload catalog

| Payload    | Calibre     | Mass   | Rate     | Capacity | F_avg  |
|------------|-------------|--------|----------|----------|--------|
| Glock 18   | 9×19        | 0.66 kg| 1200 RPM | 33       | 62 N   |
| HK416 A5   | 5.56 NATO   | 3.0 kg | 850 RPM  | 30       | 79 N   |
| M4 Carbine | 5.56 NATO   | 3.4 kg | 800 RPM  | 30       | 80 N   |
| AKM        | 7.62×39     | 3.3 kg | 600 RPM  | 30       | 73 N   |
| M249 SAW   | 5.56 NATO   | 7.5 kg | 800 RPM  | 200      | 80 N   |
| PKM        | 7.62×54R   | 7.5 kg | 650 RPM  | 100      | 125 N  |
| AA-12      | 12 ga       | 5.5 kg | 300 RPM  | 20       | 80 N   |
| M134       | 7.62 NATO   | 18 kg  | 4000 RPM | 2000     | 776 N  |

---

## 7. Why the airframe has to be large

A bare 0.5 kg racing drone with hover thrust ≈ 1.2 N per rotor would be
destabilised by an HK416's recoil (79 N average, 66× the drone's weight).
Real autonomous intercept platforms are **5–25 kg** for exactly this reason.
The defaults here:

| Quantity        | Value           |
|-----------------|-----------------|
| Airframe mass   | 2.5 kg          |
| Arm length      | 0.35 m          |
| Iₓₓ, I_yy       | 0.05 kg·m²      |
| I_zz            | 0.09 kg·m²      |
| Max thrust/rotor| 90 N            |
| Total max thrust| 360 N           |

Loaded with the HK416 A5 (3.0 kg + 30-round mag = 3.35 kg payload), the
drone weighs ~5.9 kg ≈ 57 N. The HK416's average recoil (79 N) exceeds the
drone's own weight — the controller must actively compensate during
engagement. This is the primary RL challenge.

---

## 8. Inertia tensor

The body inertia is the airframe's plus a parallel-axis transfer for the
rigidly mounted payload + ammo treated as a point mass at the mount offset
**r** = (m_x, m_y, m_z):

```
I_xx_total  =  I_xx_drone  +  m_payload · (m_y² + m_z²)
I_yy_total  =  I_yy_drone  +  m_payload · (m_x² + m_z²)
I_zz_total  =  I_zz_drone  +  m_payload · (m_x² + m_y²)
```

---

## 9. Stochastic disturbances

### Wind

Mean wind plus an **Ornstein-Uhlenbeck** turbulent gust process:

```
dW/dt  =  -W/τ  +  σ · √(2/τ) · η(t)
```

Stationary distribution is N(0, σ²); τ controls gust evolution rate. Apparent
wind on the drone is `v_apparent = wind − v_drone`, with quadratic drag:

```
F_drag  =  ½ · ρ · Cd·A · |v_apparent| · v_apparent
```

### Per-shot recoil noise

* `--recoil-noise σ` multiplies each shot's impulse by `1 + σ·z`, `z ~ N(0,1)`.
* `--recoil-angle-noise σ_deg` adds Gaussian wobble to the firing direction.

---

## 10. Projectiles and casings

### Projectiles

Projectiles travel at 360–940 m/s — too fast for MuJoCo collision detection
at 2 ms timesteps (a 5.56 NATO round covers 1.9 m per step). So projectiles
are simulated in **Python** with explicit ballistics:

```
F_drag  =  -½ · ρ · Cd · A · |v| · v        (per kg)
acc     =  g + F_drag / m
```

Each active projectile renders as a short line segment between its previous
and current position, giving a tracer effect. Hits are detected by **ray casting**
(`mujoco.mj_ray`) at the moment of firing — more accurate than collision detection
at these velocities.

### Casings

Casings are slow (3–8 m/s ejection), so they are full MuJoCo free-joint
rigid bodies. A pool is pre-allocated in the XML (default 60 slots) because
MuJoCo does not support runtime body creation.

### Casing-vs-propeller collision

Each rotor has a thin disc geom (collision bitmask: casings only) so spent
casings can strike the props. Any casing-vs-prop contact is logged and the
casing immediately despawned — modelling a real prop fracturing the brass on
first contact.

---

## 11. Intercept geometry

* **Yaw is "free."** Rotating about the vertical axis does not redirect thrust.
  The drone can yaw to face any threat without affecting hover.

* **Pitch is NOT free.** Tilting the airframe redirects thrust horizontally,
  causing the drone to accelerate. There is no static equilibrium at nonzero
  pitch without a balancing horizontal force.

Two real-world solutions for vertical intercept aim:

1. **Lean-and-engage**: drone briefly tilts to aim, fires a burst,
   re-stabilises. Implemented as an additional `pitch_offset` term in the
   position controller setpoint.

2. **Gimbaled mount**: separate ball joint between airframe and payload with
   its own actuators. The airframe hovers level; the payload pivots
   independently. This is what purpose-built counter-UAS platforms use.

---

## 12. Sources

* Mellinger & Kumar, *Minimum snap trajectory generation and control for
  quadrotors*, ICRA 2011 — cascaded controller derivation.
* Bouabdallah, *Design and control of quadrotors with application to
  autonomous flying*, EPFL 2007 — rotor model and inertia values.
* Modern Firearms encyclopedia (modernfirearms.net) for calibre ballistics and
  cyclic rates (payload physics parameters).
* MuJoCo documentation (mujoco.readthedocs.io) for `<motor>` actuator and
  free-joint state conventions.
