"""Setpoint generators for the two flight modes.

Each generator exposes:  step(dt, current_yaw) -> {'pos_des': np.array(3), 'yaw_des': float}
plus an `is_firing(self) -> bool` for the gun trigger.

Why numpad? The MuJoCo viewer binds almost every letter A-Z to a render-flag
toggle: W toggles wireframe, D toggles SDF, T toggles transparency, etc.
Since pynput cannot suppress the viewer's keypress handlers, the only clean
way to fly without flipping render flags every other keystroke is to use
keys the viewer doesn't bind. Numpad keys (GLFW codes 320-335) are unbound.

  * KeyboardSetpoint   - drives a virtual pilot stick from held numpad keys.
  * AutonomousSetpoint - cycles through a list of waypoints with smooth
                         (cosine-eased) interpolation between them.

Layout (NumLock ON):
        7 yaw-L   8 fwd     9 yaw-R
        4 strafe-L 5 reset  6 strafe-R
        1         2 back    3
        0 fire              .
        + climb             - descend

Falls back to chars + special keys; if pynput is missing the simulator
still runs but ignores keyboard input.
"""
import numpy as np

try:
    from pynput import keyboard as kb
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    kb = None

from config import ControlParams, SimParams


# ============================================================================
#  Keyboard mode
# ============================================================================

# pynput delivers numpad keys as KeyCode objects whose `.vk` is the OS-level
# virtual-key code. The tricky bit: that code differs between platforms.
#
#   Windows  : VK_NUMPAD0..9 = 96..105            (winuser.h)
#   Linux X11 (NumLock ON):
#              XK_KP_0..9    = 65456..65465       (keysymdef.h)
#   Linux X11 (NumLock OFF):
#              XK_KP_Insert/End/Down/...          = 65429..65439
#
# We accept ALL of them, so the bindings work on Windows native, Linux X11,
# and WSL+WSLg (which uses an X11 server). As a final fallback we accept
# the printed character ('8', '+', etc.) so NumLock-on works even if pynput
# couldn't read .vk for some reason.
NP_VK = {
    # token : set of vk codes that should map to it
    "0": {96, 65456, 65438},                # NP0  / KP_0  / KP_Insert
    "1": {97, 65457, 65436},                # NP1  / KP_1  / KP_End
    "2": {98, 65458, 65433},                # NP2  / KP_2  / KP_Down
    "3": {99, 65459, 65435},                # NP3  / KP_3  / KP_PageDown
    "4": {100, 65460, 65430},               # NP4  / KP_4  / KP_Left
    "5": {101, 65461, 65437},               # NP5  / KP_5  / KP_Begin
    "6": {102, 65462, 65432},               # NP6  / KP_6  / KP_Right
    "7": {103, 65463, 65429},               # NP7  / KP_7  / KP_Home
    "8": {104, 65464, 65431},               # NP8  / KP_8  / KP_Up
    "9": {105, 65465, 65434},               # NP9  / KP_9  / KP_PageUp
    "+": {107, 65451},                      # NP+  / KP_Add
    "-": {109, 65453},                      # NP-  / KP_Subtract
    "*": {106, 65450},                      # NP*  / KP_Multiply
    "/": {111, 65455},                      # NP/  / KP_Divide
    ".": {110, 65454, 65439},               # NP.  / KP_Decimal / KP_Delete
}
_VK_TO_TOKEN = {code: "np_" + tok for tok, codes in NP_VK.items() for code in codes}
_CHAR_TO_TOKEN = {tok: "np_" + tok for tok in NP_VK}


class KeyboardSetpoint:
    """Hold-to-move keyboard control using the numeric keypad.

    Movement is integrated into a position+yaw setpoint, so even if the
    controller can't track instantly the setpoint moves smoothly instead of
    jumping (which would otherwise spike the PID with a step input).
    """

    def __init__(self, ctrl: ControlParams, sim: SimParams, debug_keys: bool = False):
        self.c = ctrl
        self._init_pos = np.array(sim.init_pos, dtype=float)
        self._init_yaw = sim.init_yaw
        self.pos = self._init_pos.copy()
        self.yaw = self._init_yaw
        self._held = set()                    # tokens for "currently held" keys
        self._debug = debug_keys
        self.reset_flag = False

        if not HAS_PYNPUT:
            print("[modes] WARNING: pynput not installed - keyboard mode is dead.")
            print("        install with:  pip install pynput")
            return

        self.listener = kb.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.daemon = True
        self.listener.start()

    # ----- token decoding ----------------------------------------------------
    @staticmethod
    def _token(key):
        """Map a pynput key event to a stable string token. We match numpad
        keys by VK code first (so '8' on the numpad is distinct from top-row
        '8') and fall back to the printed char so NumLock-on numpads still
        work even when the platform reports an unfamiliar VK."""
        vk = getattr(key, 'vk', None)
        if vk is not None and vk in _VK_TO_TOKEN:
            return _VK_TO_TOKEN[vk]
        if hasattr(key, 'char') and key.char:
            c = key.char.lower()
            if c in _CHAR_TO_TOKEN:
                return _CHAR_TO_TOKEN[c]
            return c
        return str(key)                      # special keys: 'Key.shift', etc.

    # ----- pynput callbacks --------------------------------------------------
    def _on_press(self, key):
        t = self._token(key)
        if self._debug:
            print(f"[keys] press  vk={getattr(key, 'vk', None)} "
                  f"char={getattr(key, 'char', None)!r}  -> token={t!r}")
        self._held.add(t)
        if t == "np_5":
            self.reset_flag = True

    def _on_release(self, key):
        t = self._token(key)
        if self._debug:
            print(f"[keys] release vk={getattr(key, 'vk', None)} "
                  f"char={getattr(key, 'char', None)!r}  -> token={t!r}")
        self._held.discard(t)

    def _h(self, *toks):
        """Returns 1 if any token is held, else 0."""
        return int(any(t in self._held for t in toks))

    # ----- per-step update ---------------------------------------------------
    def step(self, dt, current_yaw):
        v_h = self.c.max_horiz_speed
        v_v = self.c.max_climb_rate
        yaw_rate = np.radians(self.c.max_yaw_rate_dps)

        # Body-frame velocity command.
        vx_body = (self._h("np_8") - self._h("np_2")) * v_h           # forward / back
        vy_body = (self._h("np_4") - self._h("np_6")) * v_h           # strafe L / R

        # Rotate body-frame velocity into world frame using the drone's
        # current yaw, so "forward" always means "where the drone is pointing".
        cy, sy = np.cos(current_yaw), np.sin(current_yaw)
        vx_w = cy * vx_body - sy * vy_body
        vy_w = sy * vx_body + cy * vy_body
        vz_w = (self._h("np_+") - self._h("np_-")) * v_v

        self.pos += np.array([vx_w, vy_w, vz_w]) * dt
        if self.pos[2] < 0.1:               # don't let the setpoint drop into the floor
            self.pos[2] = 0.1

        self.yaw += (self._h("np_7") - self._h("np_9")) * yaw_rate * dt

        return {'pos_des': self.pos.copy(), 'yaw_des': float(self.yaw)}

    def is_firing(self):
        return self._h("np_0") > 0          # Numpad-0 = trigger

    def reset(self):
        self.pos = self._init_pos.copy()
        self.yaw = self._init_yaw
        self._held.clear()                  # forget any keys that were held during reset


# ============================================================================
#  Autonomous mode
# ============================================================================

class AutonomousSetpoint:
    """Fly a closed loop of waypoints with smooth transitions.

    Each cycle: spend `transit_time` seconds easing from waypoint i to i+1,
    then `dwell_time` seconds hovering at i+1. Cosine easing means the
    setpoint starts and ends with zero velocity, so the drone doesn't get a
    step input.

    Auto-fire is gated by config.GUN.auto_fire_in_auto_mode — when True, the
    drone fires whenever it's at a hover point (dwelling), so you can watch
    the recoil disturbance and the controller's correction.
    """

    def __init__(self, sim: SimParams, auto_fire: bool = False):
        self.waypoints = [
            (np.array([0.0, 0.0, 1.5]),  0.0),
            (np.array([3.0, 0.0, 1.5]),  0.0),
            (np.array([3.0, 3.0, 2.0]),  np.pi / 2),
            (np.array([0.0, 3.0, 2.0]),  np.pi),
            (np.array([0.0, 0.0, 1.5]), -np.pi / 2),
        ]
        self.idx = 0
        self.t = 0.0
        self.transit_time = 5.0
        self.dwell_time = 3.0
        self.cycle = self.transit_time + self.dwell_time
        self._auto_fire = auto_fire

    def reset(self):
        self.idx = 0
        self.t = 0.0

    def step(self, dt, current_yaw):
        self.t += dt
        if self.t > self.cycle:
            self.t -= self.cycle
            self.idx = (self.idx + 1) % len(self.waypoints)

        cur_pos, cur_yaw = self.waypoints[self.idx]
        nxt_pos, nxt_yaw = self.waypoints[(self.idx + 1) % len(self.waypoints)]

        if self.t < self.transit_time:
            s = 0.5 - 0.5 * np.cos(np.pi * self.t / self.transit_time)
            pos = (1 - s) * cur_pos + s * nxt_pos
            dyaw = ((nxt_yaw - cur_yaw + np.pi) % (2 * np.pi)) - np.pi
            yaw = cur_yaw + s * dyaw
        else:
            pos = nxt_pos
            yaw = nxt_yaw
        return {'pos_des': pos.copy(), 'yaw_des': float(yaw)}

    def is_firing(self):
        # Fire only while dwelling at a waypoint - keeps the recoil disturbance
        # localized so you can see the controller catch it.
        return self._auto_fire and (self.t >= self.transit_time)
