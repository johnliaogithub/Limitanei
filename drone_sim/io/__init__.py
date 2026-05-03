"""Trajectory recording (logger) + offline playback (replay).

NOTE: this subpackage is named `io` but does NOT shadow the stdlib `io` module
because Python 3 uses absolute imports by default. `import io` from anywhere
inside `drone_sim` still resolves to the stdlib.
"""
from drone_sim.io.logger import TrajectoryRecorder, load_trajectory

__all__ = ["TrajectoryRecorder", "load_trajectory"]
