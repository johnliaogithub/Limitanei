"""Entry point: live MuJoCo simulation + viewer.

All logic lives in drone_sim.sim; this is a thin shim so you can still run
`python main.py ...` from the project root. See `drone_sim/sim.py` for the
full CLI flag list.
"""
from drone_sim.sim import main

if __name__ == "__main__":
    main()
