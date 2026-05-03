"""World & dynamics: weapons, projectiles, casings, targets, environmental noise."""
from drone_sim.physics.gun import Gun, GUNS, make_gun, PROPELLANT_GAS_FACTOR
from drone_sim.physics.bullets import BulletPool, bullet_drag_const
from drone_sim.physics.casings import CasingPool, casing_pool_xml, prop_disc_geom_xml
from drone_sim.physics.disturbances import WindModel, drag_force, recoil_noise
from drone_sim.physics.targets import (
    Target, TargetField, default_ring, target_xml, yaw_to_aim,
)

__all__ = [
    "Gun", "GUNS", "make_gun", "PROPELLANT_GAS_FACTOR",
    "BulletPool", "bullet_drag_const",
    "CasingPool", "casing_pool_xml", "prop_disc_geom_xml",
    "WindModel", "drag_force", "recoil_noise",
    "Target", "TargetField", "default_ring", "target_xml", "yaw_to_aim",
]
