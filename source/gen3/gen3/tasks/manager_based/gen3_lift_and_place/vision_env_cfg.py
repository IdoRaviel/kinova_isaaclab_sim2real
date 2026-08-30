"""Vision variant of the chain environment.

Identical to ``Gen3LiftAndPlaceChainEnvCfg`` except the grasp policy's two cube-pose
observation terms (``object_position``, ``object_orientation``) are computed by the
trained CubePoseNet from the workspace camera image instead of read from privileged
simulator state. Reassigning the existing terms keeps their position in the
concatenated observation vector, so the frozen grasp policy's input layout is
unchanged -- only the *source* of those numbers differs.

The checkpoint path is a term param (relative to the working directory, i.e. the repo
root); override it before ``gym.make`` to use a different trained model.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from . import mdp
from .joint_pos_env_cfg import Gen3LiftAndPlaceChainEnvCfg

VISION_CHECKPOINT_DEFAULT = "pretrained_models/cube_pose/best.pt"


@configclass
class Gen3LiftAndPlaceChainVisionEnvCfg(Gen3LiftAndPlaceChainEnvCfg):
    """Chain env where the grasp policy sees the cube through the camera network."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.object_position = ObsTerm(
            func=mdp.object_position_from_vision,
            params={"checkpoint": VISION_CHECKPOINT_DEFAULT},
        )
        self.observations.policy.object_orientation = ObsTerm(
            func=mdp.object_orientation_from_vision,
            params={"checkpoint": VISION_CHECKPOINT_DEFAULT},
        )
