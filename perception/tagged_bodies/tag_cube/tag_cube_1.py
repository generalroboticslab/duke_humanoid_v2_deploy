"""Instance: tag_cube_1 — tag IDs 577–581 matching the 3MF generator (not included in this repository)."""
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tagged_bodies.tag_cube import (
    TagCubeConfig, OUTER_FACE_DIST_M, TAG_SIZE_M, TAG_BORDER_SIZE_M,
)
from tagged_rigid_body import TagConfig, verify_bodies
import numpy as np

cfg = TagCubeConfig(
    name="tag_cube_1",
    tag_size=TAG_SIZE_M,
    tag_border_size=TAG_BORDER_SIZE_M,
    tags={
        # rot cols: [tag_right, tag_down, tag_into_face] in world frame
        # tag_right  = slab_X_world (image col direction)
        # tag_down   = -slab_Y_world (slab Y = image top → negate for down)
        # tag_into   = -slab_Z_world (slab Z = outward normal → negate for into)
        "top":   TagConfig(id=581, pos=[ OUTER_FACE_DIST_M, 0, 0],  rot=np.array([[ 0, 1, 0], [ 0, 0,-1], [-1, 0, 0]]).T),
        "front": TagConfig(id=580, pos=[ 0, 0, OUTER_FACE_DIST_M],  rot=np.array([[ 0, 1, 0], [ 1, 0, 0], [ 0, 0,-1]]).T),
        "back":  TagConfig(id=579, pos=[ 0, 0, -OUTER_FACE_DIST_M], rot=np.array([[ 0, 1, 0], [-1, 0, 0], [ 0, 0, 1]]).T),
        "left":  TagConfig(id=578, pos=[0, OUTER_FACE_DIST_M,  0],  rot=np.array([[-1, 0, 0], [ 0, 0,-1], [ 0,-1, 0]]).T),
        "right": TagConfig(id=577, pos=[ 0, -OUTER_FACE_DIST_M, 0], rot=np.array([[ 1, 0, 0], [ 0, 0,-1], [ 0, 1, 0]]).T),
    },
)

if __name__ == "__main__":
    verify_bodies(cfg)
