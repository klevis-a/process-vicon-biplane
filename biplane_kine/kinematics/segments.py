import numpy as np
from .cs import make_vec_hom, vec_transform, ht_inv


class StaticBodySegment:
    """A body segment.

    Attributes
    ----------
    cs: numpy.ndarray
        the segment frame expressed in the underlying lab frame
    static_markers: numpy.ndarray
        the tracking markers in the static trial expressed in the underlying lab frame
    static_markers_intrinsic: numpy.ndarray
        the tracking markers in the static trial expressed in the segment's frame
    """
    def __init__(self, joint_cs, static_marker_pos, tracking_marker_names):
        self.cs = joint_cs(*[static_marker_pos[marker] for marker in joint_cs.markers])
        self.static_markers = np.stack([static_marker_pos[marker] for marker in tracking_marker_names], 0)
        self.static_markers_intrinsic = vec_transform(ht_inv(self.cs), make_vec_hom(self.static_markers))[:, :3]


class StaticTorsoSegment(StaticBodySegment):
    """Torso body segment."""
    TRACKING_MARKERS = ['STRN', 'C7', 'T5', 'T10', 'CLAV']

    def __init__(self, joint_cs, static_marker_pos):
        super().__init__(joint_cs, static_marker_pos, StaticTorsoSegment.TRACKING_MARKERS)