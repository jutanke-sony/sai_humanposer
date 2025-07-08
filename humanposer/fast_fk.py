import torch.nn as nn
import torch
from humanposer.smplx.lbs import (
    lbs,
    vertices2landmarks,
    find_dynamic_lmk_idx_and_bcoords,
    batch_rigid_transform,
    blend_shapes,
    vertices2joints,
)
from humanposer.transforms.rotation_conversions import rotation_6d_to_matrix


class FastFK(nn.Module):
    def __init__(self, smpl):
        super(FastFK, self).__init__()
        self.smpl = smpl
        self.posedirs = smpl.posedirs
        self.v_template = smpl.v_template
        self.parents = smpl.parents
        self.J_regressor = smpl.J_regressor
        self.shapedirs = smpl.shapedirs
        self.expected_num_joints = smpl.NUM_JOINTS

    def forward(self, pose6d, betas=None, transl=None):
        """
        :param pose: {b x 24*6}
        :param betas: {b x 10} or {10}
        :param transl: {b x 3}
        :return: vertices, joints
        """
        assert torch.is_tensor(pose6d), "pose6d must be a tensor"
        if betas is not None:
            assert torch.is_tensor(betas)
        if transl is not None:
            assert torch.is_tensor(transl)
        if betas is None:
            betas = torch.zeros(
                (pose6d.shape[0], 10), dtype=pose6d.dtype, device=pose6d.device
            )
        if transl is None:
            transl = torch.zeros(
                (pose6d.shape[0], 3), dtype=pose6d.dtype, device=pose6d.device
            )
        batch_size = pose6d.shape[0]

        # full_pose = torch.cat([global_orient, body_pose], dim=1)
        full_pose = pose6d.reshape(pose6d.shape[0], -1, 6)
        full_pose_mat = rotation_6d_to_matrix(full_pose)

        # Add shape contribution
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)

        # Get the joints
        # NxJx3 array
        J = vertices2joints(self.J_regressor, v_shaped)

        rot_mats = full_pose_mat.view(batch_size, -1, 3, 3)

        # if we do not have enough joints (i.e. no hands etc) add them here...
        expected_num_joints = self.expected_num_joints + 1  # +1 for the global joint
        if rot_mats.shape[1] < expected_num_joints:
            I = torch.eye(3, dtype=J.dtype, device=J.device).reshape(1, 1, 3, 3)
            I = I.expand(batch_size, expected_num_joints - rot_mats.shape[1], 3, 3)
            rot_mats = torch.cat([rot_mats, I], dim=1)

        # 4. Get the global joint location
        J_transformed, _ = batch_rigid_transform(
            rot_mats, J, self.parents, dtype=pose6d.dtype
        )

        return J_transformed
