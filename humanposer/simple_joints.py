import torch
import torch.nn as nn
from humanposer.smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform,
)
from einops import rearrange, repeat


class SimpleSMPLJoints(nn.Module):
    """
    This is faster than the SMPL class from SMPLX
    """

    def __init__(self, smpl, *, betas=None):
        """
        :param betas: {10}
        """
        super().__init__()
        if betas is None:
            betas = torch.zeros((10,))
        v_shaped = smpl.v_template + blend_shapes(betas.unsqueeze(0), smpl.shapedirs)
        self.J = vertices2joints(smpl.J_regressor, v_shaped)
        self.parents = smpl.parents
        self.v_template = smpl.v_template
        self.shapedirs = smpl.shapedirs
        self.J_regressor = smpl.J_regressor

    def forward(
        self,
        transl: torch.Tensor,
        global_rot: torch.Tensor,
        pose: torch.Tensor,
        betas: torch.Tensor = None,
    ):
        """
        :param transl: {b x 3}
        :param global_rot: {b x 3}
        :param pose: {b x 23 x 3}
        :param betas: {b x 10}
        """
        assert torch.is_tensor(transl)
        assert torch.is_tensor(global_rot)
        assert torch.is_tensor(pose)
        if len(transl.shape) != 2 or transl.shape[1] != 3:
            raise ValueError(f"Expect transl to be 'b x 3' but found {transl.shape}")

        if len(global_rot.shape) != 2 or global_rot.shape[1] != 3:
            raise ValueError(
                f"Expect global_rot to be 'b x 3' but found {global_rot.shape}"
            )

        if len(pose.shape) == 2:
            pose = rearrange(pose, "b (j d) -> b j d", d=3)

        if len(pose.shape) != 3 or pose.shape[1] != 23 or pose.shape[2] != 3:
            raise ValueError(f"Expect pose to be 'b x 23 x 3' but found {pose.shape}")

        if pose.shape[0] != global_rot.shape[0] or pose.shape[0] != transl.shape[0]:
            raise ValueError(
                f"we expect all batchsizes to match: {transl.shape} vs {global_rot.shape} vs {pose.shape}"
            )

        batch_size = transl.shape[0]
        device = transl.device
        dtype = transl.dtype

        full_pose = torch.cat([global_rot.unsqueeze(1), pose], dim=1)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        if betas is None:
            J = repeat(self.J, "n j d -> (n b) j d", b=batch_size).to(device)
        else:
            assert torch.is_tensor(betas)
            assert (
                len(betas.shape) == 2
                and betas.shape[0] == batch_size
                and betas.shape[1] == 10
            ), f"Weird betas shape: {betas.shape} for expected batch size {batch_size}"
            v_shaped = blend_shapes(
                betas, self.shapedirs.to(device)
            ) + self.v_template.to(device)
            J = vertices2joints(self.J_regressor.to(device), v_shaped)

        J_transformed, _ = batch_rigid_transform(rot_mats, J, self.parents, dtype=dtype)

        return J_transformed + transl.unsqueeze(1)


class SimpleSMPLXJoints(nn.Module):
    """
    This is faster than the SMPL class from SMPLX
    """

    def __init__(self, smpl, *, betas=None):
        """
        :param betas: {10}
        """
        super().__init__()
        if betas is None:
            betas = torch.zeros((10,))
        v_shaped = smpl.v_template + blend_shapes(betas.unsqueeze(0), smpl.shapedirs)
        self.J = vertices2joints(smpl.J_regressor, v_shaped)
        self.parents = smpl.parents
        self.v_template = smpl.v_template
        self.shapedirs = smpl.shapedirs
        self.J_regressor = smpl.J_regressor

    def forward(
        self,
        transl: torch.Tensor,
        global_rot: torch.Tensor,
        pose: torch.Tensor,
        betas: torch.Tensor = None,
        left_hand_pose: torch.Tensor = None,
        right_hand_pose: torch.Tensor = None,
    ):
        """
        :param transl: {b x 3}
        :param global_rot: {b x 3}
        :param pose: {b x 23 x 3}
        :param betas: {b x 10}
        :param left_hand_pose: {b x 15 x 3}
        :param right_hand_pose: {b x 15 x 3}
        """
        assert torch.is_tensor(transl)
        assert torch.is_tensor(global_rot)
        assert torch.is_tensor(pose)
        if len(transl.shape) != 2 or transl.shape[1] != 3:
            raise ValueError(f"Expect transl to be 'b x 3' but found {transl.shape}")

        if len(global_rot.shape) != 2 or global_rot.shape[1] != 3:
            raise ValueError(
                f"Expect global_rot to be 'b x 3' but found {global_rot.shape}"
            )

        if len(pose.shape) == 2:
            pose = rearrange(pose, "b (j d) -> b j d", d=3)

        if (
            len(pose.shape) != 3
            or (pose.shape[1] != 21 and pose.shape[1] != 23)
            or pose.shape[2] != 3
        ):
            raise ValueError(f"Expect pose to be 'b x 23 x 3' but found {pose.shape}")

        if pose.shape[0] != global_rot.shape[0] or pose.shape[0] != transl.shape[0]:
            raise ValueError(
                f"we expect all batchsizes to match: {transl.shape} vs {global_rot.shape} vs {pose.shape}"
            )

        batch_size = transl.shape[0]

        if left_hand_pose is None:
            left_hand_pose = torch.zeros(
                size=(batch_size, 15, 3), dtype=pose.dtype, device=pose.device
            )
        if right_hand_pose is None:
            right_hand_pose = torch.zeros(
                size=(batch_size, 15, 3), dtype=pose.dtype, device=pose.device
            )
        jaw_pose = torch.zeros(
            size=(batch_size, 1, 3), dtype=pose.dtype, device=pose.device
        )
        leye_pose = torch.zeros(
            size=(batch_size, 1, 3), dtype=pose.dtype, device=pose.device
        )
        reye_pose = torch.zeros(
            size=(batch_size, 1, 3), dtype=pose.dtype, device=pose.device
        )
        # expression = torch.zeros(size=(batch_size, 10), dtype=pose.dtype, device=pose.device)

        # if pose.shape[1] == 21:  # b j d
        #     padding = torch.zeros((batch_size, 2, 3)).to(pose.device)
        #     pose = torch.cat([pose, padding], dim=1)

        device = transl.device
        dtype = transl.dtype

        full_pose = torch.cat(
            [
                global_rot.unsqueeze(1),
                pose,
                jaw_pose,
                leye_pose,
                reye_pose,
                left_hand_pose,
                right_hand_pose,
            ],
            dim=1,
        )
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        if betas is None:
            J = repeat(self.J, "n j d -> (n b) j d", b=batch_size).to(device)
        else:
            assert torch.is_tensor(betas)
            assert (
                len(betas.shape) == 2
                and betas.shape[0] == batch_size
                and betas.shape[1] == 10
            ), f"Weird betas shape: {betas.shape} for expected batch size {batch_size}"

            # WE DO NOT APPLY THE EXPRESSION HERE!!!
            v_shaped = blend_shapes(
                betas, self.shapedirs.to(device)
            ) + self.v_template.to(device)
            J = vertices2joints(self.J_regressor.to(device), v_shaped)

        J_transformed, _ = batch_rigid_transform(rot_mats, J, self.parents, dtype=dtype)

        return J_transformed + transl.unsqueeze(1)
