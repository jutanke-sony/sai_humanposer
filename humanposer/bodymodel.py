import numpy as np
import torch
from humanposer.smplx.body_models import SMPL, SMPLH, SMPLX
from os.path import abspath, isfile, isdir, join
from os import getcwd
from einops import rearrange, repeat


def get_smpl(
    gender: str, *, batch_size: int = 1, num_betas: int = 10, smpl_path: str = None
) -> SMPL:
    if smpl_path is None:
        smpl_path = abspath(join(getcwd(), join("bodymodels", "smpl")))
    smpl_path = smpl_path.replace("notebooks/", "")
    return SMPL(smpl_path, batch_size=batch_size, gender=gender, num_betas=num_betas)


def get_smplh(
    gender: str,
    *,
    batch_size: int = 1,
    num_betas: int = 10,
    smpl_path: str = None,
    use_compressed: bool = False,
) -> SMPLH:
    if smpl_path is None:
        smpl_path = abspath(join(getcwd(), join("bodymodels", "smplh")))
    smpl_path = smpl_path.replace("notebooks/", "")
    return SMPLH(
        smpl_path,
        batch_size=batch_size,
        gender=gender,
        use_compressed=use_compressed,
        num_betas=num_betas,
    )


def get_smplx(
    gender: str,
    *,
    batch_size: int = 1,
    num_betas: int = 10,
    smpl_path: str = None,
    use_compressed: bool = False,
) -> SMPLX:
    if smpl_path is None:
        smpl_path = abspath(join(getcwd(), join("bodymodels", "smplx")))
    smpl_path = smpl_path.replace("notebooks/", "")
    return SMPLX(
        smpl_path,
        batch_size=batch_size,
        gender=gender,
        use_compressed=use_compressed,
        num_betas=num_betas,
    )


@torch.no_grad()
def smpl_forward_no_grad(
    smpl_poses, betas, global_transl, *, smpl=None, device=None, gender="male"
):
    is_numpy = isinstance(
        smpl_poses, np.ndarray
    )  # if true we convert torch back to numpy

    if isinstance(smpl_poses, np.ndarray):
        smpl_poses = torch.from_numpy(smpl_poses).float()
    if isinstance(betas, np.ndarray):
        betas = torch.from_numpy(betas).float()
    if isinstance(global_transl, np.ndarray):
        global_transl = torch.from_numpy(global_transl).float()

    if len(smpl_poses.shape) == 3:  # b x 24 x 3 -> b x (24 * 3)
        smpl_poses = rearrange(smpl_poses, "b j d -> b (j d)")

    if isinstance(smpl_poses, np.ndarray) and not smpl_poses.flags["WRITEABLE"]:
        smpl_poses = np.array(smpl_poses, dtype=np.float32)
    if isinstance(betas, np.ndarray) and not betas.flags["WRITEABLE"]:
        betas = np.array(betas, dtype=np.float32)
    if isinstance(global_transl, np.ndarray) and not global_transl.flags["WRITEABLE"]:
        global_transl = np.array(global_transl, dtype=np.float32)

    vertices, joints = smpl_forward(
        smpl_poses=smpl_poses,
        betas=betas,
        global_transl=global_transl,
        smpl=smpl,
        device=device,
        gender=gender,
    )
    if is_numpy:
        vertices = vertices.cpu().numpy()
        joints = joints.cpu().numpy()
    return vertices, joints


def smpl_forward(
    smpl_poses, betas, global_transl, *, smpl=None, device=None, gender="male"
):
    """
    :param smpl_pose: {b x 72}
    :param betas: {10}
    :param global_transl: {b x 3}
    :param smpl: {SMPL}
    :param device: {torch.device}
    """
    assert torch.is_tensor(smpl_poses)
    assert torch.is_tensor(betas)
    assert torch.is_tensor(global_transl)

    if device is None and smpl is None:
        device = torch.device("cpu")

    # make sure that the shapes are correct
    assert len(betas.shape) == 1 and betas.shape[0] == 10, f"weird betas: {betas.shape}"
    assert (
        len(smpl_poses.shape) == 2 and smpl_poses.shape[1] == 72
    ), f"weird poses: {smpl_poses.shape}"
    assert (
        len(global_transl.shape) == 2 and global_transl.shape[1] == 3
    ), f"weird global translation: {global_transl.shape}"
    assert (
        global_transl.shape[0] == smpl_poses.shape[0]
    ), f"Inconsistent poses ({smpl_poses.shape}) and translation ({global_transl.shape})"

    batch_size = smpl_poses.shape[0]
    if smpl is None:
        smpl = get_smpl(gender=gender, batch_size=batch_size)

    betas = repeat(betas, "d -> t d", t=batch_size)

    global_orient = smpl_poses[:, :3]
    smpl_poses = smpl_poses[:, 3:]

    smpl = smpl.to(device)

    out = smpl.forward(
        global_orient=global_orient.to(device),
        body_pose=smpl_poses.to(device),
        transl=global_transl.to(device),
        betas=betas.to(device),
    )

    vertices = out.vertices
    joints = out.joints
    return vertices, joints
