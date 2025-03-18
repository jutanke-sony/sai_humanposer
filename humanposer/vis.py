import torch
import numpy as np
import matplotlib.pylab as plt
import tempfile
import shutil
import random
import string
import subprocess

from pathlib import Path
from os.path import join, isdir, isfile
from os import makedirs, remove

from einops import rearrange
from multiprocessing.pool import Pool
from humanposer.bodymodel import smpl_forward_no_grad
from tqdm import tqdm


def render_single_frame(out):
    Vs = out["V"]  # p x j x d
    xlim = out["xlim"]
    ylim = out["ylim"]
    zlim = out["zlim"]
    colors = out["colors"]
    title = out["title"]
    fname = out["fname"]
    view_init = out["view_init"]
    axis_off = out["axis_off"]
    number_of_persons = Vs.shape[0]
    plt.ioff()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(*view_init)
    if axis_off:
        ax.axis("off")
    else:
        ax.plot([10, -10], [0, 0], [0, 0], linewidth=1, color="gray", alpha=0.5)
        ax.plot([0, 0], [10, -10], [0, 0], linewidth=1, color="gray", alpha=0.5)
        ax.plot([0, 0], [0, 0], [10, -10], linewidth=1, color="gray", alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_title(title, pad=10)
    for person_id in range(number_of_persons):
        V = Vs[person_id]  # n_points x 3
        assert len(V.shape) == 2 and V.shape[1] == 3
        color = colors[person_id]
        if V.shape[0] == 45 or V.shape[0] == 24:
            plot_smpl_J(ax, V)
            ax.scatter(V[0, 0], V[0, 1], V[0, 2], alpha=1, color=color, s=10)
        else:
            ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, color=color)
    plt.savefig(fname)
    plt.close("all")


def get_lims(pts, *, buffer=0.1):
    if torch.is_tensor(pts):
        pts = pts.detach().cpu().numpy()
    if len(pts.shape) == 3 and pts.shape[2] in [2, 3]:
        pts = rearrange(pts, "t j d -> (t j) d")
    assert len(pts.shape) == 2 and pts.shape[1] in [
        2,
        3,
    ], f"weird pts shape: {pts.shape}"

    xleft = np.min(pts[:, 0])
    xright = np.max(pts[:, 0])
    xmu = (xleft + xright) / 2

    yleft = np.min(pts[:, 1])
    yright = np.max(pts[:, 1])
    ymu = (yleft + yright) / 2

    buffer_factor = 1.0 + buffer

    larger_distance = max(yright - yleft, xright - xleft)

    if pts.shape[1] == 3:
        zleft = np.min(pts[:, 2])
        zright = np.max(pts[:, 2])
        zmu = (zleft + zright) / 2
        larger_distance = max(larger_distance, zright - zleft) * buffer_factor

        xlim = [xmu - larger_distance / 2, xmu + larger_distance / 2]
        ylim = [ymu - larger_distance / 2, ymu + larger_distance / 2]
        zlim = [zmu - larger_distance / 2, zmu + larger_distance / 2]
        return xlim, ylim, zlim
    else:
        larger_distance *= buffer_factor
        xlim = [xmu - larger_distance / 2, xmu + larger_distance / 2]
        ylim = [ymu + larger_distance / 2, ymu - larger_distance / 2]
        return xlim, ylim, None


def set_lims(ax, pts, *, buffer=0.1):
    """
    set the limits for the plot, given a set of 2d/3d points
    :param pts: {n x 2|3}
    """
    xlim, ylim, zlim = get_lims(pts=pts, buffer=buffer)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)


def plot_smpl_J(
    ax,
    J,
    *,
    lcolor="cornflowerblue",
    rcolor="salmon",
    mcolor="gray",
    plot_jid=False,
    linewidth=2,
    alpha=1.0,
    highest_jid_plot=24,
):
    """
    :param ax: matplotlib axis
    :param J: {45 x 3}
    """
    if J.shape[0] == 55:
        connect = [
            (1, 0),
            (0, 2),
            (1, 4),
            (4, 7),
            (7, 10),
            (2, 5),
            (5, 8),
            (8, 11),
            (16, 18),
            (18, 20),
            (17, 19),
            (19, 21),
            # (16, 12),
            # (17, 12),
            (13, 12),
            (14, 12),
            (12, 15),
            (0, 3),
            (3, 6),
            (6, 9),
            (9, 12),
            (13, 16),
            (14, 17),
            # (20, 22),
            (22, 23),
            (22, 24),
            (23, 24),
            # --
            (20, 37),
            (37, 38),
            (38, 39),
            (20, 25),
            (25, 26),
            (26, 27),
            (20, 28),
            (28, 29),
            (29, 30),
            (20, 34),
            (34, 35),
            (35, 36),
            # -
            (20, 31),
            (31, 32),
            (32, 33),
            # ---
            (21, 52),
            (52, 53),
            (53, 54),
            # -
            (21, 40),
            (40, 41),
            (41, 42),
            # -
            (21, 43),
            (43, 44),
            (44, 45),
            # -
            (21, 49),
            (49, 50),
            (50, 51),
            # -
            (21, 46),
            (46, 47),
            (47, 48),
        ]
        lcolors = {
            22,
            20,
            18,
            16,
            13,
            1,
            4,
            7,
            10,
            37,
            38,
            39,
            25,
            26,
            27,
            29,
            30,
            36,
            35,
            33,
            32,
        }
        rcolors = {
            17,
            19,
            21,
            23,
            14,
            2,
            5,
            8,
            11,
            53,
            54,
            41,
            42,
            44,
            45,
            47,
            48,
            50,
            51,
        }
    if J.shape[0] == 22:
        connect = [
            (1, 0),
            (0, 2),
            (1, 4),
            (4, 7),
            (7, 10),
            (2, 5),
            (5, 8),
            (8, 11),
            (16, 18),
            (18, 20),
            (17, 19),
            (19, 21),
            (13, 16),
            (14, 17),
            # (16, 12),
            # (17, 12),
            (13, 12),
            (14, 12),
            (12, 15),
            (0, 3),
            (3, 6),
            (6, 9),
            (9, 12),
        ]

        lcolors = {22, 20, 18, 16, 13, 1, 4, 7, 10}
        rcolors = {17, 19, 21, 23, 14, 2, 5, 8, 11}
    else:
        connect = [
            (1, 0),
            (0, 2),
            (1, 4),
            (4, 7),
            (7, 10),
            (2, 5),
            (5, 8),
            (8, 11),
            (16, 18),
            (18, 20),
            (17, 19),
            (19, 21),
            (13, 16),
            (14, 17),
            # (16, 12),
            # (17, 12),
            (13, 12),
            (14, 12),
            (12, 15),
            (0, 3),
            (3, 6),
            (6, 9),
            (9, 12),
            # (20, 22),
            # (21, 23),
        ]

        lcolors = {22, 20, 18, 16, 13, 1, 4, 7, 10}
        rcolors = {17, 19, 21, 23, 14, 2, 5, 8, 11}

    for a, b in connect:
        if (a in lcolors and b not in rcolors) or (b in lcolors and a not in rcolors):
            color = lcolor
        elif (a in rcolors and b not in lcolors) or (b in rcolors and a not in lcolors):
            color = rcolor
        else:
            color = mcolor

        ax.plot(
            [J[a, 0], J[b, 0]],
            [J[a, 1], J[b, 1]],
            [J[a, 2], J[b, 2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

    if plot_jid:
        for jid, pt in enumerate(J):
            if jid <= highest_jid_plot:
                ax.text(pt[0], pt[1], pt[2], str(jid))


def render_sequence_from_vertices(
    video_fname: str,
    Vs: np.ndarray,
    framerate: int,
    *,
    tmp_key=None,
    view_init=(90, 270),
    axis_off=True,
    center_crop=True,
    colors=None,
    extra_render_fn=None,
    tmp_dir=None,
    titles=None,
):
    """
    :param Vs: {p x t x 6890 x 3}
    :param extra_render_fn: {function} (ax, i)
    """
    number_of_persons = Vs.shape[0]
    n_frames = Vs.shape[1]
    if colors is None:
        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "cornflowerblue",
            "salmon",
            "tab:orange",
            "tab:blue",
            "tab:red",
        ]
    assert len(colors) >= number_of_persons

    if tmp_key is None:
        tmp_key = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))

    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    tmp_videodir = join(tmp_dir, tmp_key)

    if isdir(tmp_videodir):
        shutil.rmtree(tmp_videodir)
    makedirs(tmp_videodir)

    if isfile(video_fname):
        # prevent "funny" ffmpeg bugs where it tries to write into the
        # existing file for some reason..
        remove(video_fname)

    # import pathlib
    dir_path = Path(video_fname).parent.resolve()
    if not isdir(dir_path):
        makedirs(dir_path)

    plt.ioff()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(*view_init)
    xlim, ylim, zlim = get_lims(pts=rearrange(Vs, "p t j d -> (p t j) d"), buffer=-0.1)

    if titles is None:
        titles = [f"frame {str(frame)}" for frame in range(n_frames)]

    assert len(titles) == n_frames, f"expected #titles ({len(titles)} == {n_frames})"

    if extra_render_fn is None:
        Data = []
        for frame in tqdm(range(n_frames)):
            fname = join(tmp_videodir, "%09d.png" % frame)
            Data.append(
                {
                    "axis_off": axis_off,
                    "xlim": xlim,
                    "ylim": ylim,
                    "zlim": zlim,
                    "V": Vs[:, frame],
                    "colors": colors,
                    "fname": fname,
                    "view_init": view_init,
                    "title": titles[frame],
                }
            )
        with Pool(15) as p:
            _ = list(tqdm(p.imap(render_single_frame, Data), total=len(Data)))
    else:
        for frame in tqdm(range(Vs.shape[1])):
            ax.clear()
            if axis_off:
                ax.axis("off")
            if extra_render_fn is not None:
                extra_render_fn(ax, frame)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            for person_id in range(number_of_persons):
                V = Vs[person_id, frame]  # n_points x 3
                assert len(V.shape) == 2 and V.shape[1] == 3
                color = colors[person_id]
                if V.shape[0] == 45:
                    plot_smpl_J(ax, V)
                    ax.scatter(V[0, 0], V[0, 2], V[0, 2], alpha=1, color=color, s=10)
                else:
                    ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, color=color)
            fname = join(tmp_videodir, "%09d.png" % frame)
            plt.savefig(fname)

    # ffmpeg --> create video
    subprocess.run(
        [
            "ffmpeg",
            "-framerate",
            str(framerate),
            "-i",
            join(tmp_videodir, "%09d.png"),
            "-pix_fmt",
            "yuv420p",
            video_fname,
        ]
    )

    if center_crop:
        # TODO fix this
        # ffmpeg -i video.mp4 -vf "crop=in_w/1.5:in_h/1.5:in_w/5:in_h/5" -c:a copy out.mp4
        video_fname_uncropped = video_fname.replace(".mp4", "_large.mp4")
        shutil.move(video_fname, video_fname_uncropped)
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_fname_uncropped,
                "-vf",
                # '"crop=in_w/1.5:in_h/1.5:in_w/5:in_h/5"',
                "crop=in_w/1.5:in_h/1.5:in_w/5:in_h/5",
                "-c:a",
                "copy",
                video_fname,
            ]
        )
        remove(video_fname_uncropped)
