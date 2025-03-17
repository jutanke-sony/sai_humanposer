from humanposer.bodymodel import (
    get_smpl,
    get_smplh,
    get_smplx,
    smpl_forward,
    smpl_forward_no_grad,
)
from typing import Dict
import yaml
from os import getcwd, makedirs
from os.path import join, isdir, isfile, abspath
import gdown


def download_bodymodels(config: Dict):
    if isinstance(config, str):
        # config is a path to a yaml file
        with open(config, "r") as f:
            config = yaml.safe_load(f)

        genders = ["female", "male", "neutral"]
        models = ["smpl", "smplh", "smplx"]

        bodymodels_dir = join(abspath(getcwd()), "bodymodels")
        bodymodels_dir = bodymodels_dir.replace("notebooks/", "")
        makedirs(bodymodels_dir, exist_ok=True)

        for model in models:
            subbodymodels_dir = join(bodymodels_dir, model)
            makedirs(subbodymodels_dir, exist_ok=True)

            for gender in genders:
                fname = join(subbodymodels_dir, f"{model.upper()}_{gender.upper()}.npz")
                if not isfile(fname):
                    url = config[f"{model}_{gender}"]
                    print(f"fetch {model} ({gender})")
                    gdown.download(url, fname)
                    print(f"\tsave to {fname}")
