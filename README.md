# sai_humanposer

## install
```
pip install git+https://github.com/jutanke-sony/sai_humanposer.git
```

## usage
```python
from humanposer import get_smpl, get_smplh, get_smplx

smpl = get_smpl(gender="male")
out = smpl()
V = out.vertices  # [1, 6890, 3]
J = out.joints  # [1, 45, 3]

smplh = get_smplh(gender="male")
out = smplh()
V = out.vertices  # [1, 6890, 3]
J = out.joints  # [1, 73, 3]

smplx = get_smplx(gender="male")
out = smplx()
V = out.vertices  # [1, 10475, 3]
J = out.joints  # [1, 127, 3])
```

```python
from humanposer import download_bodymodels

# downloads all bodymodels into ./bodymodels/smpl(x|h)/SMPL(H|X)_{gender}.npz
# We cannot share this file as the SMPL body models need to be downloaded
# directly from MPII.
download_bodymodels("/path/to/smpl_configs.yaml")
```
