# Full-Body Articulated Human-Object Interaction

<p align="center"><img src="figures/teaser.png" width="600px"/></p></br>

This is the code repository of **Full-Body Articulated Human-Object Interaction** - [pdf](https://arxiv.org/pdf/2212.10621.pdf) - [arxiv](http://arxiv.org/abs/2212.10621) - [project](https://jnnan.github.io/chairs/) -

# News
The **metas.json** containing the transformation matrix between each camera view and the world coordinate is uploade (which can make the human and the object on the ground).


# Environment
We tested our code with CUDA 11.1, PyTorch 1.11.0, and torchvision 0.10.0. 

After installing CUDA, PyTorch and torchvision, run the following command for installing other dependencies:
```shell
pip install -r requirements.txt
```

# Data Preparation
Please download the data and weights from [this link](https://forms.gle/t4SjmJS4RPx7AFvFA) and extract the zip file to the root directory of this repository.

You need to download the SMPL-X models from [here](https://smpl-x.is.tue.mpg.de/download.php) and unzip them to `Data/body_models/smplx/`.

Your directory should look like this: 

```
├─ Data
    ├─ AHOI_ROOT
        ├─ Meshes_wt
        ├─ Metas
        ├─ object_part_voxel_64
    ├─ DATA_FOLDER
    ├─ IMG_FOLDER
        ├─ 0001
            ├─ 1
                ├─ 0000.png
                ├─ 0005.png
                ├─ 0010.png
                    ...
            ├─ 2
            ├─ 3
            ├─ 4
        ├─ 0002
            ...
    ├─ body_models
        ├─ smplx
    ├─ checkpoint
├─ ahoi_utils.py
├─ dataloaders.py
    ...
```

