# Dataset Download

The first step is to download the data.

## Method 1: Using CMU FBX

1. Install FBX SDK (see: https://github.com/nv-tlabs/ASE/issues/61).
2. Follow [Expressive Humanoid](https://github.com/chengxuxin/expressive-humanoid) to retarget .fbx files to .npy motion format.

## Method 2: Using AMASS (recommended)

1. Download the amass dataset, and put them in the `./data/amass/`. (You can follow the steps in the [amass](https://amass.is.tue.mpg.de).)
2. Unzip the file, and organize them.

The final data folder structure and usage is detailed in [data folder readme](../data/README.md).

# Dataset Retargeting
Our repo provides retargeting methods for versatile design of robots.

# Asset Download

Assets could be found in [unitree_ros](https://github.com/unitreerobotics/unitree_ros), we also provide a download script.


Download: 

```bash
bash scripts/tools/assets/download_assets.sh g1 h1
```

Convert to usd:

```bash
cd trackerLab # Repo dir
mkdir -p data/assets/usd/g1_23d
python scripts/tools/assets/convert_urdf.py --headless data/assets/g1_description/g1_23dof.urdf data/assets/usd/g1_23d/g1_23d

mkdir -p data/assets/usd/g1_29d_loco
python scripts/tools/assets/convert_urdf.py --headless data/assets/g1_exbody/g1_29dof_loco.urdf data/assets/usd/g1_29d_loco/g1_29d_loco
```

If your robot do not have a mjcf for tpose generation, you could use:

```bash
pip install urdf2mjcf

urdf2mjcf #<path to your urdf file>
```