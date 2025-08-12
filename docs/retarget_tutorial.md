# Retarget Tutorial

## Retarget from amass

### Check Components:
1. Make sure that you have amass data downloaded and organzied in `./data/amass/`;
2. Check the retarget configuration in `./data/retarget_cfg` exist or not;
    - For example, if you want to retarget amass (`smpl`) to `h1`, the `./data/retarget_cfg/retar_smpl_2_h1.json` is corresponding config;
    - The config file contains the `joint mapping`, and other rotation terms.
3. Check the `tpose` file is exist or not, like `./data/tpose/h1_tpose.npy` and `./data/tpose/smpl_tpose.npy`.

### Start Retargeting

The `./poselib/scripts/amass` contains all the scripts for retargeting. Using `./poselib/scripts/amass/retarget_all.py` to retarget all the files.

```text 
usage: retarget_all.py [-h] --folders FOLDERS [FOLDERS ...] [--robot ROBOT] [--max_frames MAX_FRAMES] [--workers WORKERS]

options:
  -h, --help            show this help message and exit
  --folders FOLDERS [FOLDERS ...]
                        List of AMASS subfolders to scan under data root
  --robot ROBOT         robot type for retargeting
  --max_frames MAX_FRAMES
                        Max number of frames per sequence
  --workers WORKERS     Number of parallel workers (default=CPU count)
```

After this the motions will be saved in `./data/retargeted/amass_results`.

### Example

Once downloaded and unzip the data, the folder should be like:
```
data:
- amass
  - CMU
    - 01
      - 01_01_poses.npz
      ...
    ...
  ...
```

Run the script like follow, where our scripts will recursively search for the `.npz` file:

```bash
# Retarget some of the data:

python poselib/scripts/amass/retarget_all.py --folders CMU/01 CMU/02 --robot h1

# Retarget all data:

python poselib/scripts/amass/retarget_all.py --folders CMU --robot h1
```


