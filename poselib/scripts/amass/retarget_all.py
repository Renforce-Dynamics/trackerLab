import os
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from poselib.retarget.amass_loader import AMASSLoader
from poselib.retarget.retargeting_processor import RetargetingProcessor
from poselib import POSELIB_DATA_DIR

from trackerLab.utils.animation import animate_skeleton


AMASS_DATA_DIR = os.path.join(POSELIB_DATA_DIR, "amass")


def visualize(source_motion, name):
    edges = AMASSLoader.get_edge_map(source_motion.skeleton_tree)
    animate_skeleton(
        source_motion.global_translation, edges, source_motion.global_root_velocity, 
        interval=30,
        save_path=f"{RESULTS_DIR}/{name}.mp4"
    )

def process_pipeline(amass_file: str, output_file: Path, loader: AMASSLoader, retargetor: RetargetingProcessor) -> str:
    try:
        source_motion = loader.load_and_process(amass_file)
        target_motion = retargetor.retarget_base(source_motion)

        target_motion = RetargetingProcessor.adjust_motion(target_motion, 0, angle=-90, axis_rot=1)
        target_motion = RetargetingProcessor.adjust_motion(target_motion, 0, angle=-90, axis_rot=2)
        target_motion = RetargetingProcessor.reorder_translation_axes(target_motion, (1, 2, 0))
        target_motion = AMASSLoader.fill_motion_vel(target_motion)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        target_motion.to_file(str(output_file))
        return f"[OK] {amass_file}"
    except Exception as e:
        return f"[FAIL] {amass_file} - {str(e)}"

def _process(args):
    return process_pipeline(*args)

def process_all(folders: list[str], robot, max_frames: int = 1000, num_workers: int = None):
    loader = AMASSLoader(max_frames=max_frames)
    retargetor = RetargetingProcessor("smpl", robot)

    amass_files = []
    for folder in folders:
        folder_path = Path(AMASS_DATA_DIR) / folder
        if not folder_path.exists():
            print(f"[WARN] Folder not found: {folder_path}")
            continue
        for npz_file in folder_path.rglob("*.npz"):
            rel_path = npz_file.relative_to(AMASS_DATA_DIR)
            output_path = RESULTS_DIR / rel_path.with_suffix(".npy")
            amass_files.append([str(npz_file), output_path] + [loader, retargetor])

    print(f"[INFO] Total files to process: {len(amass_files)}")
    results = []

    with Pool(processes=num_workers or cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(_process, amass_files), total=len(amass_files)):
            results.append(res)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, 'w') as f:
        for line in results:
            f.write(line + "\n")
    print(f"[INFO] Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--folders', nargs='+', required=True,
                        help='List of AMASS subfolders to scan under data root')
    parser.add_argument('--robot', type=str, default="h1",
                        help='robot type for retargeting')
    parser.add_argument('--max_frames', type=int, default=10000,
                        help='Max number of frames per sequence')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default=CPU count)')
    args = parser.parse_args()

    robot = args.robot
    RESULTS_DIR = Path(POSELIB_DATA_DIR, "retargeted", "amass_results", robot)
    REPORT_FILE = RESULTS_DIR / "report.txt"

    folders = args.folders
    folders = [
        f"CMU/{tar}" for tar in ["127", "02", "01", "75", "55", "35"]
    ]

    process_all(folders, robot, max_frames=args.max_frames, num_workers=args.workers)
