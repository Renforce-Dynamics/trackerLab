from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lvhaidong/LAFAN1_Retargeting_Dataset",
    repo_type="dataset",
    local_dir="./data/datasets/LAFAN1_Retargeting_Dataset",
    local_dir_use_symlinks=False
)
