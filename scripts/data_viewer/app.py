# app.py
import os
import io
from flask import Flask, send_file, render_template, jsonify, request, abort
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server
import matplotlib.pyplot as plt

from utils import list_hdf5_files, walk_h5_groups

DATA_DIR = "./datasets"  # Folder for HDF5 dataset files
ALLOWED_EXT = (".h5", ".hdf5")

app = Flask(__name__)

@app.route("/")
def index():
    """Main page: HTML UI"""
    return render_template("index.html")

@app.route("/api/files")
def api_files():
    """Return list of available HDF5 files"""
    return jsonify(list_hdf5_files(DATA_DIR, ALLOWED_EXT))

@app.route("/api/keys")
def api_keys():
    """Return list of dataset keys in the given HDF5 file"""
    filename = request.args.get("file", "")
    if not filename:
        return jsonify([])
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return abort(404, "file not found")

    try:
        with h5py.File(path, "r") as f:
            keys = walk_h5_groups(f)
    except Exception as e:
        return abort(500, str(e))
    return jsonify(keys)

@app.route("/plot")
def plot():
    """
    Plot a dataset from the HDF5 file.
    - If data has episodes dimension, compute mean over episodes.
    - Optionally select a specific channel.
    """
    filename = request.args.get("file", "")
    key = request.args.get("key", "")
    channel = request.args.get("channel", None)

    if not filename or not key:
        return abort(400, "file and key are required")

    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return abort(404, "file not found")

    try:
        with h5py.File(path, "r") as f:
            if key not in f:
                # Try removing leading slash
                k2 = key[1:] if key.startswith("/") else "/" + key
                if k2 in f:
                    key = k2
                else:
                    raise KeyError(f"{key} not found in {filename}")
            dset = f[key]
            data = np.array(dset)
    except Exception as e:
        return abort(500, f"failed to read dataset: {e}")

    # Normalize shape to (time_steps, channels)
    if data.ndim == 1:
        series = data.reshape(-1, 1)
        time = np.arange(series.shape[0])
    elif data.ndim == 2:
        series = data
        time = np.arange(series.shape[0])
    elif data.ndim == 3:
        series = np.mean(data, axis=0)  # mean over episodes
        time = np.arange(series.shape[0])
    else:
        # Collapse extra leading dims into episodes
        episodes = int(np.prod(data.shape[:-2]))
        time_steps = data.shape[-2]
        channels = data.shape[-1]
        series = data.reshape(episodes, time_steps, channels).mean(axis=0)
        time = np.arange(series.shape[0])

    # Handle channel selection
    if channel is not None and channel != "":
        ch = int(channel)
        if ch < 0 or ch >= series.shape[1]:
            return abort(400, f"channel index out of range (0..{series.shape[1]-1})")
        plot_data = series[:, ch]
        legend = [f"channel {ch}"]
    else:
        mean_all = np.mean(series, axis=1)
        plot_data = mean_all
        legend = ["mean(all channels)"]

    # Plot figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, plot_data, linewidth=2, label=legend[0])
    if channel is None or channel == "":
        for c in range(series.shape[1]):
            ax.plot(time, series[:, c], alpha=0.35, linewidth=1)
    ax.set_xlabel("timestep")
    ax.set_ylabel("value")
    ax.set_title(f"{filename} | {key} | mean-over-episodes")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
