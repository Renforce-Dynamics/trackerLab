# 📈 Data Flow (TrackerLab)

The data flow in TrackerLab bridges human motion datasets with robotic trajectory execution and training via modular abstractions. Here's the typical pipeline:

```mermaid
graph LR
    A[AMASS / FBX Dataset] --> B[Preprocessing]
    B --> C[Retargeting<br>poselib/scripts/amass/retarget_all.py]
    C --> D[Retargeted Motions<br>data/retargeted/]
    D --> E[Trajectory Manager]
    E --> F[Skill FSM / Command Manager]
    F --> G[RL Env (e.g., H1TrackAll)]
    G --> H[Training<br>train.py]
```

### Step-by-step:

* **Dataset (AMASS / FBX):** Raw human motion sequences.
* **Preprocessing:** T-pose normalization, joint filtering, frame truncation.
* **Retargeting:** Convert motion to robot-specific format (using `retarget_all.py`).
* **Retargeted Output:** Saved in `data/retargeted/`.
* **Trajectory Manager:** Loads and manages motion playback with timing.
* **Skill FSM / Command Manager:** Composes motions into tasks via state machines.
* **RL Environment:** Registered as `H1TrackAll`, interfaces with IsaacLab simulation.
* **Training:** Uses IsaacLab’s `train.py` to optimize policies.
