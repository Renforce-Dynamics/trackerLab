import mujoco
import numpy as np
import json
import argparse


def get_joint_local_axes(xml_path: str):
    """
    Parse an MJCF file and extract the mapping from joint names to DOF local axes.
    Axes are expressed in the joint's local frame (as defined in MJCF).
    
    Args:
        xml_path (str): Path to the MJCF (XML) file.

    Returns:
        dict[str, list[list[float]]]:
            Dictionary mapping joint_name -> list of DOF local axes.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)

    joint_axes = {}

    for j_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        j_type = model.jnt_type[j_id]

        axes = []

        if j_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
            # Hinge/Slide: one local axis
            local_axis = model.jnt_axis[j_id]
            axes.append(local_axis.tolist())

        elif j_type == mujoco.mjtJoint.mjJNT_BALL:
            # Ball: 3 DOFs, by convention XYZ
            axes = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]

        elif j_type == mujoco.mjtJoint.mjJNT_FREE:
            # Free: 3 translation + 3 rotation DOFs
            axes = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]

        joint_axes[name] = axes

    return joint_axes


def axis_angle_to_quat(axis, angle):
    """
    Convert axis-angle representation to quaternion.
    
    Args:
        axis (np.ndarray): Axis vector (3,)
        angle (float): Rotation angle in radians
    
    Returns:
        np.ndarray: Quaternion [w, x, y, z]
    """
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
    
    axis = axis / norm
    half_angle = angle / 2.0
    qw = np.cos(half_angle)
    qx, qy, qz = axis * np.sin(half_angle)
    return np.array([qw, qx, qy, qz])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract joint DOF local axes from MJCF file.")
    parser.add_argument("--xml", type=str, required=True, help="Path to MJCF (XML) file.")
    parser.add_argument("--out", type=str, default="joint_local_axes.json", help="Output JSON file.")
    parser.add_argument("--test", action="store_true", help="Run a test example for axis-angle to quaternion.")

    args = parser.parse_args()
    
    import os
    from poselib import POSELIB_DOF_AXIS_DIR

    # Extract local axes
    mapping = get_joint_local_axes(args.xml)

    # Save to JSON
    with open(os.path.join(POSELIB_DOF_AXIS_DIR, args.out), "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Joint local axes mapping saved to {args.out}")

    # Test conversion if requested
    if args.test:
        print("\nTesting axis-angle → quaternion:")
        axis = [0, 0, 1]
        angle = np.pi / 2  # 90 degrees
        quat = axis_angle_to_quat(axis, angle)
        print(f"Axis: {axis}, Angle: {angle} rad → Quaternion: {quat}")
