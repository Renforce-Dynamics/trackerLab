import os
import json

# from poselib import TrackerLab_LABJOINTS_DIR

def get_lab_joint_names(robot_type: str) -> list:
    """
    Load the lab joint names for a specific robot type.
    
    Args:
        robot_type (str): The type of the robot.
        
    Returns:
        list: A list of lab joint names.
    """
    file_path = os.path.join(TrackerLab_LABJOINTS_DIR, f"{robot_type}.json")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get("lab_joint_names")