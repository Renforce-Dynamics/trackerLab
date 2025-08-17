import os
from trackerLab import TRACKERLAB_ASSETS_DIR
from etils import epath

def get_assets(robot_name) -> list[str]:
    assets = []
    assets_dir = os.path.join(TRACKERLAB_ASSETS_DIR, "data", robot_name)
    xml_dir = os.path.join(assets_dir, "mjcf")
    assets = epath.Path(xml_dir).glob("*.xml")
    return assets