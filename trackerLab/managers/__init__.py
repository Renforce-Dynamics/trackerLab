from .motion_manager import MotionManager, MotionManagerCfg
from .skill_manager import SkillManager, SkillManagerCfg
try:
    from .amp_manager import AMPManager
except:
    print("amp load error")
