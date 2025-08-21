# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


__version__ = "0.0.1"

from .core import *

import os

def config_dir(*args):
    dir = os.path.join(*args)
    os.makedirs(dir, exist_ok=True)
    return dir

# Dirs that containing data
POSELIB_REPO_DIR            = os.path.dirname(os.path.abspath(__file__))
POSELIB_DATA_DIR            = os.path.join(POSELIB_REPO_DIR, "..", "..","data")
POSELIB_AMASS_DIR           = os.path.join(POSELIB_DATA_DIR, "amass")
POSELIB_SCRIPT_DIR          = os.path.join(POSELIB_REPO_DIR, "scripts")
POSELIB_RETARGETED_DATA_DIR = config_dir(POSELIB_DATA_DIR, "retargeted")
POSELIB_MOTION_CFG_DIR      = os.path.join(POSELIB_DATA_DIR, "configs")
POSELIB_BUFFER_DIR          = os.path.join(POSELIB_DATA_DIR, "pkl_buffer")


# Dirs that containing poselib configs, where the retarget is using poselib
POSELIB_CFG_DIR = os.path.join(POSELIB_DATA_DIR, "poselib_cfg")

POSELIB_SKELETON_DIR        = os.path.join(POSELIB_CFG_DIR, "skeletons")
POSELIB_RETARGET_CFG_DIR    = os.path.join(POSELIB_CFG_DIR, "retarget_cfg")
POSELIB_MOTION_ALIGN_DIR    = os.path.join(POSELIB_CFG_DIR, "motion_align")
POSELIB_TPOSE_DIR           = os.path.join(POSELIB_CFG_DIR, "tpose")
POSELIB_TPOSE_MIDIFY_DIR    = os.path.join(POSELIB_CFG_DIR, "tpose_modify")

POSELIB_SMPL_SKELETON_PATH  = os.path.join(POSELIB_SKELETON_DIR, "skeleton_smpl.json")