#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AnyDexGrasp 配置文件
处理路径和环境设置
"""

import os
import sys


def setup_anydexgrasp_paths():
    """设置AnyDexGrasp项目的Python路径"""

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 尝试不同的可能路径
    possible_roots = [
        # 如果在ROS包中运行
        os.path.join(current_dir, "..", "..", "..", ".."),
        # 如果在inference_v2目录中运行
        os.path.join(current_dir, "..", "..", "..", ".."),
        # 直接在项目根目录
        "/home/ps/Projects/AnyDexGrasp",
    ]

    anydexgrasp_root = None
    for root in possible_roots:
        root = os.path.abspath(root)
        if (
            os.path.exists(os.path.join(root, "inference_v2"))
            and os.path.exists(os.path.join(root, "models"))
            and os.path.exists(os.path.join(root, "dataset"))
        ):
            anydexgrasp_root = root
            break

    if anydexgrasp_root is None:
        # 尝试环境变量
        if "ANYDEXGRASP_ROOT" in os.environ:
            anydexgrasp_root = os.environ["ANYDEXGRASP_ROOT"]
        else:
            anydexgrasp_root = "/home/ps/Projects/AnyDexGrasp"

    # 添加必要的路径到 sys.path
    paths_to_add = [
        anydexgrasp_root,
        os.path.join(anydexgrasp_root, "inference_v2"),
        os.path.join(anydexgrasp_root, "models"),
        os.path.join(anydexgrasp_root, "dataset"),
        os.path.join(anydexgrasp_root, "adg_utils"),
    ]

    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)

    return anydexgrasp_root


def get_default_model_paths(anydexgrasp_root):
    """获取默认的模型路径"""
    return {
        "inspire_checkpoint": os.path.join(
            anydexgrasp_root,
            "logs/model/inspire_model/final_single_point/obj140/checkpoint.tar",
        ),
        "inspire_models": os.path.join(
            anydexgrasp_root, "logs/model/inspire_model/final_single_point/obj140"
        ),
    }
