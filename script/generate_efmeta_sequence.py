#!/usr/bin/env python3
"""
根据配置文件生成 metaK 与 efSearch 的整数序列。

配置文件位于当前脚本所在目录下，文件名固定为: efmeta_config.yaml
并且已经在 gitignore 中忽略。

配置文件为 YAML 格式，其整体结构为:

```yaml
<配置项名称>:
  efSearch_start: <int>   # efSearch 起始值（包含）
  efSearch_end:   <int>   # efSearch 结束值（包含）
  efSearch_step:  <int>   # efSearch 步长（>0）
  k:              <float> # 一次函数系数 k
  b:              <float> # 一次函数截距 b

<另一个配置项名称>:
  efSearch_start: ...
  efSearch_end:   ...
  efSearch_step:  ...
  k:              ...
  b:              ...
```

命令行使用方式:

  python generate_efmeta_sequence.py <配置项名称>

脚本会读取对应配置，生成:
  efSearch 序列: 从 efSearch_start 到 efSearch_end，步长为 efSearch_step
  metaK 序列: 对应 efSearch，按 metaK = k * efSearch + b 计算，并四舍五入为整数

输出格式（示例）:

  metaK: 10 20 30 40
  efSearch: 30 40 50 60
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


CONFIG_FILE_NAME = "efmeta_config.yaml"


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    """加载所有配置项。"""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / CONFIG_FILE_NAME

    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件 YAML 解析失败: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是一个映射 (YAML mapping)")

    # 确保每个配置项也是映射
    normalized: Dict[str, Dict[str, Any]] = {}
    for name, conf in data.items():
        if not isinstance(conf, dict):
            raise ValueError(f"配置项 '{name}' 必须是一个映射 (mapping)")
        normalized[str(name)] = conf

    return normalized


def generate_sequences(conf: dict) -> tuple[list[int], list[int]]:
    """根据单个配置项生成 efSearch 和 metaK 序列（以 efSearch 为自变量）。"""
    required_keys = ["efSearch_start", "efSearch_end", "efSearch_step", "k", "b"]
    for key in required_keys:
        if key not in conf:
            raise KeyError(f"配置缺少必须字段: {key}")

    try:
        ef_start = int(conf["efSearch_start"])
        ef_end = int(conf["efSearch_end"])
        ef_step = int(conf["efSearch_step"])
        k = float(conf["k"])
        b = float(conf["b"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"配置字段类型错误: {e}") from e

    if ef_step <= 0:
        raise ValueError("efSearch_step 必须为正整数")
    if ef_end < ef_start:
        raise ValueError("efSearch_end 必须大于或等于 efSearch_start")

    efSearchs: list[int] = list(range(ef_start, ef_end + 1, ef_step))
    metaKs: list[int] = []

    for ef in efSearchs:
        val = k * ef + b
        # 使用四舍五入，并确保为非负整数
        iv = int(round(val))
        if iv < 0:
            iv = 0
        metaKs.append(iv)

    return metaKs, efSearchs


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        prog = os.path.basename(argv[0]) if argv else "generate_efmeta_sequence.py"
        print(f"用法: python {prog} <配置项名称>", file=sys.stderr)
        return 1

    config_name = argv[1]

    try:
        all_confs = load_all_configs()
    except Exception as e:
        print(f"读取配置文件失败: {e}", file=sys.stderr)
        return 1

    if config_name not in all_confs:
        print(f"配置项 '{config_name}' 不存在。可用配置项: {', '.join(all_confs.keys())}", file=sys.stderr)
        return 1

    try:
        metaKs, efSearchs = generate_sequences(all_confs[config_name])
    except Exception as e:
        print(f"生成序列失败: {e}", file=sys.stderr)
        return 1

    for metaK, efSearch in zip(metaKs, efSearchs):
        print(f"- [{metaK}, {efSearch}]")

    # # 按要求以整数形式输出到终端
    # print("metaK:", " ".join(str(x) for x in metaKs))
    # print("efSearch:", " ".join(str(x) for x in efSearchs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
