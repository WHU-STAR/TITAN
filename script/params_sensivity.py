# 参数敏感度实验

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# 与 analyze_memory_usage.py 保持一致的字体基准
FONT_SIZE = 28
MARKER_SIZE = 12

# 从 plotconf.yaml 中提取的样式配置
STYLE_CONFIG = [
    {"color": "#1f77b4", "linestyle": "-", "marker": "o"},  # 蓝色  # 实线  # 圆圈标记
    {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},  # 橙色  # 虚线  # 方块标记
    {
        "color": "#2ca02c",  # 绿色
        "linestyle": "-.",  # 点划线
        "marker": "D",  # 菱形标记
    },
    {"color": "#ff9896", "linestyle": "-", "marker": "h"},  # 粉色  # 实线  # 六边形标记
    {"color": "#9467bd", "linestyle": "-", "marker": "v"},  # 紫色  # 实线  # 下三角标记
    {"color": "#d62728", "linestyle": ":", "marker": "^"},  # 红色  # 点线  # 上三角标记
]

METAK = {
    "dataset": "deep100m",
    "csv_path": "../results/deep100m_10mbkt_similarity_parameter_test.csv",
    "target_param": "meta_k",
    "target_label": "ef_search",
    "param_range": [20, 60, 100, 140],
}

EF_SEARCH = {
    "dataset": "deep100m",
    "csv_path": "../results/deep100m_10mbkt_similarity_parameter_test.csv",
    "target_param": "ef_search",
    "target_label": "ef_search",
    "param_range": [40, 80, 120, 160],
}

IVF_THRESH = {
    "dataset": "deep100m",
    "csv_path": "../results/deep100m_10mbkt_similarity_parameter_test.csv",
    "target_param": "thresh",
    "target_label": "\u03c4",
    "param_range": [0, 2, 4, 6],
}

PRUING_LAMBDA = {
    "dataset": "deep100m",
    "csv_path": "../results/deep100m_10mbkt_similarity_parameter_test.csv",
    "target_param": "lambda",
    "target_label": "\u03bb",
    "filter": "thresh:[0,2];",
    "param_range": [0.3, 0.5, 0.7, 0.9],
}


def get_max_qps_data(df):
    """对数据按recall排序，取每个recall对应的最大qps（单调递增）"""
    df = df.sort_values(by="recall").reset_index(drop=True)

    # 反向遍历，保留qps单调递增的点
    max_qps = -1
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]["qps"] <= max_qps:
            df = df.drop(index=i)
        else:
            max_qps = df.iloc[i]["qps"]

    return df.reset_index(drop=True)


def analyze_params_sensivity(config):
    """分析参数敏感度，返回每个参数值对应的数据"""
    df = pd.read_csv(config["csv_path"])
    target_param = config["target_param"]
    param_range = config["param_range"]

    # 应用filter过滤（支持多个filter，用分号分割）
    if "filter" in config:
        filter_str = config["filter"]
        original_rows = len(df)

        # 用分号分割多个filter条件
        filter_conditions = [f.strip() for f in filter_str.split(";") if f.strip()]

        for filter_condition in filter_conditions:
            # 解析filter格式：如 "thresh:[0,2]" -> 参数名: thresh, 值列表: [0, 2]
            if (
                ":" in filter_condition
                and "[" in filter_condition
                and "]" in filter_condition
            ):
                param_name, values_str = filter_condition.split(":", 1)
                param_name = param_name.strip()
                # 提取方括号中的值
                values_str = values_str.strip()
                values_str = values_str.strip("[]")
                # 解析值列表（支持数字和字符串）
                values = [v.strip() for v in values_str.split(",")]
                # 尝试转换为数字类型
                try:
                    # 尝试转换为整数
                    values = [int(v) for v in values]
                except ValueError:
                    try:
                        # 尝试转换为浮点数
                        values = [float(v) for v in values]
                    except ValueError:
                        # 保持为字符串
                        pass

                # 应用filter：只保留指定参数值在允许列表中的数据
                if param_name in df.columns:
                    rows_before_filter = len(df)
                    df = df[df[param_name].isin(values)].copy()
                    print(
                        f"Applied filter: {param_name} in {values}, rows before: {rows_before_filter}, remaining rows: {len(df)}"
                    )

        print(
            f"Total filter applied: {len(filter_conditions)} condition(s), original rows: {original_rows}, final rows: {len(df)}"
        )

    results = {}

    for param_value in param_range:
        # 过滤出该参数值的数据
        subset = df[df[target_param] == param_value].copy()

        # 分组：对其他参数进行分组，计算平均qps
        group_cols = ["meta_k", "ef_search", "thresh", "n_probe"]
        group_cols = [
            col for col in group_cols if col in subset.columns and col != target_param
        ]

        if group_cols:
            grouped = subset.groupby(group_cols, as_index=False).agg(
                {"recall": "first", "qps": "mean"}
            )
        else:
            grouped = subset[["recall", "qps"]].copy()

        # 取最大值
        max_data = get_max_qps_data(grouped)
        results[param_value] = max_data

    return results


def plot_params_sensivity(config, results):
    """绘制参数敏感度图表，所有参数值在同一张图上用不同曲线表示"""
    target_param = config["target_param"]
    param_range = config["param_range"]
    target_label = config["target_label"]
    save_dir = "../results/old/figs"
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update({"font.size": FONT_SIZE})
    plt.figure(figsize=(7.2, 5.2))

    for idx, param_value in enumerate(param_range):
        data = results[param_value]
        # 使用配置中的样式，循环使用
        style = STYLE_CONFIG[idx % len(STYLE_CONFIG)]
        plt.plot(
            data["recall"],
            data["qps"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=MARKER_SIZE,
            label=f"{target_label} = {param_value}",
        )

    # 格式化 y 轴刻度为 k 格式（10k, 20k, 30k 等）
    def format_thousands(value, tick_number):
        if value >= 1000:
            return f"{int(value/1000)}k"
        return f"{int(value)}"

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))
    plt.xlabel("Recall@20", fontsize=FONT_SIZE)
    plt.ylabel("Throughput (QPS)", fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    # plt.title(f'Parameter Sensitivity: {target_param}', fontweight='bold')
    plt.grid(True)
    plt.ylim(0, None)
    # 增加画布内边距，并让图例自动选择相对不遮挡曲线的位置
    plt.margins(x=0.03, y=0.08)
    plt.legend(loc="best", fontsize=FONT_SIZE-10, frameon=True, framealpha=0.9)
    plt.tight_layout(pad=1.2)
    plt.savefig(f"{save_dir}/params_sensivity_{target_param}.pdf", dpi=500)
    print(f"Saved figure to {save_dir}/params_sensivity_{target_param}.pdf")
    plt.show()


def main():
    config_list = [METAK, EF_SEARCH, IVF_THRESH, PRUING_LAMBDA]

    for config in config_list:
        # 分析参数敏感度
        results = analyze_params_sensivity(config)

        # 绘制图表
        plot_params_sensivity(config, results)


if __name__ == "__main__":
    main()
