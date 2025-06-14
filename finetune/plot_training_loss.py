# plot_comparison.py
import os
import pandas as pd
from tbparse import SummaryReader
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =================================================================================
# === 1. 参数配置区 (支持多组实验对比) ===
# =================================================================================
class PlotConfig:
    """
    用于生成出版级对比图表的配置
    """
    # !!! 关键：请在此列表中配置您要对比的所有实验 !!!
    # 每个字典包含一个唯一的名字(用于图例)和日志路径
    experiments: List[Dict[str, Any]] = [
        {
            "name": "Qwen3, LR=2e-4",
            "path": "../results/qwen3-14b-lora-bf16_lr-1e-4/runs"  # 示例路径1
        },
        {
            "name": "Qwen3, LR=1e-4",
            "path": "../results/qwen3-14b-lora-bf16_lr-2e-4/runs"  # 示例路径2，请修改
        },

        # 在这里继续添加更多实验...
    ]

    # --- 输出文件配置 ---
    output_filename = "comparison_loss_curve.pdf"
    output_dpi = 300

    # --- 字体配置 (关键：与论文正文匹配) ---
    font_family = 'serif'
    font_serif = 'Times New Roman'
    label_fontsize = 12
    tick_fontsize = 10
    legend_fontsize = 10

    # --- 图像尺寸与样式配置 ---
    figure_size = (7, 5)  # 稍微加宽以容纳更多图例
    line_width = 1.8
    grid_alpha = 0.4

    # --- 曲线颜色方案 (将按顺序分配给每个实验) ---
    color_palette = ['blue', 'red', 'green', 'purple', 'orange', 'brown']


# =================================================================================

def setup_matplotlib_style(config: PlotConfig):
    """根据配置设置Matplotlib的全局样式"""
    plt.rcParams.update({
        'font.family': config.font_family,
        'font.serif': [config.font_serif],
        'font.size': config.tick_fontsize,
        'axes.labelsize': config.label_fontsize,
        'xtick.labelsize': config.tick_fontsize,
        'ytick.labelsize': config.tick_fontsize,
        'legend.fontsize': config.legend_fontsize,
        'figure.figsize': config.figure_size,
        'lines.linewidth': config.line_width,
        'grid.linestyle': '--',
        'grid.alpha': config.grid_alpha,
        'axes.grid': True,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def plot_comparison_curves(config: PlotConfig):
    """
    读取多个TensorBoard日志，并将它们的Loss曲线绘制在同一张对比图中。
    """
    logging.info("开始生成对比图...")
    setup_matplotlib_style(config)

    fig, ax = plt.subplots()

    # 检查颜色是否足够
    if len(config.experiments) > len(config.color_palette):
        logging.warning("实验数量超过预设颜色数量，颜色将会循环使用。")

    # 循环处理每个实验
    for i, exp in enumerate(config.experiments):
        exp_name = exp.get("name", f"Experiment {i + 1}")
        log_dir = exp.get("path")
        color = config.color_palette[i % len(config.color_palette)]

        logging.info(f"--- 处理实验: '{exp_name}' ---")
        if not log_dir or not os.path.isdir(log_dir):
            logging.error(f"跳过 '{exp_name}'，因路径 '{log_dir}' 无效或未提供。")
            continue

        try:
            reader = SummaryReader(log_dir, pivot=True)
            df = reader.scalars
            if df.empty:
                logging.warning(f"在 '{log_dir}' 中没有找到日志数据。")
                continue
        except Exception as e:
            logging.error(f"读取 '{log_dir}' 日志时发生错误: {e}")
            continue

        # 绘制训练损失 (实线)
        train_loss_df = df[df['train/loss'].notna()].dropna(subset=['step', 'train/loss'])
        if not train_loss_df.empty:
            ax.plot(train_loss_df["step"], train_loss_df["train/loss"],
                    label=f"{exp_name} (Train)",
                    color=color,
                    linestyle='-')  # 实线表示训练
        else:
            logging.warning(f"实验 '{exp_name}' 未找到 'train/loss' 数据。")

        # 绘制验证损失 (虚线)
        eval_loss_df = df[df['eval/loss'].notna()].dropna(subset=['step', 'eval/loss'])
        if not eval_loss_df.empty:
            ax.plot(eval_loss_df["step"], eval_loss_df["eval/loss"],
                    label=f"{exp_name} (Val)",
                    color=color,
                    linestyle='--')  # 虚线表示验证
        else:
            logging.warning(f"实验 '{exp_name}' 未找到 'eval/loss' 数据。")

    # --- 最终化配置与保存 ---
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')

    # 调整图例位置，防止遮挡曲线
    # 'best' 会自动寻找最佳位置，也可手动设为 'upper right' 等
    ax.legend(loc='best')

    fig.tight_layout()
    plt.savefig(config.output_filename, dpi=config.output_dpi, bbox_inches='tight')
    logging.info(f"✅ 对比图已成功保存至: {config.output_filename}")
    plt.close(fig)


def main():
    config = PlotConfig()
    if not config.experiments or not any(exp.get("path") for exp in config.experiments):
        logging.error("错误：请在 PlotConfig 中配置至少一个有效的实验路径。")
        return
    plot_comparison_curves(config)


if __name__ == "__main__":
    main()