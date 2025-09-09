import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_pins_comparison(PF_online_data_sec1, PF_offline_org, save_path=None):
    """
    绘制 p_ins 的并列柱状图，比较 online 和 offline 的结果。
    
    Parameters
    ----------
    PF_online_data_sec1 : pandas.DataFrame
        在线数据 DataFrame，需包含 'p_ins' 列。
    PF_offline_org : pandas.DataFrame
        离线数据 DataFrame，需包含 'p_ins' 列。
    save_path : str or None, optional
        如果为字符串，则保存为 PDF 和 SVG 格式到指定路径；
        如果为 None，则只显示不保存。
    """
    
    # 计算 sum
    online_sum = PF_online_data_sec1['p_ins'].sum()
    offline_sum = PF_offline_org['p_ins'].sum()

    # 拼成对比 DataFrame
    df_compare = pd.DataFrame({
        'Online': online_sum,
        'Offline': offline_sum
    })

    # 设置风格
    sns.set_theme(style="whitegrid", font_scale=1.2)
    colors = sns.color_palette("Set2", n_colors=2)

    # 画图
    ax = df_compare.plot(
        kind='bar',
        figsize=(12, 4),
        width=0.7,
        color=colors,
        edgecolor='black'
    )

    # 美化
    ax.set_title("Comparison of p_in Sum (Online vs Offline)", fontsize=14, weight='bold')
    ax.set_ylabel("Sum of p_in")
    ax.set_xlabel("Columns")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Dataset", loc="upper right", frameon=True)

    # 加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=9, label_type="edge", padding=2)

    plt.tight_layout()

    # 保存选项
    if isinstance(save_path, str):
        plt.savefig(f"{save_path}.pdf", format="pdf")
        plt.savefig(f"{save_path}.svg", format="svg")

    plt.show()
