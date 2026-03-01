import matplotlib.pyplot as plt
import numpy as np

def plot_performance_comparison(methods, toolbench_score, anytoolbench_score, ours_score, avg_score):
    plt.figure(figsize=(8, 5))
    plt.plot(methods, toolbench_score, marker='o', linewidth=2, color='#1f77b4', label="ToolBench")
    plt.plot(methods, anytoolbench_score, marker='s', linewidth=2, color='#ff7f0e', label="AnyToolBench")
    plt.plot(methods, ours_score, marker='^', linewidth=2, color='#2ca02c', label="Ours")
    plt.plot(methods, avg_score, marker='d', linewidth=2, color='#d62728', label="Average")

    plt.ylim(bottom=30, top=np.max([toolbench_score, anytoolbench_score, ours_score, avg_score]) + 2)

    plt.xlabel("Method", fontsize=12, fontweight='medium')
    plt.ylabel("Score", fontsize=12, fontweight='medium')
    plt.title("Performance Comparison Across Methods", fontsize=13, fontweight="bold")
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=False)
    plt.tight_layout()

    plt.savefig("performance_comparison_noGPT.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_step_comparison(methods, toolbench_step, anytoolbench_step, ours_step, avg_step):
    plt.figure(figsize=(8, 5))
    plt.plot(methods, toolbench_step, marker='o', linewidth=2, label="ToolBench")
    plt.plot(methods, anytoolbench_step, marker='s', linewidth=2, label="AnyToolBench")
    plt.plot(methods, ours_step, marker='^', linewidth=2, label="Ours")
    plt.plot(methods, avg_step, marker='d', linewidth=2, label="Average")

    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Step", fontsize=12)
    plt.title("Step Comparison Across Methods", fontsize=13, fontweight="bold")
    plt.xticks(rotation=0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig("step_comparison_updated.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    methods = ["ToolLLM", "AnyTool", "Ours"]
    toolbench_score = [36.89, 38.46, 45.79]
    anytoolbench_score = [41.67, 39.96, 55.10]
    ours_score = [48.99, 57.07, 71.68]
    avg_score = [42.51, 45.16, 59.19]

    toolbench_step = [30.84, 10.5, 11.42]
    anytoolbench_step = [25.90, 13.25, 16.14]
    ours_step = [18.58, 9.82, 4.56]
    avg_step = [25.10, 11.19, 10.70]

    plot_performance_comparison(methods, toolbench_score, anytoolbench_score, ours_score, avg_score)
    plot_step_comparison(methods, toolbench_step, anytoolbench_step, ours_step, avg_step)