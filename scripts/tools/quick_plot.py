#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )

    x_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    y_values = [57.13, 66.22, 70.35, 70.0, 67.05, 70.55, 68.48, 59.22, 42.52]
    line_output_path = Path.cwd() / "quick_plot.png"

    plt.figure(figsize=(7, 4))
    plt.plot(x_values, y_values, marker="o", linewidth=2)
    plt.xticks(x_values)
    plt.xlim(min(x_values), max(x_values))
    plt.xlabel("Friction (low friction -> high friction)")
    plt.ylabel("Success Rate")
    plt.title("Privileged Policy Across Dynamics Configurations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(line_output_path, dpi=200)
    plt.close()
    print(f"Saved plot to {line_output_path}")

    labels = ["Correct privileged info", "Noised privileged info"]
    values = [83.3, 63.5]
    errors = [0.01, 8.7]
    bar_output_path = Path.cwd() / "privileged_policy_performance.png"

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, yerr=errors, capsize=8, width=0.5, color=["tab:blue", "tab:orange"])
    plt.ylabel("Success Rate")
    plt.title("Privileged Policy Performance")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(bar_output_path, dpi=200)
    plt.close()
    print(f"Saved plot to {bar_output_path}")


if __name__ == "__main__":
    main()
