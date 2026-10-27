"""
UrbanCanopy-Opt: Visualization Module
Generates publication-quality figures for:
- LiDAR Allometry AI regression & residuals
- Monte Carlo valuation uncertainty intervals
- MILP Optimization vs Greedy benchmark
- Multi-period municipal schedule & budget allocation
Author: Candidate / Heudiasyc Internship Preparation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set elegant academic plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

def plot_allometry_evaluation(y_true, y_pred, save_path=None):
    """Figure 1: Scatter plot of true vs predicted circumference with error distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Scatter True vs Predicted
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.4, color="#1b4965", edgecolors="none", s=25, label="Urban Trees (Paris)")
    lims = [0, max(np.max(y_true), np.max(y_pred)) + 10]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Allometry (1:1)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground Truth Circumference (cm, Field Inventory)")
    ax.set_ylabel("Predicted Circumference (cm, LiDAR Allometry AI)")
    ax.set_title("LiDAR AI Allometric Model (Height & Crown -> Trunk)", fontweight="bold")
    ax.legend()
    
    # Residuals Distribution
    ax2 = axes[1]
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, ax=ax2, color="#2b9348", stat="density", bins=30)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax2.axvline(np.mean(residuals) + 1.96 * np.std(residuals), color="darkred", linestyle=":", label="95% Confidence Bounds")
    ax2.axvline(np.mean(residuals) - 1.96 * np.std(residuals), color="darkred", linestyle=":")
    ax2.set_xlabel("Prediction Residual Error (cm)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Uncertainty Distribution (Mean={np.mean(residuals):.1f} cm, Std={np.std(residuals):.1f} cm)", fontweight="bold")
    ax2.legend()
    
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "fig1_allometry_performance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Figure saved: {save_path}")

def plot_monte_carlo_valuation(mc_results, save_path=None):
    """Figure 2: Distribution of total zone economic value (VIE) under Monte Carlo error propagation."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    dist = mc_results["mc_distribution"]
    true_val = mc_results["true_total_eur"]
    exp_val = mc_results["expected_total_eur"]
    ci = mc_results["ci_95"]
    
    sns.histplot(dist, kde=True, ax=ax, color="#0077b6", stat="density", bins=35, label="Monte Carlo Draws (N=500)")
    ax.axvline(true_val, color="#d90429", linewidth=2.5, linestyle="-", label=f"Real Ground Truth Value ({true_val:,.0f} €)")
    ax.axvline(exp_val, color="#ffb703", linewidth=2.0, linestyle="--", label=f"AI Expected Value ({exp_val:,.0f} €)")
    ax.axvspan(ci[0], ci[1], alpha=0.15, color="#0077b6", label=f"95% Confidence Interval [{ci[0]:,.0f} €, {ci[1]:,.0f} €]")
    
    ax.set_xlabel("Total Zone Economic Value (VIE 2020 in Euros)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Monte Carlo Error Propagation: LiDAR Uncertainty -> District Economic Value", fontweight="bold")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "fig2_monte_carlo_valuation.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Figure saved: {save_path}")

def plot_optimization_comparison(res_milp, res_greedy, save_path=None):
    """Figure 3: Policy comparison between Neglect, Greedy Heuristic, and Optimal MILP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Policy Terminal Values
    policies = ["Do-Nothing\n(Neglect)", "Naive Greedy\nHeuristic", "Optimal MILP\n(PuLP + HiGHS)"]
    values = [
        res_milp["baseline_neglected_value_eur"],
        res_greedy["greedy_terminal_value_eur"],
        res_milp["optimal_terminal_value_eur"]
    ]
    colors = ["#adb5bd", "#f39c12", "#2ecc71"]
    
    bars = ax1.bar(policies, values, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,.0f} €",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold")
                     
    ax1.set_ylabel("Terminal Forest Value at Year 5 (VIE in €)")
    ax1.set_title("5-Year Value Comparison Across Management Policies", fontweight="bold")
    ax1.set_ylim(0, max(values) * 1.15)
    
    # Annual Schedule Breakdown from MILP
    sched = res_milp["annual_schedule"]
    years = sched["Year"]
    ax2.plot(years, sched["Prunings"], marker="o", linewidth=2.5, color="#27ae60", label="Prunings & Health Care")
    ax2.plot(years, sched["Inspections"], marker="s", linewidth=2.0, color="#2980b9", label="Inspections Conducted")
    ax2.plot(years, sched["New_Plantings"], marker="^", linewidth=2.0, color="#8e44ad", label="New Trees Planted")
    ax2.plot(years, sched["Replacements"], marker="d", linewidth=2.0, color="#c0392b", label="Diseased Replacements")
    
    ax2.set_xlabel("Planning Horizon Year")
    ax2.set_ylabel("Number of Interventions")
    ax2.set_title("Optimal Annual Action Schedule (MILP Solution)", fontweight="bold")
    ax2.set_xticks(years)
    ax2.legend()
    
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "fig3_optimization_benchmark.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Figure saved: {save_path}")

if __name__ == "__main__":
    print("[*] Testing visualizer module...")
    # Quick self-test with dummy data
    y_t = np.random.uniform(20, 200, 200)
    y_p = y_t + np.random.normal(0, 10, 200)
    plot_allometry_evaluation(y_t, y_p)
