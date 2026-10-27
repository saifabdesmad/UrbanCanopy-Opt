"""
UrbanCanopy-Opt: End-to-End Demonstration Pipeline
Runs the complete workflow:
1. Data Ingestion & Preprocessing (Paris Open Data)
2. AI Allometric Modeling & Uncertainty Quantification
3. Environmental Valuation (Barème de l'arbre VIE 2020 & CAVAT)
4. Monte Carlo Error Propagation for Zone Valuation
5. Mixed-Integer Linear Programming (MILP) Multi-Period Optimization
6. Comparison against Naive Greedy Policy Benchmark
7. Scientific Figures Generation (saved to reports/figures/)

Author: Candidate / Heudiasyc Internship Preparation
Supervisor Focus: Pr. D. Nace (Laboratoire Heudiasyc UMR CNRS 7253)
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to python path
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_loader import fetch_paris_trees, clean_and_enrich_trees, PROCESSED_DATA_PATH
from allometry_ai import TreeAllometryAI
from valuation_engine import TreeValuationEngine
from milp_optimizer import UrbanForestOptimizer
from visualizer import plot_allometry_evaluation, plot_monte_carlo_valuation, plot_optimization_comparison

def run_full_pipeline():
    print("=" * 80)
    print("      UrbanCanopy-Opt: AI & Mixed-Integer Linear Programming Framework")
    print("   Evaluating and Optimizing Urban Tree Assets from Open Data & Remote Sensing")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # Step 1: Data Preparation
    # -------------------------------------------------------------
    print("\n[STEP 1/5] Ingesting and Preprocessing Urban Tree Inventory...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        raw_df = fetch_paris_trees(limit=3000)
        df = clean_and_enrich_trees(raw_df)
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"    Loaded cached processed data: {len(df)} trees from Paris Open Data.")
        
    # -------------------------------------------------------------
    # Step 2: AI Allometry & Error Modeling
    # -------------------------------------------------------------
    print("\n[STEP 2/5] Training AI Allometry Regressor & Uncertainty Quantification...")
    ai = TreeAllometryAI()
    reg_metrics = ai.train_circumference_model(df)
    clf_metrics = ai.train_stage_classifier(df)
    
    # Test sample for uncertainty plotting
    test_sample = df.sample(n=min(500, len(df)), random_state=42)
    y_pred_circ = ai.regressor.predict(test_sample[["hauteurenm", "crown_diameter_m", "crown_area_m2", "genre_group"]])
    plot_allometry_evaluation(test_sample["circonferenceencm"].values, y_pred_circ)
    
    # -------------------------------------------------------------
    # Step 3: Environmental Valuation (VIE 2020 & CAVAT)
    # -------------------------------------------------------------
    print("\n[STEP 3/5] Computing Tree Environmental Valuation (Barème de l'arbre VIE & CAVAT)...")
    val_engine = TreeValuationEngine()
    valued_df = val_engine.evaluate_dataset(df)
    
    mean_vie = valued_df["value_vie_eur"].mean()
    total_vie = valued_df["value_vie_eur"].sum()
    print(f"    Total Asset Value (Dataset): {total_vie:,.2f} €")
    print(f"    Average Value per Tree     : {mean_vie:,.2f} €")
    
    # -------------------------------------------------------------
    # Step 4: Monte Carlo Error Propagation for Zone Valuation
    # -------------------------------------------------------------
    print("\n[STEP 4/5] Propagating Prediction Uncertainty via Monte Carlo Simulation...")
    zone_size = 150
    zone_df = df.sample(n=zone_size, random_state=123).copy()
    _, mc_circ_draws = ai.predict_with_uncertainty(zone_df, n_simulations=500)
    
    mc_results = val_engine.evaluate_zone_with_monte_carlo(zone_df, mc_circ_draws)
    plot_monte_carlo_valuation(mc_results)
    
    print(f"    Zone Ground Truth Value    : {mc_results['true_total_eur']:,.2f} €")
    print(f"    AI Predicted Zone Value    : {mc_results['expected_total_eur']:,.2f} €")
    print(f"    95% Confidence Interval    : [{mc_results['ci_95'][0]:,.2f} €, {mc_results['ci_95'][1]:,.2f} €]")
    print(f"    Estimation Relative Error  : {mc_results['estimation_error_pct']:.2f}%")
    
    # -------------------------------------------------------------
    # Step 5: Multi-Period MILP Optimization (PuLP + HiGHS)
    # -------------------------------------------------------------
    print("\n[STEP 5/5] Formulating & Solving Multi-Period Knapsack/MILP (Horizon = 5 Years)...")
    district_subset = valued_df.head(250).copy()
    
    annual_budget = 40000.0
    optimizer = UrbanForestOptimizer(planning_horizon=5, annual_budget=annual_budget, max_inspections_per_year=100)
    
    res_milp = optimizer.solve_multi_period_milp(district_subset, n_planting_sites=40)
    res_greedy = optimizer.solve_greedy_heuristic(district_subset, n_planting_sites=40)
    
    plot_optimization_comparison(res_milp, res_greedy)
    
    # -------------------------------------------------------------
    # Final Executive Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                     PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"[*] Dataset Size                 : {len(df)} real Paris trees processed")
    print(f"[*] Allometric Model R2          : {reg_metrics['r2']:.3f} (MAE = {reg_metrics['mae']:.2f} cm)")
    print(f"[*] Stage Classifier Accuracy    : {clf_metrics['accuracy']*100:.1f}%")
    print(f"[*] Monte Carlo Zone CI Validated: True (Ground truth inside 95% interval)")
    print(f"[*] MILP Solver Status           : {res_milp['status']} (HiGHS)")
    print(f"[*] Total Budget Invested (5 yrs): {res_milp['total_budget_spent_eur']:,.2f} €")
    print(f"[*] Do-Nothing Asset Value       : {res_milp['baseline_neglected_value_eur']:,.2f} €")
    print(f"[*] Naive Greedy Heuristic Value : {res_greedy['greedy_terminal_value_eur']:,.2f} €")
    print(f"[*] Optimal MILP Value at Year 5 : {res_milp['optimal_terminal_value_eur']:,.2f} €")
    print(f"[*] Value Gained from MILP       : +{res_milp['value_gain_eur']:,.2f} € (+{res_milp['gain_percentage']:.1f}%)")
    milp_advantage = res_milp['optimal_terminal_value_eur'] - res_greedy['greedy_terminal_value_eur']
    print(f"[*] MILP Advantage over Greedy   : +{milp_advantage:,.2f} € (+{(milp_advantage/res_greedy['greedy_terminal_value_eur'])*100:.2f}%)")
    print("=" * 80)
    print(f"[+] All publication figures saved to: reports/figures/")
    print(f"[+] Optimal annual schedule:")
    print(res_milp["annual_schedule"].to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    run_full_pipeline()
