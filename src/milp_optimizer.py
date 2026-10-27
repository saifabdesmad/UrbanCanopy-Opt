"""
UrbanCanopy-Opt: Mixed-Integer Linear Programming (MILP) Optimizer
Formulates and solves the multi-period urban tree maintenance, replacement,
and planting problem with dynamic growth and budgetary constraints.
Solved using PuLP with the HiGHS state-of-the-art solver.
Author: Candidate / Heudiasyc Internship Preparation
"""

import os
import numpy as np
import pandas as pd
import pulp

class UrbanForestOptimizer:
    def __init__(self, planning_horizon=5, annual_budget=50000.0, max_inspections_per_year=120):
        """
        Args:
            planning_horizon (int): Number of years (T).
            annual_budget (float): Municipal budget in Euros per year.
            max_inspections_per_year (int): Maximum team inspection capacity.
        """
        self.T = planning_horizon
        self.annual_budget = annual_budget
        self.max_inspections = max_inspections_per_year

    def solve_multi_period_milp(self, df_trees, n_planting_sites=30):
        """
        Builds and solves the Multi-Period Mixed-Integer Linear Program.
        
        Args:
            df_trees (pd.DataFrame): Existing trees with 'value_vie_eur', 'health_score', 'hauteurenm', 'circonferenceencm'.
            n_planting_sites (int): Number of available planting spots in the district.
            
        Returns:
            dict containing:
                - status: Optimization status (Optimal, Feasible, etc.)
                - objective_value: Total urban forest value at horizon T
                - total_cost: Total budget spent across all T years
                - annual_schedule: DataFrame of actions per year
                - summary_stats: Dict of action counts
        """
        N = len(df_trees)
        M = n_planting_sites
        T = self.T
        
        print(f"[*] Formulating Multi-Period Urban Forestry MILP (T={T} years, N={N} trees, M={M} planting sites)...")
        
        # Operational Unit Costs
        cost_inspect = 60.0  # Euros
        cost_plant_young = 750.0  # Planting young sapling at new site
        
        # Pruning cost scales with tree height: 100 + 4 * H
        cost_prune = (100.0 + 4.0 * df_trees["hauteurenm"].values).astype(float)
        
        # Replacement cost scales with trunk size: 500 + 3 * C
        cost_replace = (500.0 + 3.0 * df_trees["circonferenceencm"].values).astype(float)
        
        # Baseline tree values
        v0 = df_trees["value_vie_eur"].values.astype(float)
        health = df_trees["health_score"].values.astype(float)
        
        # Growth and degradation rates
        g_mature = 0.035  # +3.5% per year if maintained
        delta_decay = 0.07  # -7.0% per year if neglected or diseased
        v_young_init = 450.0
        g_young = 0.12  # +12% rapid annual growth for young trees
        
        # Compute terminal values at horizon T based on action time t:
        # If tree i is neglected for all T years:
        v_neglected_T = v0 * ((1.0 - delta_decay * (1.1 - health)) ** T)
        
        # Incremental value boost if pruned in year t:
        boost_prune = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                years_remaining = T - (t + 1)
                # Maintaining prevents decay and adds growth
                val_with_prune = v0[i] * (1.0 + g_mature) ** years_remaining
                boost_prune[i, t] = max(0.0, val_with_prune - v_neglected_T[i])
                
        # Incremental value boost if replaced in year t:
        boost_replace = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                years_remaining = T - (t + 1)
                val_with_replace = v_young_init * ((1.0 + g_young) ** years_remaining)
                boost_replace[i, t] = max(0.0, val_with_replace - v_neglected_T[i])
                
        # Terminal value of planting at site j in year t:
        val_planted = np.zeros((M, T))
        for j in range(M):
            for t in range(T):
                years_remaining = T - (t + 1)
                val_planted[j, t] = v_young_init * ((1.0 + g_young) ** years_remaining)
                
        # --- MILP Model Formulation with PuLP ---
        model = pulp.LpProblem("Urban_Forest_Canopy_Optimization", pulp.LpMaximize)
        
        # Decision Variables
        # x_insp[i, t] in {0, 1}
        # x_prune[i, t] in {0, 1}
        # x_repl[i, t] in {0, 1}
        # y_plant[j, t] in {0, 1}
        x_insp = {}
        x_prune = {}
        x_repl = {}
        for i in range(N):
            for t in range(T):
                x_insp[i, t] = model.add_variable(f"insp_{i}_{t}", cat="Binary")
                x_prune[i, t] = model.add_variable(f"prune_{i}_{t}", cat="Binary")
                x_repl[i, t] = model.add_variable(f"repl_{i}_{t}", cat="Binary")
                
        y_plant = {}
        for j in range(M):
            for t in range(T):
                y_plant[j, t] = model.add_variable(f"plant_{j}_{t}", cat="Binary")
                
        # Objective Function: Maximize Terminal Value at Year T
        # Objective = Baseline Neglected Value + sum(boosts from actions)
        obj_expr = pulp.lpSum([
            boost_prune[i, t] * x_prune[i, t] + boost_replace[i, t] * x_repl[i, t]
            for i in range(N) for t in range(T)
        ]) + pulp.lpSum([
            val_planted[j, t] * y_plant[j, t]
            for j in range(M) for t in range(T)
        ])
        model.setObjective(obj_expr)
        
        # Constraint 1: Annual Municipal Budget Limits
        for t in range(T):
            budget_expr = pulp.lpSum([
                cost_inspect * x_insp[i, t] +
                cost_prune[i] * x_prune[i, t] +
                cost_replace[i] * x_repl[i, t]
                for i in range(N)
            ]) + pulp.lpSum([
                cost_plant_young * y_plant[j, t]
                for j in range(M)
            ])
            model += budget_expr <= self.annual_budget, f"Budget_Year_{t+1}"
            
        # Constraint 2: Mutual Exclusivity per tree per year (Cannot prune and replace simultaneously)
        for i in range(N):
            for t in range(T):
                model += x_prune[i, t] + x_repl[i, t] <= 1, f"Excl_Act_{i}_{t}"
                
        # Constraint 3: At most one replacement per tree over the entire horizon
        for i in range(N):
            model += pulp.lpSum([x_repl[i, t] for t in range(T)]) <= 1, f"Max_Repl_{i}"
            
        # Constraint 4: At most one planting per site over the horizon
        for j in range(M):
            model += pulp.lpSum([y_plant[j, t] for t in range(T)]) <= 1, f"Max_Plant_{j}"
            
        # Constraint 5: Crew inspection capacity per year
        for t in range(T):
            model += pulp.lpSum([x_insp[i, t] for i in range(N)]) <= self.max_inspections, f"Crew_Cap_{t}"
            
        # Constraint 6: Safety policy - An intervention (prune/replace) requires inspection
        # either in the current year or in previous years
        for i in range(N):
            for t in range(T):
                recent_inspections = pulp.lpSum([x_insp[i, tau] for tau in range(max(0, t - 1), t + 1)])
                model += x_prune[i, t] + x_repl[i, t] <= recent_inspections, f"Insp_Prereq_{i}_{t}"
                
        # --- Solve with HiGHS Solver ---
        print(f"[*] Solving MILP with HiGHS ({model.numVariables()} variables, {model.numConstraints()} constraints)...")
        solver = pulp.HiGHS(msg=False)
        solve_res = model.solve(solver)
        
        status_name = "Optimal" if solve_res.status == 1 else str(solve_res.status_str)
        print(f"[+] Optimization Finished: Status = {status_name}")
        
        # Collect Results
        action_records = []
        total_costs_per_year = [0.0] * T
        
        for t in range(T):
            year_cost = 0.0
            n_insp = 0
            n_prun = 0
            n_repl = 0
            n_plan = 0
            
            for i in range(N):
                if x_insp[i, t].value() > 0.5:
                    n_insp += 1
                    year_cost += cost_inspect
                if x_prune[i, t].value() > 0.5:
                    n_prun += 1
                    year_cost += cost_prune[i]
                if x_repl[i, t].value() > 0.5:
                    n_repl += 1
                    year_cost += cost_replace[i]
                    
            for j in range(M):
                if y_plant[j, t].value() > 0.5:
                    n_plan += 1
                    year_cost += cost_plant_young
                    
            total_costs_per_year[t] = year_cost
            action_records.append({
                "Year": t + 1,
                "Budget_Available_EUR": self.annual_budget,
                "Budget_Spent_EUR": round(year_cost, 2),
                "Inspections": n_insp,
                "Prunings": n_prun,
                "Replacements": n_repl,
                "New_Plantings": n_plan
            })
            
        schedule_df = pd.DataFrame(action_records)
        baseline_neglected_total = float(np.sum(v_neglected_T))
        optimal_terminal_value = baseline_neglected_total + float(solve_res.objective)
        total_spent = sum(total_costs_per_year)
        
        return {
            "status": status_name,
            "optimal_terminal_value_eur": optimal_terminal_value,
            "baseline_neglected_value_eur": baseline_neglected_total,
            "value_gain_eur": optimal_terminal_value - baseline_neglected_total,
            "gain_percentage": ((optimal_terminal_value - baseline_neglected_total) / baseline_neglected_total) * 100,
            "total_budget_spent_eur": total_spent,
            "annual_schedule": schedule_df,
            "model_stats": {
                "n_variables": model.numVariables(),
                "n_constraints": model.numConstraints()
            }
        }

    def solve_greedy_heuristic(self, df_trees, n_planting_sites=30):
        """
        Baseline naive greedy heuristic to compare against the MILP.
        In each year, greedily prioritizes the oldest/biggest trees or random planting
        until annual budget is exhausted.
        """
        N = len(df_trees)
        M = n_planting_sites
        T = self.T
        
        cost_inspect = 60.0
        cost_plant_young = 750.0
        cost_prune = (100.0 + 4.0 * df_trees["hauteurenm"].values).astype(float)
        v0 = df_trees["value_vie_eur"].values.astype(float)
        health = df_trees["health_score"].values.astype(float)
        delta_decay = 0.07
        g_mature = 0.035
        v_neglected_T = v0 * ((1.0 - delta_decay * (1.1 - health)) ** T)
        
        # Sort existing trees by highest immediate value
        sorted_indices = np.argsort(-v0)
        
        total_heuristic_terminal_value = float(np.sum(v_neglected_T))
        spent_per_year = []
        
        pruned_set = set()
        for t in range(T):
            b_left = self.annual_budget
            years_remaining = T - (t + 1)
            
            for i in sorted_indices:
                if i not in pruned_set and b_left >= (cost_prune[i] + cost_inspect):
                    b_left -= (cost_prune[i] + cost_inspect)
                    pruned_set.add(i)
                    val_boost = v0[i] * ((1.0 + g_mature) ** years_remaining) - v_neglected_T[i]
                    total_heuristic_terminal_value += max(0.0, val_boost)
                    
            spent_per_year.append(self.annual_budget - b_left)
            
        return {
            "greedy_terminal_value_eur": total_heuristic_terminal_value,
            "greedy_total_spent_eur": sum(spent_per_year)
        }

if __name__ == "__main__":
    from valuation_engine import TreeValuationEngine
    clean_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "paris_trees_clean.csv")
    df = pd.read_csv(clean_path)
    
    # Evaluate a test district of 250 trees
    engine = TreeValuationEngine()
    district_trees = engine.evaluate_dataset(df.head(250))
    
    opt = UrbanForestOptimizer(planning_horizon=5, annual_budget=40000.0, max_inspections_per_year=100)
    res_milp = opt.solve_multi_period_milp(district_trees, n_planting_sites=40)
    res_greedy = opt.solve_greedy_heuristic(district_trees, n_planting_sites=40)
    
    print("\n" + "="*70)
    print("           MILP OPTIMIZATION RESULTS VS GREEDY BENCHMARK")
    print("="*70)
    print(f"Status                       : {res_milp['status']}")
    print(f"Do-Nothing (Neglect) Value   : {res_milp['baseline_neglected_value_eur']:,.2f} €")
    print(f"Naive Greedy Policy Value    : {res_greedy['greedy_terminal_value_eur']:,.2f} €")
    print(f"Optimal MILP Forest Value    : {res_milp['optimal_terminal_value_eur']:,.2f} €")
    print(f"Net Gain from Optimization   : +{res_milp['value_gain_eur']:,.2f} € (+{res_milp['gain_percentage']:.1f}%)")
    milp_vs_greedy_gain = res_milp['optimal_terminal_value_eur'] - res_greedy['greedy_terminal_value_eur']
    print(f"MILP Superiority over Greedy : +{milp_vs_greedy_gain:,.2f} € (+{(milp_vs_greedy_gain/res_greedy['greedy_terminal_value_eur'])*100:.2f}%)")
    print(f"Total 5-Year Budget Invested : {res_milp['total_budget_spent_eur']:,.2f} €")
    print("\nAnnual Optimization Schedule:")
    print(res_milp["annual_schedule"].to_string(index=False))
