"""
Unit Tests for UrbanCanopy-Opt Pipeline
Verifies core components: Allometry AI, Valuation Engine, and MILP Optimizer.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from valuation_engine import TreeValuationEngine
from milp_optimizer import UrbanForestOptimizer

class TestUrbanCanopyOpt(unittest.TestCase):
    def setUp(self):
        self.engine = TreeValuationEngine()
        # Mock dataframe of 10 trees
        self.mock_df = pd.DataFrame({
            "circonferenceencm": [40, 80, 120, 160, 200, 50, 90, 110, 150, 70],
            "hauteurenm": [5, 10, 12, 15, 18, 6, 9, 11, 14, 8],
            "genre_group": ["Platanus", "Acer", "Tilia", "Aesculus", "Quercus", "Betula", "Platanus", "Acer", "Tilia", "Autre"],
            "stage": ["Jeune", "Jeune_Adulte", "Adulte", "Adulte", "Mature", "Jeune", "Adulte", "Adulte", "Mature", "Jeune_Adulte"],
            "health_score": [0.95, 0.9, 0.85, 0.8, 0.75, 0.9, 0.88, 0.82, 0.78, 0.92],
            "remarquable": ["NON"] * 10,
            "crown_diameter_m": [4.0, 6.0, 8.0, 10.0, 12.0, 4.5, 6.5, 7.5, 9.5, 5.5],
            "crown_area_m2": [12.5, 28.2, 50.2, 78.5, 113.1, 15.9, 33.1, 44.1, 70.8, 23.7]
        })
        self.mock_df = self.engine.evaluate_dataset(self.mock_df)

    def test_vie_valuation_positive(self):
        """Test that VIE valuation produces strictly positive, realistic values."""
        values = self.mock_df["value_vie_eur"].values
        self.assertTrue(np.all(values > 150.0))
        # Bigger trees should generally have higher value
        self.assertGreater(values[4], values[0])

    def test_cavat_valuation_positive(self):
        """Test CAVAT calculation."""
        values = self.mock_df["value_cavat_eur"].values
        self.assertTrue(np.all(values > 100.0))

    def test_milp_optimizer_feasibility(self):
        """Test that the MILP model solves to optimality and respects budget constraints."""
        opt = UrbanForestOptimizer(planning_horizon=3, annual_budget=5000.0, max_inspections_per_year=10)
        res = opt.solve_multi_period_milp(self.mock_df, n_planting_sites=5)
        
        self.assertEqual(res["status"], "Optimal")
        self.assertGreater(res["optimal_terminal_value_eur"], res["baseline_neglected_value_eur"])
        # Check budget constraint satisfaction for every year
        for _, row in res["annual_schedule"].iterrows():
            self.assertLessEqual(row["Budget_Spent_EUR"], row["Budget_Available_EUR"] + 1e-4)

if __name__ == "__main__":
    unittest.main()
