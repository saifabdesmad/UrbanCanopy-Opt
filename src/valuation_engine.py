"""
UrbanCanopy-Opt: Environmental & Economic Valuation Engine
Implements the French official "Barème de l'arbre" (VIE 2020) and the UK CAVAT method.
Carries prediction errors through Monte Carlo simulations to assess zone-level valuation uncertainty.
Author: Candidate / Heudiasyc Internship Preparation
"""

import os
import numpy as np
import pandas as pd

class TreeValuationEngine:
    def __init__(self):
        # Species rating (Coefficient d'essence ce)
        self.genre_coefficients = {
            "Platanus": 1.3,
            "Quercus": 1.4,
            "Tilia": 1.25,
            "Aesculus": 1.15,
            "Acer": 1.1,
            "Sophora": 1.1,
            "Fraxinus": 1.05,
            "Carpinus": 1.1,
            "Betula": 0.9,
            "Populus": 0.8,
            "Prunus": 0.95,
            "Pinus": 1.15,
            "Pyrus": 0.95,
            "Autre": 1.0
        }
        
        # Development stage coefficient (cd)
        self.stage_coefficients = {
            "Jeune": 0.8,
            "Jeune_Adulte": 1.0,
            "Adulte": 1.3,
            "Mature": 1.6
        }

    def compute_vie(self, circumference_cm, genre, stage, health_score=0.9, remarkable=False, crown_area_m2=25.0):
        """
        Computes the monetary value according to the French "Barème de l'arbre" (VIE 2020).
        
        Formula:
            VIE = V_base * ce * cd * cs * cr + V_eco
        where:
            - V_base: Base nursery & planting replacement value based on basal area
            - ce: Species coefficient
            - cd: Development stage coefficient
            - cs: Health/phytosanitary coefficient
            - cr: Situation / Remarquable coefficient
            - V_eco: Annualized ecosystem services (carbon, stormwater, cooling)
        """
        # Trunk basal cross-section S in cm2 = C^2 / (4 * pi)
        basal_area_cm2 = (circumference_cm ** 2) / (4.0 * np.pi)
        
        # Base nursery replacement value (fixed tree installation + basal area scaling)
        base_value = 220.0 + 2.2 * basal_area_cm2
        
        # Coefficients
        ce = self.genre_coefficients.get(genre, 1.0)
        cd = self.stage_coefficients.get(stage, 1.1)
        cs = np.clip(health_score, 0.1, 1.0)
        cr = 1.4 if (remarkable == "OUI" or remarkable is True) else 1.05
        
        # Ecosystem service valuation (approx. 4.5 € / m2 of canopy crown per year over amenity life)
        v_eco = 4.5 * np.maximum(crown_area_m2, 1.0)
        
        total_vie = (base_value * ce * cd * cs * cr) + v_eco
        return np.maximum(total_vie, 150.0)

    def compute_cavat(self, circumference_cm, health_score=0.9, cti=1.25, accessibility=0.95):
        """
        Computes the Capital Asset Value for Amenity Trees (CAVAT - UK method).
        
        Formula:
            CAVAT = BasalArea (cm2) * UVF * CTI * Accessibility * HealthFactor
        where:
            - UVF: Unit Value Factor (~18.50 €/cm2)
            - CTI: Community Tree Index (Paris urban density factor = 1.25)
        """
        basal_area_cm2 = (circumference_cm ** 2) / (4.0 * np.pi)
        uvf = 18.50  # Euros per cm2
        cs = np.clip(health_score, 0.1, 1.0)
        
        cavat_value = basal_area_cm2 * uvf * cti * accessibility * cs
        return np.maximum(cavat_value, 100.0)

    def evaluate_dataset(self, df, circumference_col="circonferenceencm"):
        """
        Applies both VIE and CAVAT valuation to each tree in a DataFrame.
        """
        df_val = df.copy()
        
        vie_values = []
        cavat_values = []
        
        for _, row in df_val.iterrows():
            circ = row[circumference_col]
            genre = row.get("genre_group", "Autre")
            stage = row.get("stage", "Adulte")
            health = row.get("health_score", 0.9)
            rem = row.get("remarquable", False)
            crown_a = row.get("crown_area_m2", 20.0)
            
            v = self.compute_vie(circ, genre, stage, health, rem, crown_a)
            c = self.compute_cavat(circ, health)
            
            vie_values.append(v)
            cavat_values.append(c)
            
        df_val["value_vie_eur"] = np.round(vie_values, 2)
        df_val["value_cavat_eur"] = np.round(cavat_values, 2)
        return df_val

    def evaluate_zone_with_monte_carlo(self, df_zone, mc_samples_circumference):
        """
        Propagates prediction uncertainty to the entire zone valuation.
        
        Args:
            df_zone: DataFrame with N trees in a zone
            mc_samples_circumference: np.ndarray of shape (N, n_simulations)
            
        Returns:
            dict containing:
                - true_zone_value: total ground truth VIE in euros
                - expected_zone_value: mean predicted VIE
                - mc_distribution: 1D array of total zone values across simulations
                - ci_95: [lower_bound, upper_bound] 95% confidence interval
        """
        n_trees, n_sims = mc_samples_circumference.shape
        simulated_zone_totals = np.zeros(n_sims)
        
        for s in range(n_sims):
            sim_circs = mc_samples_circumference[:, s]
            sim_vies = [
                self.compute_vie(
                    circumference_cm=sim_circs[i],
                    genre=df_zone.iloc[i].get("genre_group", "Autre"),
                    stage=df_zone.iloc[i].get("stage", "Adulte"),
                    health_score=df_zone.iloc[i].get("health_score", 0.9),
                    remarkable=df_zone.iloc[i].get("remarquable", False),
                    crown_area_m2=df_zone.iloc[i].get("crown_area_m2", 20.0)
                )
                for i in range(n_trees)
            ]
            simulated_zone_totals[s] = np.sum(sim_vies)
            
        # Ground truth valuation
        true_df = self.evaluate_dataset(df_zone, circumference_col="circonferenceencm")
        true_total = float(true_df["value_vie_eur"].sum())
        
        expected_total = float(np.mean(simulated_zone_totals))
        q025 = float(np.percentile(simulated_zone_totals, 2.5))
        q975 = float(np.percentile(simulated_zone_totals, 97.5))
        
        return {
            "true_total_eur": true_total,
            "expected_total_eur": expected_total,
            "ci_95": [q025, q975],
            "mc_distribution": simulated_zone_totals,
            "estimation_error_pct": ((expected_total - true_total) / true_total) * 100
        }

if __name__ == "__main__":
    from allometry_ai import TreeAllometryAI
    
    clean_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "paris_trees_clean.csv")
    df = pd.read_csv(clean_path)
    
    engine = TreeValuationEngine()
    df_valued = engine.evaluate_dataset(df.head(10))
    print("\n[+] Sample Tree Valuations (VIE 2020 vs CAVAT):")
    print(df_valued[["genre", "hauteurenm", "circonferenceencm", "stage", "value_vie_eur", "value_cavat_eur"]])
    
    # Run Monte Carlo on a test zone of 100 trees
    zone_test = df.iloc[:100].copy()
    ai = TreeAllometryAI()
    _, mc_draws = ai.predict_with_uncertainty(zone_test, n_simulations=300)
    
    mc_results = engine.evaluate_zone_with_monte_carlo(zone_test, mc_draws)
    print("\n[+] Monte Carlo Zone Valuation Results (100 Trees in Paris):")
    print(f"    Ground Truth Real Value  : {mc_results['true_total_eur']:,.2f} €")
    print(f"    AI Predicted Zone Value  : {mc_results['expected_total_eur']:,.2f} €")
    print(f"    95% Confidence Interval  : [{mc_results['ci_95'][0]:,.2f} €, {mc_results['ci_95'][1]:,.2f} €]")
    print(f"    Estimation Error         : {mc_results['estimation_error_pct']:.2f}%")
    is_inside = mc_results['ci_95'][0] <= mc_results['true_total_eur'] <= mc_results['ci_95'][1]
    print(f"    Ground Truth inside 95% CI: {'OUI (Validé)' if is_inside else 'NON'}")
