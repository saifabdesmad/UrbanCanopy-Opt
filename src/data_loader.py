"""
UrbanCanopy-Opt: Data Loader Module
Fetches real urban tree data from the official Paris Open Data API.
Author: Candidate / Heudiasyc Internship Preparation
"""

import os
import json
import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "paris_trees_raw.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "paris_trees_clean.csv")

PARIS_OPENDATA_API = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/les-arbres/records"

def fetch_paris_trees(limit=5000, arrondissement=None, save_raw=True):
    """
    Fetch trees from Paris Open Data API.
    Handles pagination (up to 100 per page in API v2.1).
    """
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
    
    records = []
    page_size = 100
    offset = 0
    
    print(f"[*] Downloading {limit} trees from Paris Open Data API...")
    
    while len(records) < limit:
        batch_size = min(page_size, limit - len(records))
        params = {
            "limit": batch_size,
            "offset": offset
        }
        if arrondissement:
            params["refine"] = f'arrondissement:"{arrondissement}"'
            
        try:
            resp = requests.get(PARIS_OPENDATA_API, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("results", [])
            if not batch:
                break
            records.extend(batch)
            offset += len(batch)
            print(f"    Progress: {len(records)} / {limit} records fetched...", end="\r")
        except Exception as e:
            print(f"\n[!] Network or API warning: {e}. Stopping fetch.")
            break
            
    print(f"\n[+] Total fetched: {len(records)} trees.")
    df = pd.DataFrame(records)
    
    # Flatten geo_point_2d if present
    if "geo_point_2d" in df.columns:
        df["latitude"] = df["geo_point_2d"].apply(lambda p: p.get("lat") if isinstance(p, dict) else np.nan)
        df["longitude"] = df["geo_point_2d"].apply(lambda p: p.get("lon") if isinstance(p, dict) else np.nan)
        df = df.drop(columns=["geo_point_2d"])
        
    if save_raw:
        df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8")
        print(f"[+] Raw data saved to {RAW_DATA_PATH}")
        
    return df

def clean_and_enrich_trees(df=None):
    """
    Cleans raw data and enriches it with:
    - Crown diameter estimation from allometric standards (reflecting what LiDAR measures)
    - Stage of development standardization
    - Species group clustering
    """
    if df is None:
        if os.path.exists(RAW_DATA_PATH):
            df = pd.read_csv(RAW_DATA_PATH)
        else:
            df = fetch_paris_trees(limit=3000)
            
    print("[*] Cleaning and standardizing urban tree dataset...")
    
    # Filter realistic physical values
    # Height between 1m and 45m; Circumference between 10cm and 700cm
    df_clean = df.copy()
    
    df_clean = df_clean.dropna(subset=["circonferenceencm", "hauteurenm"])
    df_clean = df_clean[
        (df_clean["hauteurenm"] >= 1.5) & 
        (df_clean["hauteurenm"] <= 40.0) & 
        (df_clean["circonferenceencm"] >= 15.0) & 
        (df_clean["circonferenceencm"] <= 500.0)
    ]
    
    # Standardize stage of development
    stage_mapping = {
        "Jeune (arbre)": "Jeune",
        "Jeune (arbre)Adulte": "Jeune_Adulte",
        "Adulte": "Adulte",
        "Mature": "Mature"
    }
    df_clean["stage"] = df_clean["stadedeveloppement"].map(stage_mapping).fillna("Adulte")
    
    # Fill missing genres
    df_clean["genre"] = df_clean["genre"].fillna("Inconnu").str.strip()
    df_clean["espece"] = df_clean["espece"].fillna("sp.").str.strip()
    
    # Top tree genres in Paris (Platanus, Acer, Tilia, Aesculus, Sophora, Prunus, Fraxinus...)
    top_genres = df_clean["genre"].value_counts().nlargest(8).index.tolist()
    df_clean["genre_group"] = df_clean["genre"].apply(lambda g: g if g in top_genres else "Autre")
    
    # Simulated LiDAR Crown Diameter (in meters):
    # In reality LiDAR HD provides the crown polygon diameter D_crown.
    # Allometric relationship from urban forestry literature (e.g., Pretzsch et al. 2015):
    # D_crown ≈ alpha * (Circumference / pi)^beta + noise
    # We generate the crown diameter with realistic allometric variance to emulate the LiDAR extraction.
    np.random.seed(42)
    dbh_m = (df_clean["circonferenceencm"] / 100.0) / np.pi # DBH in meters
    # Realistic allometric crown ratio: crown diameter typically 15 to 30 times DBH with species variation
    species_factor = df_clean["genre_group"].map({
        "Platanus": 22.0, "Aesculus": 20.0, "Acer": 18.0, "Tilia": 19.0,
        "Sophora": 17.0, "Prunus": 15.0, "Fraxinus": 18.0, "Pinus": 14.0, "Autre": 17.5
    }).fillna(17.5)
    
    noise = np.random.normal(1.0, 0.12, size=len(df_clean))
    df_clean["crown_diameter_m"] = np.clip(dbh_m * species_factor * noise + 0.2 * df_clean["hauteurenm"], 1.5, 25.0)
    df_clean["crown_area_m2"] = np.pi * (df_clean["crown_diameter_m"] / 2.0) ** 2
    
    # Assign health score (0.0 to 1.0) with realistic distribution (mostly healthy trees, some degraded)
    health_scores = np.random.beta(a=8, b=2, size=len(df_clean))
    df_clean["health_score"] = np.round(np.clip(health_scores, 0.2, 1.0), 2)
    
    # Save processed dataset
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8")
    print(f"[+] Clean dataset created: {len(df_clean)} trees retained.")
    print(f"[+] Saved to {PROCESSED_DATA_PATH}")
    
    return df_clean

if __name__ == "__main__":
    df = fetch_paris_trees(limit=3000)
    clean_df = clean_and_enrich_trees(df)
    print("\nSample processed data:")
    print(clean_df[["genre", "hauteurenm", "circonferenceencm", "crown_diameter_m", "stage", "health_score"]].head())
