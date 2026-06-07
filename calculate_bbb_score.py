#!/usr/bin/env python3
"""
BBB Score Calculator for PHGDH Inhibitors
=========================================
Calculates the Blood-Brain Barrier (BBB) Score based on Gupta et al. 2019
DOI: 10.1021/acs.jmedchem.9b01220

This script:
1. Reads toxicity prediction results from ProTox-3
2. Calculates additional molecular descriptors using RDKit
3. Implements the BBB Score algorithm
4. Filters compounds by optimal BBB-penetration criteria
5. Generates summary statistics and writes per-criterion CSVs

Inputs (in the script's working directory):
    results.csv    Toxicity / physicochemistry table; must contain at minimum
                   the columns:
                     - smiles
                     - molecular_weight
                     - number_of_hydrogen_bond_acceptors
                     - number_of_hydrogen_bond_donors
                     - topological_polar_surface_area_tpsa
                     - octanol_water_partition_coefficient_logp
                     - number_of_rotatable_bonds

Outputs (written under OUTPUT_DIR, default ./outputs):
    results_with_bbb_score.csv   Full table + BBB sub-scores + criteria flags
    top_50_bbb_score.csv         Top 50 by BBB Score
    strict_bbb_criteria.csv      Compounds passing strict (TPSA <= 70) criteria
    relaxed_bbb_criteria.csv     Compounds passing relaxed (TPSA <= 90) criteria
    elite_bbb_candidates.csv     BBB Score > 4 AND relaxed criteria
    bbb_analysis_summary.csv     Per-category counts

Configure the output directory by editing OUTPUT_DIR below, or by setting
the BBB_OUTPUT_DIR environment variable.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import sys
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = "results.csv"
OUTPUT_DIR = Path(os.environ.get("BBB_OUTPUT_DIR", "./outputs"))


def calculate_rdkit_descriptors(smiles):
    """
    Calculate additional molecular descriptors needed for BBB Score.

    Parameters:
    -----------
    smiles : str
        SMILES string of the molecule

    Returns:
    --------
    dict : Dictionary containing calculated descriptors
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            'aromatic_ring_count': None,
            'heavy_atom_count': None,
            'fraction_aromatic': None
        }

    # Calculate descriptors
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())

    # Fraction of aromatic atoms (avoid division by zero)
    fraction_aromatic = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0

    return {
        'aromatic_ring_count': aromatic_rings,
        'heavy_atom_count': heavy_atoms,
        'fraction_aromatic': fraction_aromatic,
        'aromatic_atoms': aromatic_atoms
    }


def estimate_ionization_state(smiles, mol=None):
    """
    Estimate ionization state based on functional groups.

    Since we don't have a pKa calculator, we use SMARTS patterns to identify
    likely acidic, basic, or neutral/zwitterionic compounds.

    Parameters:
    -----------
    smiles : str
        SMILES string
    mol : RDKit Mol object (optional)
        Pre-computed molecule object

    Returns:
    --------
    str : 'Acid', 'Base', 'Neutral', or 'Zwitterion'
    float : Estimated pKa (used for BBB Score calculation)
    """
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return 'Neutral', 8.81  # Default

    # SMARTS patterns for common ionizable groups
    # Acidic groups (carboxylic acids, sulfonic acids, phosphoric acids)
    acid_patterns = [
        'C(=O)[OH]',           # Carboxylic acid
        'S(=O)(=O)[OH]',       # Sulfonic acid
        'P(=O)([OH])[OH]',     # Phosphoric acid
    ]

    # Basic groups (amines, guanidines, amidines)
    base_patterns = [
        '[NX3;H2,H1;!$(NC=O)]',  # Primary/secondary amine (not amide)
        '[NX3;H0;!$(NC=O)]',     # Tertiary amine (not amide)
        'c1ncnc1',               # Imidazole-like
        '[$(NC(N)=N)]',          # Guanidine
    ]

    # Count matches
    n_acidic = sum(
        len(mol.GetSubstructMatches(Chem.MolFromSmarts(p)))
        for p in acid_patterns if Chem.MolFromSmarts(p) is not None
    )
    n_basic = sum(
        len(mol.GetSubstructMatches(Chem.MolFromSmarts(p)))
        for p in base_patterns if Chem.MolFromSmarts(p) is not None
    )

    # Determine ionization state
    if n_acidic > 0 and n_basic > 0:
        return 'Zwitterion', 8.81  # Default for zwitterions
    elif n_acidic > 0:
        return 'Acid', 4.5         # Typical carboxylic acid pKa
    elif n_basic > 0:
        return 'Base', 9.5         # Typical amine pKa
    else:
        return 'Neutral', 8.81     # Default from original paper


def calculate_bbb_score(row, rdkit_data):
    """
    Calculate BBB Score based on the algorithm from Gupta et al. 2019.

    The BBB Score ranges from 0-6, with higher scores indicating better BBB
    penetration. Scores > 4 generally indicate good BBB penetration.

    Parameters:
    -----------
    row : pandas Series
        Row from the dataframe with existing descriptors
    rdkit_data : dict
        Dictionary with RDKit-calculated descriptors

    Returns:
    --------
    dict : BBB Score and component scores
    """
    # Extract values
    ar_rings = rdkit_data['aromatic_ring_count']
    hac = rdkit_data['heavy_atom_count']
    mw = row['molecular_weight']
    hba = row['number_of_hydrogen_bond_acceptors']
    hbd = row['number_of_hydrogen_bond_donors']
    tpsa = row['topological_polar_surface_area_tpsa']
    pka = rdkit_data['estimated_pka']

    # Check for None values
    if any(v is None for v in [ar_rings, hac, mw, hba, hbd, tpsa, pka]):
        return {
            'bbb_score': None,
            'ar_score': None,
            'hac_score': None,
            'mwhbn_score': None,
            'tpsa_score': None,
            'pka_score': None
        }

    # 1. Aromatic Ring Score
    if ar_rings == 0:
        ar_score = 0.336376
    elif ar_rings == 1:
        ar_score = 0.816016
    elif ar_rings == 2:
        ar_score = 1.0
    elif ar_rings == 3:
        ar_score = 0.691115
    elif ar_rings == 4:
        ar_score = 0.199399
    else:  # > 4
        ar_score = 0.0

    # 2. Heavy Atom Count (HAC) Score
    if hac < 6 or hac > 45:
        hac_score = 0.0
    else:
        hac_score = (1 / 0.624231) * (
            0.0000443 * hac ** 3 - 0.004556 * hac ** 2 + 0.12775 * hac - 0.463
        )

    # 3. TPSA Score
    if tpsa == 0 or tpsa > 120:
        tpsa_score = 0.0
    else:
        tpsa_score = (1 / 0.9598) * (-0.0067 * tpsa + 0.9598)

    # 4. MWHBN Score (combines MW, HBA, HBD)
    mwhbn = (mw ** -0.5) * (hba + hbd)
    if mwhbn <= 0.05 or mwhbn > 0.45:
        mwhbn_score = 0.0
    else:
        mwhbn_score = (1 / 0.72258) * (
            26.733 * mwhbn ** 3 - 31.495 * mwhbn ** 2 + 9.5202 * mwhbn - 0.1358
        )

    # 5. pKa Score
    if pka <= 3 or pka > 11:
        pka_score = 0.0
    else:
        pka_score = (1 / 0.597488) * (
            0.00045068 * pka ** 4 - 0.016331 * pka ** 3
            + 0.18618 * pka ** 2 - 0.71043 * pka + 0.8579
        )

    # Calculate total BBB Score
    bbb_score = (
        ar_score + hac_score
        + (1.5 * mwhbn_score)
        + (2.0 * tpsa_score)
        + (0.5 * pka_score)
    )

    return {
        'bbb_score': bbb_score,
        'ar_score': ar_score,
        'hac_score': hac_score,
        'mwhbn_score': mwhbn_score,
        'tpsa_score': tpsa_score,
        'pka_score': pka_score,
        'mwhbn_value': mwhbn
    }


def check_bbb_criteria(row):
    """
    Check if compound meets optimal BBB-penetration criteria.

    Criteria thresholds:
    - MW <= 450 Da
    - logP: 1.5 - 3.5
    - TPSA <= 70 A^2 (strict CNS criteria) or <= 90 A^2 (general BBB)
    - HBD <= 3
    - HBA <= 7
    - Rotatable bonds <= 7

    Returns:
    --------
    dict : Pass/fail for each criterion and overall assessment
    """
    criteria = {
        'mw_pass': row['molecular_weight'] <= 450,
        'logp_pass': 1.5 <= row['octanol_water_partition_coefficient_logp'] <= 3.5,
        'tpsa_strict_pass': row['topological_polar_surface_area_tpsa'] <= 70,
        'tpsa_relaxed_pass': row['topological_polar_surface_area_tpsa'] <= 90,
        'hbd_pass': row['number_of_hydrogen_bond_donors'] <= 3,
        'hba_pass': row['number_of_hydrogen_bond_acceptors'] <= 7,
        'rotatable_bonds_pass': row['number_of_rotatable_bonds'] <= 7
    }

    # Overall assessment (using strict TPSA)
    criteria['bbb_criteria_strict'] = all([
        criteria['mw_pass'],
        criteria['logp_pass'],
        criteria['tpsa_strict_pass'],
        criteria['hbd_pass'],
        criteria['hba_pass'],
        criteria['rotatable_bonds_pass']
    ])

    # Overall assessment (using relaxed TPSA)
    criteria['bbb_criteria_relaxed'] = all([
        criteria['mw_pass'],
        criteria['logp_pass'],
        criteria['tpsa_relaxed_pass'],
        criteria['hbd_pass'],
        criteria['hba_pass'],
        criteria['rotatable_bonds_pass']
    ])

    # Count how many criteria passed
    criteria['criteria_passed'] = sum([
        criteria['mw_pass'],
        criteria['logp_pass'],
        criteria['tpsa_strict_pass'],
        criteria['hbd_pass'],
        criteria['hba_pass'],
        criteria['rotatable_bonds_pass']
    ])

    return criteria


def main():
    """Main function to process compounds and calculate BBB Scores."""

    print("=" * 80)
    print("BBB Score Calculator for PHGDH Inhibitors")
    print("=" * 80)
    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the CSV file
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    print(f"Reading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} compounds")
    print()

    # Calculate RDKit descriptors
    print("Calculating molecular descriptors with RDKit...")
    rdkit_descriptors = []

    for idx, row in df.iterrows():
        smiles = row['smiles']

        # Calculate descriptors
        desc = calculate_rdkit_descriptors(smiles)
        ionization, pka = estimate_ionization_state(smiles)

        desc['ionization_state'] = ionization
        desc['estimated_pka'] = pka

        rdkit_descriptors.append(desc)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} compounds...")

    print(f"  Completed all {len(df)} compounds")
    print()

    # Add RDKit descriptors to dataframe
    for key in rdkit_descriptors[0].keys():
        df[key] = [d[key] for d in rdkit_descriptors]

    # Calculate BBB Scores
    print("Calculating BBB Scores...")
    bbb_results = []

    for idx, row in df.iterrows():
        rdkit_data = rdkit_descriptors[idx]
        bbb_data = calculate_bbb_score(row, rdkit_data)
        bbb_results.append(bbb_data)

    # Add BBB Score data to dataframe
    for key in bbb_results[0].keys():
        df[key] = [b[key] for b in bbb_results]

    print("  BBB Scores calculated")
    print()

    # Check BBB penetration criteria
    print("Checking BBB penetration criteria...")
    criteria_results = []

    for idx, row in df.iterrows():
        criteria = check_bbb_criteria(row)
        criteria_results.append(criteria)

    # Add criteria to dataframe
    for key in criteria_results[0].keys():
        df[key] = [c[key] for c in criteria_results]

    print("  Criteria assessment completed")
    print()

    # Generate summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    # BBB Score statistics
    valid_scores = df['bbb_score'].dropna()
    print(f"BBB Score Statistics (n={len(valid_scores)}):")
    print(f"  Mean:   {valid_scores.mean():.3f}")
    print(f"  Median: {valid_scores.median():.3f}")
    print(f"  Std:    {valid_scores.std():.3f}")
    print(f"  Min:    {valid_scores.min():.3f}")
    print(f"  Max:    {valid_scores.max():.3f}")
    print()

    # BBB Score categories (from literature: >4 = good BBB penetration)
    high_bbb = (df['bbb_score'] > 4).sum()
    medium_bbb = ((df['bbb_score'] > 2) & (df['bbb_score'] <= 4)).sum()
    low_bbb = (df['bbb_score'] <= 2).sum()

    print(f"BBB Score Categories:")
    print(f"  High (>4):     {high_bbb:4d} ({100 * high_bbb / len(df):.1f}%)")
    print(f"  Medium (2-4):  {medium_bbb:4d} ({100 * medium_bbb / len(df):.1f}%)")
    print(f"  Low (<=2):     {low_bbb:4d} ({100 * low_bbb / len(df):.1f}%)")
    print()

    # Criteria-based assessment
    strict_pass = df['bbb_criteria_strict'].sum()
    relaxed_pass = df['bbb_criteria_relaxed'].sum()

    print(f"BBB Penetration Criteria:")
    print(f"  Strict (TPSA <= 70):  {strict_pass:4d} ({100 * strict_pass / len(df):.1f}%)")
    print(f"  Relaxed (TPSA <= 90): {relaxed_pass:4d} ({100 * relaxed_pass / len(df):.1f}%)")
    print()

    # Individual criteria statistics
    print(f"Individual Criteria Pass Rates:")
    print(f"  MW <= 450:           {df['mw_pass'].sum():4d} ({100 * df['mw_pass'].sum() / len(df):.1f}%)")
    print(f"  logP 1.5-3.5:        {df['logp_pass'].sum():4d} ({100 * df['logp_pass'].sum() / len(df):.1f}%)")
    print(f"  TPSA <= 70:          {df['tpsa_strict_pass'].sum():4d} ({100 * df['tpsa_strict_pass'].sum() / len(df):.1f}%)")
    print(f"  TPSA <= 90:          {df['tpsa_relaxed_pass'].sum():4d} ({100 * df['tpsa_relaxed_pass'].sum() / len(df):.1f}%)")
    print(f"  HBD <= 3:            {df['hbd_pass'].sum():4d} ({100 * df['hbd_pass'].sum() / len(df):.1f}%)")
    print(f"  HBA <= 7:            {df['hba_pass'].sum():4d} ({100 * df['hba_pass'].sum() / len(df):.1f}%)")
    print(f"  Rotatable bonds <= 7:{df['rotatable_bonds_pass'].sum():4d} ({100 * df['rotatable_bonds_pass'].sum() / len(df):.1f}%)")
    print()

    # Ionization state distribution
    print(f"Ionization State Distribution:")
    ionization_counts = df['ionization_state'].value_counts()
    for state, count in ionization_counts.items():
        print(f"  {state:12s}: {count:4d} ({100 * count / len(df):.1f}%)")
    print()

    # Save full results
    output_file = OUTPUT_DIR / "results_with_bbb_score.csv"
    df.to_csv(output_file, index=False)
    print(f"Full results saved to: {output_file}")
    print()

    # Create filtered datasets
    # 1. Top BBB Score candidates
    top_bbb = df.nlargest(50, 'bbb_score')
    top_bbb_file = OUTPUT_DIR / "top_50_bbb_score.csv"
    top_bbb.to_csv(top_bbb_file, index=False)
    print(f"Top 50 by BBB Score saved to: {top_bbb_file}")

    # 2. Compounds meeting strict BBB criteria
    strict_criteria = df[df['bbb_criteria_strict'] == True].copy()
    strict_criteria = strict_criteria.sort_values('bbb_score', ascending=False)
    strict_file = OUTPUT_DIR / "strict_bbb_criteria.csv"
    strict_criteria.to_csv(strict_file, index=False)
    print(f"Compounds meeting strict BBB criteria saved to: {strict_file}")
    print(f"  (n={len(strict_criteria)} compounds)")

    # 3. Compounds meeting relaxed BBB criteria
    relaxed_criteria = df[df['bbb_criteria_relaxed'] == True].copy()
    relaxed_criteria = relaxed_criteria.sort_values('bbb_score', ascending=False)
    relaxed_file = OUTPUT_DIR / "relaxed_bbb_criteria.csv"
    relaxed_criteria.to_csv(relaxed_file, index=False)
    print(f"Compounds meeting relaxed BBB criteria saved to: {relaxed_file}")
    print(f"  (n={len(relaxed_criteria)} compounds)")

    # 4. Combined: High BBB Score AND meets criteria
    elite_candidates = df[(df['bbb_score'] > 4) & (df['bbb_criteria_relaxed'] == True)].copy()
    elite_candidates = elite_candidates.sort_values('bbb_score', ascending=False)
    elite_file = OUTPUT_DIR / "elite_bbb_candidates.csv"
    elite_candidates.to_csv(elite_file, index=False)
    print(f"Elite BBB candidates (Score >4 AND criteria) saved to: {elite_file}")
    print(f"  (n={len(elite_candidates)} compounds)")
    print()

    # Create summary table
    summary_data = {
        'Metric': [
            'Total Compounds',
            'High BBB Score (>4)',
            'Medium BBB Score (2-4)',
            'Low BBB Score (<=2)',
            'Strict Criteria Pass',
            'Relaxed Criteria Pass',
            'Elite Candidates'
        ],
        'Count': [
            len(df),
            high_bbb,
            medium_bbb,
            low_bbb,
            strict_pass,
            relaxed_pass,
            len(elite_candidates)
        ],
        'Percentage': [
            100.0,
            100 * high_bbb / len(df),
            100 * medium_bbb / len(df),
            100 * low_bbb / len(df),
            100 * strict_pass / len(df),
            100 * relaxed_pass / len(df),
            100 * len(elite_candidates) / len(df)
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_file = OUTPUT_DIR / "bbb_analysis_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Analysis summary saved to: {summary_file}")
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Key findings:")
    if len(elite_candidates) > 0:
        print(f"  - {len(elite_candidates)} elite BBB-penetrant candidates identified")
        print(f"  - Top BBB Score: {df['bbb_score'].max():.3f}")
        print(f"  - Best candidate: {elite_candidates.iloc[0]['Compound']}")
    else:
        print(f"  - No compounds meet both high BBB Score and criteria thresholds")
        print(f"  - Consider examining top BBB Score compounds or relaxed criteria matches")
    print()


if __name__ == "__main__":
    main()
