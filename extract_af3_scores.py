#!/usr/bin/env python3
"""
AF3 Score Extractor for PHGDH Ligand Screening
================================================
Place this script in the same directory as all phgdh_ligand_XXXX/ folders.

Expected directory layout:
    ./
    ├── extract_af3_scores.py             <- this script
    ├── phgdh_ligand_0000/
    │   ├── phgdh_ligand_0000_summary_confidences.json
    │   ├── phgdh_ligand_0000_data.json
    │   ├── phgdh_ligand_0000_confidences.json
    │   ├── ranking_scores.csv
    │   └── seed-1_sample-{0..4}/
    ├── phgdh_ligand_0001/
    │   └── ...
    ├── ...
    ├── results_with_bbb_score.csv        <- BBB/tox data (from calculate_bbb_score.py)
    ├── seedc1_tamgen_logprobs.tsv        <- TamGen log-prob table (smiles \t log_prob)
    └── seedc1_positive_control_row.csv   <- Single-row positive-control entry (optional)

Outputs (written to same directory):
    af3_scores.csv                raw AF3 metrics for every compound
    full_composite_scores.csv     all pillars merged and ranked
    top_candidates.csv            top 50 by composite score

Composite score (gated, tunable at top of this file):
    The composite score is calibrated on gate-passing compounds only
    (those that satisfy all Tier 1 gates: has_clash == 0,
    bbb_criteria_relaxed == True, iptm >= IPTM_MIN, predicted_toxicity >= 3).
    Compounds outside this set are scored against the same normalization
    reference but should not be interpreted as competitive candidates.

    Weights:
        55%  iptm       AlphaFold3 binding interface confidence
        25%  BBB score  Blood-brain barrier penetration (Gupta 2019)
        20%  toxicity   ProTox-3 class (class 6 = safest, class 1 = fatal)
                        Mapped as (class - 1) / 5, fixed (not min-max).

    This replaces an earlier library-wide 40/25/20/15 composite with a
    clash-penalty term. Rationale:
      - TamGen log_prob had near-zero correlation with the rest of the
        composite (r = -0.035) and the SEEDC1 positive control has no
        log_prob value at all, so its weight is redistributed to iptm.
      - Clash is now handled as a Tier 1 gate rather than a soft penalty,
        matching the filter-first-then-rank framework.
      - Library-wide normalization rewarded gate-failing compounds whose
        raw profiles were strong; gate-passer normalization aligns the
        composite with the tier ranking.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit weights here if needed
# ─────────────────────────────────────────────────────────────────────────────

# Composite score weights (sum to 1.0; log_prob dropped — see header)
WEIGHTS = {
    "iptm": 0.55,
    "bbb":  0.25,
    "tox":  0.20,
}

# Tier 1 gate thresholds — also duplicated in prioritize_compounds.py.
# Kept in sync because the gated composite needs to know which rows are
# gate-passers for its min-max normalization reference.
IPTM_MIN      = 0.70   # AF3 binding confidence minimum
TOX_MIN_HARD  = 3      # ProTox-3 class: 1-2 fails, 3+ passes

# Input filenames (relative to this script)
BBB_CSV        = "results_with_bbb_score.csv"
TSV_FILE       = "seedc1_tamgen_logprobs.tsv"
POS_CTRL_CSV   = "seedc1_positive_control_row.csv"   # optional positive control row

FOLDER_PREFIX    = "phgdh_ligand_"
READ_CONFIDENCES = False   # Set True to extract ligand pLDDT (slow — reads large files)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_folder(folder: Path) -> dict:
    """
    Read one phgdh_ligand_XXXX/ folder and return a dict of AF3 metrics.
    """
    cid    = folder.name   # e.g. phgdh_ligand_0042
    # Derive 0-based Compound label: phgdh_ligand_0042 -> Compound_42
    folder_num = int(cid.split("_")[-1])
    compound_label = f"Compound_{folder_num}"
    result = {"compound_id": cid, "compound_label": compound_label, "error": None}

    # ── summary_confidences.json (best model) ────────────────────────────────
    sc_path = folder / f"{cid}_summary_confidences.json"
    if sc_path.exists():
        with open(sc_path) as f:
            sc = json.load(f)

        result["iptm"]                = sc.get("iptm")
        result["ptm"]                 = sc.get("ptm")
        result["ranking_score"]       = sc.get("ranking_score")
        result["has_clash"]           = sc.get("has_clash", 0.0)
        result["fraction_disordered"] = sc.get("fraction_disordered", 0.0)

        # protein->ligand interface score
        cpi = sc.get("chain_pair_iptm", [])
        result["chain_pair_iptm_01"] = (
            cpi[0][1] if len(cpi) > 0 and len(cpi[0]) > 1 else None
        )

        # min PAE between protein and ligand chains (Angstroms, lower = better)
        cpm = sc.get("chain_pair_pae_min", [])
        result["chain_pair_pae_min_01"] = (
            cpm[0][1] if len(cpm) > 0 and len(cpm[0]) > 1 else None
        )
    else:
        result["error"] = f"missing {cid}_summary_confidences.json"

    # ── ranking_scores.csv (statistics across 5 seeds) ───────────────────────
    rs_path = folder / "ranking_scores.csv"
    if rs_path.exists():
        rdf    = pd.read_csv(rs_path)
        scores = rdf["ranking_score"].values
        result["best_sample_ranking_score"] = float(scores.max())
        result["ranking_score_mean"]        = float(scores.mean())
        result["ranking_score_std"]         = float(scores.std())
        result["best_sample_idx"]           = int(scores.argmax())
    else:
        result["error"] = (result.get("error") or "") + " | missing ranking_scores.csv"

    # ── data.json -> SMILES ──────────────────────────────────────────────────
    dj_path = folder / f"{cid}_data.json"
    if dj_path.exists():
        with open(dj_path) as f:
            dj = json.load(f)
        smiles = None
        for seq in dj.get("sequences", []):
            if "ligand" in seq:
                smiles = seq["ligand"].get("smiles")
                break
        result["smiles"] = smiles
    else:
        result["error"] = (result.get("error") or "") + " | missing data.json"

    # ── confidences.json -> mean ligand pLDDT (optional, large file) ─────────
    # Skipped by default because this file can be 50-100MB per compound.
    # Set READ_CONFIDENCES = True at the top of this file to enable.
    if READ_CONFIDENCES:
        cf_path = folder / f"{cid}_confidences.json"
        if cf_path.exists():
            with open(cf_path) as f:
                cf = json.load(f)
            chain_ids  = cf.get("atom_chain_ids", [])
            plddts     = cf.get("atom_plddts", [])
            lig_plddts = [p for c, p in zip(chain_ids, plddts) if c == "B"]
            result["ligand_mean_plddt"] = (
                float(np.mean(lig_plddts)) if lig_plddts else None
            )
            result["ligand_atom_count"] = len(lig_plddts)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Composite scoring (gated)
# ─────────────────────────────────────────────────────────────────────────────

def identify_gate_passers(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask for compounds that pass ALL Tier 1 hard gates.
    This is the normalization reference set for the gated composite.

    Tier 1 gates (matches prioritize_compounds.py):
      - has_clash == 0
      - bbb_criteria_relaxed == True
      - iptm >= IPTM_MIN
      - predicted_toxicity >= TOX_MIN_HARD  (class 1-2 fail)
    """
    has_clash_ok = (df["has_clash"].fillna(1) == 0)
    bbb_ok       = (df.get("bbb_criteria_relaxed",
                           pd.Series(False, index=df.index)) == True)
    iptm_ok      = (df["iptm"].fillna(0) >= IPTM_MIN)
    tox_ok       = (df["predicted_toxicity"].fillna(0) >= TOX_MIN_HARD)
    return has_clash_ok & bbb_ok & iptm_ok & tox_ok


def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the gated composite score.

    Normalization: min-max over gate-passing compounds only, so scores
    reflect relative quality among viable candidates. Non-gate-passers
    are scored against the same reference and may end up with values
    below 0 (if their iptm / BBB is worse than the gate-passer minimum)
    — that's meaningful, not an error.

    Toxicity uses a fixed (class - 1) / 5 mapping, not min-max, because
    ProTox-3 class is a discrete categorical scale where the mapping to
    [0, 1] is well-defined regardless of which compounds are in the set.
    """
    df = df.copy()

    # Identify the reference set
    gate_passers_mask = identify_gate_passers(df)
    n_gate_passers = gate_passers_mask.sum()
    print(f"  Gate-passing compounds (normalization reference): "
          f"{n_gate_passers} / {len(df)}")

    if n_gate_passers == 0:
        print("  WARNING: no gate-passing compounds found. Composite "
              "will be computed against the full library as a fallback.")
        ref_df = df
    else:
        ref_df = df[gate_passers_mask]

    # ── iptm_norm (gated) ─────────────────────────────────────────────────
    ref_iptm = ref_df["iptm"].fillna(0)
    iptm_min, iptm_max = ref_iptm.min(), ref_iptm.max()
    df["iptm_norm"] = (df["iptm"].fillna(0) - iptm_min) / (iptm_max - iptm_min + 1e-9)

    # ── bbb_norm (gated) ──────────────────────────────────────────────────
    ref_bbb = ref_df["bbb_score"].fillna(0)
    bbb_min, bbb_max = ref_bbb.min(), ref_bbb.max()
    df["bbb_norm"] = (df["bbb_score"].fillna(0) - bbb_min) / (bbb_max - bbb_min + 1e-9)

    # ── tox_norm (fixed mapping, not gated) ───────────────────────────────
    # ProTox-3 class 1 (fatal) -> 0.0; class 6 (non-toxic) -> 1.0
    # fillna with 4 (harmful if swallowed) as a mid-range default
    df["tox_norm"] = (df["predicted_toxicity"].fillna(4) - 1) / 5

    # ── Composite ─────────────────────────────────────────────────────────
    df["composite_score"] = (
        WEIGHTS["iptm"] * df["iptm_norm"] +
        WEIGHTS["bbb"]  * df["bbb_norm"]  +
        WEIGHTS["tox"]  * df["tox_norm"]
    )

    # Convenience flag (same as before): passes BBB + no clash.
    # Note: drug_candidate is a relaxed version of the gate (doesn't check
    # iptm or tox). prioritize_compounds.py applies the full gate logic
    # when assigning tiers.
    df["drug_candidate"] = (
        (df.get("bbb_criteria_relaxed",
                pd.Series(False, index=df.index)) == True) &
        (df["has_clash"].fillna(1) == 0)
    )

    return df.sort_values("composite_score", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent.resolve()

    # Discover all phgdh_ligand_XXXX/ folders sitting next to this script
    folders = sorted(
        p for p in here.iterdir()
        if p.is_dir() and p.name.startswith(FOLDER_PREFIX)
    )

    if not folders:
        print(f"ERROR: No '{FOLDER_PREFIX}*' folders found in {here}")
        print("Make sure this script is in the same directory as the compound folders.")
        sys.exit(1)

    print(f"Found {len(folders)} compound folders in {here}")
    print("Extracting AF3 metrics...\n")

    # ── Extract ──────────────────────────────────────────────────────────────
    records = []
    errors  = []

    for i, folder in enumerate(folders):
        rec = extract_from_folder(folder)
        records.append(rec)
        if rec.get("error"):
            errors.append((folder.name, rec["error"]))
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  [{i+1:>4}/{len(folders)}]  {folder.name}  "
                  f"iptm={rec.get('iptm', 'N/A')}")

    print(f"  [{len(folders)}/{len(folders)}]  done.\n")

    af3_df  = pd.DataFrame(records)
    af3_csv = here / "af3_scores.csv"
    af3_df.to_csv(af3_csv, index=False)
    print(f"AF3 scores saved  ->  {af3_csv}")

    if errors:
        print(f"\nWarning: {len(errors)} folders had issues:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # ── Load BBB / tox ───────────────────────────────────────────────────────
    bbb_path = here / BBB_CSV
    if not bbb_path.exists():
        print(f"\nWARNING: {BBB_CSV} not found — saving AF3-only output.")
        af3_df.to_csv(here / "full_composite_scores.csv", index=False)
        return

    bbb_df = pd.read_csv(bbb_path)
    # Re-index Compound labels from 1-based to 0-based to match folder numbering
    # Compound_1 -> Compound_0, Compound_2 -> Compound_1, etc.
    if "Compound" in bbb_df.columns:
        bbb_df["Compound"] = bbb_df["Compound"].apply(
            lambda x: f"Compound_{int(x.split('_')[1]) - 1}" if isinstance(x, str) and x.startswith("Compound_") else x
        )
    print(f"Loaded BBB/tox data  ->  {len(bbb_df)} compounds")

    # ── Load TamGen log-prob ─────────────────────────────────────────────────
    tsv_path = here / TSV_FILE
    if tsv_path.exists():
        tsv_df = pd.read_csv(tsv_path, sep="\t", header=0)
        tsv_df.columns = ["smiles", "log_prob"]
        bbb_df = bbb_df.merge(tsv_df, on="smiles", how="left")
        print(f"Merged TamGen log-prob  ->  {bbb_df['log_prob'].notna().sum()} matched")

    # ── Merge on SMILES ──────────────────────────────────────────────────────
    af3_cols = [
        "smiles", "iptm", "ptm", "ranking_score", "has_clash",
        "fraction_disordered", "chain_pair_iptm_01", "chain_pair_pae_min_01",
        "best_sample_ranking_score", "ranking_score_mean", "ranking_score_std",
        "ligand_mean_plddt", "ligand_atom_count", "error"
    ]
    merged = bbb_df.merge(
        af3_df[[c for c in af3_cols if c in af3_df.columns]],
        on="smiles", how="left"
    )
    n_af3 = merged["iptm"].notna().sum()
    print(f"Merged dataset  ->  {len(merged)} compounds, {n_af3} with AF3 scores\n")

    # ── Tag generated compounds as non-control, then append SEEDC1 positive control
    merged["is_positive_control"] = False

    pos_path = here / POS_CTRL_CSV
    if pos_path.exists():
        pos_ctrl = pd.read_csv(pos_path)
        # Keep only the columns that exist in `merged` so the append is clean
        # (the positive-control CSV may carry extra columns like
        # composite_score_gated from earlier workflow stages; those are
        # discarded here and will be recomputed by compute_composite below).
        keep_cols = [c for c in pos_ctrl.columns if c in merged.columns]
        pos_ctrl = pos_ctrl[keep_cols].copy()
        pos_ctrl["is_positive_control"] = True
        # Drop any stale composite/norm values from the positive-control row — they'll
        # be recomputed below so the row uses the same weights/normalization as
        # the rest of the library.
        for stale in ["iptm_norm", "bbb_norm", "tox_norm", "logprob_norm",
                      "composite_score", "drug_candidate"]:
            if stale in pos_ctrl.columns:
                pos_ctrl = pos_ctrl.drop(columns=[stale])
        merged = pd.concat([merged, pos_ctrl], ignore_index=True, sort=False)
        print(f"Appended positive control  ->  {pos_ctrl['Compound'].iloc[0]}  "
              f"total rows now {len(merged)}")
    else:
        print(f"Note: {POS_CTRL_CSV} not found — pipeline will run without "
              f"positive control row.")

    # ── Composite ranking ────────────────────────────────────────────────────
    merged = compute_composite(merged)

    full_csv = here / "full_composite_scores.csv"
    merged.to_csv(full_csv, index=False)
    print(f"Full composite scores  ->  {full_csv}")

    top50_csv = here / "top_candidates.csv"
    merged.head(50).to_csv(top50_csv, index=False)
    print(f"Top 50 candidates      ->  {top50_csv}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total compounds         {len(merged)}")
    print(f"  With AF3 scores         {n_af3}")
    print(f"  Has clash (Tier 1 fail) {(merged['has_clash'] > 0).sum()}")
    print(f"  Drug candidates         {merged['drug_candidate'].sum()}  "
          f"(relaxed BBB + no clash)")
    print(f"  Composite weights       "
          f"iptm {WEIGHTS['iptm']:.0%} / bbb {WEIGHTS['bbb']:.0%} / "
          f"tox {WEIGHTS['tox']:.0%}")
    print(f"  Normalization           min-max on gate-passers "
          f"(has_clash=0, BBB-relaxed, iptm>={IPTM_MIN}, tox>={TOX_MIN_HARD})")

    if n_af3 > 0:
        af3_sub = merged[merged["iptm"].notna()]
        print(f"  iptm  mean / max        "
              f"{af3_sub['iptm'].mean():.3f} / {af3_sub['iptm'].max():.3f}")
        print(f"  PAE   mean / min        "
              f"{af3_sub['chain_pair_pae_min_01'].mean():.2f} / "
              f"{af3_sub['chain_pair_pae_min_01'].min():.2f} A")

    print(f"\nTop 10 by composite score:")
    show_cols = ["Compound", "composite_score", "iptm", "bbb_score",
                 "predicted_toxicity", "has_clash", "drug_candidate",
                 "is_positive_control"]
    present = [c for c in show_cols if c in merged.columns]
    print(merged[present].head(10).to_string(index=False))

    # ── Positive control placement ──────────────────────────────────────────
    if "is_positive_control" in merged.columns and merged["is_positive_control"].any():
        pc = merged[merged["is_positive_control"]]
        merged_sorted = merged.sort_values("composite_score",
                                           ascending=False).reset_index(drop=True)
        print(f"\n--- Positive control placement ---")
        for _, ctrl in pc.iterrows():
            rank = int(merged_sorted[
                merged_sorted["Compound"] == ctrl["Compound"]
            ].index[0]) + 1
            pct = 100 * (1 - (rank - 1) / len(merged_sorted))
            print(f"  {ctrl['Compound']:<15} composite={ctrl['composite_score']:.4f}  "
                  f"iptm={ctrl['iptm']}  bbb={ctrl['bbb_score']:.3f}  "
                  f"tox_class={int(ctrl['predicted_toxicity'])}")
            print(f"  {'':<15} ranks {rank} / {len(merged_sorted)}  "
                  f"({pct:.1f}th percentile)")

    print("=" * 60)


if __name__ == "__main__":
    main()
