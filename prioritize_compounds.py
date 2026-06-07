#!/usr/bin/env python3
"""
PHGDH Compound Prioritization
==============================
Implements a tiered prioritization framework for PHGDH inhibitor candidates.

Place this script in the same directory as full_composite_scores.csv
(the output of extract_af3_scores.py).

Framework (as specified):
--------------------------
Tier 1 — Primary filters (hard gates, applied first):
    - has_clash == 0              (invalid poses discarded immediately)
    - bbb_criteria_relaxed        (CNS gate: must reach the brain)
    - iptm >= 0.70                (meaningful binding confidence threshold;
                                   raised from 0.5 because median iptm in the
                                   TamGen-seeded library is 0.82, so 0.5 did
                                   not act as a filter)
    - predicted_toxicity >= 3     (ProTox-3 Class 1-2 = LD50 <= 50 mg/kg,
                                   incompatible with chronic AD dosing)

Secondary flags (trigger Tier B, not Tier C):
    - ranking_score_std > 0.05    (inconsistent AF3 predictions across 5 seeds)
    - predicted_toxicity == 3     (acceptable with caution; flagged for review)
    - TPSA > 70 Å²                (above strict CNS threshold)
    - LogP outside 1.5–3.5        (outside optimal CNS window)
    - MW > 450 Da                 (above CNS size limit)

Output tiers:
    Tier A  — Pass all Tier 1 gates + reliable AF3 + good physicochemistry
              + tox class 4-6
    Tier B  — Pass all Tier 1 gates but flagged for reliability, physico,
              or tox class 3
    Tier C  — Fail exactly one Tier 1 gate (borderline, worth reviewing)
    Tier D  — Fail 2+ Tier 1 gates (deprioritized)

Composite score:
    Read directly from extract_af3_scores.py output (column `composite_score`
    in full_composite_scores.csv). That upstream script uses gated min-max
    normalization on Tier 1 gate-passers, with weights 55% iptm / 25% BBB /
    20% toxicity. This script reuses the value rather than recomputing, so
    there is a single source of truth for the scoring formula.

    Within each tier, compounds are sorted by composite_score DESC, so
    binding confidence, BBB quality, and toxicity all contribute to the
    within-tier ranking rather than iptm alone.

Outputs:
    prioritized_compounds.csv   — full dataset with tier assignments and flags
    tier_A_leads.csv            — Tier A leads, ranked by composite_score
    tier_B_promising.csv        — Tier B compounds
    tier_C_investigate.csv      — Tier C borderline compounds
    prioritization_summary.csv  — counts and key stats per tier
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds — edit here if needed
# ─────────────────────────────────────────────────────────────────────────────

IPTM_MIN           = 0.70   # Raised from 0.5 — at 0.5 the gate filtered
                            # only 1 / 1987 compounds because TamGen seeds
                            # from known PHGDH binders. 0.70 filters ~100.
TOX_MIN_HARD       = 3      # Tox class < 3 (i.e. Class 1-2) is a HARD gate
                            # failure for chronic AD dosing
TOX_FLAG_CLASS     = 3      # Class 3 passes the gate but is flagged -> Tier B
TPSA_MAX           = 90.0   # Å² — relaxed BBB threshold (hard watch)
TPSA_STRICT        = 70.0   # Å² — strict BBB threshold (flag if above this)
LOGP_MIN           = 1.5    # Optimal CNS LogP lower bound
LOGP_MAX           = 3.5    # Optimal CNS LogP upper bound
MW_MAX             = 450.0  # Da — maximum molecular weight
STD_FLAG_THRESHOLD = 0.05   # ranking_score_std above this = unreliable AF3

# Note: the composite score (column `composite_score` in the input CSV) is
# computed by extract_af3_scores.py using gated min-max normalization on
# Tier 1 gate-passers. We reuse it directly for within-tier ranking rather
# than recomputing, which keeps a single source of truth for the scoring.

INPUT_CSV  = "full_composite_scores.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def flag_reason(row):
    """Return a human-readable string listing any flags for a compound."""
    flags = []

    # Hard disqualifiers (Tier 1 gate failures)
    if pd.notna(row.get("has_clash")) and row["has_clash"] > 0:
        flags.append("has_clash")
    if row.get("bbb_criteria_relaxed") != True:
        flags.append("fails_relaxed_BBB")
    if pd.notna(row.get("iptm")) and row["iptm"] < IPTM_MIN:
        flags.append(f"low_iptm_{row['iptm']:.2f}")

    tox = row.get("predicted_toxicity")
    if pd.notna(tox) and tox < TOX_MIN_HARD:
        # Class 1-2 is now a hard gate failure, not just a caution
        flags.append(f"high_tox_class_{int(tox)}_HARD_FAIL")
    elif pd.notna(tox) and tox == TOX_FLAG_CLASS:
        # Class 3 passes the gate but is flagged -> Tier B
        flags.append("tox_class_3_caution")

    # Reliability
    if pd.notna(row.get("ranking_score_std")) and row["ranking_score_std"] > STD_FLAG_THRESHOLD:
        flags.append(f"unstable_pose_std={row['ranking_score_std']:.3f}")

    # Physicochemical
    tpsa = row.get("topological_polar_surface_area_tpsa")
    logp = row.get("octanol_water_partition_coefficient_logp")
    mw   = row.get("molecular_weight")

    if pd.notna(tpsa) and tpsa > TPSA_MAX:
        flags.append(f"TPSA>{TPSA_MAX}({tpsa:.1f})")
    elif pd.notna(tpsa) and tpsa > TPSA_STRICT:
        flags.append(f"TPSA_relaxed_only({tpsa:.1f})")

    if pd.notna(logp) and not (LOGP_MIN <= logp <= LOGP_MAX):
        flags.append(f"LogP_outside_window({logp:.2f})")

    if pd.notna(mw) and mw > MW_MAX:
        flags.append(f"MW>{MW_MAX}({mw:.0f}Da)")

    return "; ".join(flags) if flags else "none"


def assign_tier(row):
    """
    Tier A: passes all Tier 1 gates AND no reliability/physico/tox-3 flags
    Tier B: passes all Tier 1 gates BUT has reliability, physico, or tox-3 flags
    Tier C: fails exactly one Tier 1 gate
    Tier D: fails two or more Tier 1 gates

    Tier 1 gates (all must pass for A or B):
        - has_clash == 0
        - bbb_criteria_relaxed == True
        - iptm >= IPTM_MIN  (0.70)
        - predicted_toxicity >= TOX_MIN_HARD  (3, i.e. Class 1-2 fail)

    Secondary flags (any trigger Tier B):
        - ranking_score_std > STD_FLAG_THRESHOLD (unreliable pose)
        - predicted_toxicity == TOX_FLAG_CLASS (Class 3 — acceptable with caution)
        - TPSA > TPSA_STRICT  (i.e. > 70 Å²)
        - LogP outside [LOGP_MIN, LOGP_MAX]
        - MW > MW_MAX
    """
    tier1_failures = 0

    if pd.notna(row.get("has_clash")) and row["has_clash"] > 0:
        tier1_failures += 1
    if row.get("bbb_criteria_relaxed") != True:
        tier1_failures += 1
    if pd.notna(row.get("iptm")) and row["iptm"] < IPTM_MIN:
        tier1_failures += 1
    tox = row.get("predicted_toxicity")
    if pd.notna(tox) and tox < TOX_MIN_HARD:
        tier1_failures += 1

    # If AF3 scores not yet available, we can't assign A/B — mark as Unscored
    if pd.isna(row.get("iptm")):
        return "Unscored"

    if tier1_failures >= 2:
        return "D"
    if tier1_failures == 1:
        return "C"

    # All Tier 1 gates passed — check reliability, physicochemistry, and tox class 3
    has_secondary_flag = False

    if pd.notna(row.get("ranking_score_std")) and row["ranking_score_std"] > STD_FLAG_THRESHOLD:
        has_secondary_flag = True

    if pd.notna(tox) and tox == TOX_FLAG_CLASS:
        has_secondary_flag = True

    tpsa = row.get("topological_polar_surface_area_tpsa")
    logp = row.get("octanol_water_partition_coefficient_logp")
    mw   = row.get("molecular_weight")

    if pd.notna(tpsa) and tpsa > TPSA_STRICT:
        has_secondary_flag = True
    if pd.notna(logp) and not (LOGP_MIN <= logp <= LOGP_MAX):
        has_secondary_flag = True
    if pd.notna(mw) and mw > MW_MAX:
        has_secondary_flag = True

    return "B" if has_secondary_flag else "A"


def physico_summary(row):
    """One-line physicochemical status for the key BBB drivers."""
    tpsa = row.get("topological_polar_surface_area_tpsa")
    logp = row.get("octanol_water_partition_coefficient_logp")
    mw   = row.get("molecular_weight")

    tpsa_str = f"TPSA={tpsa:.1f}" if pd.notna(tpsa) else "TPSA=N/A"
    logp_str = f"LogP={logp:.2f}" if pd.notna(logp) else "LogP=N/A"
    mw_str   = f"MW={mw:.0f}" if pd.notna(mw) else "MW=N/A"

    return f"{tpsa_str} | {logp_str} | {mw_str}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent.resolve()

    input_path = here / INPUT_CSV
    if not input_path.exists():
        print(f"ERROR: {INPUT_CSV} not found in {here}")
        print("Run extract_af3_scores.py first to generate this file.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} compounds from {input_path.name}")
    print(f"AF3 scores available for: {df['iptm'].notna().sum()} compounds\n")

    # ── Apply flags and tiers ─────────────────────────────────────────────────
    print("Applying Tier 1 filters, reliability checks, and physicochemical flags...")
    df["flags"]            = df.apply(flag_reason, axis=1)
    df["tier"]             = df.apply(assign_tier, axis=1)
    df["physico_summary"]  = df.apply(physico_summary, axis=1)
    df["reliable_pose"]    = df["ranking_score_std"].apply(
        lambda x: "yes" if pd.notna(x) and x <= STD_FLAG_THRESHOLD
                  else ("no" if pd.notna(x) else "unscored")
    )

    # ── Sort: within each tier, rank by composite_score DESC ────────────────
    # composite_score is produced by extract_af3_scores.py using gated
    # min-max normalization on Tier 1 gate-passers. Because the normalization
    # reference is the gate-passing subset, within-tier rankings are coherent
    # with the gate structure: higher composite_score means better profile
    # among viable candidates.
    tier_order = {"A": 0, "B": 1, "C": 2, "D": 3, "Unscored": 4}
    df["_tier_sort"] = df["tier"].map(tier_order)
    df = df.sort_values(
        ["_tier_sort", "composite_score"],
        ascending=[True, False]
    ).drop(columns=["_tier_sort"])

    # ── Save full prioritized dataset ─────────────────────────────────────────
    full_out = here / "prioritized_compounds.csv"
    df.to_csv(full_out, index=False)
    print(f"Full prioritized dataset  ->  {full_out}")

    # ── Save per-tier files ───────────────────────────────────────────────────
    tier_files = {
        "A": "tier_A_leads.csv",
        "B": "tier_B_promising.csv",
        "C": "tier_C_investigate.csv",
    }

    display_cols = [
        "Compound", "is_positive_control", "tier", "composite_score",
        "iptm", "bbb_score",
        "predicted_toxicity", "has_clash", "reliable_pose", "ranking_score_std",
        "topological_polar_surface_area_tpsa",
        "octanol_water_partition_coefficient_logp",
        "molecular_weight", "bbb_criteria_strict", "bbb_criteria_relaxed",
        "chain_pair_pae_min_01", "ligand_mean_plddt", "log_prob",
        "physico_summary", "flags"
    ]
    present_cols = [c for c in display_cols if c in df.columns]

    for tier_label, fname in tier_files.items():
        subset = df[df["tier"] == tier_label][present_cols]
        out_path = here / fname
        subset.to_csv(out_path, index=False)
        print(f"Tier {tier_label} ({len(subset)} compounds)          ->  {out_path}")

    # ── Save positive control separately (regardless of its tier) ─────────────
    if "is_positive_control" in df.columns and df["is_positive_control"].any():
        pc_subset = df[df["is_positive_control"]][present_cols]
        pc_path = here / "positive_control.csv"
        pc_subset.to_csv(pc_path, index=False)
        print(f"Positive control ({len(pc_subset)} compound)       ->  {pc_path}")

    # ── Summary stats per tier ────────────────────────────────────────────────
    tier_counts = df["tier"].value_counts().reindex(
        ["A", "B", "C", "D", "Unscored"], fill_value=0
    )

    summary_rows = []
    for tier_label in ["A", "B", "C", "D", "Unscored"]:
        sub = df[df["tier"] == tier_label]
        summary_rows.append({
            "Tier":              tier_label,
            "Count":             len(sub),
            "Pct_of_library":    f"{100*len(sub)/len(df):.1f}%",
            "composite_mean":    (round(sub["composite_score"].mean(), 3)
                                  if "composite_score" in sub.columns
                                  and sub["composite_score"].notna().any()
                                  else "N/A"),
            "iptm_mean":         round(sub["iptm"].mean(), 3) if sub["iptm"].notna().any() else "N/A",
            "iptm_max":          round(sub["iptm"].max(), 3) if sub["iptm"].notna().any() else "N/A",
            "bbb_score_mean":    round(sub["bbb_score"].mean(), 3),
            "tox_class_median":  sub["predicted_toxicity"].median(),
            "reliable_pose_pct": (
                f"{100*(sub['reliable_pose']=='yes').sum()/len(sub):.0f}%"
                if len(sub) > 0 else "N/A"
            ),
            "Description": {
                "A":        "Passes all gates (incl. tox>=4), reliable pose, good physico",
                "B":        "Passes all gates but flagged (tox=3, or physico, or unstable pose)",
                "C":        "Fails exactly one Tier 1 gate — borderline",
                "D":        "Fails 2+ Tier 1 gates — deprioritized",
                "Unscored": "No AF3 data yet",
            }[tier_label]
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_out = here / "prioritization_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Summary table             ->  {summary_out}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("PRIORITIZATION SUMMARY")
    print("="*65)
    print(summary_df[["Tier","Count","Pct_of_library","composite_mean","iptm_mean","iptm_max","Description"]].to_string(index=False))

    print("\n--- Tier 1 gate breakdown ---")
    n_af3 = df["iptm"].notna().sum()
    if n_af3 > 0:
        af3 = df[df["iptm"].notna()]
        print(f"  has_clash > 0:              {(af3['has_clash'] > 0).sum():>5} compounds")
        print(f"  fails relaxed BBB:          {(af3['bbb_criteria_relaxed'] != True).sum():>5} compounds")
        print(f"  iptm < {IPTM_MIN}:              {(af3['iptm'] < IPTM_MIN).sum():>5} compounds")
        print(f"  tox class < {TOX_MIN_HARD} (Class 1-2): {(af3['predicted_toxicity'] < TOX_MIN_HARD).sum():>5} compounds  [HARD GATE]")
        print(f"  unreliable pose (std>0.05): {(af3['ranking_score_std'] > STD_FLAG_THRESHOLD).sum():>5} compounds  [flag only]")
        print(f"\n  Tox class distribution:")
        print(f"    class I-II  (HARD FAIL):   {(af3['predicted_toxicity'] <= 2).sum():>5} compounds")
        print(f"    class III   (flag -> B):   {(af3['predicted_toxicity'] == 3).sum():>5} compounds")
        print(f"    class IV    (acceptable):  {(af3['predicted_toxicity'] == 4).sum():>5} compounds")
        print(f"    class V-VI  (preferred):   {(af3['predicted_toxicity'] >= 5).sum():>5} compounds")

    print("\n--- Physicochemical (all compounds) ---")
    print(f"  TPSA > 90 Å²  (hard fail):  {(df['topological_polar_surface_area_tpsa'] > TPSA_MAX).sum():>5} compounds")
    print(f"  TPSA 70-90 Å² (relaxed):    {((df['topological_polar_surface_area_tpsa'] > TPSA_STRICT) & (df['topological_polar_surface_area_tpsa'] <= TPSA_MAX)).sum():>5} compounds")
    print(f"  LogP outside 1.5-3.5:       {((df['octanol_water_partition_coefficient_logp'] < LOGP_MIN) | (df['octanol_water_partition_coefficient_logp'] > LOGP_MAX)).sum():>5} compounds")
    print(f"  MW > 450 Da:                {(df['molecular_weight'] > MW_MAX).sum():>5} compounds")

    if (df["tier"] == "A").sum() > 0:
        print("\n--- Top Tier A leads (by composite_score) ---")
        tier_a = df[df["tier"] == "A"][present_cols].head(10)
        show = ["Compound", "composite_score", "iptm", "bbb_score",
                "predicted_toxicity", "ranking_score_std", "physico_summary", "flags"]
        show_present = [c for c in show if c in tier_a.columns]
        print(tier_a[show_present].to_string(index=False))
    else:
        print("\nNote: No Tier A compounds yet — AF3 scores needed for full tiering.")
        print("Once extract_af3_scores.py is run, re-run this script on the output.")

    # ── Positive control placement ────────────────────────────────────────────
    if "is_positive_control" in df.columns and df["is_positive_control"].any():
        pc = df[df["is_positive_control"]]
        print("\n--- POSITIVE CONTROL PLACEMENT ---")
        for _, ctrl in pc.iterrows():
            cname = ctrl["Compound"]
            ctier = ctrl["tier"]
            ccomp = ctrl["composite_score"]
            # Rank within its tier and library-wide
            within_tier = df[df["tier"] == ctier].sort_values(
                "composite_score", ascending=False
            ).reset_index(drop=True)
            tier_rank = int(within_tier[within_tier["Compound"] == cname].index[0]) + 1

            lib_sorted = df.sort_values(
                "composite_score", ascending=False
            ).reset_index(drop=True)
            lib_rank = int(lib_sorted[lib_sorted["Compound"] == cname].index[0]) + 1
            lib_pct = 100 * (1 - (lib_rank - 1) / len(df))

            print(f"  {cname}:")
            print(f"    Tier:               {ctier}")
            print(f"    composite_score:    {ccomp:.4f}")
            print(f"    iptm:               {ctrl['iptm']}")
            print(f"    bbb_score:          {ctrl['bbb_score']:.3f}")
            print(f"    predicted_toxicity: {int(ctrl['predicted_toxicity'])}")
            print(f"    Rank within Tier {ctier}:  {tier_rank} / {len(within_tier)}")
            print(f"    Library rank:       {lib_rank} / {len(df)}  ({lib_pct:.1f}th percentile)")
            print(f"    Flags:              {ctrl['flags']}")
            if ctier in ("C", "D"):
                n_tier_a = (df["tier"] == "A").sum()
                n_a_beat = (df[df["tier"] == "A"]["composite_score"] > ccomp).sum()
                print(f"    Tier A compounds outscoring this control: {n_a_beat} / {n_tier_a}")

    print("="*65)


if __name__ == "__main__":
    main()