#!/usr/bin/env python3
"""
Compute composite_score_gated for the SEEDC1 positive control row.

The gated composite (matching prioritize_compounds.py) uses:
  - Weights: 55% iptm / 25% BBB / 20% tox  (log_prob dropped)
  - Normalization: min-max over Tier A+B compounds only
  - No clash penalty (has_clash is a hard Tier 1 gate instead)

Rationale: filter first (Tier 1 gates), then rank among the survivors.

Inputs (in the script's working directory):
    prioritized_compounds.csv               Full library with tier assignments
    seedc1_positive_control_row.csv         Single-row positive-control entry

Output:
    seedc1_positive_control_row.csv         Same file, updated in-place with
                                            iptm_norm_gated, bbb_norm_gated,
                                            tox_norm_gated, composite_score_gated
"""
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PRIORITIZED_CSV = "prioritized_compounds.csv"
POS_CTRL_CSV    = "seedc1_positive_control_row.csv"

# Composite weights — must match extract_af3_scores.py
W_IPTM, W_BBB, W_TOX = 0.55, 0.25, 0.20


# Load the full library with tier assignments
prioritized = pd.read_csv(PRIORITIZED_CSV)
# Load positive-control row
pos_row = pd.read_csv(POS_CTRL_CSV)

# ─────────────────────────────────────────────────────────────────────────────
# Compute gated composite
# ─────────────────────────────────────────────────────────────────────────────

# Normalization reference: Tier A+B compounds only
gate_passers = prioritized[prioritized["tier"].isin(["A", "B"])]
print(f"Normalization reference: {len(gate_passers)} gate-passing compounds (Tier A+B)")


def gated_norm(value, col, fill=0):
    ref = gate_passers[col].fillna(fill)
    mn, mx = ref.min(), ref.max()
    return float((value - mn) / (mx - mn + 1e-9))


# Extract positive-control values
pos = pos_row.iloc[0]
pos_iptm      = pos["iptm"]
pos_bbb       = pos["bbb_score"]
pos_tox_class = pos["predicted_toxicity"]

# Gated norms
pos_iptm_norm_gated = gated_norm(pos_iptm, "iptm")
pos_bbb_norm_gated  = gated_norm(pos_bbb, "bbb_score")
# Toxicity uses fixed (class-1)/5 mapping, not min-max (it's a discrete scale)
pos_tox_norm_gated  = (pos_tox_class - 1) / 5

pos_composite_gated = (
    W_IPTM * pos_iptm_norm_gated +
    W_BBB  * pos_bbb_norm_gated  +
    W_TOX  * pos_tox_norm_gated
)

print(f"\nPositive control gated composite components:")
print(f"  iptm:       {pos_iptm}   -> iptm_norm_gated = {pos_iptm_norm_gated:.4f}")
print(f"  bbb_score:  {pos_bbb:.4f} -> bbb_norm_gated  = {pos_bbb_norm_gated:.4f}")
print(f"  tox_class:  {pos_tox_class}      -> tox_norm_gated  = {pos_tox_norm_gated:.4f}")
print(f"\n  composite_score_gated = "
      f"0.55*{pos_iptm_norm_gated:.3f} + 0.25*{pos_bbb_norm_gated:.3f} + 0.20*{pos_tox_norm_gated:.3f}")
print(f"                       = {pos_composite_gated:.4f}")

# Note: iptm_norm_gated can be < 0 if the positive control's iptm is below the
# gate-passer min. Same for bbb_norm_gated.
if pos_iptm_norm_gated < 0:
    gate_min_iptm = gate_passers["iptm"].min()
    print(f"\n  NOTE: iptm_norm_gated is negative because the positive control's iptm")
    print(f"  ({pos_iptm}) is BELOW the gate-passer minimum ({gate_min_iptm}). This")
    print(f"  is meaningful — it indicates the control scores worse on iptm than")
    print(f"  any compound that passed the gates.")
if pos_bbb_norm_gated < 0:
    gate_min_bbb = gate_passers["bbb_score"].min()
    print(f"\n  NOTE: bbb_norm_gated is negative because the positive control's")
    print(f"  bbb_score is BELOW the gate-passer minimum ({gate_min_bbb:.3f}).")

# ─────────────────────────────────────────────────────────────────────────────
# Compare to original composite and to library
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("POSITIVE CONTROL COMPOSITE SCORES — SIDE BY SIDE")
print("=" * 70)
print(f"  composite_score (original):         {pos['composite_score']:.4f}")
print(f"    (40/25/20/15, library-wide norm, clash penalty)")
print(f"  composite_score_gated (preferred):  {pos_composite_gated:.4f}")
print(f"    (55/25/20, Tier A+B norm, no clash penalty)")

# Where does the positive control's gated score fall?
prioritized_gated = prioritized.copy()
prioritized_gated["iptm_n_g"] = prioritized_gated["iptm"].apply(
    lambda x: gated_norm(x, "iptm") if pd.notna(x) else np.nan
)
prioritized_gated["bbb_n_g"] = prioritized_gated["bbb_score"].apply(
    lambda x: gated_norm(x, "bbb_score") if pd.notna(x) else np.nan
)
prioritized_gated["tox_n_g"] = (prioritized_gated["predicted_toxicity"].fillna(4) - 1) / 5
prioritized_gated["comp_gated"] = (
    W_IPTM * prioritized_gated["iptm_n_g"] +
    W_BBB  * prioritized_gated["bbb_n_g"]  +
    W_TOX  * prioritized_gated["tox_n_g"]
)

print(f"\nLibrary statistics on composite_score_gated:")
print(f"  Min:    {prioritized_gated['comp_gated'].min():.4f}")
print(f"  Median: {prioritized_gated['comp_gated'].median():.4f}")
print(f"  Max:    {prioritized_gated['comp_gated'].max():.4f}")

# Percentile
n_below = (prioritized_gated['comp_gated'] < pos_composite_gated).sum()
total = prioritized_gated['comp_gated'].notna().sum()
print(f"\nPositive control gated composite ({pos_composite_gated:.4f}):")
print(f"  Compounds with gated composite < positive control: {n_below} / {total}"
      f" ({100 * n_below / total:.1f}th percentile)")

# Within-tier comparison
tier_a = prioritized_gated[prioritized_gated["tier"] == "A"]
n_a_below = (tier_a['comp_gated'] < pos_composite_gated).sum()
print(f"  Tier A compounds with gated composite < positive control: {n_a_below} / {len(tier_a)}")
tier_d = prioritized_gated[prioritized_gated["tier"] == "D"]
if len(tier_d) > 0:
    print(f"  Tier D range: {tier_d['comp_gated'].min():.4f} to {tier_d['comp_gated'].max():.4f}")
    in_d_range = tier_d['comp_gated'].min() <= pos_composite_gated <= tier_d['comp_gated'].max()
    print(f"  -> Positive control sits {'inside' if in_d_range else 'outside'} the Tier D range")

# ─────────────────────────────────────────────────────────────────────────────
# Save updated row with BOTH composite columns
# ─────────────────────────────────────────────────────────────────────────────
pos_row["iptm_norm_gated"]       = pos_iptm_norm_gated
pos_row["bbb_norm_gated"]        = pos_bbb_norm_gated
pos_row["tox_norm_gated"]        = pos_tox_norm_gated
pos_row["composite_score_gated"] = pos_composite_gated

pos_row.to_csv(POS_CTRL_CSV, index=False)

print(f"\nUpdated row saved: {POS_CTRL_CSV}")
print(f"Columns now: {len(pos_row.columns)} (original + 4 gated columns)")
