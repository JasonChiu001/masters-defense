# PHGDH Inhibitor Prioritization Pipeline

A filter-first-then-rank framework for prioritizing AI-generated small-molecule PHGDH inhibitor candidates for chronic CNS dosing. The pipeline takes a TamGen-generated compound library plus AlphaFold3 binding predictions and ProTox-3 toxicity estimates, applies hard gates encoding domain non-negotiables (binding confidence, blood-brain barrier permeability, toxicity, structural validity), and produces a tiered candidate list with a composite ranking inside each tier.

This repository accompanies the master's thesis *Computational Strategies for Targeting PHGDH: From RNA-Binding Mechanism to AI-Generated Inhibitor Prioritization* (UCSD Bioengineering, 2026). The thesis describes the rationale for filter-first-then-rank and reports the framework's behavior on a 1,987-compound library seeded on a published PHGDH inhibitor scaffold.

## Pipeline overview

Four scripts run in sequence:

1. **`calculate_bbb_score.py`** computes the Gupta 2019 BBB Score and physicochemical pass/fail flags for every compound, given a CSV of ProTox-3 output (SMILES plus toxicity and basic descriptors).
2. **`extract_af3_scores.py`** walks a directory of per-compound AlphaFold3 output folders, extracts ipTM, pTM, ranking score, PAE, and clash flag for each, joins them to the BBB and toxicity data, optionally appends a positive-control row, and computes a gated composite score.
3. **`prioritize_compounds.py`** applies the Tier 1 hard gates (no clash, relaxed BBB, ipTM at least 0.70, toxicity class at least 3) and assigns each compound to Tier A, B, C, or D. Within each tier, compounds are ranked by the composite score from step 2.
4. **`gated_composite_scoring.py`** (optional) recomputes the gated composite for a positive-control row against the Tier A and B normalization reference, useful for benchmarking a known compound against the framework after the fact.

The composite score is weighted 55 percent ipTM, 25 percent BBB score, 20 percent toxicity, and is min-max normalized over Tier 1 gate-passers only. Earlier weight schemes are documented inside the scripts.

## Dependencies

Python 3.9 or later, plus:

```
pandas
numpy
rdkit
```

Install with pip:

```
pip install pandas numpy rdkit
```

RDKit is only required by `calculate_bbb_score.py`. The other three scripts use only pandas and numpy.

## Directory layout

The pipeline assumes a flat layout with all four scripts and the input files in one directory, and per-compound AlphaFold3 output as subfolders:

```
project_root/
├── calculate_bbb_score.py
├── extract_af3_scores.py
├── prioritize_compounds.py
├── gated_composite_scoring.py
│
├── results.csv                            # ProTox-3 output + RDKit descriptors
├── seedc1_tamgen_logprobs.tsv             # TamGen generation log-probabilities
├── seedc1_positive_control_row.csv        # (optional) positive-control row
│
├── phgdh_ligand_0000/                     # one folder per compound
│   ├── phgdh_ligand_0000_summary_confidences.json
│   ├── phgdh_ligand_0000_data.json
│   └── ranking_scores.csv
├── phgdh_ligand_0001/
│   └── ...
└── ...
```

Compound IDs in `results.csv` are expected to be 1-indexed (`Compound_1`, `Compound_2`, ...) while AlphaFold3 folder numbers are 0-indexed (`phgdh_ligand_0000`, `phgdh_ligand_0001`, ...). `extract_af3_scores.py` reconciles this when it merges.

## Usage

Run the scripts in order. Each script writes its output to the working directory (except `calculate_bbb_score.py`, which writes to `./outputs/` by default; override with `BBB_OUTPUT_DIR`).

```
# Step 1: compute BBB scores from ProTox-3 output
python calculate_bbb_score.py
# -> outputs/results_with_bbb_score.csv

# Move or symlink the BBB output next to the AF3 folders
mv outputs/results_with_bbb_score.csv .

# Step 2: extract AF3 metrics and compute composite scores
python extract_af3_scores.py
# -> af3_scores.csv, full_composite_scores.csv, top_candidates.csv

# Step 3: apply tier framework
python prioritize_compounds.py
# -> prioritized_compounds.csv, tier_A_leads.csv, tier_B_promising.csv,
#    tier_C_investigate.csv, positive_control.csv, prioritization_summary.csv

# Step 4 (optional): recompute gated composite for the positive control
python gated_composite_scoring.py
# -> seedc1_positive_control_row.csv (updated in place)
```

## Input file formats

**`results.csv`** must contain at least these columns:

```
Compound, smiles, predicted_ld50_mg_kg, predicted_toxicity,
average_similarity_%, prediction_accuracy_%,
molecular_weight, number_of_hydrogen_bond_acceptors,
number_of_hydrogen_bond_donors, number_of_atoms, number_of_bonds,
number_of_rotatable_bonds, molecular_refractivity,
topological_polar_surface_area_tpsa,
octanol_water_partition_coefficient_logp
```

This matches the schema ProTox-3 returns via its web interface, with RDKit-computed descriptors merged in.

**`seedc1_tamgen_logprobs.tsv`** is a tab-separated file with two columns: `smiles` and the normalized log-probability assigned by TamGen at generation time. The header row is required.

**`seedc1_positive_control_row.csv`** (optional) is a one-row CSV in the same schema as `results.csv` plus AlphaFold3 columns (`iptm`, `bbb_score`, `predicted_toxicity`, `has_clash`, `ranking_score_std`, etc.). Used to spike a known compound into the pipeline as a benchmark.

## Output files

After all four steps:

- **`prioritized_compounds.csv`**: full library with tier assignment, composite score, gate-failure flags, and physicochemical summary
- **`tier_A_leads.csv`**: compounds passing every gate with no flags
- **`tier_B_promising.csv`**: gate-passers with at least one secondary flag (tox class 3, TPSA above strict threshold, LogP outside the 1.5 to 3.5 window, MW above 450, or unstable AF3 pose)
- **`tier_C_investigate.csv`**: borderline compounds failing exactly one Tier 1 gate
- **`prioritization_summary.csv`**: per-tier counts and key statistics
- **`positive_control.csv`**: the positive-control row in isolation, with its tier and ranking
- **`top_candidates.csv`**: top 50 by composite score across the full library

Tier D (multiple gate failures) is captured in `prioritized_compounds.csv` but does not get its own CSV.

## Citing the upstream tools

If you use this pipeline, please cite the underlying tools whose output it processes:

- **AlphaFold3**: Abramson, J., Adler, J., Dunger, J., et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630, 493 to 500 (2024).
- **TamGen**: Wu, K., Karapetyan, E., Schloss, J., Vadgama, J., and Wu, Y. TamGen: target-aware molecule generation with chemical language models. *arXiv* (2023).
- **ProTox 3.0**: Banerjee, P., Kemmler, E., Dunkel, M., and Preissner, R. ProTox 3.0: a webserver for the prediction of toxicity of chemicals. *Nucleic Acids Research* 52, W513 to W520 (2024).
- **Gupta 2019 BBB Score**: Gupta, M., Lee, H. J., Barden, C. J., and Weaver, D. F. The Blood-Brain Barrier (BBB) Score. *Journal of Medicinal Chemistry* 62(21), 9824 to 9836 (2019).

## Data availability

The compound SMILES, AlphaFold3 outputs, ProTox-3 query results, and per-compound CSV files used to develop and validate this pipeline are not included in this repository due to ongoing intellectual-property considerations. The scripts will run on any TamGen plus AlphaFold3 plus ProTox-3 pipeline output that matches the schemas described above.

## License

MIT. See `LICENSE` for details.
