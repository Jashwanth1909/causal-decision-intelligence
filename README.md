# Causal AI-based data-driven decision intelligence

Code and results for the manuscript **"Causal AI-based data-driven decision intelligence for prediction and optimal intervention selection"** (submitted to *Knowledge-Based Systems*).

The framework separates three decisions that are often conflated:

1. **Outcome prediction**: what is likely to happen, `P(Y | X, T)`.
2. **Causal effect learning**: what an intervention would change, `tau(x) = E[Y(1) - Y(0) | X = x]`, estimated with S-, T-, X-, DR- and R-learners.
3. **Decision intelligence**: treat a unit only if `tau_hat(x) > c` (intervention cost), and choose among candidate policies **without ground truth** using direct-method, IPW and doubly robust off-policy value estimates.

It is evaluated in three settings where both potential outcomes are known:

| Setting | Units | Covariates | Outcome | Source |
|---|---|---|---|---|
| Synthetic | 15,000 × 10 replications | 5 | binary | confounded simulation |
| IHDP | 747 × 10 replications | 25 | continuous | Hill (2011) via CEVAE repository (pinned commit) |
| Venture | 19,814 × 10 replications | 12 real venture features | binary | Crunchbase export, 4 Dec 2015 (CC BY-NC 4.0); baseline exit model fitted to real exits, intervention simulated |

## Reproduce everything

```bash
pip install -r requirements.txt
python causal_di.py            # ~6 min on a laptop CPU; writes results/
python causal_di.py --quick    # 3 replications, ~1.5 min
```

Data files are downloaded automatically into `data/` on the first run from version-pinned URLs. If your network blocks this, the script prints the manual `git clone` commands.

## Repository layout

```
causal_di.py        # the entire pipeline: data, learners, policy evaluation, tables, figures
requirements.txt
results/
  table1_baseline_comparison.csv      # Table 1 of the paper (all methods, three settings)
  table2_paired_tests.csv             # Table 2 of the paper (paired significance tests)
  supp_table_estimation_decision.csv  # Supplementary Table S2 (means and SDs)
  supp_table_policy_selection.csv     # Supplementary Table S3 (DM / IPW / DR policy selection)
  fig1_representative_replication.*   # Figure 1 (PNG 600 dpi + vector PDF)
  fig2_cost_and_selection.*           # Figure 2
  graphical_abstract.*                # graphical abstract
  metrics.json                        # every number quoted in the text
  replications_*.csv                  # per-replication raw results
```

## Licence

Code: MIT. Crunchbase data: CC BY-NC 4.0 (non-commercial research use). IHDP: as distributed with the CEVAE repository for research use.

## Citation

J.K.G. Cheruku, Causal AI-based data-driven decision intelligence for prediction and optimal intervention selection, manuscript submitted to *Knowledge-Based Systems* (2026). Code: https://github.com/Jashwanth1909/causal-decision-intelligence
