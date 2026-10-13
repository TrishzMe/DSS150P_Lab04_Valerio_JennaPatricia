# baseline — replay of the original group project's modelling code

`replay_notebook.py` re-runs the modelling code of the original Predictive Analytics group notebook
(`Predictive_Final_Project_code.ipynb`, cells 1, 3, 5, 9, 11 and 14), copied verbatim apart from removed print and plot
statements. The original notebook never saved which rows were in its train and test sets; the replay:

- confirms that the code reproduces the notebook's saved metrics (`metadata/expected_metrics.json`);
- captures the checksums of the station-day keys in its train and test sets as `metadata/split_manifest.json`,
  which every pipeline run must reproduce (check PRD-04);
- with `--compare-curated`, checks that the pipeline's curated model-ready table is identical to the notebook's frame
  (evidence: `docs/evidence/notebook_replay.json`).

The original notebook and paper themselves are not included in this repository (the laboratory brief does not require
them).

**Attribution:** S. Grino, B. J. Perez and J. P. Valerio, *Predicting PM2.5 Air Quality Exceedances Using Machine
Learning: A Co-Pollutant Chemistry Approach*, Predictive Analytics group project, School of Data Science, Mapúa
University, 2026. This repository reuses its problem statement, target definition, feature definitions, preprocessing
steps, model families, hyperparameters and reported metrics, as allowed by section 4.1 of the laboratory brief.

Usage:  `python baseline/replay_notebook.py [--write-manifest] [--compare-curated]`
