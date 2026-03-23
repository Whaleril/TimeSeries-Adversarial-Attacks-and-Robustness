# Code Provenance

This document records where the code and data used in this repository came from, how they were adapted, and what has been changed locally.

## Purpose of This Repository

This repository is a course project workspace for:

- reproducing selected experiments from a published paper
- adapting the task to a more realistic traffic-sensor scenario
- extending the baseline with adversarial attack and robustness analysis

It is not presented as a fully original implementation from scratch.

## Upstream Source A: Original Paper Repository

Primary conceptual and code source:

- the original repository accompanying the paper
  *Adversarial Examples in Deep Learning for Multivariate Time Series Regression*
- Original repository: <https://github.com/dependable-cps/adversarial-MTSR>

Materials inherited or adapted from that line include:

- overall experiment framing
- power-consumption forecasting scripts
- LSTM baseline structure
- initial adversarial-evaluation workflow

## Upstream Source B: Community Fork / Issue-Based Fixes

A second important source was a community fork shared in the issue discussion of the original repository.

- Community fork used as reference: <https://github.com/forallx94/adversarial-MTSR>

Its role in this project was to help recover or reorganize parts that were missing or incomplete in the original codebase, especially around:

- model utilities
- attack utilities
- script organization
- TensorFlow / Keras compatibility direction

This repository uses that fork as an implementation reference and working base for further cleanup.

## Upstream Source C: Public Processed PEMS08 Repository

For the traffic experiment branch, this repository uses a public processed PEMS08 traffic dataset obtained from:

- `PEMS08-Traffic-Flow-Forecasting`
- Public repository link: <https://github.com/JvThunder/PEMS08-Traffic-Flow-Forecasting>

Local path used here:

- `Dataset/PEMS08 traffic flow dataset/traffic_dataset.csv`

This file is treated as a processed public dataset source, not as original data collected by this repository.

## License Status of Upstream Sources

At the time this repository was assembled, the following upstream sources were checked:

- the original paper code repository
- the community fork repository
- the public PEMS08 repository used for the traffic dataset

None of the above repositories was found to include an explicit `LICENSE` file.

Because of that:

- this repository keeps explicit attribution to all upstream sources
- this repository should be understood primarily as a study / course reproduction workspace
- users should independently verify redistribution requirements before reusing upstream-derived code outside a course or research context

## Local Modifications in This Repository

The following are examples of modifications made locally in this repository:

- rewriting project documentation to match the actual experimental focus
- cleaning the repository narrative to focus on the power dataset and the PEMS08 traffic dataset
- fixing local execution issues for current TensorFlow / Keras environments
- replacing fragile model loading with architecture reconstruction plus weight loading where needed
- integrating `BIM` as an additional attack baseline
- introducing a single-station traffic experiment branch
- creating:
  - `src/Traffic_LSTM.py`
  - `src/Traffic_LSTM_test.py`
- adding project-level explanatory files such as:
  - this `CODE_PROVENANCE.md`

## Data Adaptation for Traffic Experiments

The local traffic file was reduced to a simplified single-station time-series form with:

- sorted timesteps
- only the needed columns
- no irrelevant location column in the final single-station version
- input features:
  - `flow`
  - `occupy`
  - `speed`
- prediction target:
  - next-step `speed`

This adaptation was performed specifically to support:

- LSTM-based forecasting
- FGSM / BIM adversarial attacks
- robustness analysis under temporal perturbations

## Recommended Citation / Attribution Practice

If this repository is used in a course report, presentation, or derivative project, attribution should include:

1. the original paper
2. the original paper repository
3. the community fork used as a repair / reference source
4. the public processed PEMS08 repository used for the traffic dataset
