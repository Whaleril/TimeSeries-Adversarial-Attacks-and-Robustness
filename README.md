# LSTM-Based Adversarial Attacks and Robustness Analysis for Time-Series Forecasting

This repository is a course reproduction and extension project focused on:

- reproducing part of the experiments from the paper *Adversarial Examples in Deep Learning for Multivariate Time Series Regression*
- analyzing the code and experimental workflow
- adapting the work to a traffic time-series setting
- studying adversarial attacks and robustness for LSTM-based models

At the current stage, this repository mainly contains:

- a cleaned and runnable power-consumption experiment line
- a single-station PEMS08 traffic experiment line
- baseline adversarial attacks including `FGSM` and `BIM`
- documentation for code structure and provenance

## Scope

The project currently focuses on:

- `LSTM`-based forecasting
- adversarial perturbations on time-series inputs
- robustness evaluation under temporal input attacks

The power-consumption branch is currently built around the household power consumption dataset setting with:

- input features: `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`
- target: next-step `Global_active_power`
- input window: previous `60` minutes after hourly resampling (`1` timestep with `7` features in the current implementation)
- split: first `3` years for training, then the first `8` months and `2` months of the last year for validation and test

The traffic branch is currently built around a **single-station** PEMS08 setting with:

- input features: `flow`, `occupy`, `speed`
- target: next-step `speed`
- input window: previous `60` minutes (`12` timesteps at `5` minutes each)
- split: `70%` train, `10%` validation, `20%` test

## Repository Structure

- `src/`
  Core training, testing, utility, and model-definition scripts.
- `Dataset/`
  Local copies of datasets or dataset-derived files used in experiments.
- `Trained models/`
  Saved model files and generated plots.

## Main Scripts

### Power-consumption branch

- `src/Power_LSTM.py`
  Train the LSTM model for the power-consumption dataset.

- `src/Power_LSTM_test.py`
  Evaluate the trained power LSTM model under clean input, FGSM, and BIM.

### Traffic branch

- `src/Traffic_LSTM.py`
  Train the LSTM model on the cleaned single-station PEMS08 traffic dataset.

- `src/Traffic_LSTM_test.py`
  Evaluate the trained traffic LSTM model under clean input, FGSM, and BIM.

## How To Run

Run from the `src/` directory.

For the traffic flow dataset:

```powershell
cd "LSTM-Timeseries-Adversarial-Attacks\src"
python Traffic_LSTM.py
python Traffic_LSTM_test.py
```

For the power dataset:

```powershell
cd "LSTM-Timeseries-Adversarial-Attacks\src"
python Power_LSTM.py
python Power_LSTM_test.py
```

## Code Provenance

This repository is **not** an original codebase written from scratch.

It is derived from multiple upstream sources:

1. The original open-source repository accompanying the paper:
   *Adversarial Examples in Deep Learning for Multivariate Time Series Regression*
   - Original repository: <https://github.com/dependable-cps/adversarial-MTSR>

2. A community fork shared in the issue discussion of the original repository,
   which helped recover or reorganize missing implementation pieces.
   - Community fork used as reference: <https://github.com/forallx94/adversarial-MTSR>

3. A public repository containing a processed PEMS08 traffic forecasting dataset.
   - Public processed PEMS08 repository: <https://github.com/JvThunder/PEMS08-Traffic-Flow-Forecasting>

This repository further modifies, cleans, and extends those materials for course use.

Important note:

- At the time of inspection, the upstream code repositories and the public dataset repository used here did **not** provide an explicit `LICENSE` file.
- Because of that, this repository should be understood as a **course reproduction and research-study workspace**, with explicit attribution to upstream sources.

Please see [CODE_PROVENANCE.md](./CODE_PROVENANCE.md) for details.

## Dataset Provenance

### Power dataset

The power-consumption experiment is derived from the household power consumption data used by the original paper implementation.

Local path used here:

- `Dataset/Power conumption dataset/household_power_consumption.txt`

### Traffic dataset

The traffic experiment uses a cleaned single-station subset derived from a public processed PEMS08 dataset distributed in:

- `Dataset/PEMS08-Traffic-Flow-Forecasting-main/traffic_dataset.csv`

This file contains:

- `timestep`
- `flow`
- `occupy`
- `speed`

For the current project, only one station was retained and reformulated into a multivariate time-series forecasting task.

## This Repository's Contributions

Compared with the upstream materials, this repository currently adds or modifies:

- runnable notes and project introductions tailored to the course topic
- fixes for current TensorFlow/Keras execution in local experiments
- integration of `BIM` as a second attack baseline
- single-station PEMS08 preprocessing for LSTM forecasting
- new scripts:
  - `src/Traffic_LSTM.py`
  - `src/Traffic_LSTM_test.py`
- cleaned task focus around LSTM-based adversarial robustness analysis

## Academic Use

This repository is intended for:

- course reproduction
- code analysis
- robustness experiments
- method extension

If you use this project in a report or presentation, cite:

- the original paper
- the original paper repository
- the community fork that restored missing code structure
- the public processed PEMS08 repository used for the traffic dataset
