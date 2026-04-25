# Model Card

This document becomes mandatory once an ML model is proposed for portfolio construction. The repository already contains an ML research layer, so this file defines the minimum governance standard even if no model is currently approved for live portfolio use.

## Intended Use

- research on ETF expected returns, volatility, drawdown risk, and benchmark outperformance probability
- offline model comparison against simple baselines
- documented, reproducible experiments with chronological validation

Not intended for:

- automatic deployment into portfolio construction without governance approval
- replacing baseline estimators solely because a model produced an attractive in-sample chart

## Required Metadata

Each ML run should log:

- run ID
- git commit hash
- config hash
- data version
- feature version
- model version
- benchmark
- target definition
- train, validation, and test windows
- leakage checks
- output metrics
- output artifacts

## Approval Criteria

A model is eligible for portfolio construction only if it:

- beats a simple baseline out of sample
- passes leakage checks
- is stable across multiple windows
- has documented failure modes
- has reproducible training
- has a rollback path

## Current Status

Current repository status:

- ML pipeline exists
- ML is disabled by default in code and base configuration
- MLflow logging is supported when available
- governance artifacts are generated
- persisted model artifacts must state whether they were trained on the chronological train split or all eligible data
- approval remains conditional, not automatic

If a concrete model is promoted, this file should be extended with:

- model owner
- training dataset definition
- feature set
- target label construction
- validation summary
- calibration or residual analysis
- fairness or robustness concerns if relevant
- failure modes
- approval decision and approval date
