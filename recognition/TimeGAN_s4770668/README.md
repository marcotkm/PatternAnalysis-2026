# TimeGAN for AMZN Limit Order Book Simulation

## Project
COMP3710 — Section 2.7: Time-Series Generation (Hard difficulty).

## Objective
Investigate whether TimeGAN generates realistic AMZN Level 10 limit order book sequences, preserving market structure and temporal behaviour compared with a VAR(1) baseline. Evaluate whether the generated data could support trading-strategy backtesting, and identify its limitations.

## Status
- Feasibility check-off completed; VAR(1) baseline approved.
- Data auditing, chronological splitting and reversible preprocessing implemented and checked.
- VAR(1) fitted and evaluated.
- TimeGAN trained locally on CPU using reconstruction pretraining, supervisor pretraining and 10 epochs of joint training.
- Preliminary validation comparisons completed across five generation seeds, including structural violations, histogram KL divergence and absolute-return lag correlations.
- Three predefined visual case studies produced.
- Depth-heatmap SSIM and two controlled training experiments completed.
- Original 10-epoch TimeGAN checkpoint selected using validation evidence.
- Resource profiling, final held-out test evaluation and final report preparation remain outstanding.

All current comparative results are development results. The test period has not yet been used for final evaluation.

## Feasibility Review
The approved plan is available in [feasibility-review.md](feasibility-review.md). Its contents will be incorporated into this README for final submission.

## Dataset and Preprocessing
- **Dataset:** LOBSTER AMZN Level 10 sample for 21 June 2012.
- **Size:** 269,748 paired order book and message records.
- **Source audit:** No non-finite values, missing levels, negative volumes, price-ladder ordering violations, crossed/locked books or backward timestamps were detected.
- **Splits:** Chronological 70%/15%/15% division by trading-session duration, producing 176,255 training, 34,624 validation and 58,869 test records.
- **Features:** 20 per-level price log returns and 20 aligned log-transformed volumes. Standardisation parameters are fitted on training data only.
- **Windows:** Non-overlapping sequences of 100 event transitions, constructed separately within each split. Incomplete trailing windows are omitted.
- **Verification:** Price reconstruction, volume reconstruction and window-to-price alignment checks passed.

The dataset covers one trading day, limiting conclusions about generalisation across days and market conditions. Raw data remains outside the repository.

## Preliminary VAR(1) Baseline

The baseline predicts the next standardised feature vector from the current vector using a linear transformation and intercept. Generation adds sampled, centred training residual vectors at each step.

### Fitting and Validation

| Measurement | Result |
|---|---:|
| Training transitions | 176,253 |
| Regression design rank | 41/41 |
| Transition matrix spectral radius | 0.97969726 |
| Validation one-step MSE: VAR(1) | 0.192083 |
| Validation one-step MSE: persistence | 0.288920 |

MSE is measured in standardised feature units. Persistence predicts that the next feature vector equals the current one. The spectral radius indicates stable fitted linear dynamics; it does not guarantee valid reconstructed books.

### Generation Smoke Test

Generated 32 sequences of 100 steps, initialised from training observations. The audit excluded each copied initial observation, evaluating 3,168 generated snapshots without clipping volumes or sorting prices.

| Structural check | Percentage of generated snapshots |
|---|---:|
| Crossed books | 5.30% |
| Locked books | 0.00% |
| At least one ask-ordering violation | 93.72% |
| At least one bid-ordering violation | 90.53% |
| At least one negative volume | 3.03% |
| At least one ask inversion greater than $0.01 | 83.84% |
| At least one bid inversion greater than $0.01 | 82.70% |

Failure categories can overlap. The minimum spread was −$0.07035, and the minimum generated volume was approximately −0.994 shares.

**Preliminary interpretation:** Improved one-step prediction does not ensure structurally valid generated order books. The observed inversions are larger than floating-point noise. These findings describe this smoke test, not a final held-out benchmark or a comparison with TimeGAN.

## Model Selection

Three TimeGAN configurations were compared on validation data: the original 10-epoch run, a 10-epoch run with a reduced discriminator learning rate, and a 30-epoch run with that reduced rate.

The original 10-epoch checkpoint (`checkpoints/joint_cpu_02/joint_epoch_010.pt`) was selected, prioritising its lowest observed crossed-book rate. This involves a trade-off: the reduced-rate 10-epoch model had fewer bid-ordering violations and slightly better KL scores. Extending that configuration to 30 epochs substantially increased structural violations despite improving reconstruction loss.

The selection rule was adopted after inspecting validation results, not preregistered. The held-out test period was not used for selection. All configurations retain substantial limitations; selecting one does not establish suitability for backtesting.

## Current Usage
Run from this project directory with the Python environment activated:

```bash
python -m pip install -r requirements.txt

python dataset.py --data-dir /path/to/extracted/AMZN_data

python baseline.py --data-dir /path/to/extracted/AMZN_data
```

The data directory must contain the paired AMZN message and order book CSV files. The current baseline script uses fixed random seeds for its generation smoke test.

Development environment: Python 3.11.15 and NumPy 2.4.6.

## Next Steps
- Evaluate the selected model and VAR baseline against the held-out test period.
- Complete resource profiling and reproduction instructions.
- Add comparative results, training curves and three visual failure analyses.
- Incorporate the feasibility review directly into this README.
- Complete the engineering recommendation, references and submission PDF.

## Artificial Intelligence Usage Disclosure

ChatGPT assisted with planning, documentation drafts and code suggestions for data auditing, preprocessing, the VAR(1) baseline, TimeGAN components and training, evaluation metrics, and visualisation.

I executed the scripts and inspected their outputs, including chronological split checks, price and volume reconstruction, gradient and parameter-update isolation, validation losses, generated-book diagnostics and comparison figures. Experimental claims are based on the recorded outputs. Final held-out evaluation and resource profiling remain pending.