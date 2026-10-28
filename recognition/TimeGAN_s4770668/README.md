# TimeGAN for AMZN Limit Order Book Simulation

## Project
COMP3710 — Section 2.7: Time-Series Generation (Hard difficulty).

## Objective
Investigate whether TimeGAN generates realistic AMZN Level 10 limit order book sequences, preserving market structure and temporal behaviour compared with a VAR(1) baseline. Evaluate whether the generated data could support trading-strategy backtesting, and identify its limitations.

## Status
- Feasibility check-off completed; VAR(1) baseline approved.
- Data auditing, chronological splitting and reversible preprocessing completed.
- VAR(1) baseline and TimeGAN training pipeline implemented and checked.
- Structural validity, distributional fidelity, temporal dependence and depth
  SSIM evaluated across five generation seeds.
- Three predefined visual case studies produced.
- The original CPU-trained 10-epoch TimeGAN checkpoint was selected using
  validation evidence and subsequently evaluated against the test period.
- CPU generation throughput and A100 joint-training runtime and memory measured.
- Additional GPU experiments investigate training duration, discriminator
  update thresholds and update frequency.
- Final experiment comparison, report consolidation and reproducibility
  checks are in progress.

The original model selection preceded test evaluation. Additional GPU tuning was undertaken after inspecting those test results, using validation data for subsequent comparisons. These later experiments are exploratory; the test period cannot be described as untouched throughout development.

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

## Evaluation Protocol

### Shared Generation Setup

Each model generates 32 sequences of 100 new event transitions for each
of five generation seeds (42–46), giving 16,000 generated snapshots per
model. Initial price anchors are fixed observations selected from the
training period and shared between models.

VAR(1) generation conditions on a training feature state. TimeGAN generates
features from random noise and uses the shared price anchors only during
price reconstruction. This difference limits the equivalence of their
conditioning.

Generated sequences are compared against chronological validation or test
reference windows. They are independent simulations, not forecasts paired
with the observed future of a particular reference window.

Reported mean ± sample standard deviation describes variation across
generation seeds for a fixed trained checkpoint. It does not measure
variation across independently trained models or constitute a confidence
interval. Snapshots within a generated sequence are temporally dependent.

### Structural Validity

Audits use reconstructed outputs without sorting price levels, clipping
volumes or repairing spreads. Each snapshot is checked for:

- Crossed books: best bid exceeds best ask.
- Locked books: best bid equals best ask.
- Ask-ladder violations: adjacent ask prices are not strictly increasing.
- Bid-ladder violations: adjacent bid prices are not strictly decreasing.
- Negative volume at any of the 20 depth positions.

Violation categories can overlap. Additional diagnostics measure minimum
spread, minimum adjacent price gaps and inversions exceeding $0.01.

### Distributional Fidelity

Spread and mid-price log-return distributions are compared using
KL(real reference || synthetic), expressed in nats.

The primary histogram protocol uses 50 equal-width bins spanning the
training-data range, plus two tail bins. Histogram probabilities receive
additive smoothing of 1e-6 and are renormalised. Bin boundaries are shared
between real and synthetic data and are not fitted to the test period.

Sensitivity checks with 25 and 100 interior bins show that KL values
depend materially on binning. In particular, real event-time returns have
a large mass at exactly zero, whereas the continuous generators do not
reproduce that mass. The 50-bin protocol is retained consistently rather
than choosing the most favourable bin count.

### Temporal Dependence

Volatility clustering is assessed using correlations of absolute
mid-price log returns at event lags 1, 5, 10 and 20. Lagged pairs are
constructed within sequences, without joining independent windows.

The mean absolute difference between real and synthetic correlations
summarises mismatch across these four lags. These are event-time
diagnostics, not correlations at fixed elapsed-time intervals.

### Depth-Heatmap SSIM

Depth is represented separately for asks and bids as 10-level × 100-step
arrays. Volumes are log-transformed and scaled using a training-derived
maximum. Copies used for SSIM are clipped to [0, 1], with the clipped
fraction reported separately. Structural audits retain the raw outputs.

SSIM uses a 7 × 7 window and data_range=1. Real and synthetic windows are
paired independently using a reproducible pairing procedure shared
between models. A separate real-versus-real comparison provides context.

The reported mean averages the ask and bid scores. Standard deviations
are calculated across generation-seed/pairing repetitions.

Because these are independent sequences, SSIM is sensitive to pairing
and temporal alignment. The low real-versus-real score demonstrates that
the absolute target of 0.6 is difficult to interpret under this protocol.
It does not establish that a synthetic model meets the target.

### Qualitative Diagnostics

Three predefined validation cases compare real and independently generated
sequences using depth heatmaps, mid-price returns, spreads and minimum
adjacent price gaps. Shared scales within each figure support comparison.

Depth heatmaps alone cannot establish price-ladder validity because their
vertical axes represent level rank rather than the actual generated price.
The accompanying price-gap diagnostics expose inversions that may be
visually hidden in the depth plots.

## Initial Model Selection and Subsequent Experiments

Three TimeGAN configurations were compared on validation data: the original 10-epoch run, a 10-epoch run with a reduced discriminator learning rate, and a 30-epoch run with that reduced rate.

The original 10-epoch checkpoint (`checkpoints/joint_cpu_02/joint_epoch_010.pt`) was selected, prioritising its lowest observed crossed-book rate. This involves a trade-off: the reduced-rate 10-epoch model had fewer bid-ordering violations and slightly better KL scores. Extending that configuration to 30 epochs substantially increased structural violations despite improving reconstruction loss.

The selection rule was adopted after inspecting validation results, not preregistered. The held-out test period was not used for initial selection. All configurations retain substantial limitations; selecting one does not establish suitability for backtesting.

After evaluating the initially selected model on the test period, further experiments were conducted on an NVIDIA A100 GPU. These examined 50 epochs of joint training and a discriminator loss threshold of 0.15. An additional experiment changes the discriminator update interval.

The 50-epoch experiments improved reconstruction error without consistently improving generated-book validity or distributional fidelity. Subsequent checkpoint comparisons use validation data. The original CPU checkpoint is retained as the reference model while these exploratory experiments are assessed.

## TimeGAN Architecture and Training

The model generates sequences of 100 event transitions, each represented
by 40 standardised features: 20 per-level price log returns and 20
log-transformed depth volumes.

### Architecture

Five recurrent networks are implemented using GRUs with a hidden
dimension of 24.

| Component | Purpose | GRU layers | Output |
|---|---|---:|---|
| Embedder | Encode real feature sequences | 2 | 24-dimensional latent vector, sigmoid |
| Recovery | Decode latent sequences into features | 2 | 40-dimensional feature vector, linear |
| Generator | Transform Gaussian noise into latent sequences | 2 | 24-dimensional latent vector, sigmoid |
| Supervisor | Model next-step latent dynamics | 1 | 24-dimensional latent vector, sigmoid |
| Discriminator | Distinguish real and generated latent sequences | 2 | One logit per time step |

The recovery output is linear because standardised features are not
restricted to [0, 1]. Generation follows generator → supervisor → recovery.
Noise has 24 independent standard-normal components per time step.

Generated prices are reconstructed from per-level log returns using
training-based initial price anchors. The model does not explicitly
condition its generated returns on those anchors. No sorting of prices,
spread repair or volume clipping is applied to the structural audit.

### Training Stages

1. **Reconstruction pretraining:** train the embedder and recovery network
   for 20 epochs to reconstruct real training windows. The selected
   checkpoint achieved validation feature MSE of 0.385112, compared with
   0.641442 for a training-mean predictor.
2. **Supervisor pretraining:** freeze the embedder and train the supervisor
   for 20 epochs to predict the next latent state. The best validation
   latent MSE was 0.002671, compared with 0.003277 for latent persistence.
3. **Joint training:** alternate discriminator, generator/supervisor and
   embedder/recovery updates. The initial reference model used 10 epochs;
   subsequent experiments examined longer training and altered
   discriminator optimisation.

Joint training uses Adam, batch size 32 and gradient-norm clipping at 1.0.
The original learning rate is 0.001 for each optimiser. There are 55
training batches per epoch; the incomplete final batch is omitted.

### Joint Objectives

The discriminator loss sums binary cross-entropy terms for real latent
sequences, raw generator outputs and supervisor-refined generator outputs.

The generator/supervisor objective combines:

- Adversarial losses for raw and refined generated latent sequences.
- 100 times the square root of the supervised next-step latent MSE,
  with a small numerical stabiliser.
- 100 times a moment-matching loss comparing real and synthetic batch
  means and standard deviations at each time position and feature.

The supervised temporal term is calculated on real encoded sequences.
It directly trains the supervisor; generator learning comes through
the adversarial and moment-matching paths.

The embedder/recovery objective combines 10 times the square root of
feature reconstruction MSE, with a small numerical stabiliser, and
0.1 times the supervised latent MSE.

Inactive networks have frozen parameters during each update, while
gradients can still propagate through them to the active networks.
Validation reconstruction uses only the embedder and recovery network;
it is not a direct measure of synthetic generation quality.

None of these objectives explicitly guarantees positive spreads,
ordered price ladders or non-negative reconstructed volumes. Their
preservation is therefore assessed empirically.

## Test Results for the Initially Selected Model

The original CPU-trained checkpoint,
`checkpoints/joint_cpu_02/joint_epoch_010.pt`, was selected using validation
results before this test evaluation. The table below reports that fixed
model; it does not describe the later GPU checkpoints.

The test split contains 58,869 records, providing 588 complete windows
and 58,800 reference transitions.

### Distributional and Temporal Results

| Metric | VAR(1) | TimeGAN | Target |
|---|---:|---:|---|
| Spread KL | 0.481982 ± 0.048793 | 0.316629 ± 0.062030 | ≤ 0.1 |
| Mid-price return KL | 0.515427 ± 0.015366 | 0.687588 ± 0.028828 | ≤ 0.1 |
| Absolute-return correlation error | 0.077968 ± 0.012651 | 0.261913 ± 0.008778 | Lower is better |
| Mean depth SSIM | 0.119944 ± 0.006698 | 0.126209 ± 0.009634 | > 0.6 |

Real-versus-real mean depth SSIM was 0.123889 ± 0.012100.
SSIM variability includes changes in reference pairing.

TimeGAN improved spread KL relative to VAR(1), but had worse return KL
and temporal correlation error. Neither model met the KL or SSIM targets.

### Structural Results for the Generated Samples

| Violation | VAR(1) | TimeGAN |
|---|---:|---:|
| Crossed books | 5.77% ± 1.17 pp | 1.46% ± 1.36 pp |
| Locked books | 0.00% ± 0.00 pp | 0.00% ± 0.00 pp |
| Ask ordering violations | 93.86% ± 1.10 pp | 76.44% ± 4.13 pp |
| Bid ordering violations | 91.49% ± 0.94 pp | 60.02% ± 4.30 pp |
| Negative-volume books | 2.87% ± 0.34 pp | 0.00% ± 0.00 pp |

These structural rates describe the same fixed generated samples used
in the validation and test-reference comparisons. They are unchanged
when the reference split changes and are not a separate test-conditioned
generation experiment.

TimeGAN reduced structural violations relative to VAR(1), but price
ladders remained invalid in many snapshots. Zero observed negative-volume
violations does not provide a mathematical guarantee of non-negativity.

### Preliminary Engineering Recommendation

The evaluated raw generators are not sufficiently reliable for
trading-strategy backtesting. Invalid price ladders and crossed books
could distort execution opportunities, while return-distribution and
temporal mismatches could distort simulated risk.

The results support using this implementation as an experimental
prototype for studying failure modes. Further work would need to address
structural validity and validate behaviour across additional trading days.

## Computational Resources

The model contains 36,377 trainable parameters across five components. Generation uses the generator, supervisor and recovery networks, containing 20,200 parameters in total.

### CPU Generation Benchmark

Measurements were taken on an Apple M1 MacBook Pro with 8 GB unified memory, Python 3.11.15, NumPy 2.4.6 and PyTorch 2.14.1. PyTorch used two CPU threads. Each timed batch generated 32 sequences of 100 transitions. Three warm-up runs preceded 30 measured repetitions.

| Measurement | VAR(1) | TimeGAN |
|---|---:|---:|
| Median batch generation time | 1.725 ms | 13.011 ms |
| 95th-percentile batch time | 2.007 ms | 13.317 ms |
| Sequences per second | 18,556 | 2,459 |

Timing includes random sampling and standardised feature generation. It excludes loading, fitting, residual-bank centring, price reconstruction and evaluation. VAR uses NumPy float64; TimeGAN uses PyTorch float32.

Combined process peak resident memory was 732.4 MiB. This includes both models and supporting data and is not a per-model measurement or GPU VRAM.

### GPU Joint Training

Rangpur experiments used an NVIDIA A100 PCIe 40 GB, PyTorch 2.13.0 and
CUDA runtime 13.0.

| Run | Epochs | Joint-stage wall time | Peak allocated memory | Peak reserved memory |
|---|---:|---:|---:|---:|
| Original update schedule | 10 | 11.314 s | 58.68 MiB | 66.00 MiB |
| Original update schedule | 50 | 57.704 s | 58.68 MiB | 66.00 MiB |
| Discriminator threshold 0.15 | 50 | 55.047 s | 87.81 MiB | 100.00 MiB |
| Threshold 0.15; discriminator update every second batch | 50 | 53.110 s | 88.98 MiB | 100.00 MiB |

Wall time includes joint training, validation, fixed-noise monitoring and checkpoint saving. It excludes data preparation and the reconstruction and supervisor pretraining previously performed on CPU. Memory figures come from PyTorch's CUDA allocator and do not represent total device usage.

GPU experiments produce separate checkpoints from the original CPU model. Their resource measurements must not be presented as measurements of training that CPU checkpoint.

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
- Complete the validation comparison of the GPU update-frequency experiment.
- Consolidate validation and test results into clearly separated tables.
- Add the model architecture, training objectives and evaluation protocol.
- Include training curves and three visual failure analyses.
- Incorporate the approved feasibility review directly into this README.
- Complete reproduction instructions and verify them.
- Finalise the engineering recommendation, references and submission PDF.

## Artificial Intelligence Usage Disclosure

ChatGPT assisted with planning, documentation drafts and code suggestions for data auditing, preprocessing, the VAR(1) baseline, TimeGAN components and training, evaluation metrics, and visualisation.

I executed the scripts and inspected their outputs, including chronological split checks, price and volume reconstruction, gradient and parameter-update isolation, validation losses, generated-book diagnostics and comparison figures. Test evaluation of the initially selected model and resource profiling have been completed. Subsequent GPU experiments and final reporting remain in progress.