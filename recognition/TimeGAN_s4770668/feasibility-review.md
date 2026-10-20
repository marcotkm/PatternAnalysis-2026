# Feasibility Review — TimeGAN for Limit Order Book Simulation
**Student:** s4770668 | **Project 2.7:** Time-Series Generation | **Difficulty:** Hard | **Tutor Consultation Date & Time:** 20 Oct 2026 9:30am

### 1. User Need, Scope & Acceptance Criteria
**User and purpose:** A quantitative researcher needs realistic synthetic order book sequences for investigating trading strategies. This project connects my Finance and Computer Science studies by evaluating synthetic AMZN Level 10 data; it will not implement a complete trading simulator or establish profitability.

**Engineering dilemma:** Does TimeGAN’s combination of supervised temporal learning and adversarial learning preserve order book constraints and realistic temporal behaviour better than a simpler baseline generator? Based on structural violations, distribution similarity and computational cost, are the generated sequences sufficiently reliable to support trading-strategy backtesting, and what limitations remain?

**Acceptance criteria:**
- Target KL divergence ≤ 0.1 for spread and mid-price return distributions.
- Target SSIM > 0.6 on depth heatmaps under a documented comparison protocol.
- Audit 3–5 representative simulation periods using real-versus-synthetic depth heatmaps. Measure non-positive spreads, incorrect price-ladder ordering and negative volumes before corrective processing.
- Compare volatility clustering, return-tail behaviour and sample diversity against real data and the baseline. Use these findings to recommend whether the generated books are suitable for backtesting and identify remaining limitations.
- Train and generate within one A100 GPU’s available memory; measure peak VRAM, training time and generation throughput to assess computational practicality.

### 2. Model Choice & Course Concepts
- **Dataset:** LOBSTER AMZN Level 10 data from 21 June 2012. Use short event-time sequences containing prices and volumes at all 10 levels on both sides.
- **Configuration:** Split the session chronologically into training, validation and test periods (proposed 70%/15%/15%). Construct windows within each split and fit preprocessing on training data only. Use price log returns and volume normalisation with a verified reconstruction procedure.
- **Model:** TimeGAN (Hard difficulty), supported by Yoon, Jarrett and van der Schaar’s *Time-series Generative Adversarial Networks* (NeurIPS, 2019).
- **Course concepts and rationale:** Representation learning reduces the adversarial learning space; recurrent networks model sequences; supervised temporal learning encourages realistic transitions. These objectives motivate TimeGAN but do not guarantee market constraints, which must be evaluated explicitly.
- **Proposed baseline:** VAR(1) with sampled residual noise provides a simpler temporal generator for comparison.

### 3. Preliminary Feasibility Evidence
- A completed Python audit found **269,748 paired records**, with **40 order book features** and **6 message fields**, covering approximately **09:30–16:00**.
- No non-finite values, missing levels, negative volumes, price-ordering violations, crossed/locked books, backward timestamps or trading-halt events were detected.
- Spreads ranged from **$0.01 to $0.77**, averaging **$0.1309**. There were **8,483 adjacent timestamp repetitions**, retained as separate events.
- These findings support proceeding with preprocessing. Training feasibility and GPU resource consumption remain unmeasured.

### 4. Risks, Budget & Fallback
- **Risks:** GAN instability or mode collapse; invalid reconstructed books; misleading heatmap comparisons; limited generalisation from one trading day.
- **Budget:**  Approximately 27.5 GPU-hours +. Provisionally allocate 0.5 GPU-hours to a pilot, 19 to training and focused comparisons, 3 to evaluation, and 5 as contingency. Revise allocations using measured pilot throughput.
- **Next experiment:** Validate chronological splitting and reconstruction of transformed prices, then run the baseline and a small TimeGAN pilot.
- **Fallback:** Reduce model size, batch size, sequence length or training sample count while retaining all 10 levels. If performance remains inadequate, preserve the baseline comparison and analyse TimeGAN’s failures honestly. Discuss any architecture change with the tutor.

*AI assistance: ChatGPT supported planning, drafting and the audit script; I executed the data audit on Rangpur.*