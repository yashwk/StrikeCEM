# ADR-0007: Master-Coordinated Adaptive Coarse-to-Refine Sampling

## Status
Accepted

## Context
Full electromagnetic sphere or angular domain sweeps at millimeter-wave radar frequencies often require millions of directional evaluations. However, large portions of angular space have near-zero diffuse scattering, while narrow specular lobes and geometric flashpoints exhibit extreme gradients (> 10 dB/degree) requiring fine sub-degree sampling.

Per `FULL_IMPLEMENTATION.md` §11.4:
- A coarse full-domain sweep is executed first across the full angular domain.
- Local peaks and sharp gradients are detected across the coarse result.
- Angular refinement windows $[\theta - W, \theta + W]$ are expanded and overlapping windows merged.
- Mid/fine refinement stages sample the disjoint interest intervals at higher density.
- Adaptive stages cannot be independently discovered by decoupled workers: distributed execution coordinates coarse results before emitting refinement tasks.
- Every derived refinement stage plan records its parent plan hash (`parent_plan_hash`).

## Decision
1. **Adaptive Configuration Schema (`angles.adaptive`):**
   - `coarse_step_deg`: Angular increment for Stage 0 coarse grid sweep.
   - `refine_step_deg`: Angular increment for Stage 1 fine refinement sweep within interest windows.
   - `peak_threshold_dbsm` (optional): Minimum peak level to trigger window refinement.
   - `gradient_threshold_db_per_deg` (default 2.0 dB/deg): Spatial RCS derivative $|RCS(\theta_{i+1}) - RCS(\theta_i)| / \Delta\theta$ threshold.
   - `window_half_width_deg` (default 5.0 deg): Expansion half-width around detected features.
   - `max_refinement_stages` (default 1): Number of hierarchical refinement iterations.

2. **Window Expansion & Disjoint Interval Merging:**
   For all angles $\theta^*$ triggering peak or gradient conditions:
   $$I_k = [\max(\theta_{\min}, \theta^* - W), \min(\theta_{\max}, \theta^* + W)]$$
   Sorted intervals that overlap or touch ($\max(I_k) \ge \min(I_{k+1})$) are unioned into disjoint maximal intervals.

3. **Deterministic Direction Ordering & Deduplication:**
   - Coarse sample directions are kept in the final combined dataset.
   - Refinement directions within disjoint windows strictly omit points already sampled in the coarse stage (distance $< 10^{-6}$ deg).
   - All sample directions maintain canonical 3D Cartesian vectors $\hat{k}$ and stable IDs.

4. **Multi-Stage Provenance & Output Representation:**
   - Sample output tables record the `stage` of each sample (`0` for coarse, `1` for refinement).
   - Refinement plans record `parent_plan_hash`.
   - Result database is complete and self-contained, valid for StrikeEngine ingestion.
