# SFV/dSB Two-Channel Transport (Baryon–DM Ratio) — Reproducibility README

This repository contains the scripts and parameter files used to compute the **two-channel transport ratio**
between a bulk dark-matter-like species **χ (LSP)** and **baryons B** during bubble-wall passage in the
SFV/dSB framework.

The key quantity the wall-transport layer naturally outputs is the **number ratio**
\[
\frac{n_\chi}{n_B} = \frac{1-P_{\chi\to B}}{P_{\chi\to B}},
\]
while cosmology (Planck) commonly quotes the **energy-density ratio**
\[
\frac{\rho_{\rm DM}}{\rho_b} \simeq \frac{\Omega_c}{\Omega_b} \approx 5.36.
\]
Because \( \rho = m n\), these are related by
\[
\frac{\rho_{\rm DM}}{\rho_b} = \frac{m_\chi}{m_p}\frac{n_\chi}{n_B},
\qquad
\Rightarrow
\qquad
\left(\frac{n_\chi}{n_B}\right)_{\rm target}
=
\left(\frac{\rho_{\rm DM}}{\rho_b}\right)_{\rm Planck}\frac{m_p}{m_\chi}.
\]

> **Important:** early notes introduced an “O(1) settling factor ~0.94”. This arose primarily from targeting
> the wrong observable (comparing a number ratio to an energy ratio) when \(m_\chi \neq m_p\). With the
> correct number target, that factor disappears.

---

## Directory / Files

Core scripts:

- `transport_from_profile.py`  
  Computes the wall conversion probability **P(χ→B)** and the implied **number ratio** \(n_\chi/n_B\) using
  a local Landau–Zener estimator at the crossing extracted from the bounce profile.

- `extended_LZ_lambda_v2.py`  
  Computes a **distributed overlap integral** (finite-energy kernel across ξ) and outputs a
  **local-equivalent effective mixing** `lambda_eff_equiv` that can be used in the local estimator.

- `xi_overlap_finiteE_v2.py`  
  Diagnostic script that computes a finite-energy WKB-like overlap and prints a thermally weighted
  `<lambda_tr_eff>_MB`. This is **not** directly the same as `transport_from_profile.py`’s local mixing
  parameter unless mapped consistently.

Parameter files (examples):

- `transport_params.json`  
  Inputs for `transport_from_profile.py` (CSV path, wall center, couplings, etc.)

- `extended_LZ_lambda_params.json`  
  Inputs for `extended_LZ_lambda_v2.py` (target ratio, energy distribution, overlap settings, etc.)

- `xi_overlap_finiteE_params.json`  
  Inputs for `xi_overlap_finiteE_v2.py` (energy distribution, side, cutoffs, etc.)

Typical outputs:

- `transport_result.json`
- `extended_LZ_lambda_out.json` (or similar)
- `xi_overlap_finiteE_report.json`

---

## Dependencies

Python 3.9+ recommended. The scripts typically require:

- `numpy`
- `scipy`
- `json` (stdlib)

Install (example):

```bash
pip install numpy scipy
