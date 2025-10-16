
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute χ→B conversion probability across the SFV→brane wall directly from the O(4) bounce profile.
Implements a Landau–Zener estimate with all profile-dependent pieces taken from background_profile.csv.
Outputs both console text and a JSON report.

Usage:
  python transport_from_profile.py --params /path/to/transport_params.json

The JSON file should include:
  - profile_csv: path to CSV with columns for radius r and background fields (phi, Phi)
  - columns: optional explicit names for r, phi, Phi (auto-detected if empty)
  - R0: wall center in r-tilde units; used to define xi = r - R0
  - w: (optional) wall thickness scale for search window
  - couplings: { y_B, y_chi, lambda_tr_eff }
  - wall: { v_w }
  - average_over_k: bool (if True, applies a mild dispersion correction F<=1; here we keep F=1 as placeholder)
"""

import json, argparse, sys, math, numpy as np, pandas as pd

def _to_float(x):
    try:
        y = float(x)
        if math.isfinite(y):
            return y
        return float('nan')
    except Exception:
        return float('nan')

def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k,v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (int, float, np.floating, np.integer)):
        y = _to_float(obj)
        # replace NaN with a sentinel
        return 0.0 if math.isnan(y) else y
    else:
        return obj

from dataclasses import dataclass

# Try SciPy for smooth derivatives; fall back to numpy finite differences
try:
    from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

@dataclass
class Columns:
    r: str
    phi: str
    Phi: str


def autodetect_columns(df):
    names = list(df.columns)
    # Prefer exact canonical names if available
    if "r" in names and "phi" in names and "Phi" in names:
        return "r","phi","Phi"
    # Case-insensitive fallbacks without conflating Phi with phi
    def find_exact(name):
        for c in names:
            if c == name:
                return c
        return None
    def find_case_insensitive(target):
        for c in names:
            if c.lower() == target.lower():
                return c
        return None
    r = find_exact("r") or find_case_insensitive("r") or find_case_insensitive("radius") or names[0]
    ph = find_exact("phi") or find_case_insensitive("phi")
    PH = find_exact("Phi") or find_case_insensitive("Phi")
    # As a last resort, try to guess distinct scalar columns
    if PH is None:
        for c in names:
            if c.lower().startswith("phi") and c != ph:
                PH = c
                break
    return r, ph, PH


def spline_abs(x, y):
    # Spline of absolute value |y|; ensure monotone x
    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.abs(np.asarray(y)[order])
    if SCIPY_OK and len(x) > 8:
        # smoothing spline with small smoothing to tame noise
        s = UnivariateSpline(x, y, s=1e-6 if len(x) > 50 else 0, k=3)
        ds = s.derivative(1)
        return s, ds
    else:
        # linear interp + finite-diff derivative
        def f(z):
            return np.interp(z, x, y)
        def df(z, h=1e-3):
            return (f(z+h)-f(z-h))/(2*h)
        return f, df

def find_root(f, a, b, ngrid=2001):
    # robust sign-change or min-abs search
    xs = np.linspace(a, b, ngrid)
    vals = f(xs)
    # sign-change
    s = np.sign(vals)
    sign_change = np.where(np.diff(s) != 0)[0]
    if len(sign_change) > 0:
        i = sign_change[0]
        # bisection
        left, right = xs[i], xs[i+1]
        fl, fr = vals[i], vals[i+1]
        for _ in range(80):
            mid = 0.5*(left+right)
            fm = f(mid)
            if np.sign(fl)*np.sign(fm) <= 0:
                right, fr = mid, fm
            else:
                left, fl = mid, fm
        return 0.5*(left+right)
    # else: pick min of |vals|
    j = int(np.argmin(np.abs(vals)))
    return xs[j]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="JSON parameter file")
    ap.add_argument("--out", default="transport_result.json")
    args = ap.parse_args()

    with open(args.params, "r") as f:
        P = json.load(f)

    csv_path = P["profile_csv"]
    df = pd.read_csv(csv_path)
    # Allow awkward CSVs with unnamed first column = index; drop if purely index-like
    if df.shape[1] >= 2 and df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    # Columns
    r_name = P.get("columns", {}).get("r") or None
    phi_name = P.get("columns", {}).get("phi") or None
    Phi_name = P.get("columns", {}).get("Phi") or None
    if not (r_name and phi_name and Phi_name):
        auto_r, auto_phi, auto_Phi = autodetect_columns(df)
        r_name = r_name or auto_r
        phi_name = phi_name or auto_phi
        Phi_name = Phi_name or auto_Phi

    if not (r_name and phi_name and Phi_name):
        raise RuntimeError(f"Could not detect columns. Found: {list(df.columns)}")

    r = df[r_name].to_numpy(dtype=float)
    phi = df[phi_name].to_numpy(dtype=float)
    Phi = df[Phi_name].to_numpy(dtype=float)

    # Prefer CSV-provided R_peak if requested
    R0 = None
    if P.get("prefer_csv_R_peak", False):
        # try to use any column that looks like R_peak
        for name in df.columns:
            if name.lower() in ("r_peak","rpeak","r_pk","rpk"):
                try:
                    R0 = float(df[name].iloc[0])
                    break
                except Exception:
                    pass
    if R0 is None:
        R0 = float(P.get("R0", np.median(r)))
    w  = float(P.get("w", (np.max(r)-np.min(r))/10.0))
    xi = r - R0

    # Spline |phi| and |Phi|
    s_phi, ds_phi = spline_abs(xi, phi)
    s_Phi, ds_Phi = spline_abs(xi, Phi)

    # Build Delta(xi) = y_B*|phi| - y_chi*|Phi|
    yB   = float(P["couplings"]["y_B"])
    ychi = float(P["couplings"]["y_chi"])
    def Delta(z):
        return yB*s_phi(z) - ychi*s_Phi(z)

    # Root near xi=0 within a few widths
    xi_min, xi_max = float(np.min(xi)), float(np.max(xi))
    L = min(max(3.0*w, 0.4*(xi_max-xi_min)), 0.9*(xi_max - 0.0), 0.9*(0.0 - xi_min))
    xi_star = find_root(Delta, -L, +L)

    # Slopes at crossing
    dphi = float(np.asarray(ds_phi(xi_star)).reshape(-1)[0])
    dPhi = float(np.asarray(ds_Phi(xi_star)).reshape(-1)[0])
    Delta_prime = yB*dphi - ychi*dPhi

    # Mixing at crossing: m_mix = lambda_tr_eff * phi(xi*)
    lam_tr = float(P["couplings"]["lambda_tr_eff"])
    v_w    = float(P["wall"]["v_w"])
    phi_star = float(np.asarray(s_phi(xi_star)).reshape(-1)[0])

    # LZ adiabaticity parameter δ and probability
    # Optional F(k) averaging -> set to 1 for now
    F_k = 1.0
    den = max(v_w,1e-8) * max(abs(Delta_prime), 1e-8)
    delta = (lam_tr**2 * phi_star**2) / den * F_k
    P_conv = 1.0 - math.exp(-2.0*math.pi*delta)
    ratio = (1.0 - P_conv) / max(P_conv, 1e-12)

    # Debug prints for types/values
    print("[DEBUG] xi_star:", xi_star, type(xi_star))
    print("[DEBUG] phi_star:", phi_star, type(phi_star))
    print("[DEBUG] Delta_prime:", Delta_prime, type(Delta_prime))
    print("[DEBUG] delta:", delta, type(delta))
    print("[DEBUG] P_conv:", P_conv, type(P_conv))
    print("[DEBUG] ratio:", ratio, type(ratio))

    # Pack results
    out = {
        "inputs": {
            "csv": csv_path,
            "columns_used": {"r": r_name, "phi": phi_name, "Phi": Phi_name},
            "R0": R0, "w": w,
            "y_B": yB, "y_chi": ychi,
            "lambda_tr_eff": lam_tr,
            "v_w": v_w
        },
        "derived": {
            "xi_star": xi_star,
            "phi_star": phi_star,
            "Delta_prime_at_star": Delta_prime
        },
        "results": {
            "delta_LZ": delta,
            "P_chi_to_B": P_conv,
            "n_LSP_over_n_B": ratio
        },
        "notes": "Replace lambda_tr_eff and v_w with first-principles values (portal overlap and wall dynamics). Everything else comes directly from the bounce profile."
    }

    out_clean = _sanitize_for_json(out)
    with open(args.out, "w") as f:
        json.dump(out_clean, f, indent=2, allow_nan=False)

    # Console summary
    print("=== Transport from Bounce Profile ===")
    print(f"CSV: {csv_path}")
    print(f"Columns: r={r_name}, phi={phi_name}, Phi={Phi_name}")
    print(f"Wall center R0={R0:.6g}, search half-window L={L:.6g}")
    print(f"Crossing xi*={out['derived']['xi_star']:.6g}")
    print(f"|phi(xi*)|={out['derived']['phi_star']:.6g}")
    print(f"Delta'(xi*)={out['derived']['Delta_prime_at_star']:.6g}")
    print(f"delta_LZ={out['results']['delta_LZ']:.6g}")
    print(f"P(chi->B)={out['results']['P_chi_to_B']:.6g}")
    print(f"n_LSP/n_B={out['results']['n_LSP_over_n_B']:.6g}")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
