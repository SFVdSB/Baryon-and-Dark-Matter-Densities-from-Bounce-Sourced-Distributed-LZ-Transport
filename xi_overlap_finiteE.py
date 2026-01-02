
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute finite-energy χ amplitude at the wall crossing and derive lambda_tr_eff = g_portal * <f_chi(xi*, E)>_E.
Then, for convenience, output the implied Landau–Zener probability and n_LSP/n_B using the same bounce profile.
This removes the E=0 WKB over-suppression and is appropriate for a moderate–thick wall.

Params file fields (see template):
- profile_csv, columns, prefer_csv_R_peak
- y_chi, g_portal
- E_dist: {type: "MB_1D", T_reh, E_min, E_max_mode, E_max_fraction}
- xi_side, xi_cutoff_mult
- nE
"""

import json, math, numpy as np, pandas as pd, argparse

def pick(df, keys):
    low = {c.lower(): c for c in df.columns}
    for k in keys:
        if k.lower() in low: return low[k.lower()]
    # fallback exact
    for c in df.columns:
        if c in keys: return c
    return None

def spline_abs(x, y):
    # linear interp + finite diff derivative; avoid SciPy dependency
    x = np.asarray(x); y = np.abs(np.asarray(y))
    order = np.argsort(x)
    x = x[order]; y = y[order]
    def f(z):
        return np.interp(z, x, y)
    def df(z, h=1e-3):
        return (f(z+h)-f(z-h))/(2*h)
    return f, df

def find_root(f, a, b, ngrid=4001):
    xs = np.linspace(a, b, ngrid)
    vals = f(xs)
    s = np.sign(vals)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) > 0:
        i = idx[0]
        lo, hi = xs[i], xs[i+1]
        flo, fhi = vals[i], vals[i+1]
        for _ in range(80):
            mid = 0.5*(lo+hi)
            fm = f(mid)
            if np.sign(flo)*np.sign(fm) <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return 0.5*(lo+hi)
    # fallback: min |vals|
    j = int(np.argmin(np.abs(vals)))
    return xs[j]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default="xi_overlap_finiteE_report.json")
    args = ap.parse_args()

    P = json.load(open(args.params,"r"))
    df = pd.read_csv(P["profile_csv"])
    # columns
    r_col   = P.get("columns",{}).get("r")   or pick(df, ["r","radius"])
    Phi_col = P.get("columns",{}).get("Phi") or pick(df, ["Phi","phi1","Phi1"])
    phi_col = P.get("columns",{}).get("phi") or pick(df, ["phi","phi2"])
    if r_col is None or Phi_col is None or phi_col is None:
        raise RuntimeError("Need r, Phi, phi columns in profile CSV")
    # R_peak / FWHM
    Rpk = None
    if P.get("prefer_csv_R_peak", False):
        Rpk = pick(df, ["R_peak","r_peak","rwall","r_wall"])
        if Rpk is not None:
            Rpk = float(np.nanmean(df[Rpk]))
    if Rpk is None:
        Rpk = float(np.median(df[r_col]))
    FWHM_col = pick(df, ["w_FWHM","fwhm","width"])
    FWHM = float(np.nanmean(df[FWHM_col])) if FWHM_col else max(1.0, 0.1*(df[r_col].max()-df[r_col].min()))

    r   = df[r_col].to_numpy(float)
    Phi = df[Phi_col].to_numpy(float)
    phi = df[phi_col].to_numpy(float)

    xi = r - Rpk
    # choose SFV side
    if P.get("xi_side","SFV_positive").lower() == "sfv_negative":
        xi = -xi

    # splines
    s_phi, ds_phi = spline_abs(xi, phi)
    s_Phi, ds_Phi = spline_abs(xi, Phi)

    ychi = float(P.get("y_chi", 1.0))
    g    = float(P.get("g_portal", 1.0))

    # Find crossing xi* for Delta = y_B|phi| - y_chi|Phi|; set y_B=ychi for geometry-only crossing
    def Delta(z): return s_phi(z) - s_Phi(z)
    L = max(3.0*float(P.get("xi_cutoff_mult",3.0))*FWHM, 0.4*(xi.max()-xi.min()))
    xi_star = find_root(Delta, -L, +L)

    # Local values
    phi_star = s_phi(xi_star)
    dphi = ds_phi(xi_star)
    dPhi = ds_Phi(xi_star)
    Delta_prime = dphi - dPhi  # with y_B=y_chi=1; otherwise scale

    # Finite-energy WKB: kappa(ξ,E) = sqrt( max(m_chi(ξ)^2 - E^2, 0) ), with m_chi = ychi*|Phi|.
    def kappa(z, E):
        m = ychi*s_Phi(z)
        if m <= E: return 0.0
        return math.sqrt(max(m*m - E*E, 0.0))

    # Energy grid and weights
    Ed = P.get("E_dist",{})
    T  = float(Ed.get("T_reh", 0.5))
    Emin = float(Ed.get("E_min", 0.0))
    mchi_max = ychi*np.max(np.abs(Phi))
    if Ed.get("E_max_mode","fraction_of_mchi_max") == "absolute":
        Emax = float(Ed.get("E_max", mchi_max*0.9))
    else:
        frac = float(Ed.get("E_max_fraction", 0.9))
        Emax = frac * mchi_max
    nE = int(P.get("nE", 100))
    Es = np.linspace(Emin, Emax, nE)

    # 1D MB weights ~ exp(-E/T); normalize
    w = np.exp(-(Es - Emin)/max(T,1e-6))
    w /= w.sum()

    # Compute finite-energy amplitude at xi* via WKB integral I(E)=∫_{xi*}^{xi_max} kappa dxi
    xi_max = float(P.get("xi_cutoff_mult",3.0))*FWHM
    # Build a fine grid to integrate
    def I_of_E(E):
        # sample points from xi* to xi_max on available domain
        xgrid = np.linspace(xi_star, xi_max, 1200)
        k = np.array([kappa(z,E) for z in xgrid])
        return float(np.trapz(k, xgrid))

    Ivals = np.array([I_of_E(E) for E in Es])
    f_wall = np.exp(-Ivals)  # amplitude at the crossing
    lam_E = g * f_wall       # effective mixing at the crossing
    lam_eff = float(np.sum(lam_E * w))

    # Landau–Zener with v_w placeholder (report a few values)
    def delta_of(lam, v_w):
        return (lam*lam * phi_star*phi_star) / (max(v_w,1e-8)*max(abs(Delta_prime),1e-8))
    def P_of(lam, v_w): return 1.0 - math.exp(-2.0*math.pi*delta_of(lam, v_w))

    vws = [0.10, 0.30, 0.50]
    Plist = {str(v): P_of(lam_eff, v) for v in vws}
    ratio = {str(v): (1.0-Plist[str(v)])/max(Plist[str(v)],1e-12) for v in vws}

    out = {
        "inputs": {
            "csv": P["profile_csv"],
            "columns_used": {"r": r_col, "Phi": Phi_col, "phi": phi_col},
            "R_peak": Rpk, "FWHM": FWHM,
            "y_chi": ychi, "g_portal": g,
            "xi_star": xi_star, "phi_star": phi_star, "Delta_prime": Delta_prime
        },
        "energy_grid": {"Emin": Emin, "Emax": Emax, "nE": nE, "T_reh": T},
        "amplitude": {
            "I_vals_sample": [float(Ivals[0]), float(Ivals[nE//2]), float(Ivals[-1])],
            "f_wall_sample": [float(f_wall[0]), float(f_wall[nE//2]), float(f_wall[-1])]
        },
        "lambda_tr_eff": {
            "per_E_mean": float(np.mean(lam_E)),
            "per_E_weighted_MB": lam_eff
        },
        "transport_preview": {
            "P_chi_to_B_vs_vw": Plist,
            "nLSP_over_nB_vs_vw": ratio
        },
        "notes": "Finite-energy WKB at the crossing xi*. Use lambda_tr_eff = per_E_weighted_MB. Then run transport_from_profile.py with that lambda and your chosen v_w."
    }
    with open(args.out,"w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}. <lambda_tr_eff>_MB ≈ {lam_eff:.6g}. For v_w=0.3 => P≈{Plist['0.3']:.4f}, nLSP/nB≈{ratio['0.3']:.2f}")

if __name__ == "__main__":
    main()
