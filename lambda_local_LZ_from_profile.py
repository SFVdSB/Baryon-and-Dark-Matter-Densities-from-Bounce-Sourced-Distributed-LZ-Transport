#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# lambda_local_LZ_from_profile.py  (fixed)
#
# Thick-brane → local LZ mapping:
#   1) Find crossing xi* from your bounce CSV.
#   2) Build finite-energy WKB amplitude f_chi(ξ; E) along the wall-normal.
#   3) Average f_chi(ξ;E)*phi(ξ) over a Gaussian window around xi*.
#   4) Map extended mixing to local LZ constant: lambda_eff = g * [⟨f·phi⟩ / phi(xi*)].
#
# Params JSON must provide:
# {
#   "profile_csv": "background_profile.csv",
#   "columns": {"r":"r","Phi":"Phi","phi":"phi"},
#   "prefer_csv_R_peak": true,
#   "y_chi": 1.0,
#   "g_portal": 2.313019,
#   "xi_side": "SFV_positive",
#   "E": 0.2,
#   "window_sigma_factor": 1.0
# }

import json, math, numpy as np, pandas as pd, argparse

def pick(df, keys):
    low = {c.lower(): c for c in df.columns}
    for k in keys:
        if k.lower() in low: return low[k.lower()]
    for c in df.columns:
        if c in keys: return c
    return None

def spline_abs(x, y):
    x = np.asarray(x); y = np.abs(np.asarray(y))
    order = np.argsort(x); x = x[order]; y = y[order]
    def f(z): return np.interp(z, x, y)
    def df(z, h=1e-3): return (f(z+h)-f(z-h))/(2*h)
    return f, df

def find_root(f, a, b, ngrid=4001):
    xs = np.linspace(a, b, ngrid)
    v = f(xs)
    s = np.sign(v)
    idx = np.where(np.diff(s)!=0)[0]
    if len(idx)>0:
        lo, hi = xs[idx[0]], xs[idx[0]+1]
        for _ in range(80):
            mid = 0.5*(lo+hi); fm = f(mid)
            if np.sign(f(lo))*np.sign(fm) <= 0: hi = mid
            else: lo = mid
        return 0.5*(lo+hi)
    return xs[int(np.argmin(np.abs(v)))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default="lambda_local_LZ_report.json")
    args = ap.parse_args()

    P = json.load(open(args.params,"r"))

    df = pd.read_csv(P["profile_csv"])
    r_col   = P["columns"]["r"]
    Phi_col = P["columns"]["Phi"]
    phi_col = P["columns"]["phi"]

    # Center at R_peak if present, else median(r)
    Rpk = None
    if P.get("prefer_csv_R_peak", True):
        name = pick(df, ["R_peak","r_peak","r_wall","rwall"])
        if name is not None:
            Rpk = float(np.nanmean(df[name]))
    if Rpk is None:
        Rpk = float(np.median(df[r_col]))

    # Wall width (FWHM) if present, else 10% of span as a fallback
    FWHM_name = pick(df, ["w_FWHM","fwhm","width"])
    if FWHM_name:
        FWHM = float(np.nanmean(df[FWHM_name]))
    else:
        FWHM = max(1.0, 0.1*(df[r_col].max()-df[r_col].min()))

    r   = df[r_col].to_numpy(float)
    Phi = df[Phi_col].to_numpy(float)
    phi = df[phi_col].to_numpy(float)

    xi = r - Rpk
    if P.get("xi_side","SFV_positive").lower()=="sfv_negative":
        xi = -xi

    s_phi, ds_phi = spline_abs(xi, phi)
    s_Phi, ds_Phi = spline_abs(xi, Phi)

    # Find crossing for y_B=y_chi=1 (pure geometry)
    def Delta(z): return s_phi(z) - s_Phi(z)
    Lsearch = max(3.0*FWHM, 0.4*(xi.max()-xi.min()))
    xi_star = find_root(Delta, -Lsearch, +Lsearch)

    phi_star = s_phi(xi_star)
    Delta_prime = ds_phi(xi_star) - ds_Phi(xi_star)

    # Finite-energy WKB amplitude profile f_chi(ξ;E)
    ychi = float(P.get("y_chi", 1.0))
    E = float(P.get("E", 0.2))
    def kappa(z):
        m = ychi*s_Phi(z)
        if m <= E: return 0.0
        return math.sqrt(max(m*m - E*E, 0.0))

    # Window around xi*
    sigma = (FWHM/2.355) * float(P.get("window_sigma_factor",1.0))
    x = np.linspace(xi_star - 5*sigma, xi_star + 5*sigma, 1601)

    # Right-going integral of kappa from x to xi_star+3*FWHM
    xi_max = xi_star + 3.0*FWHM
    xR = np.linspace(min(x.min(), xi_star), xi_max, 2000)
    kR = np.array([kappa(z) for z in xR])

    # cumulative ∫ kappa dξ from right to left
    I = np.zeros_like(xR)
    for i in range(len(xR)-2, -1, -1):
        dx = xR[i+1]-xR[i]
        I[i] = I[i+1] + 0.5*(kR[i+1]+kR[i])*dx

    # interpolate I onto x, then f=exp(-I)
    I_on_x = np.interp(x, xR, I, left=I[0], right=0.0)
    f = np.exp(-I_on_x)

    # Normalized Gaussian window W(ξ)
    W = np.exp(-0.5*((x - xi_star)/max(sigma,1e-8))**2)
    W /= np.trapz(W, x)

    g = float(P.get("g_portal", 1.0))
    # Local-LZ equivalent lambda
    numer = np.trapz(W * f * s_phi(x), x)
    lam_local = g * numer / max(phi_star,1e-12)

    out = {
        "inputs": {
            "csv": P["profile_csv"],
            "columns_used": P["columns"],
            "R_peak": Rpk, "FWHM": FWHM,
            "xi_star": xi_star, "phi_star": phi_star, "Delta_prime": Delta_prime,
            "E": E, "sigma": sigma
        },
        "result": {
            "lambda_tr_eff_local_LZ": lam_local
        },
        "notes": "Local-LZ mapping of thick-brane mixing. For distributed-LZ without this mapping, use the extended_LZ approach we discussed."
    }
    json.dump(out, open(args.out,"w"), indent=2)
    print(f"Wrote {args.out}. lambda_tr_eff_local_LZ ≈ {lam_local:.6g}")

if __name__ == "__main__":
    main()
