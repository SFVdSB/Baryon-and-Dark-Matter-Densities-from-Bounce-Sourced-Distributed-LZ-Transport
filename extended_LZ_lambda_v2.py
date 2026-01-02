
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

def find_root(f, a, b, n=4001):
    xs = np.linspace(a,b,n); v=f(xs); s=np.sign(v)
    idx = np.where(np.diff(s)!=0)[0]
    if len(idx):
        lo,hi = xs[idx[0]], xs[idx[0]+1]
        for _ in range(80):
            mid=0.5*(lo+hi); fm=f(mid)
            if np.sign(f(lo))*np.sign(fm) <= 0: hi=mid
            else: lo=mid
        return 0.5*(lo+hi)
    return xs[int(np.argmin(np.abs(v)))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default="extended_LZ_lambda_out.json")
    args = ap.parse_args()

    P = json.load(open(args.params,"r"))
    df = pd.read_csv(P["profile_csv"])

    r_col   = P["columns"]["r"];  Phi_col = P["columns"]["Phi"];  phi_col = P["columns"]["phi"]

    # center & width
    Rpk = None
    if P.get("prefer_csv_R_peak", True):
        nm = pick(df, ["R_peak","r_peak","r_wall","rwall"])
        if nm is not None: Rpk = float(np.nanmean(df[nm]))
    if Rpk is None: Rpk = float(np.median(df[r_col]))

    FWHM_nm = pick(df, ["w_FWHM","fwhm","width"])
    if FWHM_nm is not None:
        FWHM = float(np.nanmean(df[FWHM_nm]))
    else:
        FWHM = max(1.0, 0.1*(df[r_col].max()-df[r_col].min()))

    r   = df[r_col].to_numpy(float)
    Phi = df[Phi_col].to_numpy(float)
    phi = df[phi_col].to_numpy(float)

    xi = r - Rpk
    # Choose which physical side of the wall to integrate into (do NOT reflect the profile).
    side = P.get("xi_side","SFV_positive").lower()
    sgn = +1.0 if side in ("sfv_positive","positive","+","pos") else -1.0

    s_phi, ds_phi = spline_abs(xi, phi)
    s_Phi, ds_Phi = spline_abs(xi, Phi)

    yB   = float(P.get("y_B",1.0))
    ychi = float(P.get("y_chi",1.0))

    def Delta(z): return yB*s_phi(z) - ychi*s_Phi(z)
    Lsearch = max(3.0*FWHM, 0.4*(xi.max()-xi.min()))
    xi_star = find_root(Delta, -Lsearch, +Lsearch)
    phi_star = s_phi(xi_star)
    Delta_prime = yB*ds_phi(xi_star) - ychi*ds_Phi(xi_star)

    # energy grid and weights (1D MB)
    Ed = P.get("E_dist", {})
    T  = float(Ed.get("T_reh", 0.2))
    Emin = float(Ed.get("E_min", 0.0))
    mchi_max = ychi*np.max(np.abs(Phi))
    if Ed.get("E_max_mode","fraction") == "absolute":
        Emax = float(Ed.get("E_max", 0.6*mchi_max))
    else:
        Emax = float(Ed.get("E_max_fraction", 0.6)) * mchi_max
    nE = int(P.get("nE", 160))
    Es = np.linspace(Emin, Emax, nE)
    wE = np.exp(-(Es-Emin)/max(T,1e-8)); wE /= wE.sum()

    # xi-grid on SFV side
    L = float(P.get("xi_span_mult", 3.0))*FWHM
    xi_end = float(np.clip(xi_star + sgn*L, xi.min(), xi.max()))
    x = np.linspace(xi_star, xi_end, 1401)
    mchi_x = ychi*np.abs([s_Phi(z) for z in x])

    f2_avg = np.zeros_like(x)
    for E, w in zip(Es, wE):
        k = np.sqrt(np.clip(mchi_x*mchi_x - E*E, 0.0, None))
        I = np.zeros_like(x)
        for i in range(len(x)-2, -1, -1):
            dx = abs(x[i+1]-x[i])
            I[i] = I[i+1] + 0.5*(k[i+1]+k[i])*dx
        f = np.exp(-I)
        f2_avg += w * (f*f)

    phi_x = np.abs([s_phi(z) for z in x])
    I2 = float(abs(np.trapz(phi_x*phi_x * f2_avg, x)))

    ratio = float(P.get("target_ratio", 5.7))
    Pconv = 1.0/(1.0+ratio)
    delta_tgt = -math.log(1.0-Pconv)/(2.0*math.pi)

    v_w = float(P.get("v_w", 0.3))

    lam0_req = math.sqrt(delta_tgt * v_w * abs(Delta_prime) / max(I2,1e-16))
    lam_eff = lam0_req * math.sqrt(I2) / max(phi_star, 1e-12)

    out = {
        "inputs": {
            "csv": P["profile_csv"],
            "columns_used": P["columns"],
            "R_peak": Rpk, "FWHM": FWHM,
            "xi_star": xi_star, "xi_side": side, "xi_end": float(x[-1]), "phi_star": phi_star, "Delta_prime": Delta_prime,
            "v_w": v_w,
            "energy_grid": {"Emin": float(Emin), "Emax": float(Emax), "nE": nE, "T_reh": T},
            "xi_span": L,
            "target_ratio": ratio
        },
        "integrals": {"I2_phi2_f2": I2},
        "result": {"lambda0_required": lam0_req, "lambda_eff_equiv": lam_eff},
        "notes": "lambda0_required is for distributed LZ; lambda_eff_equiv is a local-LZ constant that reproduces the same δ."
    }
    json.dump(out, open(args.out,"w"), indent=2)
    print(f"lambda0_required ≈ {lam0_req:.6g},  lambda_eff_equiv ≈ {lam_eff:.6g}")

if __name__ == "__main__":
    main()
