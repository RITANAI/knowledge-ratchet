"""
figures.py — Regenerate Figures 1-6 of Yan (2026), "The Knowledge Ratchet".

Usage:
    python figures.py              # full runs, writes PDFs to figures/
    python figures.py --quick      # reduced run counts (fast, noisier curves)

Requires: numpy, scipy, matplotlib.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from knowledge_ratchet import (Params, run_s4ai, run_nolearn, run_random,
                               run_periodic, run_online_cl, run_gp_ucb,
                               run_turbo1, run_oracle_threshold)

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "black": "#000000"}
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                     "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "figure.dpi": 200, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})


def fig1(out):
    fig, ax = plt.subplots(figsize=(6.9, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 5.6)

    def box(x, y, w, h, title, lines, fc, fs=7.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    fc=fc, ec="k", lw=0.9))
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center",
                fontsize=8.4, fontweight="bold")
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.70 - i * 0.40, ln, ha="center",
                    va="center", fontsize=fs)

    box(0.25, 2.55, 3.35, 2.45, "Generative model",
        [r"capability $\theta_t$", "selects $M$ hypotheses", "",
         r"checkpoint rate",
         r"$\phi_t=\Phi\!\left((\Phi^{-1}(\alpha_0\theta_t)-c^*)/\sigma_\varepsilon\right)$"],
        "#EAF2FA")
    box(4.05, 2.55, 3.30, 2.45, "Physical checkpoint",
        [r"$N_t\sim\mathrm{Binomial}(M,\phi_t)$", "",
         r"validation $p_t=\min\{p_{\max},\;\alpha_0\theta_t$",
         r"$+\;\gamma_0 K_t/(H+K_t)\}$",
         r"$\Delta K_t\sim\mathrm{Binomial}(N_t,p_t)$"], "#FDF3E3")
    box(7.80, 2.55, 2.55, 2.45, "Knowledge stock",
        [r"$K_{t+1}=K_t+\Delta K_t$", "", "ratchet: non-decreasing,",
         "irreversible, cumulative"], "#EAF6EF")

    def arr(x1, y1, x2, y2, color="k", lw=1.2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=lw, color=color))

    arr(3.62, 4.30, 4.03, 4.30)
    ax.text(3.83, 4.52, "proposes", ha="center", fontsize=7.2,
            bbox=dict(fc="white", ec="none", pad=0.4))
    arr(7.37, 4.30, 7.78, 4.30)
    ax.text(7.58, 4.52, r"$\Delta K_t$", ha="center", fontsize=7.6,
            bbox=dict(fc="white", ec="none", pad=0.4))
    arr(9.05, 2.42, 9.05, 1.30)
    arr(9.05, 1.30, 1.90, 1.30)
    arr(1.90, 1.30, 1.90, 2.42)
    ax.text(5.5, 1.58, "RLPE update:  "
            r"$\theta_{t+1}=\min\{\bar{\theta},\;\theta_t+\eta_0\,\Delta K_t/N_t\}$",
            ha="center", fontsize=8.4, color=OI["blue"],
            bbox=dict(fc="white", ec="none", pad=0.6))
    ax.text(5.5, 0.96,
            r"experience feedback:  $\gamma_0 K_t/(H+K_t)$ enters $p_t$",
            ha="center", fontsize=7.6, color=OI["blue"])
    arr(1.90, 5.48, 1.90, 5.12)
    ax.text(1.90, 5.30, "new hypotheses", ha="center", fontsize=7.2,
            bbox=dict(fc="white", ec="none", pad=0.4))
    ax.plot([6.10, 5.92, 6.10, 5.92], [2.40, 1.98, 1.98, 1.56],
            color=OI["red"], lw=1.6)
    ax.annotate("", xy=(5.92, 1.47), xytext=(5.92, 1.60),
                arrowprops=dict(arrowstyle="-|>", color=OI["red"], lw=1.6))
    ax.text(6.34, 1.94,
            r"forgetting: w.p. $\lambda_{\mathrm{CF}}$, "
            r"$\theta\leftarrow(1-\delta_{\mathrm{CF}})\theta$",
            fontsize=7.4, color=OI["red"], ha="left",
            bbox=dict(fc="white", ec="none", pad=0.4))
    ax.text(5.3, 0.28,
            "Physical evidence closes the loop: capability improves only "
            "through validated experimental outcomes.",
            ha="center", fontsize=7.8, style="italic", color="#444444")
    fig.savefig(f"{out}/fig1_schematic.pdf")
    plt.close(fig)


def fig2(out, quick):
    R = 2_000 if quick else 10_000
    sweeps = {
        "p_max": (np.linspace(0.6, 0.95, 8), "p_max", 0.8),
        "eta_0": (np.linspace(0.01, 0.06, 8), "eta_0", 0.03),
        "gamma_0": (np.linspace(0.05, 0.35, 8), "gamma_0", 0.20),
        "alpha_0": (np.linspace(0.45, 0.85, 8), "alpha_0", 0.65),
        "lambda_CF": (np.linspace(0, 0.3, 8), "lambda_CF", 0.0),
    }
    labels = {"p_max": r"$p_{\max}$", "eta_0": r"$\eta_0$",
              "gamma_0": r"$\gamma_0$", "alpha_0": r"$\alpha_0$",
              "lambda_CF": r"$\lambda_{\mathrm{CF}}$"}
    base = run_s4ai(R=R)["K_final"]
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.3))
    for ax, (key, (grid, fld, bval)) in zip(axes.flat[:5], sweeps.items()):
        ys = []
        for x in grid:
            kw = {fld: float(x)}
            if fld == "lambda_CF":
                kw["delta_CF"] = 0.3
            ys.append(run_s4ai(R=R, par=Params(**kw))["K_final"])
        ax.plot(grid, ys, "-o", ms=3.5, color=OI["blue"], lw=1.3)
        ax.axvline(bval, color="gray", ls="--", lw=0.8)
        ax.axhline(base, color=OI["orange"], ls=":", lw=0.8)
        ax.set_xlabel(labels[key])
        ax.set_ylabel(r"mean $K_{100}$")
    axes.flat[5].axis("off")
    axes.flat[5].text(-0.08, 0.92,
                      "Each panel varies one\nparameter with all others\n"
                      "at baseline.\n\nDashed gray: baseline value.\n"
                      "Dotted orange: baseline\n$K_{100}=2{,}812$.\n\n"
                      "$\\lambda_{\\mathrm{CF}}$ panel: "
                      "$\\delta_{\\mathrm{CF}}=0.3$.",
                      fontsize=7.2, va="top")
    fig.suptitle("Parameter sensitivity of cumulative validated discoveries",
                 y=1.0, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{out}/fig2_sensitivity.pdf")
    plt.close(fig)


def fig3(out, quick):
    R = 2_000 if quick else 10_000
    Rbo = 60 if quick else 200
    res = {
        "S4AI (RLPE)": run_s4ai(R=R, ret_traj=True),
        "TuRBO-1 (BO)": run_turbo1(R=Rbo, ret_traj=True),
        "GP-UCB (BO)": run_gp_ucb(R=Rbo, ret_traj=True),
        "Periodic retrain ($\\tau$=10)": run_periodic(R=R),
        "Online CL": run_online_cl(R=R),
        "No learning": run_nolearn(R=R),
        "Random": run_random(R=R),
    }
    oracle = run_oracle_threshold(R=R)["K_final"]
    cols = [OI["blue"], OI["green"], OI["sky"], OI["orange"], OI["purple"],
            OI["red"], OI["black"]]
    t = np.arange(1, 101)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 3.0),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    for (name, r), c in zip(res.items(), cols):
        a1.plot(t, r["K_mean"], color=c, lw=1.4, label=name)
    a1.axhline(oracle, color=OI["green"], ls=":", lw=1.0)
    a1.text(98, oracle * 1.025,
            f"oracle threshold-policy ceiling ({oracle:,.0f})",
            fontsize=6.6, color=OI["green"], ha="right", va="bottom",
            bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
    a1.set_xlabel("round $t$")
    a1.set_ylabel(r"cumulative discoveries $K_t$")
    a1.set_ylim(0, 3050)
    a1.legend(frameon=False, loc="center left", fontsize=6.6,
              bbox_to_anchor=(0.0, 0.60))
    a1.set_title("(a) Accumulation trajectories", loc="left", fontsize=8.5)
    names = ["S4AI", "Oracle ceiling", "TuRBO-1", "GP-UCB", "Periodic",
             "Online CL", "No learn", "Random"]
    vals = [res["S4AI (RLPE)"]["K_final"], oracle,
            res["TuRBO-1 (BO)"]["K_final"], res["GP-UCB (BO)"]["K_final"],
            res["Periodic retrain ($\\tau$=10)"]["K_final"],
            res["Online CL"]["K_final"], res["No learning"]["K_final"],
            res["Random"]["K_final"]]
    cols2 = [OI["blue"], "#7F7F7F", OI["green"], OI["sky"], OI["orange"],
             OI["purple"], OI["red"], OI["black"]]
    s4 = vals[0]
    bars = a2.barh(names[::-1], vals[::-1], color=cols2[::-1], height=0.62)
    for b, v in zip(bars, vals[::-1]):
        a2.text(v + 40, b.get_y() + b.get_height() / 2,
                f"{v:,.0f} ({s4 / v:.2f}$\\times$)", va="center", fontsize=6.4)
    a2.set_xlim(0, 3800)
    a2.set_xlabel(r"mean $K_{100}$")
    a2.set_title("(b) 100-round cumulative output", loc="left", fontsize=8.5)
    a2.tick_params(axis="y", labelsize=7.0)
    fig.tight_layout()
    fig.savefig(f"{out}/fig3_baselines.pdf")
    plt.close(fig)


def fig4(out, quick):
    T, R = (300, 500) if quick else (1000, 2000)
    base = run_s4ai(T=T, R=R)
    forg = run_s4ai(T=T, R=R, par=Params(lambda_CF=0.035, delta_CF=0.3),
                    ret_traj=True)
    tt = np.arange(1, T + 1)
    sl_b = np.polyfit(np.arange(T // 2, T), base["K_mean"][T // 2:], 1)[0]
    sl_f = np.polyfit(np.arange(T // 2, T), forg["K_mean"][T // 2:], 1)[0]
    tail = forg["theta_all"][forg["theta_all"] < 0.945]
    atom = (forg["theta_all"] >= 0.945).mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 3.0))
    a1.plot(tt, base["K_mean"], color=OI["blue"], lw=1.4,
            label=rf"$\lambda_{{\mathrm{{CF}}}}=0$: slope {sl_b:.1f}/round")
    a1.plot(tt, forg["K_mean"], color=OI["red"], lw=1.4,
            label=rf"$\lambda_{{\mathrm{{CF}}}}=0.035$: slope {sl_f:.1f}/round")
    a1.set_xlabel("round $t$")
    a1.set_ylabel(r"cumulative discoveries $K_t$")
    a1.legend(frameon=False, loc="upper left", fontsize=7.0)
    a1.set_title("(a) Accumulation under forgetting", loc="left", fontsize=8.5)
    a2.plot(tt, forg["theta_mean"], color=OI["red"], lw=1.0)
    a2.axhline(0.95, color="gray", ls="--", lw=0.7)
    a2.text(T - 15, 0.958, r"ceiling $\bar{\theta}=0.95$", fontsize=6.4,
            color="gray", ha="right")
    a2.set_xlabel("round $t$")
    a2.set_ylabel(r"mean capability $\theta_t$")
    a2.set_ylim(0.55, 1.02)
    a2.set_title("(b) Capability dynamics and stationary law", loc="left",
                 fontsize=8.5)
    axin = a2.inset_axes([0.34, 0.15, 0.44, 0.38])
    axin.hist(tail, bins=28, color=OI["sky"], edgecolor="white", lw=0.3,
              density=True)
    axin.set_title(f"stationary $\\theta$ below ceiling\n({atom:.0%} atom at "
                   "$\\bar{\\theta}$ excluded)", fontsize=5.6, pad=1.5)
    axin.tick_params(labelsize=5.4)
    axin.set_yticks([])
    axin.set_xlim(0.3, 1.0)
    fig.tight_layout()
    fig.savefig(f"{out}/fig4_forgetting.pdf")
    plt.close(fig)


def fig5(out, quick):
    R = 2_000 if quick else 10_000
    t = np.arange(1, 101)
    KmA = run_s4ai(R=R)["K_mean"]
    KmP1 = run_s4ai(R=R, par=Params(zeta_0=0.05, kappa=0.1))["K_mean"]
    KmP2 = run_s4ai(R=R, par=Params(zeta_0=0.05, kappa=0.2))["K_mean"]
    KmR1 = run_s4ai(R=R, par=Params(rho_d=0.05))["K_mean"]
    KmR2 = run_s4ai(R=R, par=Params(rho_d=0.20))["K_mean"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 3.0))
    a1.plot(t, KmA, color=OI["blue"], lw=1.8,
            label=rf"no pruning ($\kappa=0$): {KmA[-1]:,.0f}")
    a1.plot(t, KmP1, color=OI["orange"], lw=1.2,
            label=rf"$\kappa=0.1$: {KmP1[-1]:,.0f} "
                  rf"($-${100 * (1 - KmP1[-1] / KmA[-1]):.1f}%)")
    a1.plot(t, KmP2, color=OI["red"], lw=1.2,
            label=rf"$\kappa=0.2$: {KmP2[-1]:,.0f} "
                  rf"($-${100 * (1 - KmP2[-1] / KmA[-1]):.1f}%)")
    a1.set_xlabel("round $t$")
    a1.set_ylabel(r"cumulative discoveries $K_t$")
    a1.legend(frameon=False, loc="upper left", fontsize=6.9)
    a1.set_title(r"(a) Pruning ablation ($\zeta_0=0.05$)", loc="left",
                 fontsize=8.5)
    a2.plot(t, KmA, color=OI["blue"], lw=1.8,
            label=rf"irreversible ($\rho_d=0$): {KmA[-1]:,.0f}")
    a2.plot(t, KmR1, color=OI["orange"], lw=1.2,
            label=rf"$\rho_d=0.05$: {KmR1[-1]:,.0f} "
                  rf"($-${100 * (1 - KmR1[-1] / KmA[-1]):.0f}%)")
    a2.plot(t, KmR2, color=OI["red"], lw=1.2,
            label=rf"$\rho_d=0.20$: {KmR2[-1]:,.0f} "
                  rf"($-${100 * (1 - KmR2[-1] / KmA[-1]):.0f}%)")
    a2.set_xlabel("round $t$")
    a2.set_ylabel(r"knowledge stock $K_t$")
    a2.legend(frameon=False, loc="upper left", fontsize=6.9)
    a2.set_title("(b) Reversibility ablation", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{out}/fig5_ablations.pdf")
    plt.close(fig)


def fig6(out):
    q = np.linspace(0, 1.3, 131)
    ns = np.array([2, 3, 4])
    Z = 10 * np.log2((1 + 0.42 * q)[None, :] ** ns[:, None])
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    im = ax.imshow(Z, aspect="auto", origin="lower", extent=[0, 1.3, 1.5, 4.5],
                   cmap="viridis")
    cs = ax.contour(q, [2, 3, 4], Z, levels=[5, 10, 15, 20, 25],
                    colors="white", linewidths=0.7)
    ax.clabel(cs, fmt="%d yr", fontsize=6.6)
    qq = np.array([0.25, 0.5, 0.75, 1.0, 1.25])
    for n in ns:
        for qv in qq:
            ax.plot(qv, n, "o", ms=2.6, color="white", mec="k", mew=0.4)
            ax.annotate(f"{10 * n * np.log2(1 + 0.42 * qv):.1f}", (qv, n),
                        fontsize=5.6, ha="center", va="center", color="k",
                        xytext=(qv, n + 0.15))
    ax.set_xlabel(r"implementation quality $q$ (fraction of conservative "
                  r"uplift $f=1.42$)", fontsize=8)
    ax.set_ylabel("pipeline stages $n$")
    ax.set_yticks([2, 3, 4])
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Eroom-years offset", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(r"Eroom-years offset, $t=10\,n\,\log_2(1+0.42q)$", loc="left",
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{out}/fig6_eroom.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig1(args.out)
    print("fig1 done")
    fig2(args.out, args.quick)
    print("fig2 done")
    fig3(args.out, args.quick)
    print("fig3 done")
    fig4(args.out, args.quick)
    print("fig4 done")
    fig5(args.out, args.quick)
    print("fig5 done")
    fig6(args.out)
    print("fig6 done")
    print(f"All figures written to {args.out}/")


if __name__ == "__main__":
    main()
