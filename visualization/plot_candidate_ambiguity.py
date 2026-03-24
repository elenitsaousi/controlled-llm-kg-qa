# visualization/plot_candidate_ambiguity.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_candidate_ambiguity(
    ambiguity_scores,
    baseline_ranks,
    improved_ranks,
    ground_truth_ranks,
    baseline_loss,
    improved_loss,
    constraint_loss
):
    """
    Thesis-grade diagnostic visualization inspired by anomaly plots
    in physics-informed ML.
    """

    # --- Convert everything to numpy arrays (IMPORTANT) ---
    ambiguity_scores = np.asarray(ambiguity_scores)
    baseline_ranks = np.asarray(baseline_ranks)
    improved_ranks = np.asarray(improved_ranks)
    ground_truth_ranks = np.asarray(ground_truth_ranks)
    baseline_loss = np.asarray(baseline_loss)
    improved_loss = np.asarray(improved_loss)
    constraint_loss = np.asarray(constraint_loss)

    baseline = baseline_ranks
    improved = improved_ranks
    both_mask = baseline == improved

    both_series = np.where(both_mask, baseline, np.nan)
    both_loss_series = np.where(both_mask, baseline_loss, np.nan)

    N = len(ambiguity_scores)
    x = np.arange(N)

    fig, ax = plt.subplots(figsize=(16, 6))

    # --- Background: ambiguity map ---
    norm = mcolors.Normalize(
        vmin=min(ambiguity_scores),
        vmax=max(ambiguity_scores)
    )
    cmap = cm.Reds

    for i, a in enumerate(ambiguity_scores):
        ax.axvspan(
            i,
            i + 1,
            color=cmap(norm(a)),
            alpha=0.25,
            linewidth=0,
            zorder=1
        )


    # --- Selection outcomes ---
    ax.plot(
        x + 0.5, baseline,
        label="Baseline Selection",
        color="#1f77b4", linewidth=2.2, zorder=3
    )
    ax.plot(
        x + 0.5, improved,
        label="Constraint-Aware Selection",
        color="#2ca02c", linewidth=2.2, zorder=4
    )
    ax.plot(
        x + 0.5, both_series,
        label="Baseline = Constraint-Aware",
        color="#9467bd", linewidth=3.0, alpha=0.9, zorder=5
    )

    


    ax.set_xlabel("Query Index")
    ax.set_ylabel("Selected Candidate Rank")
    ax.set_title("Candidate Ambiguity Timeline with Selection Outcomes- SPARQL")
    ax.set_ylim(5.5, 0.5)
    ax.legend(loc="upper left", frameon=False)

    # --- Colorbar ---
    norm = mcolors.Normalize(vmin=min(ambiguity_scores), vmax=max(ambiguity_scores))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # --- Compact colorbar (top-right, small) ---
    cax = fig.add_axes([0.92, 0.45, 0.015, 0.28])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Candidate Ambiguity", fontsize=9)
    cbar.ax.tick_params(labelsize=8)


    # --- Inset: loss decomposition ---
    # --- Inset: loss comparison (minimal & readable) ---
    inset = fig.add_axes([0.64, 0.13, 0.30, 0.20])

    inset.plot(
        x, baseline_loss,
        color="#1f77b4", linewidth=2.0, alpha=0.9,
        label="Baseline Error"
    )

    inset.plot(
        x, improved_loss,
        color="#2ca02c", linewidth=2.2, alpha=0.9,
        label="Constraint-Aware Error"
    )
    inset.plot(
        x, both_loss_series,
        color="#9467bd", linewidth=2.6, alpha=0.9,
        label="Baseline = Constraint-Aware"
    )

    inset.set_yscale("log")
    inset.set_title("Error Comparison", fontsize=9)
    inset.set_xlabel("Query Index", fontsize=8)
    inset.set_ylabel("Error", fontsize=8)

    inset.grid(alpha=0.15)
    inset.tick_params(labelsize=7)
    for spine in inset.spines.values():
        spine.set_alpha(0.3)




    plt.tight_layout()
    plt.show()
