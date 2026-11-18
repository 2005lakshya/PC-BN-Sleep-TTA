"""Draw the system architecture figure used in the README (figures/architecture.png)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY, PURPLE, GOLD, LIGHT, GREY, RED = "#1B1F4B", "#5B4FCF", "#F2B134", "#EEF0FA", "#6E7299", "#D9483B"
boxes = [
    ("New night", "EEG recording\nno labels", LIGHT, NAVY, False),
    ("Pre-process", "30-s pieces\nscale signal", LIGHT, NAVY, False),
    ("1D CNN", "trained on young\npeople (frozen)", PURPLE, "white", False),
    ("PC-BN", "adapt BatchNorm\nstage by stage", GOLD, NAVY, True),
    ("HMM", "smooth sudden\nstage jumps", GOLD, NAVY, True),
    ("Sleep report", "a stage for\nevery 30 s", NAVY, "white", False),
]
fig, ax = plt.subplots(figsize=(13, 4.2), dpi=160)
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.2)
ax.axis("off")
W, Hb, gap, y = 1.75, 1.55, 0.37, 1.35
for i, (title, sub, fill, txt, ours) in enumerate(boxes):
    x = 0.3 + i * (W + gap)
    ax.add_patch(FancyBboxPatch((x, y), W, Hb, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fill, ec="none", zorder=2))
    ax.text(x + W / 2, y + Hb - 0.42, title, ha="center", va="center", fontsize=12.5, weight="bold", color=txt, zorder=3)
    ax.text(x + W / 2, y + 0.55, sub, ha="center", va="center", fontsize=10.5, color=txt, zorder=3)
    if ours:
        ax.add_patch(FancyBboxPatch((x + W / 2 - 0.38, y + Hb - 0.05), 0.76, 0.3, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=RED, ec="none", zorder=4))
        ax.text(x + W / 2, y + Hb + 0.1, "OURS", ha="center", va="center", fontsize=9, weight="bold", color="white", zorder=5)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x + W + 0.04, y + Hb / 2), (x + W + gap - 0.04, y + Hb / 2),
                                     arrowstyle="-|>", mutation_scale=18, color=GREY, lw=2, zorder=1))
# loop: PC-BN re-runs the CNN
xc, xp = 0.3 + 2 * (W + gap) + W / 2, 0.3 + 3 * (W + gap) + W / 2
ax.add_patch(FancyArrowPatch((xp, y - 0.05), (xc, y - 0.05), connectionstyle="arc3,rad=-0.45",
                             arrowstyle="-|>", mutation_scale=16, color=GOLD, lw=2))
ax.text((xc + xp) / 2, 0.45, "re-run the CNN with new BatchNorm numbers (2 rounds)", ha="center", fontsize=10, color=GREY, style="italic")
ax.text(0.3, 3.85, "System architecture: adapting a sleep-staging CNN to one new night, without labels",
        fontsize=13, weight="bold", color=NAVY)
fig.savefig("figures/architecture.png", bbox_inches="tight", facecolor="white")
print("saved figures/architecture.png")
