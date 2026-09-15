import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# file1 = "runs_E2_V0/exp_label_eff_our_v0.csv"
# file2 = "runs_E2_V0/exp_label_eff_baseline_v0.csv"

# file1 = "runs_E2/exp_label_eff_baseline_more.csv"
# file2 = "runs_E2/exp_label_eff_our_SimCLR_more.csv"
file1 = "runs_E2/label_eff_baseline_ava.csv"
file2 = "runs_E2/label_eff_our_ava.csv"

file3 = "runs_E2/exp_label_eff_our_MoCo.csv"
file4 = "runs_E2/exp_label_eff_our_BYOL.csv"
file5 = "runs_E2/exp_label_eff_our_SimSiam.csv"

df_baseline = pd.read_csv(file1)
df_our_SimCLR = pd.read_csv(file2)
df_our_MoCo = pd.read_csv(file3)
df_our_BYOL = pd.read_csv(file4)
df_our_SimSiam = pd.read_csv(file5)

df_baseline = df_baseline.sort_values("N")
df_our_SimCLR = df_our_SimCLR.sort_values("N")
df_our_MoCo = df_our_MoCo.sort_values("N")
df_our_BYOL = df_our_BYOL.sort_values("N")
df_our_SimSiam = df_our_SimSiam.sort_values("N")

fig, ax = plt.subplots(figsize=(6,4))

# baseline - whole model fine-tune
ax.plot(
    df_baseline["N"],
    df_baseline["s_acc"],
    label="Supervised LFBA",
    color="red",
    linestyle="-",
    marker="o",
    linewidth = 2,
    markersize=6,
)

# Our method - SimCLR
ax.plot(
    df_our_SimCLR["N"],
    df_our_SimCLR["s_acc"],
    label="DC-LFBA (SimCLR)",
    color="blue",
    linestyle="-",
    linewidth = 2,
    marker="s",
    markersize=6,
)
# Our method - MoCo
# plt.plot(
#     df_our_MoCo["N"],
#     df_our_MoCo["s_acc"],
#     label="DC-LFBA (MoCo)",
#     color="g",
#     linestyle="-",
#     linewidth = 1,
#     marker="^",
#     markersize=6,
# )
# Our method - BYOL
# plt.plot(
#     df_our_BYOL["N"],
#     df_our_BYOL["s_acc"],
#     label="DC-LFBA (BYOL)",
#     color="y",
#     linestyle="-",
#     linewidth = 1,
#     marker="D",
#     markersize=6,
# )

# Our method - SimSiam
# plt.plot(
#     df_our_SimSiam["N"],
#     df_our_SimSiam["s_acc"],
#     label="DC-LFBA (SimSiam)",
#     color="c",
#     linestyle="-",
#     linewidth = 1,
#     marker="x",
#     markersize=6,
# )

ax.set_xscale('log')
ax.set_xticks([1, 10, 50, 100,200,500])
# ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.set_xticklabels(["1", "10", "50", "100","200","500"])

ax.set_xlabel("Labeled Samples per Class")
ax.set_ylabel("S-Acc")
# plt.title("Experiment 2: Label Efficiency Evaluation")
ax.legend(loc="lower right")
ax.grid(True, linestyle="--", alpha=0.4)

# axs[0].set_xticks([1,10,100])
# axs[0].set_xticklabels(["1", "10", "100"])

plt.tight_layout()
plt.savefig("runs_E2/label_efficiency_ava.png", dpi=300)
plt.savefig("runs_E2/label_efficiency_ava.pdf", bbox_inches="tight")
print("Plot saved as exp_2_results")
plt.show()