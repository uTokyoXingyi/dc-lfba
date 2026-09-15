import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import pandas as pd


# file1 = "runs_E1/exp_11_random_init_our_lr0.3.csv"

file1 = "runs_E1/exp_11_random_init_Baseline_lr0.01.csv"
file2 = "runs_E1/exp_11_random_init_SimCLR.csv"
file3 = "runs_E1/exp_11_random_init_BYOL.csv"
file4 = "runs_E1/exp_11_random_init_MoCo.csv"
file5 = "runs_E1/exp_11_random_init_SimSiam.csv"

df_baseline = pd.read_csv(file1)
df_our_M1 = pd.read_csv(file2)
df_our_M2 = pd.read_csv(file3)
df_our_M3 = pd.read_csv(file4)
df_our_M4 = pd.read_csv(file5)

df_baseline = df_baseline.sort_values("sup_epochs")
df_our_M1 = df_our_M1.sort_values("ssl_epochs")
df_our_M2 = df_our_M2.sort_values("ssl_epochs")
df_our_M3 = df_our_M3.sort_values("ssl_epochs")
df_our_M4 = df_our_M4.sort_values("ssl_epochs")

baseline_LFBA = 0.9877564311027527
baseline_SimCLR = 0.9675711393356323

# create two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# plot a
# axs[0].plot(df_baseline["sup_epochs"], df_baseline["s_acc"], marker='o', markersize=4, color="r", linestyle="-", linewidth=1, label="Baseline")
# axs[0].plot(df_baseline["sup_epochs"], df_baseline["b_acc"], marker='^', markersize=4, color="r", linestyle="--", linewidth=1, label="Baseline")
axs[0].plot(df_our_M1["ssl_epochs"], df_our_M1["s_acc"], marker='o', markersize=5, color="blue", linestyle="-", linewidth=2, label="DC-LFBA (SimCLR)")
# axs[0].plot(df_our_M1["ssl_epochs"], df_our_M1["b_acc"], marker='x', markersize=4, color="blue", linestyle="--", linewidth=1, label="SimCLR ()")
axs[0].axhline(baseline_LFBA, color="r", linewidth=2, linestyle='--', label="Baseline (Supervised LFBA)")
plt.xscale('log')

# axs[0].plot(df_baseline["sup_epochs"], df_baseline["s_acc"], marker='o',color="r", linestyle="-", label="Baseline (Std Acc)")
# axs[0].plot(df_baseline["sup_epochs"], df_baseline["b_acc"], marker='x', color="r", linestyle="--", label="Baseline (Bal Acc)")
# axs[0].plot(df_our_M1["ssl_epochs"], df_our_M1["s_acc"], marker='*',  color="blue", linestyle="-", label="SimCLR + Head (50 Epochs) (Std Acc)")
# axs[0].plot(df_our_M1["ssl_epochs"], df_our_M1["b_acc"], marker='s', color="blue", linestyle="--",label="SimCLR + Head (50 Epochs) (Bal Acc)")
axs[0].set_xscale('log')
axs[0].set_title("(a) SimCLR training behavior")
axs[0].set_xlabel("Contrastive Learning Epochs")
axs[0].set_ylabel("S-Acc")
axs[0].grid(True)
axs[0].legend()
axs[0].set_xticks([1,10,100])
axs[0].set_xticklabels(["1", "10", "100"])

# plot b
# axs[1].plot(df_baseline["sup_epochs"], df_baseline["s_acc"], marker='o', markersize=4, color="r", linestyle="-", linewidth=1, label="Baseline")
# axs[1].plot(df_our_M1["ssl_epochs"], df_our_M1["s_acc"], marker='*', markersize=4, color="blue", linestyle="-", linewidth=1, label="SimCLR")
axs[1].plot(df_our_M2["ssl_epochs"], df_our_M2["s_acc"], marker='D', markersize=5, color="g", linewidth=2, label="DC-LFBA (BYOL)")
axs[1].plot(df_our_M3["ssl_epochs"], df_our_M3["s_acc"], marker='s', markersize=5, color="y", linewidth=2, label="DC-LFBA (MoCo)")
axs[1].plot(df_our_M4["ssl_epochs"], df_our_M4["s_acc"], marker='^', markersize=5, color="c", linewidth=2, label="DC-LFBA (SimSiam)")
axs[1].axhline(baseline_SimCLR, color="blue", linewidth=2, linestyle='--', label="SimCLR (Best)")
axs[1].axhline(baseline_LFBA, color="r", linewidth=2, linestyle='--', label="Baseline (Best)")

# axs[1].plot(epochs, moco_acc, marker='^', label="MoCo")
# axs[1].plot(epochs, byol_acc, marker='s', label="BYOL")
# axs[1].plot(epochs, simsiam_acc, marker='D', label="SimSiam")
axs[1].set_xscale('log')
axs[1].set_title("(b) MoCo, BYOL and SimSiam training behavior")
axs[1].set_xlabel("Contrastive Learning Epochs")
# axs[1].set_ylabel("Test Accuracy (S-Acc)")
axs[1].grid(True)
axs[1].legend()
axs[1].set_xticks([1,10,100])
axs[1].set_xticklabels(["1", "10", "100"])

plt.tight_layout()
plt.savefig("E11_ssl_comparison.png", dpi=300)
plt.savefig("E11_ssl_comparison.pdf", bbox_inches="tight")
plt.show()