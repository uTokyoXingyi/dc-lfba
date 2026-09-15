import matplotlib.pyplot as plt
import pandas as pd

# file1 = "runs_E1/exp_11_random_init_our_lr0.3.csv"

file1 = "runs_E1/exp_11_random_init_Baseline_lr0.01.csv"
file2 = "runs_E1/exp_11_random_init_SimCLR_lr0.3.csv"
file3 = "runs_E1/exp_11_random_init_BYOL.csv"
file4 = "runs_E1/exp_11_random_init_MoCo.csv"
file5 = "runs_E1/exp_11_random_init_SimSiam_all.csv"

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


plt.figure(figsize=(6, 4))

# baseline - whole model fine-tune
plt.plot(
    df_baseline["sup_epochs"],
    df_baseline["s_acc"],
    label="Baseline",
    color="red",
    linestyle="-",
    marker="o",
    markersize=8,
)
# plt.plot(
#     df_baseline["sup_epochs"],
#     df_baseline["b_acc"],
#     label="Baseline - Supervised (Bal Acc)",
#     color="red",
#     linestyle="--",
#     marker="x",
#     markersize=8,
# )
# Our method - SimCLR
plt.plot(
    df_our_M1["ssl_epochs"],
    df_our_M1["s_acc"],
    label="SimCLR",
    color="blue",
    linestyle="-",
    marker="s",
    markersize=8,
)
# plt.plot(
#     df_our_M1["ssl_epochs"],
#     df_our_M1["b_acc"],
#     label="SSL Encoder + Head (50 Epochs) (Bal Acc)",
#     color="blue",
#     linestyle="--",
#     marker="s",
#     markersize=8,
# )

# Our method - BYOL
plt.plot(
    df_our_M2["ssl_epochs"],
    df_our_M2["s_acc"],
    label="BYOL",
    color="green",
    linestyle="-",
    marker="*",
    markersize=8,
)
# plt.plot(
#     df_our_M2["ssl_epochs"],
#     df_our_M2["b_acc"],
#     label="SSL Encoder + Head (50 Epochs) (Bal Acc)",
#     color="green",
#     linestyle="--",
#     marker="s",
#     markersize=8,
# )

# Our method - MoCo
plt.plot(
    df_our_M3["ssl_epochs"],
    df_our_M3["s_acc"],
    label="MoCo",
    color="y",
    linestyle="-",
    marker="x",
    markersize=8,
)
# plt.plot(
#     df_our_M3["ssl_epochs"],
#     df_our_M3["b_acc"],
#     label="SSL Encoder + Head (50 Epochs) (Bal Acc)",
#     color="blue",
#     linestyle="--",
#     marker="s",
#     markersize=8,
# )
# # Our method - SimSiam
plt.plot(
    df_our_M4["ssl_epochs"],
    df_our_M4["s_acc"],
    label="SimSiam",
    color="c",
    linestyle="-",
    marker="D",
    markersize=8,
)
# plt.plot(
#     df_our_M4["ssl_epochs"],
#     df_our_M4["b_acc"],
#     label="SSL Encoder + Head (50 Epochs) (Bal Acc)",
#     color="y",
#     linestyle="--",
#     marker="s",
#     markersize=8,
# )
plt.xscale('log')
plt.xlabel("SSL Epochs/Supervised Epoches")
plt.ylabel("Test Accuracy")
plt.title("Experiment 1.1: Random Setting")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("exp_11_comparison.png", dpi=300)
plt.savefig("exp_11__comparison.pdf", bbox_inches="tight")
print("Plot saved as exp_11_comparison.png")
plt.show()