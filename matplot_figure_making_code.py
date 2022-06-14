import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("pdf")

plt.rc("font", family="serif", serif="Times")
plt.rc("text", usetex=True)
plt.rc("ytick", labelsize=8)
plt.rc("axes", labelsize=8)

# width as measured in inkscape
width = 3.487
height = width / 1.618


alpha_value = 0.625
fig, ax = plt.subplots(figsize=(width, height))
plt.suptitle(
    "Percentage of Opinions Shared For Different Algorithms",
    fontsize=9,
)

ax.set_ylim([0.0, 1.0])

x = np.arange(3)
incorrect_shared = [0.21, 0.20, 0.14]
incorrect_shared_error = [0.20, 0.08, 0.13]
correct_shared = [0.5, 0.79, 0.86]
correct_shared_error = [0.12, 0.15, 0.09]

ax.bar(
    x - 0.2,
    incorrect_shared,
    yerr=incorrect_shared_error,
    align="center",
    alpha=alpha_value,
    ecolor="black",
    capsize=5,
    color=["r"],
    width=0.4,
)
ax.bar(
    x + 0.2,
    correct_shared,
    yerr=correct_shared_error,
    align="center",
    alpha=alpha_value,
    ecolor="black",
    capsize=5,
    color=["g"],
    width=0.4,
)
plt.xticks(
    x,
    [
        "Random",
        "Algorithm",
        "Skill 2.0",
    ],
    fontsize=7,
)
plt.yticks(
    fontsize=7,
)

plt.ylabel("Percentage of Opinions Shared", fontsize=7)

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

colors = {
    "Incorrect Opinions": "r",
    "Correct Opinions": "g",
}
labels = list(colors.keys())
handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[label], alpha=alpha_value)
    for label in labels
]
plt.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    fontsize=7,
    handletextpad=0.1,
)

fig.savefig("plot.pdf")
