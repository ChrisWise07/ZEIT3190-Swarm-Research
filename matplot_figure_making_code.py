import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("pdf")

plt.rc("font", family="serif", serif="Times")
plt.rc("text", usetex=True)
plt.rc("ytick", labelsize=8)
plt.rc("axes", labelsize=8)

# width as measured in inkscape
width = 3.487 * 2
height = width / 1.618


alpha_value = 0.625
fig, ax = plt.subplots(figsize=(width, height))
plt.suptitle(
    "Percentage of Correctly Commitment Agents for Different Algorithms",
    fontsize=10,
)

ax.set_ylim([0.0, 1.0])

x = np.arange(3)
accuracy_no_mal_01 = [0.9974, 0.9976, 1.0000]
accuracy_no_mal_01_error = [0.0031, 0.0047, 0.0000]

accuracy_mal_01 = [0.9788, 0.9815, 0.9893]
accuracy_mal_01_error = [0.0047, 0.0084, 0.0039]

accuracy_no_mal_04 = [0.9944, 0.9969, 0.9993]
accuracy_no_mal_04_error = [0.0042, 0.0032, 0.0015]

accuracy_mal_04 = [0.9796, 0.9784, 0.9784]
accuracy_mal_04_error = [0.0071, 0.0068, 0.0038]

accuracy_no_mal_10 = [0.9870, 0.9924, 0.9956]
accuracy_no_mal_10_error = [0.0045, 0.0040, 0.0037]

accuracy_mal_10 = [0.9638, 0.9786, 0.9547]
accuracy_mal_10_error = [0.0093, 0.0069, 0.0127]


ax.bar(
    x - 0.35,
    accuracy_no_mal_01,
    yerr=accuracy_no_mal_01_error,
    align="center",
    alpha=alpha_value,
    ecolor="black",
    capsize=4,
    color=["b"],
    width=0.1,
)
ax.bar(
    x - 0.25,
    accuracy_mal_01,
    yerr=accuracy_mal_01_error,
    align="center",
    alpha=alpha_value,
    ecolor="black",
    capsize=4,
    color=["g"],
    width=0.1,
)
ax.bar(
    x - 0.05,
    accuracy_no_mal_04,
    yerr=accuracy_no_mal_04_error,
    align="center",
    alpha=alpha_value,
    ecolor="black",
    capsize=4,
    color=["r"],
    width=0.1,
)
ax.bar(
    x + 0.05,
    accuracy_mal_04,
    yerr=accuracy_mal_04_error,
    align="center",
    alpha=alpha_value,
    ecolor="black",
    capsize=4,
    color=["c"],
    width=0.1,
)
ax.bar(
    x + 0.25,
    accuracy_no_mal_10,
    yerr=accuracy_no_mal_10_error,
    align="center",
    alpha=alpha_value,
    ecolor="black",
    capsize=4,
    color=["m"],
    width=0.1,
)
ax.bar(
    x + 0.35,
    accuracy_mal_10,
    yerr=accuracy_mal_10_error,
    align="center",
    alpha=alpha_value,
    ecolor="black",
    capsize=4,
    color=["y"],
    width=0.1,
)

plt.xticks(
    x,
    [
        "Static",
        "Skill 4.0",
        "Equation-based",
        "Inverted-equation-based",
    ],
    fontsize=8,
)
plt.yticks(
    fontsize=8,
)


plt.ylabel("Percentage of correct commitments", fontsize=8)

box = ax.get_position()
ax.set_position(
    [
        box.x0 - box.width * 0.05,
        box.y0 + box.height * 0.075,
        box.width * 1.15,
        box.height * 0.975,
    ]
)

colors = {
    "Non Malicious max opinion weight: 0.1": "b",
    "Malicious max opinion weight: 0.1": "g",
    "Non Malicious max opinion weight: 0.4": "r",
    "Malicious max opinion weight: 0.4": "c",
    "Non Malicious max opinion weight: 1.0": "m",
    "Malicious max opinion weight: 1.0": "y",
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
    bbox_to_anchor=(0.475, -0.075),
    ncol=3,
    fontsize=8,
    handletextpad=0.1,
)

fig.savefig("plot.pdf")
