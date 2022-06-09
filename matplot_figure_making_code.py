import numpy as np
import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt

plt.rc("font", family="serif", serif="Times")
plt.rc("text", usetex=True)
plt.rc("xtick", labelsize=8)
plt.rc("ytick", labelsize=8)
plt.rc("axes", labelsize=6)

# width as measured in inkscape
width = 3.487
height = width / 1.618

import numpy as np
import matplotlib.pyplot as plt


# creating the dataset
data = {
    "Free region: random action": 42.7,
    "Free region: after learning skill 1.1": 170.7,
    "Populated region: after learning skill 1.1": 172.632,
    "Populated region: after learning skill 1.2": 209.62,
    "Populated region: navigation algorithm": 205.38,
}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(10, 4))

alpha_value = 0.625

# creating the bar plot
plt.bar(
    courses,
    values,
    yerr=[14.86, 41.7, 6.82, 3.71, 7.62],
    align="center",
    alpha=alpha_value,
    ecolor="black",
    capsize=5,
    color=["b", "g", "r", "c", "m"],
    width=0.4,
)
plt.tick_params(labelbottom=False)

plt.ylabel("Number of unique cells visited")
plt.title(
    "Total Number of Unique Cells Visited for Different Algorithms",
    fontsize=8,
)
colors = {
    "Free region: random action": "b",
    "Populated region: after learning skill 1.1": "r",
    "Populated region: navigation algorithm": "m",
    "Free region: after learning skill 1.1": "g",
    "Populated region: after learning skill 1.2": "c",
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
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
    fontsize=5,
)


fig.set_size_inches(width, height)
fig.autofmt_xdate()
fig.savefig("plot.pdf")
