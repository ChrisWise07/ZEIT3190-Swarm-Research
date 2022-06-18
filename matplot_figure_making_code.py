import numpy as np
import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt

plt.rc("font", family="serif", serif="Times")
plt.rc("text", usetex=True)
plt.rc("ytick", labelsize=8)
plt.rc("axes", labelsize=8)

# width as measured in inkscape
width = 3.487
height = width / 1.618

import numpy as np
import matplotlib.pyplot as plt


# creating the dataset
data = {
    "Random action": 0.64,
    "Quorum threshold": 0.62,
    "Skill 3.0": 0.59,
}
courses = list(data.keys())
values = list(data.values())


alpha_value = 0.625
fig = plt.figure(figsize=(width, height))
plt.suptitle(
    "Percentage of Correctly Commitment Agents for Different Algorithms",
    fontsize=9,
)
ax = plt.subplot(111)
# creating the bar plot
ax.bar(
    courses,
    values,
    yerr=[0.04, 0.03, 0.04],
    align="center",
    alpha=alpha_value,
    ecolor="black",
    capsize=5,
    color=["b", "g", "r"],
    width=0.4,
)
ax.set_ylim([0, 1.0])
plt.ylabel("Percentage of correct commitments", fontsize=7)


# fig.autofmt_xdate()
fig.savefig("plot.pdf")
