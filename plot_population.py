import matplotlib.pyplot as plt

from CNS.spinalcord.izhikevich_cpg import IzhikevichCPG


cpg = IzhikevichCPG(sim_dt=0.002)

duration = 5.0
steps = int(duration / cpg.sim_dt)

times = []

lf_values = []
le_values = []
rf_values = []
re_values = []


for k in range(steps):
    out = cpg.step()

    t = k * cpg.sim_dt

    times.append(t)

    lf_values.append(out.left_flexor)
    le_values.append(out.left_extensor)
    rf_values.append(out.right_flexor)
    re_values.append(out.right_extensor)


plt.figure(figsize=(10, 5))

plt.plot(times, lf_values, label="LF - Left Flexor")
plt.plot(times, le_values, label="LE - Left Extensor")
plt.plot(times, rf_values, label="RF - Right Flexor")
plt.plot(times, re_values, label="RE - Right Extensor")

plt.xlabel("Time [s]")
plt.ylabel("Normalized population activity")
plt.title("Activity of Four Izhikevich CPG Populations")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "./images/cpg_four_populations.png",
    dpi=300,
    bbox_inches="tight",
)

# WSL / Agg backend 下不用 plt.show()