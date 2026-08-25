import matplotlib.pyplot as plt

from CNS.spinalcord.izhikevich_cpg import IzhikevichCPG


cpg = IzhikevichCPG(sim_dt=0.002)

duration = 5.0
steps = int(duration / cpg.sim_dt)

times = []
left_values = []
right_values = []

for k in range(steps):
    out = cpg.step()

    times.append(k * cpg.sim_dt)
    left_values.append(out.left)
    right_values.append(out.right)


plt.figure(figsize=(10, 4))

plt.plot(times, left_values, label="Left CPG")
plt.plot(times, right_values, label="Right CPG")

plt.xlabel("Time [s]")
plt.ylabel("Motor primitive")
plt.title("Left and Right CPG Output")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cpg_left_right.png", dpi=300)
plt.show()