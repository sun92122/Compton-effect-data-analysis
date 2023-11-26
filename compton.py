import numpy as np
import matplotlib.pyplot as plt

font = {"family": "Arial", "weight": "bold", "size": 24}
# font = {"weight": "bold", "size": 24} # if no font 'Arial'

plt.rc("font", **font)


def star(x, t=90e-6):
    return x / (1 - t * x)


# comprehensive diffraction experiment
data = np.loadtxt("2.txt", dtype=np.float64)
theta = data[:, 0]
count = data[:, 1]

# peak
peak_degree = np.array([20.7, 23, 44.3, 50.5])
peak_index = np.array([np.where(theta == i)[0][0] for i in peak_degree])
peak_count = count[peak_index]

plt.figure(figsize=(9.5, 8))
plt.plot(theta, count, linewidth=4)
for i in range(len(peak_degree)):
    plt.annotate(
        f"( {peak_degree[i]:.1f}, {peak_count[i]:.0f} )",
        xy=(peak_degree[i], peak_count[i] + 20),
        xytext=(peak_degree[i] - 15, peak_count[i] - 100),
        arrowprops=dict(facecolor="black", shrink=0.05, width=2),
        fontsize=18,
        fontweight="bold",
    )
plt.xticks(fontsize=26, fontweight="bold")
plt.yticks(fontsize=26, fontweight="bold")
plt.xlabel("theta (degree)", fontsize=30, fontweight="bold")
plt.ylabel("intensity (Imp/s)", fontsize=30, fontweight="bold")
ax = plt.gca()
ax.tick_params(top="on", right="on", direction="in", length=6, width=3)
ax.spines["bottom"].set_linewidth(3)
ax.spines["top"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.spines["right"].set_linewidth(3)

plt.tight_layout()
# plt.show()  # DEBUG


# plot the figure of intensity vs theta, with and without Al filter
data = np.loadtxt("3.txt", dtype=np.float64)
theta_with_Al = data[:, 0]
count_with_Al = data[:, 1]
count_with_Al_star = count_with_Al / (1 - 90e-6 * count_with_Al)
count_with_Al_star_err = np.ones(len(count_with_Al_star)) * 0.1 / np.sqrt(12)

data = np.loadtxt("4.txt", dtype=np.float64)
theta_without_Al = data[:, 0]
count_without_Al = data[:, 1]
count_without_Al_star = star(count_without_Al)
count_without_Al_star_err = np.ones(len(count_without_Al_star)) * 0.1 / np.sqrt(12)

plt.figure(figsize=(9.5, 8))
plt.plot(theta_with_Al, count_with_Al_star, linewidth=4, label="with Al filter")
plt.plot(
    theta_without_Al, count_without_Al_star, linewidth=4, label="without Al filter"
)

plt.xticks(fontsize=26, fontweight="bold")
plt.yticks(fontsize=26, fontweight="bold")
plt.xlabel("theta (degree)", fontsize=30, fontweight="bold")
plt.ylabel("intensity (Imp/s)", fontsize=30, fontweight="bold")
ax = plt.gca()
ax.tick_params(top="on", right="on", direction="in", length=6, width=3)
ax.spines["bottom"].set_linewidth(3)
ax.spines["top"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.spines["right"].set_linewidth(3)

plt.legend(prop=font)
plt.tight_layout()

# calculate the transmission and wave length
# wave_length = 2 * d * sin(theta), d = 201.4 pm

transmission = count_with_Al_star / count_without_Al_star
transmission_err = transmission * np.sqrt(
    (count_with_Al_star_err / count_with_Al_star) ** 2
    + (count_without_Al_star_err / count_without_Al_star) ** 2
)
wave_length = 2 * 201.4 * np.sin(np.deg2rad(theta_with_Al))
wave_length_err = (
    2
    * 201.4
    * np.cos(np.deg2rad(np.ones(len(theta_with_Al)) * 0.1 / np.sqrt(12)))
    * np.deg2rad(0.1 / np.sqrt(12))
)

transmission_prime = transmission[12:]
wave_length_prime = wave_length[12:]
wave_length_err_prime = wave_length_err[12:]

# plot the figure of transmission vs wave length

plt.figure(figsize=(9.5, 8))
plt.scatter(wave_length, transmission, linewidths=4)
# plt.scatter(wave_length, transmission, linewidths=4)
plt.xticks(fontsize=26, fontweight="bold")
plt.yticks(fontsize=26, fontweight="bold")
plt.xlabel("wave length (pm)", fontsize=30, fontweight="bold")
plt.ylabel("transmission", fontsize=30, fontweight="bold")
ax = plt.gca()
ax.tick_params(top="on", right="on", direction="in", length=6, width=3)
ax.spines["bottom"].set_linewidth(3)
ax.spines["top"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.spines["right"].set_linewidth(3)

plt.tight_layout()

# Nb = [6, 6, 6]
# N3 = [200, 199, 201]
# N4 = [91, 92, 93]
# N5 = [76, 75, 76]

Nb, N3, N4, N5 = (
    np.array([6, 6, 6]),
    np.array([200, 199, 201]),
    np.array([91, 92, 93]),
    np.array([76, 75, 76]),
)
Nb_star, N3_star, N4_star, N5_star = (
    star(Nb.mean()),
    star(N3.mean()),
    star(N4.mean()),
    star(N5.mean()),
)
Nb_star_err, N3_star_err, N4_star_err, N5_star_err = (
    np.sqrt(Nb_star),
    np.sqrt(N3_star),
    np.sqrt(N4_star),
    np.sqrt(N5_star),
)
print(f"Nb = {Nb.mean():.5f}\tNb* = {Nb_star:.5f}\terr = {Nb_star_err:.5f}")
print(f"N3 = {N3.mean():.5f}\tN3* = {N3_star:.5f}\terr = {N3_star_err:.5f}")
print(f"N4 = {N4.mean():.5f}\tN4* = {N4_star:.5f}\terr = {N4_star_err:.5f}")
print(f"N5 = {N5.mean():.5f}\tN5* = {N5_star:.5f}\terr = {N5_star_err:.5f}")

transmission_1 = (N4_star - Nb_star) / (N3_star - Nb_star)
transmission_2 = (N5_star - Nb_star) / (N3_star - Nb_star)
print(f"T1 = {transmission_1:.5f}\t\tT2 = {transmission_2:.5f}")
transmission_1_err = transmission_1 * np.sqrt(
    (N4_star_err**2 + Nb_star_err**2) / (N4_star - Nb_star) ** 2
    + (N3_star_err**2 + Nb_star_err**2) / (N3_star - Nb_star) ** 2
)
transmission_2_err = transmission_2 * np.sqrt(
    (N5_star_err**2 + Nb_star_err**2) / (N5_star - Nb_star) ** 2
    + (N3_star_err**2 + Nb_star_err**2) / (N3_star - Nb_star) ** 2
)
print(f"T1 err = {transmission_1_err:.5f}\tT2 err = {transmission_2_err:.5f}")


def wave_length(T, T_err):
    ma = np.where(transmission_prime <= T)[0][0]
    mi = np.where(transmission_prime >= T)[0][-1]
    x1 = wave_length_prime[ma]
    x2 = wave_length_prime[mi]
    y1 = transmission_prime[ma]
    y2 = transmission_prime[mi]

    x1_err = wave_length_err_prime[ma]
    x2_err = wave_length_err_prime[mi]
    y1_err = transmission_err[ma]
    y2_err = transmission_err[mi]

    x1_x2_err = np.sqrt(x1_err**2 + x2_err**2)
    y1_y2_err = np.sqrt(y1_err**2 + y2_err**2)
    T_y2_err = np.sqrt(T_err**2 + y2_err**2)

    wave_length = (x1 - x2) / (y1 - y2) * (T - y2) + x2
    wave_length_err = np.sqrt(
        (
            (x1_x2_err / (x1 - x2)) ** 2
            + (y1_y2_err / (y1 - y2)) ** 2
            + (T_y2_err / (T - y2)) ** 2
        )
        * ((x1 - x2) / (y1 - y2) * (T - y2)) ** 2
        + x2_err**2
    )
    # # DEBUG
    # print(f"y1_err = {y1_err:.6f}\ty2_err = {y2_err:.6f}")
    # print(f"x1_err = {x1_err:.6f}\tx2_err = {x2_err:.6f}")
    # print(f"x1 = {x1:.3f}\tx2 = {x2:.3f}\ty1 = {y1:.3f}\ty2 = {y2:.3f}")
    return wave_length, wave_length_err


# calculate the wave length of T1 and T2
wave_length_1, wave_length_1_err = wave_length(transmission_1, transmission_1_err)
wave_length_2, wave_length_2_err = wave_length(transmission_2, transmission_2_err)
print(
    f"T1 wave length = {wave_length_1:.5f}\tT1 wave length err = {wave_length_1_err:.5f}"
)
print(
    f"T2 wave length = {wave_length_2:.5f}\tT2 wave length err = {wave_length_2_err:.5f}"
)

# calculate the wave length of T1-T2
wave_length_diff = wave_length_2 - wave_length_1
print(f"wave length 2 - wave length 1 = {wave_length_diff:.5f}")
# h / m0 / c
true_wave_length_diff = 6.626e-34 / 9.109e-31 / 2.998e8 * 1e12
print(
    f"relative error = {(wave_length_diff - true_wave_length_diff) / true_wave_length_diff * 100:.5f} %"
)

# calculate the error of wave length
print(
    f"wave length diff err = {np.sqrt(wave_length_1_err**2 + wave_length_2_err**2):.5f}"
)


plt.figure(figsize=(9.5, 8))
plt.scatter(wave_length_prime, transmission_prime, linewidths=4)
# draw a line showing the transmission of T1, T2
wave_length_1_max = (wave_length_1 - plt.xlim()[0]) / (plt.xlim()[1] - plt.xlim()[0])
wave_length_2_max = (wave_length_2 - plt.xlim()[0]) / (plt.xlim()[1] - plt.xlim()[0])
transmission_1_max = (transmission_1 - plt.ylim()[0]) / (plt.ylim()[1] - plt.ylim()[0])
transmission_2_max = (transmission_2 - plt.ylim()[0]) / (plt.ylim()[1] - plt.ylim()[0])
# draw the text of T1, T2
plt.annotate(
    r"$T_1 = {:.5f}$".format(transmission_1)
    + "\n"
    + r"$\lambda_1 = {:.5f}$".format(wave_length_1),
    xy=(wave_length_1, transmission_1),
    xytext=(wave_length_1 + 3, transmission_1 + 0.05),
    arrowprops=dict(facecolor="black", shrink=0.05, width=2),
    fontsize=18,
    fontweight="bold",
)
plt.annotate(
    r"$T_2 = {:.5f}$".format(transmission_2)
    + "\n"
    + r"$\lambda_2 = {:.5f}$".format(wave_length_2),
    xy=(wave_length_2, transmission_2),
    xytext=(wave_length_2 + 3, transmission_2 + 0.05),
    arrowprops=dict(facecolor="black", shrink=0.05, width=2),
    fontsize=18,
    fontweight="bold",
)


plt.axhline(
    y=transmission_1,
    xmax=wave_length_1_max,
    color="r",
    linestyle="-",
    linewidth=2,
    clip_on=False,
)
plt.axvline(
    x=wave_length_1,
    ymax=transmission_1_max,
    color="r",
    linestyle="-",
    linewidth=2,
    clip_on=False,
)
plt.axhline(
    y=transmission_2,
    xmax=wave_length_2_max,
    color="r",
    linestyle="-",
    linewidth=2,
    clip_on=False,
)
plt.axvline(
    x=wave_length_2,
    ymax=transmission_2_max,
    color="r",
    linestyle="-",
    linewidth=2,
    clip_on=False,
)


plt.xticks(fontsize=26, fontweight="bold")
plt.yticks(fontsize=26, fontweight="bold")
plt.xlabel("wave length (pm)", fontsize=30, fontweight="bold")
plt.ylabel("transmission", fontsize=30, fontweight="bold")
ax = plt.gca()
ax.tick_params(top="on", right="on", direction="in", length=6, width=3)
ax.spines["bottom"].set_linewidth(3)
ax.spines["top"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.spines["right"].set_linewidth(3)

plt.tight_layout()

plt.show()  # DEBUG
