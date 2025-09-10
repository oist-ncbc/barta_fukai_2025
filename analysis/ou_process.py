import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if __name__ == '__main__':
    dt = 0.1  # ms
    sqrt_dt = np.sqrt(dt)
    T = 1000  # ms
    n_steps = int(T / dt)

    taue = 6.0  # ms
    taui = 6.0  # ms
    mu_e = 6.6  # nS
    mu_i = 12.2  # nS
    sigma_e = 1.6  # nS
    sigma_i = 3.2  # nS
    rho = 0.2  # correlation coefficient

    # Preallocate arrays
    ge = np.zeros(n_steps)
    gi = np.zeros(n_steps)
    ge[0] = mu_e
    gi[0] = mu_i

    # Generate uncorrelated Gaussian noise
    xi_e = np.random.randn(n_steps)
    xi_i = np.random.randn(n_steps)

    # Simulate correlated OU process
    for t in range(1, n_steps):
        correlated_noise = rho * xi_e[t-1] + np.sqrt(1 - rho**2) * xi_i[t-1]
        dge = ((mu_e - ge[t-1]) / taue) * dt + sigma_e * np.sqrt(2 / taue) * sqrt_dt * xi_e[t-1]
        dgi = ((mu_i - gi[t-1]) / taui) * dt + sigma_i * np.sqrt(2 / taui) * sqrt_dt * correlated_noise
        ge[t] = ge[t-1] + dge
        gi[t] = gi[t-1] + dgi

    time = np.linspace(0, T, n_steps)

    fig, ax = plt.subplots()
    ax.plot(time, ge, color='black')
    ax.plot(time, gi, color='gray')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Conductance (nS)')

    # Add inset
    # axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
    # axins.scatter(ge, gi, s=0.1, alpha=0.3, color='black')
    # axins.set_xlabel('ge', fontsize=8)
    # axins.set_ylabel('gi', fontsize=8)
    # axins.tick_params(labelsize=6)

    # print(stats.pearsonr(ge, gi)[0], ge.mean(), gi.mean(), ge.std(), gi.std())
    np.savetxt('plotting/data/ou.csv', np.array([time, ge, gi]))

    ax.set_xlim(0, 1000)

    # plt.tight_layout()
    plt.savefig('img/ou.png')
