import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

# DATA
M_r, abs_sig_M_r = 0.26725, 0.000005
M, abs_sig_M = 0.74407, 0.000005
L, abs_sig_L = 0.787, 0.0005
h, abs_sig_h = 0.535, 0.0005

# Time Period of Rod
T_r_samepos = 0.25 * np.array([
    52.33,
    52.31,
    52.03,
    51.81,
    52.00,
    52.03,
    51.35
])
T_r_difpos = 0.25 * np.array([
    52.33,
    52.31,
    52.17,
    52.97,
    52.14,
    51.96,
    52.47,
    52.00
])
T_r = np.concatenate((T_r_samepos, T_r_difpos))
abs_sig_T_r = np.std(T_r, ddof=1)

# Time Period of Pendulum
T_p_samepos = 0.25 * np.array([
    123.39,
    125.06,
    125.46,
    125.48,
    122.52,
    122.25,
    125.12,
    125.50
])
T_p_difpos = 0.25 * np.array([
    123.39,
    125.06,
    121.32,
    120.85,
    120.12,
    119.75,
    123.37,
    123.78
])
T_p = np.concatenate((T_p_samepos, T_p_difpos))
abs_sig_T_p = np.std(T_p, ddof=1)

# Time Period of Stiff Pendulum
T_s = 0.1 * np.array([
    17.16,
    16.69,
    16.67,
    16.78,
    17.03,
    16.75,
    16.95,
    16.63,
    16.95,
    16.72
])
abs_sig_T_s = np.std(T_s, ddof=1)


# Calculations for the inertia of the pendulum at centre of mass
I_com = (1/12) * M_r * (L**2) * (np.mean(T_p)/np.mean(T_r))**2

# Calculations for the inertia at knife edge
I = I_com + M * h**2

# Calculations for acceleration due to gravity
g = (4 * np.pi**2 * I) / (M * h * np.mean(T_s)**2)


# Uncertainty in I_com, adding percentage uncertainties
sig_time_ratio = (abs_sig_T_p/np.mean(T_p)) + (abs_sig_T_r/np.mean(T_r))
sig_I_com = (abs_sig_M_r/M_r) + 2*(abs_sig_L/L) + 2*sig_time_ratio

# Uncertainty in I, adding absolute uncertainties
abs_sig_I = sig_I_com*I_com + ((abs_sig_M/M) + 2*(abs_sig_h/h))*M*h**2

# Uncertainty in g, adding percentage uncertainties
sig_g = (abs_sig_I/I) + (abs_sig_M/M) + (abs_sig_h/h) + 2*(abs_sig_T_s/np.mean(T_s))
abs_sig_g = sig_g * g

# Uncertainties in final calculation
print(f'Inertia(%-error):{(abs_sig_I/I)},\nMass(%-error):{(abs_sig_M/M)},\nh(%-error):{(abs_sig_h/h)},\nTs(%-error):{2*(abs_sig_T_s/np.mean(T_s))},\nTotal(%-error):{sig_g}')

# Value of g and uncertainty
print(f'\ng = {g:.3f}+-{abs_sig_g:.3f} ms-2')



# Graphs
sns.set_style("whitegrid")
fig, axes = plt.subplots(2,3,figsize=(18,12))
fig2, axes2 = plt.subplots(1,1, figsize=(5,5))

def create_T_graph(pos, data, xlabel, binrange):
    axes[pos[0]][pos[1]].set_title(f'Mean={np.mean(data):.3f}, Std={np.std(data, ddof=1):.3f}, %-error={np.std(data, ddof=1)/np.mean(data):.4f}')
    axes[pos[0]][pos[1]].set_xlabel(xlabel)
    sns.histplot(data, ax=axes[pos[0]][pos[1]], kde=True,bins=10, binrange=binrange, color='steelblue')

# Time Periods for Rod
create_T_graph((0,0), T_r, 'All values of T_r (s)', (12.5,13.5))
create_T_graph((0,1), T_r_samepos, 'Same position - T_r (s)', (12.5,13.5))
create_T_graph((0,2), T_r_difpos, 'Different position - T_r (s)', (12.5,13.5))

# Time Periods for Pendulum
create_T_graph((1,0), T_p, 'All values of T_p (s)', (29.5,32))
create_T_graph((1,1), T_p_samepos, 'Same position - T_p (s)', (29.5,32))
create_T_graph((1,2), T_p_difpos, 'Different position - T_p (s)', (29.5,32))

axes2.set_title(f'Mean={np.mean(T_s):.3f}, Std={np.std(T_s):.3f}, %-error={np.std(T_s)/np.mean(T_s):.4f}')
axes2.set_xlabel('T_s (s)')
sns.histplot(T_s, ax=axes2, kde=True, bins=10, binrange=(1.5,2), color='steelblue')


fig.savefig('torsion_pendulum_graphs.png')
fig2.savefig('stiff_pendulum_graphs.png')
plt.show()

