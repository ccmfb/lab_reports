import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.optimize as opt


def main():
    # Loading in all the CSV files and assigning them to each method.
    method1, method2, method3 = load_data(
        'method_1.csv', 'method_2.csv', 'method_3_gen.csv', 'method_3_measured.csv'
    )

    '''
    GENERAL DATA PLOTS:
    plot_data(method1, method2, method3)
    
    LINEAR PLOTS AND CALCULATIONS:
    linear_calculation_m1(method1)
    linear_calculation_m2(method2)
    sine_calculation(method3)
    '''


def sine_calculation(df: pd.DataFrame):
    t_range = np.linspace(0,12, 300)
    gen_opt, gen_cov = opt.curve_fit(sine_func, df.index, df['Voltage Generated (V)'],
                                     [1,3,0,1])
    meas_opt, meas_cov = opt.curve_fit(sine_func, df.index, df['Voltage Measured (V)'],
                                       [1,3,0,1])

    print(f'M3 gen opt: {gen_opt}, \n'
          f'M3 gen cov: {gen_cov} \n')

    print(f'M3 meas opt: {meas_opt}, \n'
          f'M3 meas cov: {meas_cov}')

    sns.set_style("whitegrid")
    sns.set_context('poster')

    f, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.set_title('Method 3: Fitted Sine Curves', fontsize=35, weight='bold')
    ax.set_xlabel('Time (*10^-5 s)', fontsize=30, weight='bold')
    ax.set_ylabel('Voltage (V)', fontsize=30, weight='bold')
    sns.scatterplot(data=df, x='Time (*10^-5 s)', y='Voltage Generated (V)',
                    color='dimgray', marker='o', s=10)
    sns.scatterplot(data=df, x='Time (*10^-5 s)', y='Voltage Measured (V)',
                    color='grey', marker='o', s=10)

    sns.lineplot(t_range, sine_func(t_range, *gen_opt), color='royalblue')
    sns.lineplot(t_range, sine_func(t_range, *meas_opt), color='orange')

    #plt.savefig('Pictures/m3_fitted_sine_curves.png')
    plt.show()

    print(calc_capacitance(gen_opt[0], meas_opt[0], gen_opt[2], meas_opt[2], gen_opt[1]))


def calc_capacitance(V_g, V_m, a1, a2, omega_gen):
    a = a1 - a2
    v = omega_gen * 10**5 / (2 * np.pi)
    V_r1 = np.sqrt((V_g * np.cos(a) - V_m)**2 + (V_g * np.sin(a))**2)
    I_g = V_r1 / (6.8 * 10**3)
    cos_theta = (V_g * np.sin(a)) / V_r1
    X_c = V_m / (I_g * cos_theta)
    C_t = 1 / (2 * np.pi * v * X_c)

    return C_t


def linear_calculation_m2(df: pd.DataFrame):
    # Separate discharge curves:
    dis1 = df[df.index < 0.0005]
    dis2 = df[df.index > 0.001]
    dis2 = dis2[dis2.index < 0.0015]
    dis3 = df[df.index > 0.002]

    # Add column for discharge number
    dis1['Discharge Number'] = [1 for i in range(len(dis1.index))]
    dis2['Discharge Number'] = [2 for i in range(len(dis2.index))]
    dis3['Discharge Number'] = [3 for i in range(len(dis3.index))]

    # Move data sets to begin at t=0
    dis2.index -= dis2.index.tolist()[0]
    dis3.index -= dis3.index.tolist()[0]

    # Put them into one df
    lin_df = dis1.append(dis2)
    lin_df = lin_df.append(dis3)

    lin_df['Voltage (V)'] = np.log(lin_df['Voltage (V)'])
    lin_df = lin_df.rename(columns={'Voltage (V)': 'ln(V)'})

    # Adjust time interval and change to a better unit
    lin_df = lin_df[lin_df.index < 0.0001]
    lin_df.index *= 10**5
    lin_df.index.names = ['Time (*10^-5 s)']

    # Curve fit
    t_range = np.linspace(0,10, 200)
    popt, pcov = opt.curve_fit(lin_func, lin_df.index, lin_df['ln(V)'])
    print(f'M2 fit: {popt}, \n'
          f'M2 cov: {pcov} \n')

    # Shuffle df (makes it look nicer when plotting)
    lin_df = lin_df.sample(frac=1)

    sns.set_style("whitegrid")
    sns.set_context('poster')

    f, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.set_title('Method 2: Linear Discharge Curves (0-10 * 10^-5 s)', fontsize=35, weight='bold')
    ax.set_xlabel('Time (*10^-5 s)', fontsize=30, weight='bold')
    ax.set_ylabel('ln(V)', fontsize=30, weight='bold')
    sns.scatterplot(x='Time (*10^-5 s)', y='ln(V)', data=lin_df, hue='Discharge Number',
                    s=20, marker='o', legend=False)
    sns.lineplot(t_range, lin_func(t_range, *popt), color='crimson')

    ax.legend(labels=['Different Discharge Curves', 'Fit'])

    #plt.savefig('Pictures/m2_linear_discharge_curves.png')
    plt.show()


def linear_calculation_m1(df1: pd.DataFrame):
    lin_df = pd.DataFrame({
        'Time (s)': df1.index.tolist(),
        'ln(V)': np.log(df1.values.flatten().tolist())
    })
    lin_df = lin_df.set_index('Time (s)')
    lin_df = lin_df[lin_df.index < 20]

    t_range = np.linspace(0,20,400)
    popt, pcov = opt.curve_fit(lin_func, lin_df.index, lin_df['ln(V)'])

    print(f'M1 fit: {popt}, \n'
          f'M1 cov: {pcov} \n')

    sns.set_style("whitegrid")
    sns.set_context('poster')

    f, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.set_title('Method 1: Linear Discharge Curve (0-20s)', fontsize=35, weight='bold')
    ax.set_xlabel('Time (s)', fontsize=30, weight='bold')
    ax.set_ylabel('ln(V)', fontsize=30, weight='bold')
    sns.scatterplot(x='Time (s)', y='ln(V)', data=lin_df, s=20, marker='+')
    sns.lineplot(t_range, lin_func(t_range, *popt), color='red')
    ax.legend(labels=['Data', 'Fit'], fontsize = 'large')

    #plt.savefig('Pictures/m1_linear_discharge_curve.png')
    plt.show()


def plot_data(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
    sns.set_style("whitegrid")
    sns.set_context('poster')

    f, ax = plt.subplots(1,1,figsize=(12,8))
    '''
    METHOD 1:
    ax.set_title('Method 1: Discharge Curve', fontsize=35, weight='bold')
    ax.set_xlabel('Time (s)', fontsize=30, weight='bold')
    ax.set_ylabel('Voltage (V)', fontsize=30, weight='bold')
    sns.scatterplot(data=df1, x='Time (s)', y='Voltage (V)', marker='o', s=10)
    plt.savefig('Pictures/method1_discharge_curve.png')
    
    METHOD 2:
    ax.set_title('Method 2: Charging and Discharging Curves', fontsize=35, weight='bold')
    ax.set_xlabel('Time (s)', fontsize=30, weight='bold')
    ax.set_ylabel('Voltage (V)', fontsize=30, weight='bold')
    sns.scatterplot(data=df2, x='Time (s)', y='Voltage (V)', marker='o', s=10)
    plt.savefig('Pictures/method2_charging_discharging_curves.png')

    METHOD 3:
    ax.set_title('Method 3: Generated and Measured Voltage', fontsize=35, weight='bold')
    ax.set_xlabel('Time (*10^-5 s)', fontsize=30, weight='bold')
    ax.set_ylabel('Voltage (V)', fontsize=30, weight='bold')
    sns.scatterplot(data=df3, x='Time (*10^-5 s)', y='Voltage Generated (V)', marker='o', s=10)
    sns.scatterplot(data=df3, x='Time (*10^-5 s)', y='Voltage Measured (V)', marker='o', s=10)
    plt.savefig('Pictures/method3_gen_meas_voltage.png')
    '''

    plt.show()


def load_data(path1: str, path2: str, path3_gen: str, path3_meas: str):
    df1 = pd.read_csv(path1)
    df1 = df1.rename(columns={'in s': 'Time (s)', 'C1 in V': 'Voltage (V)'})
    df1 = df1.set_index('Time (s)')
    df1 = df1[df1.index > 34.57389]
    df1.index -= 34.57389 # Sets the t values to start from 0
    df1 = df1[::10]

    df2 = pd.read_csv(path2)
    df2 = df2.rename(columns={'in s': 'Time (s)', 'C1 in V': 'Voltage (V)'})
    df2 = df2.set_index('Time (s)')
    df2.index -= 3.249053 - 4.56*10**-7 - 1.84*10**-8
    #df2 = df2[::10]

    df3_gen = pd.read_csv(path3_gen)
    df3_meas = pd.read_csv(path3_meas)
    df3 = df3_gen.rename(columns={'in s': 'Time (*10^-5 s)', 'C2 in V': 'Voltage Generated (V)'})
    df3['Voltage Measured (V)'] = df3_meas['C1 in V']
    df3 = df3.set_index('Time (*10^-5 s)')
    df3.index -= 3.250192
    df3.index *= 10**(5)
    #df3 = df3[::10]

    return df1, df2, df3


def lin_func(t, V_0, RC):
    return np.log(V_0) - np.array(t) / (RC)


def sine_func(t, amp, freq, phi, c):
    return amp * np.sin(freq * np.array(t) + phi) + c


if __name__ == '__main__':
    main()
