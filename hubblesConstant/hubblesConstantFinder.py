import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np

'''
1.  Load in all the data + convert them to a usable dataframe (function: load_data())

2.  Plot the the intensity vs. wavelength graphs, saves them as a png, and returns the lambda values
    (with uncertainties), which then get added to the dataframe (function: plot_intensity())
    
        2.1. Uses estimates from get_estimates() to fit the function to the data points
        
3. Plot the graph of velocity vs. distance and calculate the slope (hubble constant)

        3.1. To get the velocities, I first calculated it by rearranging the equation that relates the
        observed and expected values of the wavelength with the speed (Redshift). Also added another
        column in the dataframe with the uncertainties in the speed.
        
NOTES: 
    - I tried to organize the functions in chronological order, thus the setup with a main function on top
    - The console outputs are the dataframe, including all the data for each observation,
        and the hubble constant
    - Tried to focus more on comprehensibility than efficiency. For example, it would probably be more efficient to
        remove the intensity data from the dataframe after using it. But then the dataframe would not include all the
        data.
'''


def main():
    # Loads in the data from the files. Also, saves them to one variable which stores all the wavelengths measured
    # and one which stores the data points for each observation.
    wavelen, data = load_data()

    # Plots the intensity vs. wavelength graphs and adds columns with lambda_0 and the uncertainty in lambda_0 to
    # the dataframe.
    data['Lambda_0'], data['Uncertainty Lambda_0'] = plot_intensity(wavelen, data)

    # Adds columns with velocities and uncertainty form the lambda values.
    data['v'], data['Uncertainty v'] = get_velocities(data)

    # Plots the graph of velocities vs. distance and fits a regression line, also gets the value for the slope with its
    # corresponding uncertainty (hubble's constant).
    hubble_constant = plot_velocities(data)

    # Complete data set:
    print(f'Complete Dataframe: \n'
          f'{data.to_string()}\n'
          f'---------------------------------------------------\n')

    # Hubble's constant:
    print(f'Hubble´s Constant with Weighted  Error: {hubble_constant[0]} +/- {hubble_constant[1]} km s-1 Mpc-1 \n')


# Load in files and return dataframe
def load_data():
    # Reading in csv of spectral data
    df = pd.read_csv('Data/Halpha_spectral_data.csv', skiprows=4, header=None)

    # Extract the frequencies measurements were taken at. Then converting them into wavelengths in nm.
    freq = df.iloc[[0], 1:1001].values.flatten().tolist()
    wavelen = (2.99*10**8 / np.array(freq)) * 10**9

    # Remove measurement frequencies from dataframe and setting the observations as the index of the dataframe.
    df = df.drop([0])
    df.set_index([0], inplace=True)

    # Reading in txt of distances and valid measurement, as well as making the observations the index.
    dist = np.loadtxt('Data/Distance_Mpc.txt')
    dist_df = pd.DataFrame(data=dist, columns=['Observation', 'Distance (Mpc)', 'Valid Response'])
    dist_df.set_index('Observation', inplace=True)

    # Creating a new dataframe, by filtering out invalid measurement responses. Then removing the column with
    # the measurement responses.
    data = dist_df[dist_df['Valid Response'] == 1]
    data = data.drop(['Valid Response'], axis=1)

    # Adding the all the intensity data points for each observation in a column of the dataframe.
    spec_data = []
    for obs in data.index:
        spec_data.append(df.loc[df.index == obs].values.flatten().tolist())
    data['Intensity'] = spec_data

    # Functions returns the wavelengths and the dataframe
    return wavelen, data


# Function plots all the intensity graphs and saves them as a png. It also calculates lambda_0 by fitting
# a line with a gaussian distribution to the data points.
def plot_intensity(x: list, df: pd.DataFrame):
    # Sets the font on the graph.
    plt.rcParams.update({'font.size': 22})

    # Creates as many subplots as there are observations.
    fig, axes = plt.subplots(len(df.index), 1, figsize=(8, 160))

    # Lambda_0 and sigma_lambda are the values that the function will return. Namely, they are the peak wavelength
    # and the standard deviation in the gaussian distribution.
    lambda_0 = []
    sigma_lambda = []

    # While looping through every observation, each intensity vs. wavelength graph is plotted, and the values for
    # lambda_0 and sigma_lambda are added to the list.
    for i, obs in enumerate(df.index):
        # Fits the line + gaussian to the points. Makes use of the get_estimates() function, which is defined below.
        popt, pcov = opt.curve_fit(func, x, df.at[obs, 'Intensity'], get_estimates(x, df.at[obs, 'Intensity']))
        lambda_0.append(popt[3])
        sigma_lambda.append(popt[4])

        axes[i].set_title(f'Observation: {obs}')
        axes[i].set_xlabel('Wavelength (nm)')
        axes[i].set_ylabel('Intensity')
        axes[i].scatter(x, df.at[obs, 'Intensity'], s=12)
        axes[i].plot(x, func(x, *popt), c='orange', linewidth=4)

    plt.tight_layout()
    plt.savefig('intensity_graphs.png')

    return lambda_0, sigma_lambda


# Fitted function
def func(x, m, c, a, mu, sig):
    lin = m * np.array(x) + c
    gauss = a * np.e**(-(1/2)*((np.array(x) - mu) / sig)**2)
    return lin + gauss


# This function returns estimates for the fitted function.
def get_estimates(x: list, l: list):
    # m (the gradient) is estimated as the (last intensity value - the first) / (last wavelength - first)
    m = (l[-1] - l[0]) / (x[-1] - x[0])

    # Once we have m, we can estimate c (y-intercept) by taking an intensity value and taking away the
    # corresponding wavelength times m. (By rearranging y = mx + c)
    c = l[0] - m * x[0]

    # a (value corresponding to the height of the gaussian) does not differ significantly between values.
    # taking away any point from the maximum will give the right order of magnitude for the scipy curve fit
    # function to work.
    a = max(l) - l[0]

    # Similarly sig (the standard deviation) is so similar that we can just look at any graph and give a guess.
    sig = 10

    # mu (lambda_0) can be estimated by calculating the distance from each point to the line function, and then
    # getting the wavelength when this value is the largest.
    dist_to_line = np.array(l) - (m*x + c)
    mu = x[dist_to_line.tolist().index(max(dist_to_line.tolist()))]

    return [m, c, a, mu, sig]


def get_velocities(df: pd.DataFrame):
    # By rearranging the formula relating the expected and observed wavelengths (Redshift) and converting to km/s
    v = 0.001 * (2.9979 * 10**8) * (
        (df['Lambda_0']**2 - 656.28**2) / (df['Lambda_0']**2 + 656.28**2)
    )

    # To get the uncertainty in v, we have to times the %-uncertainty in lambda_0 by 4. Then we times it by the
    # speed again to get the absolute uncertainty.
    sigma_v = 4 * (df['Uncertainty Lambda_0'] / df['Lambda_0']) * v

    return v, sigma_v


def plot_velocities(df: pd.DataFrame):
    # Make one plot.
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))

    # popt gives the weighted regression line.
    popt, pcov = opt.curve_fit(best_fit_func, df['Distance (Mpc)'], df['v'], sigma=df['Uncertainty v'],
                               absolute_sigma=True)

    # The xrange for the regression lines. Beginning at the smallest value for the wavelength and going to the largest.
    # Therefore does not include extrapolation.
    xrange = np.linspace(df['Distance (Mpc)'].min(), df['Distance (Mpc)'].max(), 400)

    # Sets labels for the graph
    axes.set_title('Velocity vs. Distance')
    axes.set_xlabel('Distance (Mpc)')
    axes.set_ylabel('Velocity (km/s)')
    axes.grid()

    # Plots the data points with error bars and the regression line.
    axes.errorbar(df['Distance (Mpc)'], df['v'], yerr=df['Uncertainty v'], fmt='o', capsize=3,
                  ms=4, label='Data')
    axes.plot(xrange, best_fit_func(xrange, *popt), c='orange', linewidth=2,
              label='Weighted Best Fit')

    axes.legend()
    plt.tight_layout()
    plt.savefig('velocity_distance.png')

    # Returns the weighted value for hubble's constant (slope of regression lines) and its uncertainty
    return [popt[0], pcov[0][0]]


# Function for the scipy curve fit optimization
def best_fit_func(x, m, c):
    return m * np.array(x) + c


if __name__ == '__main__':
    main()
