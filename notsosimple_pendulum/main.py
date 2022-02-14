import scipy.optimize as opt
import ast
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import seaborn as sns


def main():
    # Reading in position data
    dfpos = pd.read_csv('position_data.csv', sep=',', index_col=[0])
    dffps = pd.read_csv('fps_data.csv', sep=',')

    # Looping through every trial and appending frequency to array
    xvalues, yvalues = [], []
    for count, row in dfpos.iterrows():
        pos = np.array(ast.literal_eval(clean_list(row['pos'])))
        fps = np.array(dffps['fps'].iloc[count])

        xy = plot_position(pos, fps, ind=(count+1), OUTPUT=0)
        xvalues.append(xy[0])
        yvalues.append(xy[1])

    # Plotting Time period vs amplitude and finding curve fit
    plot_timeperiod(np.array(xvalues), np.array(yvalues))
    

def plot_timeperiod(xvalues, yvalues, OUTPUT=1):
    amplitudes = np.array([2,3,3,3,6,6,6,9,9,12,12,12,15,15,15,18,18,18,21,21,21])
    
    xtimeperiods = 2 * np.pi / xvalues[:,1,0]
    xerror = (xvalues[:,1,1] / xvalues[:,1,0]) * xtimeperiods # Calculates percentage error times the new value (time period)
    x = np.c_[xtimeperiods.reshape((24,1)), xerror.reshape((24,1))]
    x = np.delete(x, [1,2,11], axis=0)

    ytimeperiods = 2 * np.pi / yvalues[:,1,0]
    yerror = (yvalues[:,1,1] / yvalues[:,1,0]) * ytimeperiods # Calculates percentage error times the new value (time period)
    y = np.c_[ytimeperiods.reshape((24,1)), yerror.reshape((24,1))]
    y = np.delete(y, [1,2,11], axis=0)

    comb = np.concatenate((x[:,0],y[:,0]), axis=0)
    comb_amp = np.concatenate((amplitudes, amplitudes), axis=0)
    comb_err = np.concatenate((x[:,1],y[:,1] ), axis=0)
    print(f'Excluding outliers which are more than 1 std away, mean +/- std = {np.mean(comb)} +/- {np.std(comb)}')

    popt, pcov = opt.curve_fit(quad_func, comb_amp, comb, [1.37, 0, (1/16)], sigma=comb_err)
    print(f'T_0 = {popt[0]} +/- {np.sqrt(pcov[0,0])}, alpha = {popt[1]} +/- {np.sqrt(pcov[1,1])}, beta = {popt[2]} +/- {np.sqrt(pcov[2,2])}')

    xopt, xcov = opt.curve_fit(quad_func, amplitudes, x[:,0], [1.37, 0, (1/16)], sigma=x[:,1])
    print(f'T_0 = {xopt[0]} +/- {np.sqrt(xcov[0,0])}, alpha = {xopt[1]} +/- {np.sqrt(xcov[1,1])}, beta = {xopt[2]} +/- {np.sqrt(xcov[2,2])}')

    yopt, ycov = opt.curve_fit(quad_func, amplitudes, y[:,0], [1.37, 0, (1/16)], sigma=y[:,1])
    print(f'T_0 = {yopt[0]} +/- {np.sqrt(ycov[0,0])}, alpha = {yopt[1]} +/- {np.sqrt(ycov[1,1])}, beta = {yopt[2]} +/- {np.sqrt(ycov[2,2])}')

    if OUTPUT == 1:
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        amp_range = np.linspace(2, 21, 200)
        
        ax.errorbar(
                amplitudes, 
                y[:,0], 
                xerr=0.5,
                yerr=y[:,1],
                ms=10, 
                fmt='.',
                capsize=4.0,
                color='#619CFF', 
                label='Vertical Time Periods'
            )

        ax.errorbar(
                amplitudes, 
                x[:,0], 
                xerr=0.5,
                yerr=x[:,1],
                ms=10, 
                fmt='.',
                capsize=4.0,
                color='#F8766D', 
                label='Horizontal Time Periods'
            )
            
        ax.plot(
                amp_range, 
                quad_func(amp_range, 1.34272, 0, (1/16)), 
                color='#00BA38', 
                label='Theory'
            )

        ax.plot(
                amp_range, 
                quad_func(amp_range, xopt[0], xopt[1], xopt[2]), 
                color='black', 
                label='Curve Fit for Horizontal Points'
            )

        ax.set_xlabel(r'Amplitude $(\theta)$')
        ax.set_ylabel(r'Time Period $(T)$')
        ax.legend(loc='lower right')

        fig.savefig(f'plot_T.jpg', dpi=400)
        plt.show()


def plot_position(pos, fps, ind, OUTPUT=1):
    xdata, ydata = filter_points(pos)

    xdata[0] = xdata[0] / fps
    ydata[0] = ydata[0] / fps
    xrange = np.linspace(xdata[0][0], xdata[0][-1], 200)
    yrange = np.linspace(ydata[0][0], ydata[0][-1], 200)

    # Calculating fit
    xopt, xcov = opt.curve_fit(sine_func, xdata[0], xdata[1], initial_guess(xdata), sigma=xdata[2])
    yopt, ycov = opt.curve_fit(sine_func, ydata[0], ydata[1], initial_guess(ydata), sigma=ydata[2])
    print(f'{ind}: {xopt}')

    # Transforming resutls in a better format
    xcov_eye = np.eye(4) * xcov
    xsigma = np.matmul(xcov_eye, np.ones((4,1,)))
    xsigma = np.sqrt(xsigma)
    xresult = np.c_[np.array(xopt).reshape((4,1)), xsigma]

    ycov_eye = np.eye(4) * ycov
    ysigma = np.matmul(ycov_eye, np.ones((4,1,)))
    ysigma = np.sqrt(ysigma)
    yresult = np.c_[np.array(yopt).reshape((4,1)), ysigma]

    if OUTPUT == 1:
        fig, ax = plt.subplots(2,1,figsize=(8,7))
        
        index = np.where(pos[:,1] == -1)
        exc = np.delete(pos, index, axis=0)
        
        # Plot 1
        ax[0].scatter(
                exc[:,0]/fps, 
                exc[:,1], 
                marker='x', 
                s=12, 
                color='#619CFF', 
                label='Excluded Points'
            )

        ax[0].scatter(
                xdata[0], 
                xdata[1], 
                marker='x', 
                s=12, 
                color='#F8766D', 
                label='Included Ponits'
            )

        ax[0].fill_between(
                xdata[0], 
                xdata[1]-xdata[2], 
                xdata[1]+xdata[2], 
                color='#F8766D', 
                alpha=0.2
            )

        ax[0].plot(
                xrange, 
                sine_func(xrange, *xopt), 
                color='black', 
                label='Fitted Line'
            )

        ax[0].set_ylabel('Horizontal Distance (pixels)')

        # Plot 2
        ax[1].scatter(
                exc[:,0] / fps, 
                exc[:,2], 
                marker='x', 
                s=12, 
                color='#619CFF', 
                label='Excluded Points'
            )

        ax[1].scatter(
                ydata[0], 
                ydata[1], 
                marker='x', 
                s=12, 
                color='#F8766D', 
                label='Included Ponits'
            )

        ax[1].fill_between(
                ydata[0], 
                ydata[1]-ydata[2], 
                ydata[1]+ydata[2], 
                color='#F8766D', 
                alpha=0.2, 
                label='Uncertainty'
            )

        ax[1].plot(
                yrange, 
                sine_func(yrange, *yopt), 
                color='black', 
                label='Fitted Line'
            )

        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Vertical Distance (pixels)')
        ax[1].legend(loc='upper right')

        fig.savefig(f'plots/plot_for_{ind}.jpg', dpi=400)
        plt.close(fig)

    return xresult, yresult


def initial_guess(data):
    a = (max(data[1]) - min(data[1])) * 100
    freq = (2 * np.pi) / 1.343
    phi = np.pi / 2
    c = sum(data[1]) / data.shape[0]
    return np.array([a, freq, phi, c], dtype=float).flatten()


def sine_func(x, a, freq, phi, c):
    return a * np.sin(x*freq + phi) + c


def filter_points(pos, alpha=1.5):
    # Removing empty rows
    index = np.where(pos[:,1] == -1)
    pos = np.delete(pos, index, axis=0)

    # Removing first 20 frames
    xpos = pos[24:,1]
    xerr = pos[24:,3]
    ypos = pos[24:,2]
    yerr = pos[24:,4]
    frames = pos[24:,0]

    return np.array([frames, xpos, xerr], dtype='float64'), np.array([frames, ypos, yerr], dtype='float64')


def clean_list(l):
    #l = l.replace('[[','[')
    #l = l.replace(']]',']')
    l = l.replace('[ ', '[')
    l = l.replace('[  ', '[')
    l = l.replace('[   ', '[')
    l = l.replace('[    ', '[')
    l = ','.join(l.split())
    l = l.replace('\n ',',')
    l = l.replace('[,', '[')

    return l


def quad_func(x, t_0, a, b):
    x = (x/360) * 2 * np.pi
    return t_0 * (1 + a * x + b * x**2)
    

if __name__ == '__main__':
    plt.style.use('normalStyle.mplstyle')

    main()
