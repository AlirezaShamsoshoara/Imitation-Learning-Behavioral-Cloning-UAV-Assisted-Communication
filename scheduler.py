#################################
# Packet Scheduler function
#################################

#########################################################
# import libraries
import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import animation
from random import expovariate
from random import paretovariate
from scipy.stats import poisson

#########################################################
# Function definition


def generator_poission(lmbda, counts, users, plotflag):
    """
    This function generates arrival time based on Poisson  distribution
    :param lmbda: The lambda parameter for the Poisson  distribution
    :param counts: number of samples
    :param users: number of UEs
    :param plotflag: FLAG -> to plot or not plot data
    :return: intervals and arrival times
    """
    intervals = np.zeros([users, counts])
    timestamp = np.zeros([users, 1])
    arrival_time = np.zeros([users, counts])
    for user in range(users):
        intervals[user, :] = [expovariate(lmbda[user]) for _ in range(counts)]
        count = 0
        for t in intervals[user, :]:
            timestamp[user, 0] += t
            arrival_time[user, count] = timestamp[user, 0]
            count += 1

    if plotflag:
        fig, ax = plt.subplots(figsize=(5, 3.75))
        markerstyles = ['+', '_', '*', 'x', 'o']
        x = np.arange(0, counts)
        for user, ls, mu in zip(range(users), markerstyles, lmbda):
            # create a poisson distribution
            # we could generate a random sample from this distribution using, e.g.
            ax.plot(x, arrival_time[user, :], color='black', marker=ls, label=r'$\lambda=%i$' % mu, linestyle="None")

        plt.grid(True)
        plt.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
        plt.xlabel('Count(Packet)')
        plt.ylabel('$t$')
        plt.title('Arrival time for Poisson distribution')

        plt.figure()
        linestyles = ['+', '|', '*', 'x', 'o']
        colors = ['blue', 'green', 'red', 'black', 'magenta']
        for mu, ls, color in zip(lmbda, linestyles, colors):
            dist = poisson(mu)
            plt.plot(x, dist.pmf(x), marker=ls, linestyle='--', label=r'$\mu=%i$' % mu)
        plt.xlim(-0.5, 30)
        plt.ylim(0, 0.25)
        plt.grid(True)
        plt.xlabel('$x$', fontsize=14, fontweight="bold")
        plt.ylabel(r'$p(x|\mu)$', fontsize=14, fontweight="bold")
        plt.title('Poisson Distribution for inter-arrival time')
        plt.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')

        plt.show(block=False)

    return arrival_time, intervals


def generator_pareto(shapes, counts, users, plotflag):
    """
    This function generates arrival time based on Pareto distribution
    :param shapes: The shape parameter for Pareto distribution
    :param counts: Number of samples
    :param users: Number of UEs
    :param plotflag: FLAG -> to plot or not plot data
    :return: Intervals and arrival times
    """
    #  Shape should be greater than 0, Using Paretovariate
    intervals = np.zeros([users, counts])
    intervals_norm = np.zeros([users, counts])
    timestamp = np.zeros([users, 1])
    arrival_time = np.zeros([users, counts])
    for user in range(users):
        intervals[user, :] = [paretovariate(shapes[user]) for _ in range(counts)]
        intervals_norm[user, :] = (intervals[user, :] - intervals[user, :].min()) / (intervals[user, :].max() -
                                                                                     intervals[user, :].min())
        count = 0
        for t in intervals_norm[user, :]:
            timestamp[user, 0] += t
            arrival_time[user, count] = timestamp[user, 0]
            count += 1

    if plotflag:
        fig, ax = plt.subplots(figsize=(5, 3.75))
        markerstyles = ['x', '_', '1']
        x = np.arange(0, 100)
        for user, ls, shape in zip(range(users), markerstyles, shapes):
            plt.plot(x, arrival_time[user, :], color='black',
                     marker=ls, label=r'$shape=%i$' % shape, linestyle="None")
        plt.grid()
        plt.legend()
        plt.xlabel('Count')
        plt.ylabel('$t$')
        plt.title('Arrival time for Pareto distribution')

        plt.show(block=False)
    return arrival_time, intervals


def service_time_generator(mu, counts, users):
    """
    This function generates random service time using exponential distribution.
    :param mu: mu parameter for the exponential distribution
    :param counts: Number of samples
    :param users: Number of UEs
    :return: Numpy vector for the service time
    """
    service_time = np.zeros([users, counts])
    for user in range(users):
        service_time[user, :] = [expovariate(mu[user]) for _ in range(counts)]
    return service_time
