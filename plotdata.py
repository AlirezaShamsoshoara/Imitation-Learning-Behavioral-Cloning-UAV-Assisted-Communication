"""
#################################
# plot functions for visualization
#################################
"""

#########################################################
# import libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from config import Radius
from config import Config_Queue
from config import Config_General as General

#########################################################
# General Parameters
circle = General.get('Circle')
cbr_rate = General.get('CBR_RATE')
num_angles = General.get('NUM_ANGLE')
queue_lim = Config_Queue.get('Queue_limit')

#########################################################
# Function definition


def plotbar(queue, drop, fig, active_user):
    ues = np.arange(0 + 1, len(queue) + 1)
    if fig is None:
        plt.ion()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(211)
        ax.bar(ues, queue, align='center', alpha=1, width=0.2, tick_label=ues, color="blue")
        ax.set_ylim(0, cbr_rate)
        ax.set_xlabel("UEs", size=12, fontweight='bold')
        ax.set_ylabel("Packets in Queue", size=12, labelpad=10, fontweight='bold')
        ax.grid()

        ax2 = fig.add_subplot(212)
        ax2.bar(ues, drop, align='center', alpha=1, width=0.2, tick_label=ues, color="red")
        ylim = max(cbr_rate/2, max(drop))
        ax2.set_ylim(0, ylim)
        ax2.set_xlabel("UEs", size=12, fontweight='bold')
        ax2.set_ylabel("Dropped Packets", size=12, labelpad=10, fontweight='bold')
        ax2.grid()
        fig.canvas.draw()
        # plt.show(block=True)
    else:
        bars1 = [rect for rect in fig.axes[0].patches]
        bars2 = [rect for rect in fig.axes[1].patches]
        index = 0
        for bar1, bar2 in zip(bars1, bars2):
            bar1.set_height(queue[index])
            bar2.set_height(drop[index])
            if index == active_user:
                bar1.set_color('g')
            else:
                bar1.set_color('b')
            index += 1
        # fig.axes[0].bar(ues, queue, align='center', alpha=1, width=0.3, tick_label=ues, color="blue")
        # fig.axes[1].bar(ues, drop, align='center', alpha=1, width=0.3, tick_label=ues, color="red")
        ylim = max(cbr_rate / 2, max(drop)) + 20
        fig.axes[1].set_ylim(0, ylim)
    fig.canvas.draw()
    # plt.pause(0.000001)
    # plt.show(block=True)
    return fig


def plotpie(uav_angle, ue_angles, ue_radius, fig, active_user):
    width = np.deg2rad(circle / num_angles)
    radius_uav = Radius / 2

    if fig is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
        ax.bar(np.squeeze(np.deg2rad(uav_angle)), radius_uav, width=width, bottom=15.0, alpha=0.7,
               color='b')
        ax.bar(np.squeeze(np.deg2rad(ue_angles)), np.squeeze(ue_radius), width=width, bottom=0.0, alpha=0.5,
               color='k')
        pies = [pie for pie in fig.axes[0].patches]
        fig.axes[0].text(np.squeeze(np.deg2rad(uav_angle)), pies[0].get_height(), "UAV", fontweight='bold')
        ues = np.arange(0 + 1, len(ue_angles) + 1)
        for angle, pie, ue in zip(np.squeeze(np.deg2rad(ue_angles)), pies[1:], ues):
            fig.axes[0].text(angle, pie.get_height(), ue, fontweight='bold')
    else:
        pies = [pie for pie in fig.axes[0].patches]
        pies[0].set_x(np.squeeze(np.deg2rad(uav_angle)))
        [pie.set_color('k') for pie in pies[1:]]
        pies[active_user+1].set_color('g')
        fig.axes[0].texts[0].set_x(np.squeeze(np.deg2rad(uav_angle)))
    fig.canvas.draw()
    return fig


def plotline(energy, consumed, fig):
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
        ax_energy = fig.add_subplot(111)
        ax_consum = ax_energy.twinx()
        ax_energy.grid()
        ax_energy.set_xlabel("Events", size=12, fontweight='bold')
        ax_energy.set_ylabel("Remaining Energy [-]", size=12, fontweight='bold', color='blue')
        ax_consum.set_ylabel("Consumed Energy [--]", size=12, fontweight='bold', color='red')
        ax_energy.plot(0, energy, color="blue", linestyle='-', label='Remaining Energy', linewidth=2)
        ax_consum.plot(0, consumed, color="red", linestyle='--', label='Consumed Energy', linewidth=2)
        fig.legend()
    else:
        new_data_x = fig.axes[0].lines[0].get_xdata()[-1]+1
        new_x = np.append(fig.axes[0].lines[0].get_xdata(), new_data_x)
        new_y = np.append(fig.axes[0].lines[0].get_ydata(), energy)
        fig.axes[0].lines[0].set_xdata(new_x)
        fig.axes[0].lines[0].set_ydata(new_y)
        fig.axes[0].set_xlim(left=0, right=max(new_x))
        fig.axes[0].set_ylim(bottom=min(new_y)-10, top=max(new_y)+10)

        new_y_2 = np.append(fig.axes[1].lines[0].get_ydata(), consumed)
        fig.axes[1].lines[0].set_xdata(new_x)
        fig.axes[1].lines[0].set_ydata(new_y_2)
        fig.axes[1].set_xlim(left=0, right=max(new_x))
        fig.axes[1].set_ylim(bottom=min(new_y_2) - min(new_y_2)/2, top=max(new_y_2) + max(new_y_2)/5)

    fig.canvas.draw()
    return fig


def plotlybar(data_queue, fig):
    if fig is None:
        ues = np.arange(0 + 1, len(data_queue) + 1)
        trace1 = go.Bar(x=ues, y=data_queue, name="Queue Length", marker=dict(color='rgb(55, 83, 109)'))
        fig = go.FigureWidget(data=trace1)
    else:
        fig.update_traces(y=data_queue)
    return fig


def plot_training(result, type_model, layers_len, units_num):
    (fig, ax) = plt.subplots(2, 1, figsize=(13, 13))
    epochs = len(result.history['accuracy'])
    ax[0].set_title("Loss", fontsize=14, fontweight='bold')
    ax[0].set_xlabel("Epoch #", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Loss", fontsize=14, fontweight="bold")
    ax[0].plot(np.arange(1, epochs+1), result.history['loss'], label='Loss', linewidth=2.5, linestyle='-', marker='o',
               markersize='10', color='red')
    ax[0].plot(np.arange(1, epochs+1), result.history['val_loss'], label='Validation_loss', linewidth=2.5, marker='x',
               linestyle='--', markersize='10', color='blue')
    ax[0].grid(True)
    ax[0].legend(prop={'size': 14, 'weight': 'bold'})
    ax[0].tick_params(axis='both', which='major', labelsize=15)

    plt.subplots_adjust(hspace=0.3)

    ax[1].set_title("Accuracy", fontsize=14, fontweight="bold")
    ax[1].set_xlabel("Epoch #", fontsize=14, fontweight="bold")
    ax[1].set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax[1].plot(np.arange(1, epochs+1), result.history['accuracy'], label='Accuracy', linewidth=2.5, linestyle='-',
               marker='o', markersize='10', color='red')
    ax[1].plot(np.arange(1, epochs+1), result.history['val_accuracy'], label='Validation_accuracy', linewidth=2.5,
               linestyle='--', marker='x', markersize='10', color='blue')
    ax[1].grid(True)
    ax[1].legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    file_figobj = 'Output/FigureObject/%s_%d_EPOCH_%d_layers_%s_units_opt.fig.pickle' % (type_model, epochs, layers_len,
                                                                                         units_num)
    file_pdf = 'Output/Figures/%s_%d_EPOCH_%d_layers_%s_units_opt.pdf' % (type_model, epochs, layers_len, units_num)

    pickle.dump(fig, open(file_figobj, 'wb'))
    fig.savefig(file_pdf, bbox_inches='tight')
