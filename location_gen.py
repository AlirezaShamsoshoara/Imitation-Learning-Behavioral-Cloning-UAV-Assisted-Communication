"""
#################################
# Location Generation function
#################################
"""

#########################################################
# import libraries
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from config import Config_General as General

#########################################################
# Function definition
center_x = 0
center_y = 0
center_z = 0


def location(num_uav, num_pu, height, radius, pthdist, savefile, plot):
    """
    This function generates the location of UAVs and UEs.
    :param num_uav: number of UAVs
    :param num_pu: number of UEs
    :param height: UAV altitude
    :param radius: UAV coverage radius
    :param pthdist: Path directory to save data
    :param savefile: True/False FLAG to save or not save data
    :param plot: True/False FLAG to plot or not plot data
    :return: A dictionary of locations
    """
    radius_uav = np.floor(radius / 2)
    if savefile:

        angle_prim = 2 * math.pi * np.random.rand(num_pu, 1)
        angle_prim_deg = np.rad2deg(angle_prim)
        angle_uav = 2 * math.pi * np.random.rand(num_uav, 1)
        angle_uav_deg = np.rad2deg(angle_uav)

        random_radius1 = radius * np.sqrt(np.random.rand(num_pu, 1))

        x_put1 = np.floor(random_radius1 * np.cos(angle_prim))
        y_put1 = np.floor(random_radius1 * np.sin(angle_prim))
        z_put1 = np.zeros([num_pu, 1])

        x_uav = np.floor(radius_uav * np.cos(angle_uav))
        y_uav = np.floor(radius_uav * np.sin(angle_uav))
        z_uav = height * np.ones([num_uav, 1])

        angle2 = math.pi * np.random.rand(num_pu, 1) + math.pi
        random_radius2 = radius * np.sqrt(np.random.rand(num_pu, 1))
        x_put2 = np.floor(random_radius2 * np.cos(angle2))
        y_put2 = np.floor(random_radius2 * np.sin(angle2))
        z_put2 = np.zeros([num_pu, 1])

        return_dict = dict([('x_put1', x_put1), ('y_put1', y_put1), ('z_put1', z_put1), ('x_put2', x_put2),
                            ('y_put2', y_put2), ('z_put2', z_put2), ('x_uav', x_uav), ('y_uav', y_uav),
                            ('z_uav', z_uav), ('angle_prim', angle_prim), ('angle_uav', angle_uav),
                            ('angle_prim_deg', angle_prim_deg), ('angle_uav_deg', angle_uav_deg),
                            ('UE_radius', random_radius1)])
        sio.savemat(pthdist, return_dict)

    else:
        return_dict = sio.loadmat(pthdist)
        x_put1 = return_dict.get('x_put1')
        y_put1 = return_dict.get('y_put1')
        z_put1 = return_dict.get('z_put1')
        x_put2 = return_dict.get('x_put2')
        y_put2 = return_dict.get('y_put2')
        z_put2 = return_dict.get('z_put2')
        x_uav = return_dict.get('x_uav')
        y_uav = return_dict.get('y_uav')
        z_uav = return_dict.get('z_uav')

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(np.squeeze(x_put1), np.squeeze(y_put1), np.squeeze(z_put1), 'ro', markersize=6)
        # ax.plot(np.squeeze(x_put2), np.squeeze(y_put2), np.squeeze(z_put2), 'go', markersize=12)
        ax.set_xlim(center_x - radius - radius/10, radius + center_x + radius/10)
        ax.set_ylim(center_x - radius - radius/10, radius + center_y + radius/10)
        ax.set_zlim(0 - 10, height + 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        circle_primary = Circle((center_x, center_y), radius)
        ax.add_patch(circle_primary)
        art3d.pathpatch_2d_to_3d(circle_primary, z=0, zdir="z")
        circle_primary.set_color((0, 0.5, 1, 1))
        circle_primary.set_edgecolor((0, 0, 0, 1))

        circle_uav = Circle((center_x, center_y), radius_uav)
        ax.add_patch(circle_uav)
        art3d.pathpatch_2d_to_3d(circle_uav, z=height, zdir="z")
        circle_uav.set_color((1, 1, 1, 1))
        circle_uav.set_edgecolor((0, 0, 0, 1))
        uav_fig = ax.plot([np.squeeze(x_uav)], [np.squeeze(y_uav)], np.squeeze(z_uav), 'ko', markersize=12)
        uav_fig[0].set_data((center_x, center_y))
        uav_fig[0].set_xdata(center_x)
        uav_fig[0].set_ydata(center_y)
        fig.canvas.draw()

        plt.show(block=False)
    return return_dict


def update_angle_to_location(angle, radius):
    """
    This function finds the location based on angle and radius
    :param angle: Node's angle
    :param radius: Node's radius
    :return: longitude and latitude of a point
    """
    radius_uav = np.floor(radius / 2)
    x = np.floor(radius_uav * np.cos(np.deg2rad(angle)))
    y = np.floor(radius_uav * np.sin(np.deg2rad(angle)))
    return x, y


def update_index_to_angle(index):
    """
    :param index: Index of the UAV's location in the circle
    :return: Angle in degree
    """
    circle = General.get('Circle')
    num_angles = General.get('NUM_ANGLE')
    angle_states = np.arange(0, circle + 1, circle / num_angles)
    half_step = circle / (2*num_angles)
    angle = angle_states[index] + half_step
    return angle


def update_location(old_index, dir_taken, num_angles):
    """
    This function update location based on chosen action.
    :param old_index: Old index of the sector
    :param dir_taken: Chosen action
    :param num_angles: Number of sectors
    :return: the new location's index based on the chosen action
    """
    # 0: 'CC: Counter-clockwise', 1:'C: Clockwise', 2: 'None'
    if dir_taken == 0:
        new_index = old_index + 1
    elif dir_taken == 1:
        new_index = old_index - 1
    else:
        new_index = old_index
    new_index = np.mod(new_index, num_angles)
    return new_index
