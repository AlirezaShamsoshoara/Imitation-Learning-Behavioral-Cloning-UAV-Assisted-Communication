"""
#################################
# Utils functions
#################################
"""

#########################################################
# import libraries
import numpy as np
from copy import deepcopy
from config import Config_General as General

#########################################################
# Function definition


def distance_calculator(uav_idx, prim_idxs):
    """
    This function calculates the distance between the UAV and all UEs on the ground.
    :param uav_idx: The location's index for the UAV
    :param prim_idxs: locations' indexes for all UEs
    :return: all distance between the UAV and UEs and the direction of movement
    """
    dist = np.zeros([prim_idxs.shape[0], 1], dtype=int)
    index = 0
    direction = np.empty([prim_idxs.shape[0], 1], dtype='object')  # possible dir: cc=counter-clockwise, c=clocwise
    num_angles = General.get('NUM_ANGLE')
    max_dist = num_angles/2
    for prim_idx in prim_idxs:
        dist[index] = np.abs(uav_idx - prim_idx)
        direction[index] = 'cc' if uav_idx < prim_idx else 'c'
        if uav_idx == prim_idx:
            direction[index] = 'None'
        if dist[index] > max_dist:
            dist[index] = num_angles - dist[index]
            direction[index] = 'cc' if uav_idx > prim_idx else 'c'
        index += 1
    return dist, direction


def distance_map_attenuator(distance_vect, max_dist):
    """
    This function scales the distance between the UAV and UEs
    :param distance_vect: distance between the UAV and all UEs
    :param max_dist: Maximum possible distance
    :return: Scaled value
    """
    scaled_dist = np.zeros([distance_vect.shape[0], 1])
    index = 0
    for dist in distance_vect:
        scaled_dist[index] = 1 / ((dist/max_dist) + 1)
        index += 1
    return scaled_dist


def queue_lengths(queues):
    """
    This function returns the queue length based on pkts inside that
    :param queues: all queues for all UEs
    :return: Length of all queues
    """
    q_lentghs = np.zeros([len(queues), 1], dtype=int)
    index = 0
    for queue in queues:
        q_lentghs[index] = len(queue)
        index += 1
    return np.squeeze(q_lentghs)


def queue_drops(queues):
    """
    This function counts number of dropped packet in each queue.
    :param queues: all queues of all UEs
    :return: Number of dropped packets
    """
    q_drops = np.zeros([len(queues), 1], dtype=int)
    index = 0
    for queue in queues:
        for pkt in queue:
            if pkt.get_status() == 'Drop':
                q_drops[index] += 1
        index += 1
    return np.squeeze(q_drops)


def dir_string_to_num(dir_mat):
    """
    This function maps the direction vector from string to integer
    :param dir_mat: Direction matrix (String)
    :return: Direction matrix (integer)
    """
    # 0: 'cc: Counter-clockwise', 1:'c: Clockwise', 2: 'None'
    return_mat = deepcopy(dir_mat)
    return_mat[return_mat == 'cc'] = 0
    return_mat[return_mat == 'c'] = 1
    return_mat[return_mat == 'None'] = 2
    return return_mat
