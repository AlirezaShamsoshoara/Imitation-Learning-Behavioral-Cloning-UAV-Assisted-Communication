"""
#################################
# State Action functions
#################################
"""

#########################################################
# import libraries
import numpy as np

#########################################################
# Function definition


def location_index(angles, angle_states):
    """
    This function finds the index of the located UE.
    :param angles: The angle of the located UE.
    :param angle_states: Total number of sectors.
    :return: Sector's index
    """
    loc_index = np.zeros([angles.shape[0], 1], dtype=int)
    index = 0
    for angle in angles:
        loc_index[index] = np.max(np.where(angle > angle_states))
        index += 1
    return loc_index



