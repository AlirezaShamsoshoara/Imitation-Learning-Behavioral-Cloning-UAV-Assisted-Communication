"""
#################################
# Expert Operation functions
#################################
"""

#########################################################
# import libraries
import numpy as np
from config import Config_Queue

#########################################################
# General Parameters
queue_lim = Config_Queue.get('Queue_limit')
max_threshold = Config_Queue.get('Max_Threshold')
min_threshold = Config_Queue.get('Min_Threshold')

#########################################################
# # Function definition


def action_expert(state, num_ue, first=False):
    """
    This function returns the expert's action for the UE selection and direction based on his/her knowledge.
    :param state: State of the problem (queues' length)
    :param num_ue: Number of UEs
    :param first: True/False: if this is the first round of the simulation
    :return: Expert's actions: UE selection, movement
    """
    # Returns two actions: A1: which UE, A2: which direction
    queue = state[0:num_ue]
    dist = state[num_ue:2*num_ue]
    dirc = state[2*num_ue:3*num_ue]
    active_user = state[-1]
    if first:
        a1 = np.argmax(state[0:num_ue])
    else:
        a1 = user_policy_expert(queue, active_user)
    a2 = direction_policy_expert(a1, dist, dirc)
    return a1, a2


def user_policy_expert(queue, active_user):
    """
    This function is expert's policy regarding the UE selection.
    :param queue: all UEs' queues
    :param active_user: Active UE which is being serviced at the current time
    :return: The chosen UE
    """
    # What policy or strategy the Expert considers to switch the active user
    diffs = np.abs(queue - queue_lim)
    selected = np.argmin(diffs) if np.min(diffs) < max_threshold else active_user

    if queue[selected] < min_threshold:
        selected = np.argmax(queue)
    if np.all(queue < min_threshold):
        selected = active_user
    if queue[selected] == 0:
        selected = np.argmax(queue)
    return selected


def direction_policy_expert(selected_user, dist, direction):
    """
    This function chooses the appropriate movement direction based on the selected user.
    :param selected_user: Chosen UE to get service
    :param dist: Distance vector between the UAV and all UEs
    :param direction: direction vector
    :return: Direction for movement based on UAV's decision
    """
    dir_action = [0, 1, 2]  # 0: 'CC: Counter-clockwise', 1:'C: Clockwise', 2: 'None'
    dir_action = direction[selected_user] if dist[selected_user] > 0 else None
    if dir_action == 'cc':
        dir_action_return = 0
    elif dir_action == 'c':
        dir_action_return = 1
    else:
        dir_action_return = 2
    return dir_action_return
