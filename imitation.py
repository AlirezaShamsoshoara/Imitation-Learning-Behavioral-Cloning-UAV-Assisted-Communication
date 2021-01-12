"""
#################################
# Using the Imitated model by UAV to choose estimate the UE based on the queue states
  Keras
  GPU: Nvidia RTX 2080 Ti
  OS: Ubuntu 18.04
#################################
"""

#########################################################
# import libraries
import numpy as np
from keras.models import load_model
from expert import direction_policy_expert

#########################################################
# General Parameters
model_queue = load_model('Output/Models/model_queue_5_layers_[40, 80, 160, 80, 5]_units.model')

#########################################################
# Function definition


def action_imitation(state, num_ue):
    """
    This function chooses the UE based on imitated and trained model based on the expert's knowledge.
    :param state: State of the problem based on queues' length
    :param num_ue: Number of UEs
    :return: decision on actions for UE selection and mobility direction
    """
    x_queue_vec = state[0:num_ue]
    dist = state[num_ue:2 * num_ue]
    dirc = state[2 * num_ue:3 * num_ue]
    a1 = np.squeeze(model_queue.predict_classes(x_queue_vec.reshape((1, num_ue))))
    a2 = direction_policy_expert(a1, dist, dirc)
    return a1, a2
