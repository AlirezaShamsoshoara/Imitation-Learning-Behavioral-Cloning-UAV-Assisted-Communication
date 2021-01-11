"""
Created on Mar. 20, 2020
@author:    Alireza Shamsoshoara
@Project:   Imitation Learning (Behavioral-Cloning) for UAV-Assisted Communication in Remote Disaster Area
            Paper: ### TODO
            Arxiv: ### TODO
            YouTube Link for the designed Simulator: https://youtu.be/xYSlZac-AMM
@Northern Arizona University
This project is developed and tested with Python 3.6 using pycharm on Ubuntu 18.04 LTS machine
"""

#################################
# Main File
#################################

# ############# import libraries
# General Modules
import os
import time
import platform
import numpy as np
from copy import deepcopy

# Customized Modules
from config import Radius
from config import Config_Path
from config import Config_Queue
from config import Config_Dim as Dim
from config import Config_Power as Power
from config import Config_General as General

from location_gen import location
from location_gen import update_location
from location_gen import update_angle_to_location as angle_to_loc
from location_gen import update_index_to_angle as index_to_angle


from energy import init_energy
from energy import update_energy

from scheduler import generator_poission
from scheduler import service_time_generator

from queue_operation import enqueue
from queue_operation import dequeue

from states_actions import location_index

from utils import queue_drops
from utils import queue_lengths
from utils import distance_calculator
from utils import distance_map_attenuator

from plotdata import plotbar
from plotdata import plotpie

from expert import action_expert

from training import train
from classification import classify
from imitation import action_imitation

from results import result_newrate
from results import result_imitation
from results import result_demonstration

#########################################################
# General Flags
Flag_Print = False
Mode = General.get("Mode")
Flag_Imitation = None
#########################################################
# Scenario Definition


def main():
    print(General, "Radius = ", Radius)
    num_uav = General.get('NUM_UAV')
    num_ue = General.get('NUM_UE')
    num_frm = General.get('NUM_FRM')
    num_run = General.get('NUM_RUN')
    num_angle = General.get('NUM_ANGLE')
    num_var_dim = Dim.get('Var_Dim')
    num_actions = General.get('Actions')
    cbr_rate = General.get('CBR_RATE')
    num_event = General.get('Sim_Events')
    num_angles = General.get('NUM_ANGLE')
    circle = General.get('Circle')
    path_dist = Config_Path.get('PathDist')
    path_energy = Config_Path.get('pathEnergy')
    queue_lim = Config_Queue.get('Queue_limit')
    demonstration_plot = General.get('Demonstration_plot')
    print_flag = General.get('printFlag')
    decision_delay = General.get('DecisionDelay')

    location_init = location(num_uav, num_ue, Dim.get('Height'), Radius, path_dist, General.get('Location_SaveFile'),
                             General.get('PlotLocation'))
    energy_init = init_energy(Power.get('Min_energy'), Power.get('Max_energy'), General.get('Energy_SaveFile'),
                              path_energy)

    angle_states = np.arange(0, circle + 1, circle / num_angles)
    max_dist = int(num_angles / 2)

    lmbda_sample = np.array([3, 5, 10, 8, 7])
    mu_sample = np.array([6, 6, 6, 6, 6])

    # lmbda_sample = np.array([6, 5, 10, 8, 7]) # New_Rate1
    # lmbda_sample = np.array([14, 11, 4, 5, 3])  # New_Rate2
    # lmbda_sample = np.array([5, 1, 20, 10, 2])  # New_Rate_snapshot

    # mu_sample = np.array([2, 2, 2, 2, 2])
    # mu_sample = np.array([20, 20, 20, 20, 20])

    # dir_action = [0, 1, 2]  # 0: 'CC: Counter-clockwise', 1:'C: Clockwise', 2: 'None'

    for Run in range(0, num_run):
        # ########################################################
        # Initialization
        queue_users = [[] for _ in range(0, num_ue)]
        queue_virtual = [[] for _ in range(0, num_ue)]  # Queue with no threshold or limit(Size)
        print("queue_users = ", queue_users)

        num_features = num_ue + num_ue + num_ue + 1  # number of queues + number of dist + number of dir + active_user
        state_feature_vec = np.empty([num_frm, num_event, num_features], dtype=object)
        action_vec = np.zeros([num_frm, num_event, num_actions], dtype=int) - 1
        loc_uav_mat = np.zeros([num_frm, num_event, num_var_dim])
        dist_mat = np.zeros([num_frm, num_event, num_ue])
        dir_mat = np.empty([num_frm, num_event, num_ue], dtype=object)
        active_user = np.zeros([num_frm, num_event], dtype=int) - 1
        queue_drop_mat = np.zeros([num_frm, num_event, num_ue], dtype=int)
        queue_length_mat = np.zeros([num_frm, num_event, num_ue], dtype=int)
        remain_energy_mat = np.zeros([num_frm, num_event], dtype=float)
        energy_consumed_mat = np.zeros([num_frm, num_event], dtype=float)
        number_switch = np.zeros([num_frm, num_event], dtype=int)
        loc_uav_idx_mat = np.zeros([num_frm, num_event], dtype=int)
        arrival_time_mat = np.zeros([num_frm, num_ue, cbr_rate], dtype=float)
        service_time_mat = np.zeros([num_frm, num_ue, cbr_rate], dtype=float)

        prim_angles = location_init.get('angle_prim_deg')
        ue_radius = location_init.get('UE_radius')
        uav_angle = location_init.get('angle_uav_deg')
        remained_energy = deepcopy(energy_init)
        loc_uav_idx = location_index(uav_angle, angle_states)
        loc_prim_idx = location_index(prim_angles, angle_states)
        distance, direction = distance_calculator(loc_uav_idx, loc_prim_idx)

        loc_uav_idx_mat[0, 0] = loc_uav_idx
        loc_uav_mat[0, 0, :] = np.squeeze(location_init.get('x_uav')), np.squeeze(location_init.get('y_uav'))
        # angle_to_loc(uav_angle, radius=Radius)
        # index_to_angle(loc_uav_idx)
        remain_energy_mat[0, 0] = energy_init
        updated_number_switch = 0
        fig_queue = None
        fig_direction = None
        fig_energy = None
        updated_index_uav = loc_uav_idx
        t_sim = 5.0
        ue_selected = -1
        for Frame in range(0, num_frm):
            timer = time.clock()
            arrival_time, interval_time = generator_poission(lmbda=lmbda_sample, counts=cbr_rate, users=num_ue,
                                                             plotflag=General.get('PlotDistribution'))
            if Frame != 0:
                arrival_time = arrival_time + t_sim
            arrival_time_mat[Frame, :, :] = arrival_time
            service_time = service_time_generator(mu=mu_sample, counts=cbr_rate, users=num_ue)
            service_time_mat[Frame, :, :] = service_time
            event = 0
            first = True if (Frame == 0 and event == 0) else False

            while event < num_event:
                print("----- Run = ", Run, " ----- Frame = ", Frame, " ----- Event = ", event, ", time = ", t_sim)

                for ue in range(0, num_ue):
                    queue_users[ue], queue_virtual[ue] = enqueue(queue_users[ue], queue_virtual[ue],
                                                                 arrival_time[ue, :], t_sim, ue, Frame)

                queue_length_mat[Frame, event, :] = queue_lengths(queue_users)
                queue_drop_mat[Frame, event, :] = queue_drops(queue_virtual)
                if print_flag:
                    print('queue_len_mat[Frame={}, Event={}] = {}'.format(Frame, event,
                                                                          queue_length_mat[Frame, event, :]))
                    print('queue_drop_mat[Frame={}, Event={}] = {}'.format(Frame, event,
                                                                           queue_drop_mat[Frame, event, :]))

                active_user[Frame, event] = ue_selected
                remain_energy_mat[Frame, event] = remained_energy
                number_switch[Frame, event] = updated_number_switch

                loc_uav_mat[0, 0, :] = angle_to_loc(uav_angle, radius=Radius)
                dist_mat[Frame, event, :] = np.squeeze(distance)
                dir_mat[Frame, event, :] = np.squeeze(direction)
                loc_uav_idx_mat[Frame, event] = updated_index_uav

                state_feature_vec[Frame, event, 0:num_ue] = queue_length_mat[Frame, event, :]
                state_feature_vec[Frame, event, num_ue:2*num_ue] = dist_mat[Frame, event, :]
                state_feature_vec[Frame, event, 2*num_ue:3*num_ue] = dir_mat[Frame, event, :]
                state_feature_vec[Frame, event, -1] = active_user[Frame, event]

                if Flag_Imitation is False:
                    ue_selected, dir_taken = action_expert(state_feature_vec[Frame, event, :], num_ue, first)
                elif Flag_Imitation is True:
                    ue_selected, dir_taken = action_imitation(state_feature_vec[Frame, event, :], num_ue)
                    pass
                else:
                    print("[INFO] --------- Flag is not correct! ---------")
                    print("[INFO] --------- Exit! ---------")
                    return
                action_vec[Frame, event, :] = ue_selected, dir_taken

                # TODO: Update the Energy for the Learner (UAV) remain_energy_mat[Frame, event]
                remained_energy, energy_consumed_mat[Frame, event], updated_number_switch = \
                    update_energy(ue_selected, active_user[Frame, event], remain_energy_mat[Frame, event],
                                  number_switch[Frame, event], dir_taken)

                # TODO: Updating the distance here based on the dir_taken (Checked!)
                updated_index_uav = update_location(loc_uav_idx_mat[Frame, event], dir_taken, num_angles)
                distance, direction = distance_calculator(updated_index_uav, loc_prim_idx)
                scaled_distance = distance_map_attenuator(distance, max_dist)
                uav_angle = index_to_angle(updated_index_uav)

                queue_users[ue_selected], pkt = dequeue(queue_users[ue_selected], t_sim)
                if pkt is not None:
                    t_sim = pkt.service_scheduler(t_sim, service_time[ue_selected, :],
                                                  atten_factor=scaled_distance[ue_selected, 0])
                else:
                    t_sim += decision_delay
                    print("[INFO] --------- PKT is None ---------")

                if demonstration_plot:
                    fig_queue = plotbar(queue_length_mat[Frame, event, :], queue_drop_mat[Frame, event, :], fig_queue,
                                        ue_selected)
                    fig_direction = plotpie(uav_angle, index_to_angle(loc_prim_idx), ue_radius, fig_direction,
                                            ue_selected)
                    # fig_energy = plotline(remain_energy_mat[Frame, event], energy_consumed_mat[Frame, event],
                    #                       fig_energy)

                # t_sim += 1

                first = False
                event += 1
                # End of each event
            print(" ------- Run = %d ------- Frame = %d ------- Duration = %f " % (Run, Frame, time.clock() - timer))
            pass
            # End of each Frame
        if General.get('SaveOutput'):
            if platform.system() == "Linux":
                path_save = 'SimulationData'
            else:
                path_save = "D:"
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            if Flag_Imitation is False:
                outputfile_linux =\
                    'SimulationData/TestData/num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d' \
                    '_cbr_rate_%d_Event_%d' % (num_ue, num_angle, queue_lim, Run, num_frm, cbr_rate, num_event)
                # 'SimulationData/TestData/NewRate2/num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d' \
                # '_cbr_rate_%d_Event_%d_new_rate2' % (num_ue, num_angle, queue_lim, Run, num_frm,
                #                                      cbr_rate, num_event)
            elif Flag_Imitation is True:
                outputfile_linux = 'SimulationData/ImitatedModel/imit_num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d' \
                                   '_Frame_%d_cbr_rate_%d_Event_%d' % (num_ue, num_angle, queue_lim, Run, num_frm,
                                                                       cbr_rate, num_event)
            else:
                print("[INFO] --------- Flag is not correct! ---------")
                print("[INFO] --------- Continue! ---------")
                continue
            np.savez(outputfile_linux, location_init=location_init, energy_init=energy_init, lmbda_sample=lmbda_sample,
                     mu_sample=mu_sample, queue_users=queue_users, queue_virtual=queue_virtual, dir_mat=dir_mat,
                     state_feature_vec=state_feature_vec, action_vec=action_vec, loc_uav_mat=loc_uav_mat,
                     dist_mat=dist_mat, active_user=active_user, queue_drop_mat=queue_drop_mat,
                     queue_length_mat=queue_length_mat, remain_energy_mat=remain_energy_mat,
                     energy_consumed_mat=energy_consumed_mat, number_switch=number_switch,
                     loc_uav_idx_mat=loc_uav_idx_mat, General=General, arrival_time_mat=arrival_time_mat,
                     service_time_mat=service_time_mat)
        pass
        # End of each Run
    pass
    # End of the main


if __name__ == "__main__":
    if Mode == "Demonstration":
        Flag_Imitation = False
        main()
    elif Mode == "Training":
        train()
    elif Mode == "Classification":
        classify()
    elif Mode == "Results_demonstration":
        result_demonstration()
    elif Mode == "Imitation":
        Flag_Imitation = True
        main()
    elif Mode == "Result_imitation":
        result_imitation()
    elif Mode == "Result_imitation_newRate":
        result_newrate()
    else:
        print("Mode is not correct")