"""
#################################
# Queue Operation functions
#################################
"""

#########################################################
# import libraries
import numpy as np
from packet import Packet
from config import Config_Queue

#########################################################
# General Parameters
queue_lim = Config_Queue.get('Queue_limit')

#########################################################
# Function definition


def enqueue(queue_list, queue_virt, arr_times, t_sim, pu_id, frame):
    """
    This function enqueues packets in the queue of each UE.
    :param queue_list: The queue of each UE
    :param queue_virt: Virtual queue of each UE
    :param arr_times: Arrival time array
    :param t_sim: Current simulation TIME
    :param pu_id: UE's id
    :param frame: frame id
    :return: New queue and virtual queue
    """

    pkt_indexes = np.where(t_sim > arr_times)[0]
    for pkt_index in pkt_indexes:
        if pkt_exist(queue_list, frame, pkt_index) or pkt_exist(queue_virt, frame, pkt_index):
            continue
        pkt = Packet(arr_time=arr_times[pkt_index], user_id=pu_id, pkt_id=pkt_index, frme_id=frame)
        if len(queue_list) < queue_lim:
            pkt.set_status("Gen")
            pkt.set_qID(len(queue_list) + 1)
            queue_list.append(pkt)
        else:
            pkt.set_status("Drop")

        queue_virt.append(pkt)
    return queue_list, queue_virt


def dequeue(queue_list, t_sim):
    """
    This function dequeues the packet from the queue.
    :param queue_list: UE's queue
    :param t_sim: Current time of the simulation
    :return: UE's queue and the extracted packet
    """
    pkt = None
    if len(queue_list) > 0:
        pkt = queue_list.pop(0)
        pkt.set_depart(t_sim)
        wait_time_tmp = t_sim - pkt.get_arrival()
        pkt.set_wait(wait_time_tmp)
        pkt.set_qID(0)
        queue_list = refresh_qid(queue_list)
    return queue_list, pkt


def refresh_qid(queue):
    """
    This function refresh the ID for packets
    :param queue: the current UE's queue
    :return: Updated UE's queue
    """
    for pkt in queue:
        pkt.set_qID(pkt.get_qID() - 1)
    return queue


def lookup_pkt(queue, pkt_target):
    """
    This function looks for a pkt in a queue.
    :param queue: UE's queue
    :param pkt_target: Target packet
    :return: True/False
    """
    for pkt in queue:
        if pkt == pkt_target:
            return True
    return False


def pkt_exist(queue, frame_id, pkt_id):
    """
    This function is the same as lookup_pkt but considers the frame_id as well.
    :param queue: -
    :param frame_id: FRAME ID
    :param pkt_id: -
    :return: True/False
    """
    for pkt in queue:
        if pkt.get_fID() == frame_id:
            if pkt.get_pid() == pkt_id:
                # print("-------- Packet Exist --------- ")
                return True
    return False
