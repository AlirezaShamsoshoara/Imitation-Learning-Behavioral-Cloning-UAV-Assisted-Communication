"""
#################################
# Energy function
#################################
"""
#########################################################
# import libraries
import numpy as np
import scipy.io as sio

from config import Config_Power as Power
mob_consump = Power.get('mob_consump')
tran_consump = Power.get('tran_consump')
switch_consump = Power.get('switch_consump')
#########################################################
# Function definition


def init_energy(min_energy, max_energy, savefile, pthenergy):
    """
    This function generates random energy value for the UAV.
    :param min_energy: Minimum energy value
    :param max_energy: Maximum energy value
    :param savefile: Path directory to save data
    :param pthenergy: True/False flag to save energy value or not
    :return: Numpy array for the energy value
    """
    if savefile:
        energy = np.random.uniform(min_energy, max_energy, 1)
        energy_dict = dict([('energy', energy)])
        sio.savemat(pthenergy, energy_dict)
    else:
        energy_dict = sio.loadmat(pthenergy)
        energy = energy_dict.get('energy')
        energy = np.squeeze(energy)
    return energy


def update_energy(new_ue, old_ue, energy, number_switch_old, direction_taken):
    """
    This function updates the energy value after transmission and movement
    :param new_ue: New chosen UE
    :param old_ue: Old UE under the UAV's service
    :param energy: The energy value
    :param number_switch_old: Number of times that UAV switched UEs
    :param direction_taken: chosen action for the direction
    :return: remained_energy, consumed_energy, number_switch_new
    """
    # 0: 'CC: Counter-clockwise', 1:'C: Clockwise', 2: 'None'
    if new_ue != old_ue:
        consumed_energy = switch_consump + tran_consump
        number_switch_new = number_switch_old + 1
    else:
        consumed_energy = tran_consump
        number_switch_new = number_switch_old

    if direction_taken != 2:
        consumed_energy = consumed_energy + mob_consump

    remained_energy = energy - consumed_energy
    return remained_energy, consumed_energy, number_switch_new
