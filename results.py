"""
#################################
# plot the results and outcome from demonstration and imitated model
#################################
"""

#########################################################
# import libraries
import pickle
import platform
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
from config import Config_Queue
from config import Config_General as General

#########################################################
# General Parameters
num_ue = General.get('NUM_UE')
num_run = General.get('NUM_RUN')
# num_run = 1
num_frm = General.get('NUM_FRM')
cbr_rate = General.get('CBR_RATE')
pkt_size = General.get('PacketSize')
num_event = General.get('Sim_Events')
save_pdf_obj = General.get('SavePDF')
num_angles = General.get('NUM_ANGLE')
queue_lim = Config_Queue.get('Queue_limit')
num_pkt = cbr_rate * num_ue

queue_virtual_arr = np.empty([num_run, num_ue], dtype=object)
queue_length_mat = np.zeros([num_run, num_frm, num_event, num_ue], dtype=int)
queue_drop_mat = np.zeros([num_run, num_frm, num_event, num_ue], dtype=int)
queue_drop_mat_dif = np.zeros([num_run, num_frm, num_ue], dtype=int)
energy_consumed_mat = np.zeros([num_run, num_frm, num_event])
edt_mat = np.zeros([num_run, num_frm], dtype=float)
number_switch_mat = np.zeros([num_run, num_frm, num_event], dtype=int)

queue_virtual_arr_imit = np.empty([num_run, num_ue], dtype=object)
queue_length_mat_imit = np.zeros([num_run, num_frm, num_event, num_ue], dtype=int)
queue_drop_mat_imit = np.zeros([num_run, num_frm, num_event, num_ue], dtype=int)
queue_drop_mat_imit_dif = np.zeros([num_run, num_frm, num_ue], dtype=int)
energy_consumed_mat_imit = np.zeros([num_run, num_frm, num_event])
edt_mat_imit = np.zeros([num_run, num_frm], dtype=float)
number_switch_mat_imit = np.zeros([num_run, num_frm, num_event], dtype=int)

#########################################################
# Function definition


def result_demonstration():
    print("[INFO] --------- Results (Demo and TestData) --------- ")
    run_list = range(0, num_run)
    for run in run_list:
        if platform.system() == "Windows":
            output_file = \
                "D:\\SimulationData\\TestData\\num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                "_cbr_rate_%d_Event_%d.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate, num_event)
        elif platform.system() == "Linux":
            output_file = \
                "SimulationData/TestData/num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                "_cbr_rate_%d_Event_%d.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate, num_event)
        else:
            print("Nor Linux neither Windows")
            return
        readfile = np.load(output_file, allow_pickle=True)
        # queue_users = readfile['queue_users']
        queue_virtual_arr[run, :] = readfile['queue_virtual'].tolist()
        queue_length_mat[run, :, :, :] = readfile['queue_length_mat']
        queue_drop_mat[run, :, :, :] = readfile['queue_drop_mat']
        energy_consumed_mat[run, :, :] = readfile['energy_consumed_mat']
        number_switch_mat[run, :, :] = readfile['number_switch']

    print("[INFO] --------- End of loading Data ---------")
    calculate_edt(queue_virtual_arr)
    calculate_long_session(number_switch_mat)


def calculate_edt(queue_virt):
    run = 0
    num_delivered = np.zeros([num_frm, num_ue], dtype=int)
    num_dropped = np.zeros([num_frm, num_ue], dtype=int)
    num_passed = np.zeros([num_frm, num_ue], dtype=int)
    service_time = np.zeros([num_frm, num_ue])
    for Frame in range(0, num_frm):
        for ue in range(0, num_ue):
            num_delivered[Frame, ue], num_dropped[Frame, ue], num_passed[Frame, ue] = \
                delivered(queue_virt[run, ue][Frame*cbr_rate:(Frame+1)*cbr_rate])
            service_time[Frame, ue] = service_time_cal(queue_virt[run, ue][Frame*cbr_rate:(Frame+1)*cbr_rate])
    edt_mat[run, :] = pkt_size * np.sum(num_delivered, axis=1) / ((np.sum(num_dropped, axis=1) + 1) *
                                                                  np.sum(service_time, axis=1) *
                                                                  np.sum(energy_consumed_mat[run, :, :], axis=1))
    print("[INFO] --------- End of calculation ---------")
    fig_edt = plt.figure(figsize=(8, 8))
    ax_edt = fig_edt.add_subplot(111)
    ax_edt.set_xlabel("Frames", size=12, fontweight='bold')
    ax_edt.set_ylabel("EDT [1 / Watt]", size=12, fontweight='bold')
    ax_edt.plot(np.arange(1, num_frm)+1, edt_mat[run, 1:], color="blue", linestyle='--', marker='o',
                markersize='5', label='EDT (Expert Demonstration)', linewidth=2)
    ax_edt.grid()
    ax_edt.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
    file_figobj = 'Output/FigureObject/edt.fig.pickle' % ()
    file_pdf = 'Output/Figures/edt.pdf' % ()
    if save_pdf_obj:
        pickle.dump(fig_edt, open(file_figobj, 'wb'))
        fig_edt.savefig(file_pdf, bbox_inches='tight')

    fig_drop = plt.figure(figsize=(8, 8))
    ax_drop = fig_drop.add_subplot(111)
    ax_drop.set_xlabel("Frames", size=12, fontweight='bold')
    ax_drop.set_ylabel("Number of dropped packets", size=12, fontweight='bold')
    ax_drop.plot(np.arange(0, num_frm) + 1, np.sum(num_dropped, axis=1), color="red", linestyle='--', marker='o',
                 markersize='5', label='Packet Drop (Expert Demonstration)', linewidth=2)
    ax_drop.grid()
    ax_drop.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
    file_figobj = 'Output/FigureObject/drop.fig.pickle' % ()
    file_pdf = 'Output/Figures/drop.pdf' % ()
    if save_pdf_obj:
        pickle.dump(fig_drop, open(file_figobj, 'wb'))
        fig_drop.savefig(file_pdf, bbox_inches='tight')

    fig_energy = plt.figure(figsize=(8, 8))
    ax_energy = fig_energy.add_subplot(111)
    ax_energy.set_xlabel("Frames", size=12, fontweight='bold')
    ax_energy.set_ylabel("Consumed Energy [J]", size=12, fontweight='bold')
    ax_energy.plot(np.arange(0, num_frm) + 1, np.sum(energy_consumed_mat[run, :, :], axis=1), color="blue",
                   linestyle='--', marker='o', markersize='5', label='Consumed Energy (Expert Demonstration)',
                   linewidth=2)
    ax_energy.grid()
    ax_energy.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
    file_figobj = 'Output/FigureObject/energy.fig.pickle' % ()
    file_pdf = 'Output/Figures/energy.pdf' % ()
    if save_pdf_obj:
        pickle.dump(fig_energy, open(file_figobj, 'wb'))
        fig_energy.savefig(file_pdf, bbox_inches='tight')


def delivered(queue_frame):
    processed = 0
    dropped = 0
    passed = 0
    for pkt in queue_frame:
        if pkt.get_status() == 'Proc':
            processed += 1
        elif pkt.get_status() == 'Drop':
            dropped += 1
        else:
            passed += 1
    return processed, dropped, passed


def service_time_cal(queue_frame):
    service_time = 0
    for pkt in queue_frame:
        if pkt.get_status() == 'Proc':
            service_time += pkt.get_serv()
    return service_time


def calculate_long_session(number_switch):
    run = 0
    longest_session = np.zeros([num_frm, 1], dtype=int)
    for Frame in range(0, num_frm):
        longest_session[Frame, 0] = int(np.max(np.bincount(number_switch[run, Frame, :])))
    fig_session = plt.figure()
    ax_session = fig_session.add_subplot(111)
    ax_session.set_xlabel("Frames", size=12, fontweight='bold')
    ax_session.set_ylabel("Longest Session [Events]", size=12, fontweight='bold')
    ax_session.plot(np.arange(0, num_frm) + 1, longest_session, color="blue",
                    linestyle='--', marker='o', markersize='5', label='Longest Session (Expert Demonstration)',
                    linewidth=2)
    ax_session.grid()
    ax_session.legend(prop={'size': 10, 'weight': 'bold'}, loc='best')
    file_figobj = 'Output/FigureObject/session.fig.pickle' % ()
    file_pdf = 'Output/Figures/session.pdf' % ()
    if save_pdf_obj:
        pickle.dump(fig_session, open(file_figobj, 'wb'))
        fig_session.savefig(file_pdf, bbox_inches='tight')


def result_imitation(new_rate=False):
    print("[INFO] --------- Results (Imitated Model) --------- ")
    print("[INFO] --------- Running! ............... --------- ")
    run_list = range(0, num_run)
    for run in run_list:
        if platform.system() == "Windows":
            output_file = \
                "D:\\SimulationData\\TestData\\num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                "_cbr_rate_%d_Event_%d.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate, num_event)
            if new_rate:
                output_file = \
                    "D:\\SimulationData\\TestData\\NewRate\\num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                    "_cbr_rate_%d_Event_%d_new_rate.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate,
                                                            num_event)
        elif platform.system() == "Linux":
            output_file = \
                "SimulationData/TestData/num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                "_cbr_rate_%d_Event_%d.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate, num_event)
            if new_rate:
                output_file = \
                    "SimulationData/TestData/NewRate/num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                    "_cbr_rate_%d_Event_%d_new_rate.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate,
                                                            num_event)
        else:
            print("Nor Linux neither Windows")
            return
        readfile = np.load(output_file, allow_pickle=True)
        queue_virtual_arr[run, :] = readfile['queue_virtual'].tolist()
        queue_length_mat[run, :, :, :] = readfile['queue_length_mat']
        queue_drop_mat[run, :, :, :] = readfile['queue_drop_mat']
        energy_consumed_mat[run, :, :] = readfile['energy_consumed_mat']
        number_switch_mat[run, :, :] = readfile['number_switch']

        for Run in run_list:
            for Frame in range(0, num_frm):
                for ue in range(0, num_ue):
                    if Frame is 0:
                        queue_drop_mat_dif[Run, Frame, ue] = queue_drop_mat[Run, Frame, -1, ue]
                    else:
                        queue_drop_mat_dif[Run, Frame, ue] = queue_drop_mat[Run, Frame, -1, ue] - \
                                                             queue_drop_mat[Run, Frame-1, -1, ue]

        num_run_imit = num_run
        run_list = range(0, num_run_imit)
        for run in run_list:
            if platform.system() == "Windows":
                output_file_imit = \
                    "D:\\SimulationData\\ImitatedModel\\imit_num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                    "_cbr_rate_%d_Event_%d.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate, num_event)
                if new_rate:
                    output_file_imit = \
                        "D:\\SimulationData\\ImitatedModel\\NewRate\\imit_num_UE_%d_num_angles_%d_queue_lim_%d_" \
                        "Run_%d_Frame_%d_cbr_rate_%d_Event_%d_new_rate.npz" % (num_ue, num_angles, queue_lim, run,
                                                                               num_frm, cbr_rate, num_event)
            elif platform.system() == "Linux":
                output_file_imit = \
                    "SimulationData/ImitatedModel/imit_num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                    "_cbr_rate_%d_Event_%d.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate, num_event)
                if new_rate:
                    output_file_imit = \
                        "SimulationData/ImitatedModel/NewRate/imit_num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d" \
                        "_Frame_%d_cbr_rate_%d_Event_%d_new_rate.npz" % (num_ue, num_angles, queue_lim, run, num_frm,
                                                                         cbr_rate, num_event)
            else:
                print("Nor Linux neither Windows")
                return
            readfile_imit = np.load(output_file_imit, allow_pickle=True)
            queue_virtual_arr_imit[run, :] = readfile_imit['queue_virtual'].tolist()
            queue_length_mat_imit[run, :, :, :] = readfile_imit['queue_length_mat']
            queue_drop_mat_imit[run, :, :, :] = readfile_imit['queue_drop_mat']
            energy_consumed_mat_imit[run, :, :] = readfile_imit['energy_consumed_mat']
            number_switch_mat_imit[run, :, :] = readfile_imit['number_switch']

            for Run in run_list:
                for Frame in range(0, num_frm):
                    for ue in range(0, num_ue):
                        if Frame is 0:
                            queue_drop_mat_imit_dif[Run, Frame, ue] = queue_drop_mat_imit[Run, Frame, -1, ue]
                        else:
                            queue_drop_mat_imit_dif[Run, Frame, ue] = queue_drop_mat_imit[Run, Frame, -1, ue] - \
                                                                 queue_drop_mat_imit[Run, Frame - 1, -1, ue]

    print("[INFO] --------- End of loading Data ---------")
    calculate_edt_imit(queue_virtual_arr, queue_virtual_arr_imit, new_rate)
    calculate_long_session_imit(number_switch_mat, number_switch_mat_imit, new_rate)


def calculate_edt_imit(queue_virt, queue_virt_imit, new_rate):
    global num_run
    num_delivered = np.zeros([num_run, num_frm, num_ue], dtype=int)
    num_dropped = np.zeros([num_run, num_frm, num_ue], dtype=int)
    num_passed = np.zeros([num_run, num_frm, num_ue], dtype=int)
    service_time = np.zeros([num_run, num_frm, num_ue])

    num_delivered_imit = np.zeros([num_run, num_frm, num_ue], dtype=int)
    num_dropped_imit = np.zeros([num_run, num_frm, num_ue], dtype=int)
    num_passed_imit = np.zeros([num_run, num_frm, num_ue], dtype=int)
    service_time_imit = np.zeros([num_run, num_frm, num_ue])

    # if new_rate is False:
    #     num_run = 1
    for run in range(0, num_run):
        for Frame in range(0, num_frm):
            for ue in range(0, num_ue):
                num_delivered[run, Frame, ue], num_dropped[run, Frame, ue], num_passed[run, Frame, ue] = \
                    delivered(queue_virt[run, ue][Frame * cbr_rate:(Frame + 1) * cbr_rate])
                service_time[run, Frame, ue] = service_time_cal(queue_virt[run, ue][Frame * cbr_rate:
                                                                                    (Frame + 1) * cbr_rate])

                num_delivered_imit[run, Frame, ue], num_dropped_imit[run, Frame, ue], num_passed_imit[run, Frame, ue] =\
                    delivered(queue_virt_imit[run, ue][Frame * cbr_rate:(Frame + 1) * cbr_rate])
                service_time_imit[run, Frame, ue] = service_time_cal(queue_virt_imit[run, ue][Frame*cbr_rate:
                                                                                              (Frame+1)*cbr_rate])

        edt_mat[run, :] = pkt_size * np.sum(num_delivered[run, :, :], axis=1) / ((np.sum(num_dropped[run, :, :], axis=1)
                                                                                  + 1) *
                                                                                 np.sum(service_time[run, :, :], axis=1)
                                                                                 *
                                                                                 np.sum(energy_consumed_mat[run, :, :],
                                                                                        axis=1))

        edt_mat_imit[run, :] = pkt_size * np.sum(num_delivered_imit[run, :, :], axis=1) /\
                               ((np.sum(num_dropped_imit[run, :, :], axis=1) + 1) * np.sum(service_time_imit[run, :, :],
                                                                                           axis=1) *
                                np.sum(energy_consumed_mat_imit[run, :, :], axis=1))
    print("[INFO] --------- End of calculation ---------")
    # *****************************************   EDT Result
    fig_edt = plt.figure(figsize=(8, 8))
    ax_edt = fig_edt.add_subplot(111)
    ax_edt.set_xlabel("Frames", size=12, fontweight='bold')
    ax_edt.set_ylabel("EDT [1 / Watt]", size=12, fontweight='bold')
    if new_rate:
        ax_edt.plot(np.arange(1, num_frm) + 1, np.mean(edt_mat[:, 1:], axis=0), color="blue", linestyle='-', marker='o',
                    markersize='8', label='EDT (Expert Demonstration)', linewidth=2)
        ax_edt.plot(np.arange(1, num_frm) + 1, np.mean(edt_mat_imit[:, 1:], axis=0), color="red", linestyle='--',
                    marker='x', markersize='10', label='EDT (Behavioral Cloning)', linewidth=2)
    else:
        ax_edt.plot(np.arange(1, num_frm) + 1, np.mean(edt_mat[:, 1:], axis=0), color="blue", linestyle='-', marker='o',
                    markersize='8', label='EDT (Expert Demonstration)', linewidth=2)
        ax_edt.plot(np.arange(1, num_frm) + 1, np.mean(edt_mat_imit[:, 1:], axis=0), color="red", linestyle='--',
                    marker='x', markersize='10', label='EDT (Behavioral Cloning)', linewidth=2)
    ax_edt.grid(True)
    ax_edt.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
    file_figobj = 'Output/FigureObject/edt_compare_mean.fig.pickle' % ()
    file_pdf = 'Output/Figures/edt_compare_mean.pdf' % ()
    if new_rate:
        file_figobj = 'Output/FigureObject/edt_compare_newrate.fig.pickle'
        file_pdf = 'Output/Figures/edt_compare_newrate.pdf'
    if save_pdf_obj:
        pickle.dump(fig_edt, open(file_figobj, 'wb'))
        fig_edt.savefig(file_pdf, bbox_inches='tight')

    # *****************************************   Packet Drop Result
    fig_drop = plt.figure(figsize=(8, 8))
    ax_drop = fig_drop.add_subplot(111)
    ax_drop.set_xlabel("Frames", size=12, fontweight='bold')
    ax_drop.set_ylabel("Number of dropped packets", size=12, fontweight='bold')
    if new_rate:
        ax_drop.plot(np.arange(0, num_frm) + 1, np.mean(np.sum(num_dropped, axis=2), axis=0), color="blue",
                     linestyle='-', marker='o', markersize='8', label='Packet Drop (Expert Demonstration)', linewidth=2)
        ax_drop.plot(np.arange(0, num_frm) + 1, np.mean(np.sum(num_dropped_imit, axis=2), axis=0), color="red",
                     linestyle='--', marker='x', markersize='10', label='Packet Drop (Behavioral Cloning)', linewidth=2)
    else:
        ax_drop.plot(np.arange(0, num_frm) + 1, np.mean(np.sum(num_dropped[:, :, :], axis=2), axis=0), color="blue",
                     linestyle='-', marker='o', markersize='8', label='Packet Drop (Expert Demonstration)', linewidth=2)
        ax_drop.plot(np.arange(0, num_frm) + 1, np.mean(np.sum(num_dropped_imit[:, :, :], axis=2), axis=0), color="red",
                     linestyle='--', marker='x', markersize='10', label='Packet Drop (Behavioral Cloning)', linewidth=2)
    ax_drop.grid(True)
    ax_drop.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
    file_figobj = 'Output/FigureObject/drop_compare_mean.fig.pickle' % ()
    file_pdf = 'Output/Figures/drop_compare_mean.pdf' % ()
    if new_rate:
        file_figobj = 'Output/FigureObject/drop_compare_newrate.fig.pickle' % ()
        file_pdf = 'Output/Figures/drop_compare_newrate.pdf' % ()
    if save_pdf_obj:
        pickle.dump(fig_drop, open(file_figobj, 'wb'))
        fig_drop.savefig(file_pdf, bbox_inches='tight')

    # *****************************************   Energy consumption Result
    fig_energy = plt.figure(figsize=(8, 8))
    ax_energy = fig_energy.add_subplot(111)
    ax_energy.set_xlabel("Frames", size=12, fontweight='bold')
    ax_energy.set_ylabel("Consumed Energy [J]", size=12, fontweight='bold')
    if new_rate:
        ax_energy.plot(np.arange(0, num_frm) + 1, np.mean(np.sum(energy_consumed_mat, axis=2), axis=0), color="blue",
                       linestyle='-', marker='o', markersize='8', label='Consumed Energy (Expert Demonstration)',
                       linewidth=2)
        ax_energy.plot(np.arange(0, num_frm) + 1, np.mean(np.sum(energy_consumed_mat_imit, axis=2), axis=0),
                       color="red", linestyle='--', marker='x', markersize='10',
                       label='Consumed Energy (Behavioral Cloning)', linewidth=2)
    else:
        ax_energy.plot(np.arange(0, num_frm) + 1, np.mean(np.sum(energy_consumed_mat[:, :, :], axis=2), axis=0),
                       color="blue", linestyle='-', marker='o', markersize='8',
                       label='Consumed Energy (Expert Demonstration)', linewidth=2)
        ax_energy.plot(np.arange(0, num_frm) + 1, np.mean(np.sum(energy_consumed_mat_imit[:, :, :], axis=2), axis=0),
                       color="red", linestyle='--', marker='x', markersize='10',
                       label='Consumed Energy (Behavioral Cloning)', linewidth=2)
    ax_energy.grid(True)
    ax_energy.legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
    file_figobj = 'Output/FigureObject/energy_compare_mean.fig.pickle' % ()
    file_pdf = 'Output/Figures/energy_compare_mean.pdf' % ()
    if new_rate:
        file_figobj = 'Output/FigureObject/energy_compare_newrate.fig.pickle' % ()
        file_pdf = 'Output/Figures/energy_compare_newrate.pdf' % ()
    if save_pdf_obj:
        pickle.dump(fig_energy, open(file_figobj, 'wb'))
        fig_energy.savefig(file_pdf, bbox_inches='tight')


def calculate_long_session_imit(number_switch, number_switch_imit, new_rate):
    # run = 0
    longest_session = np.zeros([num_run, num_frm, 1], dtype=int)
    longest_session_imit = np.zeros([num_run, num_frm, 1], dtype=int)
    for run in range(0, num_run):
        for Frame in range(0, num_frm):
            longest_session[run, Frame, 0] = int(np.max(np.bincount(number_switch[run, Frame, :])))
            longest_session_imit[run, Frame, 0] = int(np.max(np.bincount(number_switch_imit[run, Frame, :])))

    # *****************************************   Long Session Result
    fig_session = plt.figure()
    ax_session = fig_session.add_subplot(111)
    ax_session.set_xlabel("Frames", size=12, fontweight='bold')
    ax_session.set_ylabel("Longest Session [Events]", size=12, fontweight='bold')
    ax_session.plot(np.arange(0, num_frm) + 1, np.mean(longest_session, axis=0), color="blue",
                    linestyle='-', marker='o', markersize='8', label='Longest Session (Expert Demonstration)',
                    linewidth=2)
    ax_session.plot(np.arange(0, num_frm) + 1, np.mean(longest_session_imit, axis=0), color="red",
                    linestyle='--', marker='x', markersize='10', label='Longest Session (Behavioral Cloning)',
                    linewidth=2)
    ax_session.grid(True)
    ax_session.legend(prop={'size': 10, 'weight': 'bold'}, loc='best')
    file_figobj = 'Output/FigureObject/session_compare_mean.fig.pickle' % ()
    file_pdf = 'Output/Figures/session_compare_mean.pdf' % ()
    if new_rate:
        file_figobj = 'Output/FigureObject/session_compare_newrate.fig.pickle' % ()
        file_pdf = 'Output/Figures/session_compare_newrate.pdf' % ()
    if save_pdf_obj:
        pickle.dump(fig_session, open(file_figobj, 'wb'))
        fig_session.savefig(file_pdf, bbox_inches='tight')


def result_newrate():
    num_features = num_ue + num_ue + num_ue + 1  # number of queues + number of dist + number of dir + active_user
    num_actions = General.get('Actions')
    x_data_state_vec = np.empty([num_run * num_frm * num_event, num_features], dtype=object)
    y_action_vec = np.zeros([num_run * num_frm * num_event, num_actions], dtype=int) - 1

    x_data_state_mat = np.empty([num_run, num_event*num_frm, num_features], dtype=object)
    y_action_mat = np.zeros([num_run, num_event*num_frm, num_actions], dtype=int) - 1

    x_data_state_imit_mat = np.empty([num_run, num_event * num_frm, num_features], dtype=object)
    y_action_imit_mat = np.zeros([num_run, num_event * num_frm, num_actions], dtype=int) - 1

    num_run_new = num_run
    print(" --------- New Rate Results --------- ")
    run_list = range(0, num_run_new)
    for run in run_list:
        if platform.system() == "Windows":
            output_file = \
                "D:\\SimulationData\\TestData\\NewRate\\num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                "_cbr_rate_%d_Event_%d_new_rate.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate,
                                                        num_event)
            output_file_imit = \
                "D:\\SimulationData\\ImitatedModel\\imit_num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                "_cbr_rate_%d_Event_%d.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate, num_event)
        elif platform.system() == "Linux":
            output_file = \
                "SimulationData/TestData/NewRate/num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                "_cbr_rate_%d_Event_%d_new_rate.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate,
                                                        num_event)
            output_file_imit = \
                "SimulationData/ImitatedModel/imit_num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                "_cbr_rate_%d_Event_%d.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate, num_event)
        else:
            print("Nor Linux neither Windows")
            return
        readfile = np.load(output_file, allow_pickle=True)
        x_data_state_vec[run * num_frm * num_event:(run + 1) * num_frm * num_event, :] = \
            readfile['state_feature_vec'].reshape(num_frm * num_event, num_features)
        y_action_vec[run * num_frm * num_event:(run + 1) * num_frm * num_event, :] = \
            readfile['action_vec'].reshape(num_frm * num_event, num_actions)

        x_data_state_mat[run, :, :] = readfile['state_feature_vec'].reshape(num_frm * num_event, num_features)
        y_action_mat[run, :, :] = readfile['action_vec'].reshape(num_frm * num_event, num_actions)

        readfile_imit = np.load(output_file_imit, allow_pickle=True)
        x_data_state_imit_mat[run, :, :] = readfile_imit['state_feature_vec'].reshape(num_frm * num_event, num_features)
        y_action_imit_mat[run, :, :] = readfile_imit['action_vec'].reshape(num_frm * num_event, num_actions)

    x_queue_vec = x_data_state_vec[:, 0:num_ue]
    x_queue_mat = x_data_state_mat[:, :, 0:num_ue]
    x_queue_imit_mat = x_data_state_imit_mat[:, :, 0:num_ue]

    print("[INFO] data matrix: ({:.2f}MB)".format(x_data_state_vec.nbytes / (1024 * 1000.0)))

    y_action_lb = LabelBinarizer()
    y_action_user = y_action_lb.fit_transform(y_action_vec[:, 0])
    y_action_user_mat = np.zeros([num_run, num_frm*num_event, num_ue], dtype=int) - 1
    y_action_user_imit_mat = np.zeros([num_run, num_frm * num_event, num_ue], dtype=int) - 1
    for run in run_list:
        y_action_user_mat[run, :, :] = y_action_lb.fit_transform(y_action_mat[run, :, 0])
        y_action_user_imit_mat[run, :, :] = y_action_lb.fit_transform(y_action_imit_mat[run, :, 0])
    model_queue = load_model('Output/Models/model_queue_5_layers_[40, 80, 160, 80, 5]_units.model')

    loss_queue, accuracy_queue = model_queue.evaluate(x_queue_vec, y_action_user)
    print('accuracy_queue: %.2f' % (accuracy_queue * 100), "loss_queue: %.5f" % loss_queue)
    predictions_queue = model_queue.predict_classes(x_queue_vec)
    predictions_queue_mat = np.zeros([num_run, num_frm*num_event], dtype=int) - 1
    predictions_queue_imit_mat = np.zeros([num_run, num_frm * num_event], dtype=int) - 1
    for run in run_list:
        predictions_queue_mat[run, :] = model_queue.predict_classes(x_queue_mat[run, :, :])
        predictions_queue_imit_mat[run, :] = model_queue.predict_classes(x_queue_imit_mat[run, :, :])
    index = 0
    accuracy = np.zeros([num_frm, 1], dtype=float)
    correctness = np.zeros([num_frm, num_event], dtype=int)

    accuracy_mat = np.zeros([num_run, num_frm, 1], dtype=float)
    correctness_mat = np.zeros([num_run, num_frm, num_event], dtype=int)

    accuracy_imit_mat = np.zeros([num_run, num_frm, 1], dtype=float)
    correctness_imit_mat = np.zeros([num_run, num_frm, num_event], dtype=int)

    for frame in range(0, num_frm):
        for event in range(0, num_event):
            if predictions_queue[index] == y_action_vec[index, 0]:
                correctness[frame, event] = 1
            index += 1
        accuracy[frame] = np.mean(correctness[frame, :])

    for run in range(0, num_run):
        index = 0
        for frame in range(0, num_frm):
            for event in range(0, num_event):
                if predictions_queue_mat[run, index] == y_action_mat[run, index, 0]:
                    correctness_mat[run, frame, event] = 1

                if predictions_queue_imit_mat[run, index] == y_action_imit_mat[run, index, 0]:
                    correctness_imit_mat[run, frame, event] = 1

                index += 1
            accuracy_mat[run, frame] = np.mean(correctness_mat[run, frame, :])
            accuracy_imit_mat[run, frame] = np.mean(correctness_imit_mat[run, frame, :])

    fig_session = plt.figure()
    ax_session = fig_session.add_subplot(111)
    ax_session.set_xlabel("Frames", size=12, fontweight='bold')
    ax_session.set_ylabel("Performance/Performance of the expert", size=12, fontweight='bold')
    ax_session.plot(np.arange(0, num_frm) + 1, np.mean(accuracy_imit_mat, axis=0), color="blue",
                    linestyle='-', marker='o', markersize='8', label='Mimic the expert(Trained rate)',
                    linewidth=1.5)
    ax_session.plot(np.arange(0, num_frm) + 1, np.mean(accuracy_mat[:, :], axis=0), color="red",
                    linestyle='-', marker='^', markersize='8', label='Mimic the expert(New rate)',
                    linewidth=1.5)

    ax_session.plot(np.arange(0, num_frm) + 1, np.ones([num_frm, 1]), color="black", linestyle='--',
                    label='Expert (New rate)', linewidth=2)
    ax_session.grid(True)
    ax_session.legend(prop={'size': 10, 'weight': 'bold'}, loc='best')
    file_figobj = 'Output/FigureObject/newrate_performance_mean.fig.pickle'
    file_pdf = 'Output/Figures/newrate_performance_mean.pdf'
    if save_pdf_obj:
        pickle.dump(fig_session, open(file_figobj, 'wb'))
        fig_session.savefig(file_pdf, bbox_inches='tight')
