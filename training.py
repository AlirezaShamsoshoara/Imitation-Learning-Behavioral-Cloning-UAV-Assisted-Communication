"""
#################################
  Training phase after demonstration
  Keras
  GPU: Nvidia RTX 2080 Ti
  OS: Ubuntu 18.04
#################################
"""

#########################################################
# import libraries
import platform
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from config import Config_Queue
from plotdata import plot_training
from utils import dir_string_to_num
from config import Config_General as General
from config import Config_learning as Learning

#########################################################
# General Parameters
num_ue = General.get('NUM_UE')
num_run = General.get('NUM_RUN')
num_frm = General.get('NUM_FRM')
cbr_rate = General.get('CBR_RATE')
dir_loss = Learning.get('dir_loss')
num_actions = General.get('Actions')
num_event = General.get('Sim_Events')
num_angles = General.get('NUM_ANGLE')
user_loss = Learning.get('user_loss')
test_size = Learning.get('test_size')
INIT_LR = Learning.get('Learning_Rate')
EPOCHS_DIR = Learning.get('Epochs_dir')
EPOCHS_USER = Learning.get('Epochs_user')
queue_lim = Config_Queue.get('Queue_limit')
num_features = num_ue + num_ue + num_ue + 1  # number of queues + number of dist + number of dir + active_user
User_loss_weight = Learning.get('User_loss_weight')
Dir_loss_weight = Learning.get('Dir_loss_weight')
x_data_state_vec = np.empty([num_run * num_frm * num_event, num_features], dtype=object)
y_action_vec = np.zeros([num_run * num_frm * num_event, num_actions], dtype=int) - 1
# x_queue_vec = np.zeros([num_run * num_frm * num_event, num_ue], dtype=int)

#########################################################
# Function definition


def train():
    """
    This function trains a DNN model based on the collected information from the expert.
    :return: None
    """
    print(" --------- Training --------- ")
    run_list = range(0, num_run)
    for run in run_list:
        if platform.system() == "Windows":
            output_file = \
                "D:\\SimulationData\\TrainData\\num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d_" \
                "cbr_rate_%d_Event_%d.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate, num_event)
        elif platform.system() == "Linux":
            output_file = \
                "SimulationData/TrainData/num_UE_%d_num_angles_%d_queue_lim_%d_Run_%d_Frame_%d" \
                "_cbr_rate_%d_Event_%d.npz" % (num_ue, num_angles, queue_lim, run, num_frm, cbr_rate, num_event)
        else:
            print("Nor Linux neither Windows")
            return
        readfile = np.load(output_file, allow_pickle=True)
        x_data_state_vec[run*num_frm*num_event:(run+1)*num_frm*num_event, :] = \
            readfile['state_feature_vec'].reshape(num_frm*num_event, num_features)
        y_action_vec[run * num_frm * num_event:(run + 1) * num_frm * num_event, :] = \
            readfile['action_vec'].reshape(num_frm * num_event, num_actions)
    x_queue_vec = x_data_state_vec[:, 0:num_ue]
    # x_dist_vec = x_data_state_vec[:, num_ue:2*num_ue]
    x_dir_vec = x_data_state_vec[:, 2*num_ue:3*num_ue]
    print("[INFO] data matrix: ({:.2f}MB)".format(
         x_data_state_vec.nbytes / (1024 * 1000.0)))
    y_action_lb = LabelBinarizer()
    y_action_user = y_action_lb.fit_transform(y_action_vec[:, 0])
    y_action_dir = y_action_lb.fit_transform(y_action_vec[:, 1])
    # 0: 'CC: Counter-clockwise', 1:'C: Clockwise', 2: 'None'

    split_queue = train_test_split(x_queue_vec, y_action_user, test_size=test_size, random_state=42)
    train_x_queue, test_x_queue, train_y_user, test_y_user = split_queue

    x_dir_vec_num = dir_string_to_num(x_dir_vec)
    x_dir_vec_con = np.concatenate((x_dir_vec_num, np.reshape(y_action_vec[:, 0], [y_action_vec[:, 0].shape[0], 1])),
                                   axis=1)
    split_dir = train_test_split(x_dir_vec_con, y_action_dir, test_size=test_size, random_state=42)
    train_x_dir, test_x_dir, train_y_dir, test_y_dir = split_dir

    losses = {"user_output": user_loss, "dir_output": dir_loss}
    # loss_weight = {"user_output": User_loss_weight, "dir_output": Dir_loss_weight}

    # ****************************************************************************************** TRAINING_USER
    print("[INFO] compiling model for user selection...")
    model_queue = UAVModel.build_queue(losses.get('user_output'))
    res_queue = model_queue.fit(train_x_queue, train_y_user, validation_data=(test_x_queue, test_y_user),
                                epochs=EPOCHS_USER, verbose=1)
    layers_len = len(model_queue.layers)
    units_num = []
    [units_num.append(model_queue.layers[i].units) for i in range(0, layers_len)]
    model_queue.name = 'model_queue'
    file_model_queue = 'Output/Models/model_queue_%d_layers_%s_units_%d_epochs.model' % (layers_len, units_num,
                                                                                         EPOCHS_USER)
    model_queue.save(file_model_queue)

    if Learning.get('TrainingPlot'):
        plot_training(res_queue, 'queue', layers_len, units_num)
    print("[INFO] --------- Stop Training --------- ")

    # ****************************************************************************************** TRAINING_DIR
    print("[INFO] compiling model for direction ...")
    model_dir = UAVModel.build_dir(losses.get('dir_output'))
    res_dir = model_dir.fit(train_x_dir, train_y_dir, validation_data=(test_x_dir, test_y_dir), epochs=EPOCHS_DIR,
                            verbose=1)
    layers_len = len(model_dir.layers)
    units_num = []
    [units_num.append(model_dir.layers[i].units) for i in range(0, layers_len)]
    model_dir.name = 'model_dir'
    file_model_dir = 'Output/Models/model_dir_%d_layers_%s_units_%d_epochs.model' % (layers_len, units_num, EPOCHS_DIR)
    model_dir.save(file_model_dir)

    if Learning.get('TrainingPlot'):
        plot_training(res_dir, 'direction', layers_len, units_num)
    print("[INFO] --------- Stop Training --------- ")


class UAVModel:
    """
    UAV class for creating model for the UAV
    """
    @staticmethod
    def build_queue(loss):
        """
        This function builds the model for the UE selection.
        :param loss: Loss function for the model compilation
        :return: Compiled model
        """
        model = Sequential()
        model.add(Dense(units=40, input_dim=num_ue, activation='relu'))
        model.add(Dense(units=80, activation='relu'))
        model.add(Dense(units=160, activation='relu'))
        model.add(Dense(units=80, activation='relu'))
        model.add(Dense(units=num_ue, activation='softmax'))
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS_USER)
        # model.compile(loss=loss, optimizer='adam', metrics=["accuracy"])
        model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
        return model

    @staticmethod
    def build_dir(loss):
        """
        This function builds the model for the direction selection.
        :param loss: Loss function for the model compilation
        :return: Compiled model
        """
        dir_action = [0, 1, 2]  # 0: 'CC: Counter-clockwise', 1:'C: Clockwise', 2: 'None'
        model = Sequential()
        model.add(Dense(units=8, input_dim=num_ue+1, activation='relu'))
        model.add(Dense(units=10, activation='relu'))
        model.add(Dense(units=len(dir_action), activation='softmax'))
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS_DIR)
        model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
        return model
