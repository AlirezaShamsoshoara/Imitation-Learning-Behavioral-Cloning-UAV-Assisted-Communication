"""
#################################
# Classification after training the Model
  Keras
  GPU: Nvidia RTX 2080 Ti
  OS: Ubuntu 18.04
#################################
"""

#########################################################
# import libraries
import pickle
import platform
import itertools
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from config import Config_Queue
from utils import dir_string_to_num
from config import Config_General as General

#########################################################
# General Parameters
num_ue = General.get('NUM_UE')
num_run = General.get('NUM_RUN')
# num_run = 1
num_frm = General.get('NUM_FRM')
cbr_rate = General.get('CBR_RATE')
num_actions = General.get('Actions')
num_event = General.get('Sim_Events')
num_angles = General.get('NUM_ANGLE')
queue_lim = Config_Queue.get('Queue_limit')
num_features = num_ue + num_ue + num_ue + 1  # number of queues + number of dist + number of dir + active_user
x_data_state_vec = np.empty([num_run * num_frm * num_event, num_features], dtype=object)
y_action_vec = np.zeros([num_run * num_frm * num_event, num_actions], dtype=int) - 1


#########################################################
# Function definition

def classify():
    """
    This function loads data from the local drive to evaluate the performance on the collected information from the
    expert.
    :return: None
    """
    print(" --------- Classification --------- ")
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
        x_data_state_vec[run * num_frm * num_event:(run + 1) * num_frm * num_event, :] = \
            readfile['state_feature_vec'].reshape(num_frm * num_event, num_features)
        y_action_vec[run * num_frm * num_event:(run + 1) * num_frm * num_event, :] = \
            readfile['action_vec'].reshape(num_frm * num_event, num_actions)

    x_queue_vec = x_data_state_vec[:, 0:num_ue]
    x_dir_vec = x_data_state_vec[:, 2 * num_ue:3 * num_ue]
    x_dir_vec_num = dir_string_to_num(x_dir_vec)
    x_dir_vec_con = np.concatenate((x_dir_vec_num, np.reshape(y_action_vec[:, 0], [y_action_vec[:, 0].shape[0], 1])),
                                   axis=1)

    print("[INFO] data matrix: ({:.2f}MB)".format(
        x_data_state_vec.nbytes / (1024 * 1000.0)))
    y_action_lb = LabelBinarizer()
    y_action_user = y_action_lb.fit_transform(y_action_vec[:, 0])
    y_action_dir = y_action_lb.fit_transform(y_action_vec[:, 1])

    model_queue = load_model('Output/Models/model_queue_5_layers_[40, 80, 160, 80, 5]_units.model')
    model_dir = load_model('Output/Models/model_dir_3_layers_[8, 10, 3]_units.model')

    loss_queue, accuracy_queue = model_queue.evaluate(x_queue_vec, y_action_user)
    print('accuracy_queue: %.2f' % (accuracy_queue * 100), "loss_queue: %.5f" % loss_queue)

    loss_dir, accuracy_dir = model_dir.evaluate(x_dir_vec_con, y_action_dir)
    print('accuracy_dir: %.2f' % (accuracy_dir * 100), "loss_dir: %.5f" % loss_dir)

    predictions_queue = model_queue.predict_classes(x_queue_vec)
    for i in range(10):
        print('%s => %d (expected %d)' % (x_queue_vec[i].tolist(), predictions_queue[i], y_action_vec[i, 0]))

    predictions_dir = model_dir.predict_classes(x_dir_vec_con)
    for i in range(10):
        print('%s => %d (expected %d)' % (x_dir_vec_con[i].tolist(), predictions_dir[i], y_action_vec[i, 1]))

    # predictions_not_rounded = model_queue.predict(x_queue_vec)
    # rounded = [round(x[0]) for x in predictions_not_rounded]

    cm = confusion_matrix(y_true=y_action_vec[:, 0], y_pred=predictions_queue)
    cm_plot_labels = ['1', '2', '3', '4', '5']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig_conf = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=12, fontweight='bold')
    plt.xlabel('Predicted label', size=12, fontweight='bold')
    # file_pdf = 'Output/Figures/confusion_matrix.pdf'
    file_figobj = 'Output/FigureObject/confusion_matrix_40_epochs.fig.pickle'
    pickle.dump(fig_conf, open(file_figobj, 'wb'))
    # plt.savefig(file_pdf)
    # plt.imsave(file_pdf)
