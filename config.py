"""
#################################
# Configuration File
#################################
"""

Radius = 100
Float_Precision = 2
Config_General = {'NUM_UAV': 1, 'NUM_UE': 5, 'NUM_FRM': 50, 'Location_SaveFile': 0, 'CSI_SaveFile': 1,
                  'Energy_SaveFile': 0, 'PlotLocation': 0, 'DF': 0, 'printFlag': 0, 'PlotResult': 1, 'NUM_RUN': 10,
                  'SaveOutput': 0, 'NUM_ANGLE': 36, 'Circle': 360, 'CBR_RATE': 220,
                  'PlotDistribution': 0, 'Sim_Events': 1000, 'Actions': 2, 'Demonstration_plot': True,
                  'Mode': "Imitation", "PacketSize": 100000, 'DecisionDelay': 0.015, 'SavePDF': 0}

# ***** Modes ==> i) Demonstration: The expert shows his/her behavior
# *****          ii) Training: Supervised learning the clone expert's behavior after demonstration
# *****         iii) Classification: Evaluation of the imitated model on the classification problem
# *****          iv) Imitation: Getting action from the imitation model
# *****           v) Results_demonstration: Results of the expert
# *****          vi) Result_imitation: Result of the agent (Behavioral cloning)
# *****         vii) Result_imitation_newRate: Result of the agent (Behavioral cloning) for new arrival rates

# Different Modes: {"Demonstration", "Training", "Classification", "Imitation", "Results_demonstration",
# "Result_imitation", "Result_imitation_newRate"}

Config_Queue = {'Queue_limit': 200, 'Max_Threshold': 40, 'Min_Threshold': 75}

Config_Params = {'T': 1, 'alpha1': 1, 'alpha2': 1, 'alpha3': 1, 'alpha4': 1, 'beta1': 1, 'beta2': 1}
Config_Dim = {'Height': 50, 'Var_Dim': 2, 'Tot_Dim': 3}
min_energy = Config_General.get('NUM_UE') * Config_General.get('NUM_FRM') * Config_General.get('CBR_RATE')
max_energy = min_energy + 1000
Config_Power = {'Max_energy': max_energy, 'Min_energy': min_energy, 'mob_consump': 0.1, 'tran_consump': 0.005,
                'switch_consump': 1}

pathDist = 'ConfigData/LocMatrix_UE_%d_Radius_%d' % (Config_General.get('NUM_UE'), Radius)
pathEnergy = 'ConfigData/Energy_UE_%d_Radius_%d' % (Config_General.get('NUM_UE'), Radius)
Config_Path = {'PathDist': pathDist, 'pathEnergy': pathEnergy}

Config_learning = {"User_loss_weight": 1.0, "Dir_loss_weight": 1.0, "user_loss": "categorical_crossentropy",
                   "dir_loss": "categorical_crossentropy", "Learning_Rate": 1e-3, "Epochs_user": 40, "Epochs_dir": 20,
                   "BatchSize": 32, "test_size": 0.2, "TrainingPlot": True}
