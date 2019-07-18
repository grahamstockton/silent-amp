"""
Example script using the tensorflow version of the NN model
"""

import os,sys
import ase.io
from amp import Amp
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
from amp.descriptor.cutoffs import Cosine,Polynomial
#from amp.model.neuralnetwork import NeuralNetwork
from amp.model.tflow import NeuralNetwork
from amp.model import LossFunction

Gs = {"H": [{"type":"G2", "element":"H", "eta":0.05, "Rs":0.},
            {"type":"G2", "element":"H", "eta":8., "Rs":0.},
            {"type":"G2", "element":"H", "eta":20., "Rs":0.},
            {"type":"G2", "element":"H", "eta":40., "Rs":0.},
            {"type":"G2", "element":"H", "eta":80., "Rs":0.},
            {"type":"G2", "element":"H", "eta":600., "Rs":2.6},
            {"type":"G2", "element":"H", "eta":800., "Rs":2.65},
            {"type":"G2", "element":"H", "eta":600., "Rs":2.8},
            {"type":"G2", "element":"H", "eta":100., "Rs":3.0},
            {"type":"G2", "element":"H", "eta":100., "Rs":4.0},
            {"type":"G2", "element":"H", "eta":200., "Rs":4.5},
            {"type":"G2", "element":"H", "eta":100., "Rs":5.0},
            {"type":"G2", "element":"H", "eta":100., "Rs":6.0},
            {"type":"G2", "element":"Pd", "eta":10., "Rs":0.},
            {"type":"G2", "element":"Pd", "eta":40., "Rs":0.},
            {"type":"G2", "element":"Pd", "eta":100., "Rs":0.},
            {"type":"G2", "element":"Pd", "eta":120., "Rs":0.},
            {"type":"G2", "element":"Pd", "eta":160., "Rs":0.},
            {"type":"G2", "element":"Pd", "eta":300., "Rs":0.},
            {"type":"G2", "element":"Pd", "eta":700., "Rs":1.1},
            {"type":"G2", "element":"Pd", "eta":800., "Rs":1.25},
            {"type":"G2", "element":"Pd", "eta":300., "Rs":2.0},
            {"type":"G2", "element":"Pd", "eta":800., "Rs":2.2},
            {"type":"G2", "element":"Pd", "eta":400., "Rs":2.5},
            {"type":"G2", "element":"Pd", "eta":400., "Rs":2.7},
           ],
      "Pd": [
            {"type":"G2", "element":"Pd", "eta":300., "Rs":0.},
            {"type":"G2", "element":"Pd", "eta":700., "Rs":1.1},
            {"type":"G2", "element":"Pd", "eta":800., "Rs":1.25},
            {"type":"G2", "element":"Pd", "eta":300., "Rs":2.0},
            {"type":"G2", "element":"Pd", "eta":800., "Rs":2.2},
             {"type":"G2", "element":"Pd", "eta":160, "Rs":0.},
             {"type":"G2", "element":"Pd", "eta":120., "Rs":0.5},
             {"type":"G2", "element":"Pd", "eta":120., "Rs":1.0},
             {"type":"G2", "element":"Pd", "eta":40., "Rs":2.0},
             {"type":"G2", "element":"Pd", "eta":800., "Rs":3.2},
             {"type":"G2", "element":"Pd", "eta":400., "Rs":3.5},
             {"type":"G2", "element":"Pd", "eta":400., "Rs":4.0},
             {"type":"G2", "element":"Pd", "eta":400., "Rs":5.0},
             {"type":"G2", "element":"Pd", "eta":400., "Rs":6.0},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":-1.0, "zeta":100.0, "theta_s":-1.0},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":-1.0, "zeta":100.0, "theta_s":-1.5},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":-1.0, "zeta":100.0, "theta_s":-1.75},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":-1.0, "zeta":100.0, "theta_s":-2.0},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":-1.0, "zeta":300.0, "theta_s":-2.15},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":-1.0, "zeta":300.0, "theta_s":-2.25},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":1.0, "zeta":160.0, "theta_s":0.0},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":1.0, "zeta":320.0, "theta_s":0.25},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":1.0, "zeta":640.0, "theta_s":0.35},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":1.0, "zeta":160.0, "theta_s":0.5},
            {"type":"G4", "elements":["Pd", "Pd"],"eta":0.005, "gamma":1.0, "zeta":160.0, "theta_s":0.75},
             ]}


calc = Amp(descriptor=Gaussian(Gs=Gs,
                               cutoff=Cosine(6.0)
                               ),
           cores=24,
           model=NeuralNetwork(hiddenlayers=(50,), activation='sigmoid',
                               maxTrainingEpochs=20000,
                               energy_coefficient=1.0,
                               force_coefficient=0.1,
                               optimizationMethod='l-BFGS-b',
                               convergenceCriteria={'energy_rmse': 0.05,
                                                    'force_rmse': 0.05}))
calc.train(images='train.traj',overwrite=True)

