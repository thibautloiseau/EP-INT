from torch.utils.tensorboard import SummaryWriter
import os
import subprocess
import datetime
import json
import numpy as np


class Visualizer(SummaryWriter):
    """Monitor training with tensorboard"""

    def __init__(self, net, args):
        # We create the path where we are going to store results
        path = os.path.join(os.getcwd(), "SAVE-" + args.archi + "-" + args.dataset, str(datetime.date.today()))

        if os.path.exists(path):
            try:
                noFolder = str(max([int(element[1:]) for element in os.listdir(path) if 'S' in element]) + 1)
                self.path = os.path.join(os.getcwd(), "SAVE-" + args.archi + "-" + args.dataset, str(datetime.date.today()), 'S' + noFolder)
            except:
                self.path = os.path.join(os.getcwd(), "SAVE-" + args.archi + "-" + args.dataset, str(datetime.date.today()), 'S1')

        else:
            self.path = os.path.join(os.getcwd(), "SAVE-" + args.archi + "-" + args.dataset, str(datetime.date.today()), 'S1')

        super(Visualizer, self).__init__(self.path)

        self.args = args
        self.net = net


    def launch(self):
        """Launch the tensorboard server on localhost"""
        subprocess.Popen("tensorboard --logdir=\"" + os.path.dirname(self.path) + "\"", shell=True)

        return 0

    def saveHyperParameters(self):
        """Save all arguments given by user in json file"""
        with open(os.path.join(self.path, 'hyperparameters.json'), 'w') as file:
            json.dump(vars(self.args), file, indent=4)

        return 0

    def addTraining(self, aveTrainError, singleTrainError, trainLoss, epoch):
        """Adding training loss curve on tensorboard"""
        self.add_scalar("Average training error", aveTrainError, epoch)
        self.add_scalar("Single training error", singleTrainError, epoch)
        self.add_scalar("Training loss", trainLoss, epoch)

        return 0

    def addTesting(self, aveTestError, singleTestError, testLoss, epoch):
        """Adding testing loss curve on tensorboard"""
        self.add_scalar("Average testing error", aveTestError, epoch)
        self.add_scalar("Single testing error", singleTestError, epoch)
        self.add_scalar("Testing loss", testLoss, epoch)

        return 0

    def addNbChanges(self, nbChanges, epoch):
        """Adding the number of weight flips to monitor training"""
        for layer in range(len(self.args.layersList) - 1):
            nbWeights = self.net.W[layer].weight.numel()
            piLayerEpoch = np.log(nbChanges[layer] / nbWeights + np.exp(-9))
            self.add_scalar("Weight flips for layer " + str(layer), piLayerEpoch, epoch)

        return 0
