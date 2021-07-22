from torch.utils.tensorboard import SummaryWriter
import os
import subprocess
import datetime
import json
import numpy as np
import shutil


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
        """Save code and all arguments given by user in json file"""
        with open(os.path.join(self.path, 'hyperparameters.json'), 'w') as file:
            json.dump(vars(self.args), file, indent=4)

        # Copy code used for training to see modifications
        code = [file for file in os.listdir(os.getcwd()) if '.py' in file]

        os.makedirs(os.path.join(self.path, 'code'))

        for file in code:
            shutil.copyfile(file, os.path.join(self.path, 'code', file))

        return 0

    def addTraining(self, aveTrainError, singleTrainError, trainLoss, epoch):
        """Adding training loss curve on tensorboard"""
        self.add_scalar("training/Average training error", aveTrainError, epoch)
        self.add_scalar("training/Single training error", singleTrainError, epoch)
        self.add_scalar("training/Training loss", trainLoss, epoch)

        return 0

    def addTesting(self, aveTestError, singleTestError, testLoss, epoch):
        """Adding testing loss curve on tensorboard"""
        self.add_scalar("testing/Average testing error", aveTestError, epoch)
        self.add_scalar("testing/Single testing error", singleTestError, epoch)
        self.add_scalar("testing/Testing loss", testLoss, epoch)

        return 0

    def addNbChanges(self, nbChanges, epoch, nbChangesConv=None):
        """Adding the number of weight flips to monitor training"""
        if self.args.archi == 'fc':

            for layer in range(len(self.args.layersList) - 1):
                nbWeights = self.net.W[layer].weight.numel()
                piLayerEpoch = np.log(nbChanges[layer] / nbWeights + np.exp(-9))
                self.add_scalar("flips/Weight flips for layer " + str(layer), piLayerEpoch, epoch)

        elif self.args.archi == 'conv':

            for layer in range(len(self.args.layersList)):
                nbWeights = self.net.fc[layer].weight.numel()
                piLayerEpoch = np.log(nbChanges[layer] / nbWeights + np.exp(-9))
                self.add_scalar("flips/Weight flips for FC layer " + str(layer), piLayerEpoch, epoch)

            for layer in range(len(self.args.convList) - 1):
                nbWeights = self.net.conv[layer].weight.numel()
                piLayerEpoch = np.log(nbChangesConv[layer] / nbWeights + np.exp(-9))
                self.add_scalar("flips/Weight flips for conv layer " + str(layer), piLayerEpoch, epoch)

        return 0

    def addGrad(self, epoch):
        """Adding the gradients mean after each epoch"""
        if self.args.archi == 'fc':
            for layer in range(len(self.args.layersList) - 1):
                self.add_scalar("gradients/Weight gradients for layer " + str(layer), self.net.gradWInt[layer], epoch)

        if self.args.archi == 'conv':
            for layer in range(len(self.args.layersList)):
                self.add_scalar("gradients/Weight gradients for FC layer " + str(layer), self.net.fcGrad[layer], epoch)

            for layer in range(len(self.args.convList) - 1):
                self.add_scalar("gradients/Weight gradients for conv layer " + str(layer), self.net.convGrad[layer], epoch)

        return 0


