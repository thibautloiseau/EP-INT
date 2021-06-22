import numpy as np
import torch

from main import *
import matplotlib.pyplot as plt
import pickle

# Define classes to import data
class ReshapeTransform:
    """Transform the input in the appropriate way"""
    def __init__(self, newSize):
        self.newSize = newSize

    def __call__(self, img):
        return torch.reshape(img, self.newSize)


class ReshapeTransformTarget:
    """Transform target output in the appropriate way"""
    def __init__(self, noClasses, expandOutput):
        self.noClasses = noClasses
        self.expandOutput = expandOutput

    def __call__(self, target):
        target = torch.tensor(target).unsqueeze(0).unsqueeze(1)
        targetOneHot = torch.zeros((1, self.noClasses))

        return targetOneHot.scatter_(1, target, 1).repeat_interleave(self.expandOutput).squeeze(0)


# For fc architecture
class DataSet():
    """Create the data_loader for training and for testing"""
    def __init__(self, args):
        self.trainBatchSize = args.trainBatchSize
        self.testBatchSize = args.testBatchSize
        self.fcTransforms = [transforms.ToTensor(), ReshapeTransform((-1,))]
        self.expandOutput = args.expandOutput

    def __call__(self):
        """We return a tuple with both dataloader for train and test"""
        return datasets.MNIST(
                    root='./data', train=True, download=True,
                    transform=transforms.Compose(self.fcTransforms),
                    target_transform=ReshapeTransformTarget(10, self.expandOutput)
                )

########################################################################################################################
# Test on alpha values
def main1():
    Nin = 784.
    Nout = 10.

    n = np.arange(1, 20)

    def overflow():
        return ((3 * 2**(n-1) - 2) / (Nin + Nout)).astype(int)

    def alpha():
        return (1 / (2 * np.sqrt(Nin)) * (2**(n-1) - 1)).astype(int)

    over = overflow()
    alphas = alpha()

    plt.plot(n, over, 'b', label='Max alpha for overflow')
    plt.plot(n, alphas, 'r', label='Ideal alpha')
    plt.xlabel('Number of bits')
    plt.ylabel('Values of alpha')
    plt.legend()
    plt.show()

# main1()

########################################################################################################################
# Test on stepper

def main2():
    args.layersList.reverse()
    for i in range(10):
        net = FCbinWAInt(args)
        # Creating dataloaders only for training first
        trainLoader, _ = Data_Loader(args)()

        if net.cuda:
            net.to(net.device)

        batch_idx, (data, targets) = next(iter(enumerate(trainLoader)))

        # We initialize the first layer with input data
        state = net.initHidden(data)

        # Sending tensors to GPU if available
        if net.cuda:
            targets = targets.to(net.device)
            net.beta = net.beta.to(net.device)

            for i in range(len(state)):
                state[i] = state[i].to(net.device)

        # Keep track of the states during free phase to see evolution
        # For output

        save = []

        for k in range(len(state) - 1):
            save.append([])
            save[-1].append(state[k][0][7].item())

        #####
        # Free phase
        T = 10

        for step in range(T):
            state = net.stepper(state)

            for k in range(len(state) - 1):
                save[k].append(state[k][0][7].item())

        freeState = state.copy()

        # See evolution of free state
        for k in range(len(freeState) - 1):
            print(k)
            print(save[k])
        #     plt.plot(range(T + 1), save[k])
        # plt.show()

        #####
        # Nudged phase
        # Kmax = 12
        # beta = 2
        #
        # for step in range(Kmax):
        #     state = net.stepper(state, target=targets, beta=beta)
        #     for k in range(len(state) - 1):
        #         save[k].append(state[k][0][7].item())
        #
        # # Plotting state evolution
        # # For first hidden layer in [0, 1]
        # for k in range(1):
        #     plt.plot(range(T + 1 + Kmax), save[k])
        # plt.show()
        #
        # # For next layers in [0, 2**(n-1) - 1]
        # for k in range(1, 3):
        #     plt.plot(range(T + 1 + Kmax), save[k])
        # plt.show()



    # We compute the gradient
    # gradW, _, _ = net.computeGradients(freeState, state)
    #
    # freeBinState = net.getBinState(freeState)
    # nudgedBinState = net.getBinState(state)

    # for i in range(len(state)):
    #     print('layer ' + str(i))
    #     print(torch.count_nonzero(freeBinState[i][52]).item())

# main2()

########################################################################################################################
# See initialized int weights

def main3():
    net = FCbinWAInt(args)

    for i in range(len(net.Wint)):
        print(net.Wint[i].weight)
        print(net.W[i].weight)

# main3()

########################################################################################################################
# Seeing number of bits for which to code BOP parameters

def main4():
    n = np.arange(2, 25)

    def scaling():
        return (2e-7 * (2**(n-1))).astype(int)

    y = scaling()
    plt.plot(n, y)
    plt.show()

# main4()

########################################################################################################################
# Numbers of bit with alpha

def main5():
    n = np.arange(2, 16)
    Nin = 8192

    def scaling():
        return (1 / (2*np.sqrt(Nin)) * (2**(n-1))).astype(int)

    y = scaling()
    print(y)
    plt.plot(n, y)
    plt.show()

# main5()

########################################################################################################################
# Tuning BOP hyperparameters

def main6():
    args.layersList.reverse()
    for i in range(10):
        net = FCbinWAInt(args)

        # Creating dataloaders only for training first
        trainLoader, _ = Data_Loader(args)()

        if net.cuda:
            net.to(net.device)

        batch_idx, (data, target) = next(iter(enumerate(trainLoader)))

        # We initialize the first layer with input data
        state = net.initHidden(data)

        # Sending tensors to GPU if available
        if net.cuda:
            target = target.to(net.device)
            net.beta = net.beta.to(net.device)

            for k in range(len(state)):
                state[k] = state[k].to(net.device)

        # Keep track of the states during free phase to see evolution
        # For output

        save = []

        for k in range(len(state)):
            save.append([])
            save[-1].append(state[k][0][99].item())

        #####
        # Free phase
        T = 16

        for step in range(T):
            state = net.stepper(state)

            for k in range(len(state)):
                save[k].append(state[k][0][99].item())

        freeState = state.copy()

        # # See evolution of free state
        # for k in range(len(freeState)):
        #     plt.plot(save[k])
        #     plt.title('layer ' + str(k))
        #     plt.show()

        #####
        # Nudged phase
        K = 16

        for step in range(K):
            state = net.stepper(state, target=target, beta=net.beta)

            for k in range(len(state)):
                save[k].append(state[k][0][99].item())

        # # See evolution of states
        # for k in range(len(freeState)):
        #     plt.plot(save[k])
        #     plt.title('layer ' + str(k))
        #     plt.show()

        nudgedState = state.copy()

        gradW, _, _ = net.computeGradients(freeState, nudgedState)
        # for k in range(len(gradW)):
        #     print('layer ' + str(k))
        #     grad = gradW[k].tolist()
        #     plt.plot(grad, '+')
        #     plt.title('layer ' + str(k))
        #     plt.show()


# main6()

########################################################################################################################
# Testing dataloader to recode target as integers

def main7():
    trainLoader, testloader = Data_Loader(args)()

    _, (data, target) = next(iter(enumerate(trainLoader)))

    print(np.log(1 + np.exp(-9)))
    return 0


main7()

