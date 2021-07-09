import os

import numpy as np
import torch

from main import *
import matplotlib.pyplot as plt
import time

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

        # See evolution of states
        for k in range(len(freeState)):
            plt.plot(save[k])
            plt.title('layer ' + str(k))
            plt.show()

        # nudgedState = state.copy()
        #
        # gradW, gradBias = net.computeGradients(freeState, nudgedState)
        #
        # for k in range(len(gradW)):
        #     print('Weights layer ' + str(k))
        #     gradw = gradW[k].tolist()
        #     plt.plot(gradw, '+')
        #     plt.title('Weights layer ' + str(k))
        #     plt.show()


        # for k in range(len(gradBias)):
        #     print('Biases layer ' + str(k))
        #     gradb = gradBias[k].tolist()
        #     plt.plot(gradb, '+')
        #     plt.title('Biases layer ' + str(k))
        #     plt.show()


# main6()

########################################################################################################################
# See end values of network weights and biases

def main7():
    trainLoader, testloader = Data_Loader(args)()

    for dir, subdir, files in os.walk(os.getcwd()):
        for file in files:
            cpath = os.path.join(dir, file)
            if '.pt' in cpath and 'S7' in cpath and '24' in cpath:
                model = (torch.load(cpath))

    print(model['modelStateDict'])

    return 0

# main7()

########################################################################################################################
# Number of bits to have a granularity of 1e-7

def main8():
    print(6*np.log(10)/np.log(2))
    # print(2**15)
    return 0

# main8()

########################################################################################################################
# Testing code copy

def main9():
    net = FCbinWAInt(args)

    # Create visualizer for tensorboard and save training
    visualizer = Visualizer(net, args)
    visualizer.saveHyperParameters()

# main9()

########################################################################################################################
# Testing values of accumulated gradients

def main10():
    args.layersList.reverse()

    trainLoader, _ = Data_Loader(args)()

    net = FCbinWAInt(args)

    if net.cuda:
        net.to(net.device)

    for i in range(2):
        for batch, (data, target) in enumerate(tqdm(trainLoader)):

            state = net.initHidden(data)

            if net.cuda:
                targets = target.to(net.device)
                net.beta = net.beta.to(net.device)

                for j in range(len(state)):
                    state[j] = state[j].to(net.device)

            state = net.forward(state)
            freeState = state.copy()

            nudgedState = net.forward(state, beta=net.beta, target=targets)

            gradW, _ = net.computeGradients(freeState, nudgedState)
            _ = net.updateWeights(freeState=freeState, nudgedState=state)


    for k in range(len(gradW)):
        print('Weights layer ' + str(k))
        gradw = gradW[k].tolist()
        plt.plot(gradw, '+')
        plt.title('Weights layer ' + str(k))
        plt.show()

# main10()

########################################################################################################################
# Number of bits for states with binary weights only and no scaling factors

def main11():
    print(np.log((8192*2 + 2) / 3) / np.log(2) + 1)

# main11()

########################################################################################################################
# Testing conv arch

def main12():
    args.layersList.reverse()
    args.convList.reverse()

    for l in range(10):

        trainLoader, _ = Data_Loader(args)()
        net = ConvWAInt(args)

        if net.cuda:
            net.to(net.device)

        print("Running on " + net.deviceName)

        batch_idx, (data, target) = next(iter(enumerate(trainLoader)))

        state, indices = net.initHidden(data)

        if net.cuda:
            data, target = data.to(net.device), target.to(net.device)
            net.beta = net.beta.to(net.device)

            for i in range(len(state)):
                state[i] = state[i].to(net.device)

        ##############################
        T = 50
        Kmax = 50

        save = [[] for k in range(len(state))]

        with torch.no_grad():
            # Free phase
            for t in range(T):
                state, indices = net.stepper(state, indices, data)

                save[0].append(state[0][32][350].item())
                save[1].append(state[1][32][0][0][0].item())
                save[2].append(state[2][32][0][0][0].item())

            freeState = state.copy()

            # Nudged phase
            for k in range(Kmax):
                state, indices = net.stepper(state, indices, data, target=target, beta=net.beta, pred=freeState[0])

                save[0].append(state[0][32][350].item())
                save[1].append(state[1][32][0][0][0].item())
                save[2].append(state[2][32][0][0][0].item())

        for layer in range(len(save)):
            plt.plot(save[layer])

        plt.show()

# main12()

########################################################################################################################
# Init of weights on FC arch

def main13():
    args.layersList = [100, 8192, 784]

    net = FCbinWAInt(args)

    print(net.W[0].weight.data)

    net.W[0].weight.data[:, ::2] = torch.ones(size=net.W[0].weight.data[:, ::2].shape)
    net.W[0].weight.data[:, 1::2] = -torch.ones(size=net.W[0].weight.data[:, 1::2].shape)

    print(net.W[0].weight.data)
    print(net.W[0].weight.data.shape)

    return

# main13()

########################################################################################################################
# Testing "analytical" solution for relaxation

def main14():
    args.layersList = [100, 8192, 784]
    args.archi = 'fc'

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
            save[-1].append(state[k][0][50].item())

        #####
        # Free phase
        T = 16

        analyticalState = net.analytical(state)
        print(analyticalState)

        for step in range(T):
            state = net.stepper(state)

            for k in range(len(state)):
                save[k].append(state[k][0][50].item())

        print(state)

        freeState = state.copy()

        # # See evolution of free state
        # for k in range(len(freeState)):
        #     plt.plot(save[k])
        #     plt.title('layer ' + str(k))
        #     plt.show()

    return

# main14()

########################################################################################################################
# Generation of data with trained model by going backward

def main15():
    model = torch.load("./SAVE-fc-MNIST/2021-06-30/S7/checkpoint.pt")['modelStateDict']
    net = FCbinWAInt(args)

    print(model)

    def stepper(state):
        """Evolution of the state during free phase or nudged phase"""
        preAct = state.copy()
        binState = net.getBinState(state)
        binStateP = net.getBinStateP(state)

        # We compute the pre-activation for each layer
        preAct[0] = binStateP[0] * model['W.0.weight'](binState[1])

        for layer in range(1, len(state)):
            # Previous layer contribution
            preAct[layer] = binStateP[layer] * model.W[layer](binState[layer + 1])
            # Next layer contribution
            preAct[layer] += binStateP[layer] * torch.mm(binState[layer - 1], model.W[layer - 1].weight)
            # Updating, filtering, and clamping the pre-activations
            state[layer] = (0.5 * (state[layer] + preAct[layer])).int().float().clamp(0, net.maxIntState)

        state[0] = (0.5 * (state[0] + preAct[0])).int().float().clamp(0, net.maxIntState)

        return state

    state = []

    # Init state
    for layer in range(len(net.layersList)):
        state.append(torch.zeros(net.trainBatchSize, net.layersList[layer], requires_grad=False))

    T = 16

    for i in range(T):
        state = stepper(state)

    print(state[-1])

    return

# main15()

########################################################################################################################
# Testing for accumulated gradients

def main16():
    # a = torch.randn(size=(10, 10))
    # print(a)
    # print(a[torch.where(torch.abs(a) > 1)])
    print(4*784)

    return

# main16()

########################################################################################################################
# See weights

def main17():
    model = torch.load("SAVE-fc-MNIST/2021-07-09/S1/checkpoint.pt")
    print(model['modelStateDict']['W.1.weight'])
    dropout = torch.nn.Dropout(p=0.5)
    print(dropout(model['modelStateDict']['W.0.weight']))

    return

main17()