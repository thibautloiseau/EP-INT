import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import json

# ======================================================================================================================
# ================================================ Training FC architecture ============================================
# ======================================================================================================================

def trainFC(net, trainLoader, epoch, args):
    """Train the network with FC architecture for one epoch"""
    net.train()

    criterion = nn.MSELoss(reduction='sum')
    aveFalsePred, singleFalsePred, trainLoss = 0, 0, 0
    nbChanges = [0. for k in range(len(net.layersList) - 1)]

    # Decay
    if epoch % 5 == 0 and epoch != 0:
        for k in range(len(args.tauInt)):
            args.tauInt[k] = args.tauInt[k] * args.decay

    for batch_idx, (data, targets) in enumerate(tqdm(trainLoader)):
        # We set beta's sign to be random for each batch
        if net.randomBeta == 1:
            net.beta = torch.sign(torch.randn(1)) * args.beta

        # We initialize the first layer with input data
        state = net.initHidden(data)

        # Sending tensors to GPU if available
        if net.cuda:
            targets = targets.to(net.device)
            net.beta = net.beta.to(net.device)

            for i in range(len(state)):
                state[i] = state[i].to(net.device)

        # Free phase
        state = net.forward(state)
        freeState = state.copy()

        # We calculate the loss after inference and accumulate it for the considered epoch
        lossBatch = 1 / (2 * net.trainBatchSize) * criterion(state[0], targets)
        trainLoss += lossBatch

        # Nudged phase
        state = net.forward(state, beta=net.beta, target=targets)

        # We track the number of weight changes for the considered batch
        nbChangesBatch = net.updateWeights(freeState=freeState, nudgedState=state)

        # We accumulate the number of changes
        for i in range(len(net.layersList) - 1):
            nbChanges[i] += nbChangesBatch[i]

        # We compute the error
        avePred = torch.stack([item.sum(1) for item in freeState[0].split(args.expandOutput, dim=1)], 1) / args.expandOutput
        targetsRed = torch.stack([item.sum(1) for item in targets.split(args.expandOutput, dim=1)], 1) / args.expandOutput
        aveFalsePred += (torch.argmax(targetsRed, dim=1) != torch.argmax(avePred, dim=1)).int().sum(dim=0)

        # compute error computed on the first neuron of each sub class
        singlePred = torch.stack([item[:, 0] for item in freeState[0].split(args.expandOutput, dim=1)], 1)
        singleFalsePred += (torch.argmax(targetsRed, dim=1) != torch.argmax(singlePred, dim=1)).int().sum(dim=0)

    # We compute the error for the whole epoch
    aveTrainError = aveFalsePred.float() / float(len(trainLoader.dataset)) * 100
    singleTrainError = singleFalsePred.float() / float(len(trainLoader.dataset)) * 100

    trainLoss = trainLoss / len(trainLoader.dataset)

    return nbChanges, aveTrainError.item(), singleTrainError.item(), trainLoss.item(), state


def testFC(net, testLoader, args):
    """Test the network with FC architecture after each epoch"""
    net.eval()

    criterion = nn.MSELoss(reduction='sum')
    aveFalsePred, singleFalsePred, testLoss = 0, 0, 0

    for batch_idx, (data, targets) in enumerate(tqdm(testLoader)):
        # We initialize the first layer with input data
        state = net.initHidden(data)

        # Sending tensors to GPU if available
        if net.cuda:
            targets = targets.to(net.device)

            for i in range(len(state)):
                state[i] = state[i].to(net.device)

        # Free phase
        state = net.forward(state)

        # We calculate the loss after inference and accumulate it for the considered epoch
        testLoss += 1 / (2 * net.testBatchSize) * criterion(state[0], targets)

        # We compute the error
        avePred = torch.stack([item.sum(1) for item in state[0].split(args.expandOutput, dim=1)], 1) / args.expandOutput
        targetsRed = torch.stack([item.sum(1) for item in targets.split(args.expandOutput, dim=1)], 1) / args.expandOutput
        aveFalsePred += (torch.argmax(targetsRed, dim=1) != torch.argmax(avePred, dim=1)).int().sum(dim=0)

        # Compute error computed on the first neuron of each sub class
        singlePred = torch.stack([item[:, 0] for item in state[0].split(args.expandOutput, dim=1)], 1)
        singleFalsePred += (torch.argmax(targetsRed, dim=1) != torch.argmax(singlePred, dim=1)).int().sum(dim=0)

        # We compute the error for the whole epoch
        aveTestError = aveFalsePred.float() / float(len(testLoader.dataset)) * 100
        singleTestError = singleFalsePred.float() / float(len(testLoader.dataset)) * 100

        testLoss = testLoss / len(testLoader.dataset)

    return aveTestError.item(), singleTestError.item(), testLoss.item()

# ======================================================================================================================
# ================================================ Training Conv architecture ==========================================
# ======================================================================================================================

def trainConv(net, trainLoader, epoch, args):
    """Train convolutional arch for one epoch"""
    net.train()

    criterion = nn.MSELoss(reduction='sum')
    avePred, singlePred, trainLoss = 0, 0, 0
    nbChangesFC = [0. for k in range(len(net.fcList) - 1)]
    nbChangesConv = [0. for k in range(len(net.convList) - 1)]

    for batch_idx, (data, targets) in enumerate(tqdm(trainLoader)):
        if args.randomBeta == 1:
            net.beta = torch.sign(torch.randn(1)) * args.beta

        state, indices = net.initHidden(data)

        if net.cuda:
            data, targets = data.to(net.device), targets.to(net.device)
            net.beta = net.beta.to(net.device)

            for i in range(len(state)):
                state[i] = state[i].to(net.device)

        # Free phase
        state, indices = net.forward(state, indices, data)

        freeState = state.copy()
        freeIndices = indices.copy()

        # Accumulating loss
        lossBatch = (1 / (2 * state[0].size(0))) * criterion(state[0], targets)
        trainLoss += lossBatch

        # Nudged phase
        state, indices = net.forward(state, indices, data, beta=net.beta, target=targets, pred=freeState[0])

        # Update and track changing weights of the network
        nbChangesFCBatch, nbChangesConvBatch = net.updateWeight(freeState, state, freeIndices, indices, data)

        # Accumulating the number of changes for FC and conv part of the network
        nbChangesFC = [x1 + x2 for (x1, x2) in zip(nbChangesFC, nbChangesFCBatch)]
        nbChangesConv = [x1 + x2 for (x1, x2) in zip(nbChangesConv, nbChangesConvBatch)]

        # Compute error
        # Compute averaged error over the sub-classes
        predAve = torch.stack([item.sum(1) for item in freeState[0].split(args.expandOutput, dim=1)], 1) / args.expandOutput
        targetsRed = torch.stack([item.sum(1) for item in targets.split(args.expandOutput, dim=1)], 1) / args.expandOutput
        avePred += (torch.argmax(targetsRed, dim=1) != torch.argmax(predAve, dim=1)).int().sum(dim=0)

        # Compute error computed on the first neuron of each sub class for single error
        predSingle = torch.stack([item[:, 0] for item in freeState[0].split(args.expandOutput, dim=1)], 1)
        singlePred += (torch.argmax(targetsRed, dim=1) != torch.argmax(predSingle, dim=1)).int().sum(dim=0)

    aveTrainError = (avePred.float() / float(len(trainLoader.dataset))) * 100
    singleTrainError = (singlePred.float() / float(len(trainLoader.dataset))) * 100

    trainLoss = trainLoss / len(trainLoader.dataset)

    return aveTrainError.item(), singleTrainError.item(), trainLoss.item(), nbChangesFC, nbChangesConv

def testConv(net, testLoader, args):
    """Testing network with conv arch for one epoch"""
    net.eval()

    criterion = nn.MSELoss(reduction='sum')
    avePred, singlePred, testLoss = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(testLoader)):

            state, indices = net.initHidden(data)

            if net.cuda:
                data, targets = data.to(net.device), targets.to(net.device)
                for i in range(len(state)):
                    state[i] = state[i].to(net.device)

            # Free phase
            state, indices = net.forward(state, indices, data)

            # Loss
            loss = (1 / (2 * state[0].size(0))) * criterion(state[0], targets)
            testLoss += loss

            # Compute error
            # Compute averaged error over the sub-classes
            predAve = torch.stack([item.sum(1) for item in state[0].split(args.expandOutput, dim=1)], 1) / args.expandOutput
            targetsRed = torch.stack([item.sum(1) for item in targets.split(args.expandOutput, dim=1)], 1) / args.expandOutput
            avePred += (torch.argmax(targetsRed, dim=1) != torch.argmax(predAve, dim=1)).int().sum(dim=0)

            # compute error computed on the first neuron of each sub class
            predSingle = torch.stack([item[:, 0] for item in state[0].split(args.expandOutput, dim=1)], 1)
            singlePred += (torch.argmax(targetsRed, dim=1) != torch.argmax(predSingle, dim=1)).int().sum(dim=0)

    aveTestError = (avePred.float() / float(len(testLoader.dataset))) * 100
    singleTestError = (singlePred.float() / float(len(testLoader.dataset))) * 100

    testLoss = testLoss / len(testLoader.dataset)

    return aveTestError.item(), singleTestError.item(), testLoss.item()

