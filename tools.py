import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import json

# ======================================================================================================================
# ================================================ Training FC architecture ============================================
# ======================================================================================================================

def trainFC(net, trainLoader, args):
    """Train the network with FC architecture for one epoch"""
    net.train()

    criterion = nn.MSELoss(reduction='sum')
    aveFalsePred, singleFalsePred, trainLoss = 0, 0, 0
    nbChanges = [0. for k in range(len(net.layersList) - 1)]

    for batch_idx, (data, targets) in enumerate(tqdm(trainLoader)):
        # We set beta's sign to be random for each batch
        # if net.randomBeta == 1:
        #     net.beta = torch.sign(torch.randn(1)) * args.beta

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
    aveFalsePred, singleFalsePred,testLoss = 0, 0, 0

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
