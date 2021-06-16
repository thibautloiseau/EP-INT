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

    nbits = np.arange(1., 16.)

    def mini(Nin=Nin, Nout=Nout, nbits=nbits):
        return ((-3 * 2**(nbits-1) + 1) / (Nin + Nout)).astype(int)

    def maxi(Nin=Nin, Nout=Nout, nbits=nbits):
        return ((3 * 2**(nbits-1) - 2) / (Nin + Nout)).astype(int)

    ymin = mini()
    ymax = maxi()

    plt.plot(nbits, ymin)
    plt.plot(nbits, ymax)
    plt.show()

# main1()

########################################################################################################################
# Test on stepper
# Creating network
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
        saveO = [state[0][0][5].item()]

        # For hidden
        saveH = [state[1][0][100].item()]

        # Free phase
        T = 10

        for step in range(T):
            state = net.stepper(state)

            saveO.append(state[0][0][5].item())
            saveH.append(state[1][0][100].item())

        freeState = state.copy()

        # plt.plot(range(T + 1), saveO)
        # plt.plot(range(T + 1), saveH, 'b')
        # plt.show()

        # Nudged phase
        Kmax = 12
        beta = 2

        for step in range(Kmax):
            state = net.stepper(state, target=targets, beta=beta)
            saveO.append(state[0][0][5].item())
            saveH.append(state[1][0][100].item())

        nudgedState = state.copy()

        plt.plot(range(T + 1 + Kmax), saveO)
        plt.plot(range(T + 1 + Kmax), saveH)
        plt.show()

        # Compute gradients
        coef = int(((1 / (beta * data.shape[0])) * net.maxInt))
        gradW = []

        with torch.no_grad():
            # We get the binary states of each neuron
            freeBinState, nudgedBinState = net.getBinState(freeState), net.getBinState(nudgedState)

            for layer in range(len(freeState) - 1):
                gradW.append(coef * (
                        torch.mm(torch.transpose(nudgedBinState[layer], 0, 1), nudgedBinState[layer + 1]) -
                        torch.mm(torch.transpose(freeBinState[layer], 0, 1), freeBinState[layer + 1])
                ))

            # We initialize the accumulated gradients for first iteration
            if net.accGradients == []:
                net.accGradients = [net.gamma[i] * w for (i, w) in enumerate(gradW)]

            # We compute momentum with BOP algorithm if we already have gradients
            else:
                net.accGradients = [torch.add(net.gamma[i] * g, (1 - net.gamma[i]) * m) for i, (m, g) in enumerate(zip(net.accGradients, gradW))]

            gradW = net.accGradients

        # Updating weights
        with torch.no_grad():
            for layer in range(len(freeState) - 1):
                tauTensor = net.tau[layer] * torch.ones(net.W[layer].weight.shape).to(net.device)
                modifyWeights = -1 * torch.sign( (-1 * torch.sign(net.W[layer].weight) * gradW[layer] > tauTensor).int() - 0.5)
                net.W[layer].weight.data = torch.mul(net.W[layer].weight.data, modifyWeights)

main2()

########################################################################################################################
# Testing for input layer

def main3():
    n = np.arange(2, 25)
    Nin = 784
    Nout = 10

    def mini1(n=n, Nin=Nin, Nout=Nout):
        return ((2**n + 2**(n-1) - 1 - Nout * ((3 * 2**(n-1) - 2) / (Nin + Nout)).astype(int)) / (Nin * 2**(n-1))).astype(int)

    def mini2(n=n, Nin=Nin, Nout=Nout):

        return ((2**n + 2**(n-1) - 2 - Nout * ((3 * 2**(n-1) - 2) / (Nin + Nout)).astype(int)) / (Nin * (2**(n-1) - 1))).astype(int)


    y1 = mini1()
    y2 = mini2()
    # plt.plot(n, y1)
    plt.plot(n, y2)
    plt.show()

# main3()
