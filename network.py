import torch

from tools import *

# ======================================================================================================================
# ==================================== FC architecture - Binarized weights and synapses ================================
# ======================================================================================================================

class FCbinWAInt(nn.Module):
    """Network for fully connected architecture with binary weights and activations"""

    def __init__(self, args):
        super(FCbinWAInt, self).__init__()

        # Checking device
        if torch.cuda.is_available():
            self.cuda = True
            self.device = args.device
            self.deviceName = torch.cuda.get_device_name()

        else:
            self.cuda = False
            self.device = torch.device("cpu")

        self.layersList = args.layersList

        # Parameters for EP
        self.T = args.T
        self.Kmax = args.Kmax
        self.beta = torch.tensor(2**args.beta)
        self.randomBeta = args.randomBeta
        self.constNudge = args.constNudge

        self.hasBias = args.hasBias

        # Batch sizes
        self.trainBatchSize = args.trainBatchSize
        self.testBatchSize = args.testBatchSize

        # Scaling factors
        self.alphasInt = []

        # Int coding
        # For states
        self.bitsState = args.bitsState
        self.maxIntState = 2**(self.bitsState - 1) - 1

        # Parameters of BOP
        self.tauInt = args.tauInt

        # Initialize the accumulated gradients for the batch, clamping and reinitializing gradients
        self.accGradientsInt = []
        self.clampMom = args.clampMom
        self.reinitGrad = args.reinitGrad

        # Initialize the network according to the layersList given with XNOR-net method
        self.W = nn.ModuleList(None)

        with torch.no_grad():
            for i in range(len(self.layersList) - 1):
                if self.hasBias:
                    self.W.extend([nn.Linear(self.layersList[i+1], self.layersList[i], bias=True)])
                    self.W[-1].bias.data = torch.zeros(size=self.W[-1].bias.data.shape)
                    # self.W[-1].bias.data = int(1 / (2 * np.sqrt(self.layersList[i+1])) * self.maxIntState) * torch.sign(self.W[-1].bias)

                else:
                    self.W.extend([nn.Linear(self.layersList[i+1], self.layersList[i], bias=False)])

                alphaInt = int(1 / (2 * np.sqrt(self.layersList[i+1])) * self.maxIntState)
                self.alphasInt.append(alphaInt)
                self.W[-1].weight.data = alphaInt * torch.sign(self.W[-1].weight)


    def initHidden(self, data):
        """Initialize the neurons"""
        state = []
        size = data.shape[0]

        for layer in range(len(self.layersList)):
            state.append(self.maxIntState / 2 * torch.ones(size, self.layersList[layer], requires_grad=False))

        # We initialize the input neurons with stochastic binary input data
        for i in range(8):
            randV = torch.rand(size=data.shape)
            stoch = (randV < data).int()
            # first_layer = self.W[-2](stoch).to(self.device)
            # print(first_layer)

        # We initialize the inputs with the last stochastic example, as the dynamics depends on the input
        state[-1] = data.float()

        return state

    def activ(self, x):
        """Activation function for other layers (integers)"""
        return (x >= self.maxIntState / 2).int().float()
        # return torch.sign(x - self.maxIntState / 2).int().float()

    def activP(self, x):
        """Derivative of activation function for other layers (integer)"""
        return ((x >= 0) & (x <= self.maxIntState)).int().float()
        # return 2 * (((x >= 0) & (x <= self.maxIntState)).int() - 0.5).int().float()

    def getBinState(self, state):
        """Get the binary activation of the pre-activations"""
        binState = state.copy()

        for layer in range(len(state) - 1):
            binState[layer] = self.activ(state[layer])

        return binState

    def getBinStateP(self, state):
        """Get the derivative binary activations of the pre-activations"""
        binStateP = state.copy()

        for layer in range(len(state) - 1):
            binStateP[layer] = self.activP(state[layer])

        return binStateP

    def stepper(self, state, target=None, beta=0, nudge=None):
        """Evolution of the state during free phase or nudged phase"""
        preAct = state.copy()
        binState = self.getBinState(state)
        binStateP = self.getBinStateP(state)

        # We compute the pre-activation for each layer
        preAct[0] = binStateP[0] * self.W[0](binState[1])

        if beta != 0:
            if self.constNudge:
                preAct[0] += beta * (target - nudge[0])
            else:
                preAct[0] += beta * (target - state[0])

        for layer in range(1, len(state) - 1):
            # Previous layer contribution
            preAct[layer] = binStateP[layer] * self.W[layer](binState[layer + 1])
            # Next layer contribution
            preAct[layer] += binStateP[layer] * torch.mm(binState[layer - 1], self.W[layer - 1].weight)
            # Updating, filtering, and clamping the pre-activations
            state[layer] = (0.5 * (state[layer] + preAct[layer])).int().float().clamp(0, self.maxIntState)

        state[0] = (0.5 * (state[0] + preAct[0])).int().float().clamp(0, self.maxIntState)

        return state

    def forward(self, state, beta=0, target=None):
        """Two state evolution for free and nudged phase"""
        with torch.no_grad():
            # Free phase
            if beta == 0:
                for i in range(self.T):
                    state = self.stepper(state)

            # Nudged phase
            else:
                nudge = state.copy()
                for i in range(self.Kmax):
                    state = self.stepper(state, target=target, beta=self.beta, nudge=nudge)

        return state

    def computeGradients(self, freeState, nudgedState):
        """Compute the gradients for the considered batch, according to all learnable parameters"""
        gradWInt, gradBias = [], []

        with torch.no_grad():
            # We get the binary states of each neuron
            freeBinState, nudgedBinState = self.getBinState(freeState), self.getBinState(nudgedState)

            for layer in range(len(freeState) - 1):
                gradWInt.append(
                                (torch.mm(torch.transpose(nudgedBinState[layer], 0, 1), nudgedBinState[layer + 1]) -
                                 torch.mm(torch.transpose(freeBinState[layer], 0, 1), freeBinState[layer + 1])).clamp(-8, 7)
                                )

                if self.hasBias:
                    gradBias.append((nudgedBinState[layer] - freeBinState[layer]).sum(0).clamp(-8, 7))

            # We initialize the accumulated gradients for first iteration or BOP
            if self.accGradientsInt == []:
                self.accGradientsInt = gradWInt

            else:
                self.accGradientsInt = [(g.int() + m.int()).clamp(-self.clampMom[i], self.clampMom[i]) for i, (g, m) in enumerate(zip(gradWInt, self.accGradientsInt))]

            gradWInt = self.accGradientsInt

        return gradWInt, gradBias

    def updateWeights(self, freeState, nudgedState):
        """Updating parameters after having computed the gradient. We return the number of weight changes in the process"""
        gradWInt, gradBias = self.computeGradients(freeState, nudgedState)

        noChanges = []

        with torch.no_grad():
            for layer in range(len(freeState) - 1):
                # Weights updates
                tauTensorInt = self.tauInt[layer] * torch.ones(self.W[layer].weight.shape).to(self.device)
                modifyWeightsInt = -1 * torch.sign((-1 * torch.sign(self.W[layer].weight) * gradWInt[layer] > tauTensorInt).int() - 0.5)
                noChanges.append(torch.sum(abs(modifyWeightsInt - 1)).item() / 2)

                self.W[layer].weight.data = torch.mul(self.W[layer].weight.data, modifyWeightsInt)

                # We reinitialize the accumulated gradients to 0 if the weight has been flipped
                if self.reinitGrad:
                    modifyGradInt = ((modifyWeightsInt + 1) / 2).int()
                    self.accGradientsInt[layer] = torch.mul(self.accGradientsInt[layer], modifyGradInt)

                # Biases updates
                if self.hasBias:
                    self.W[layer].bias += gradBias[layer]

        return noChanges

# ======================================================================================================================
# ==================================== Conv architecture - Binarized weights and synapses ==============================
# ======================================================================================================================

class ConvWAInt(nn.Module):
    def __init__(self, args):
        super(ConvWAInt, self).__init__()

        # Checking device
        if torch.cuda.is_available():
            self.cuda = True
            self.device = args.device
            self.deviceName = torch.cuda.get_device_name()

        else:
            self.cuda = False
            self.device = torch.device("cpu")

        # Dynamics parameters
        self.T = args.T
        self.Kmax = args.Kmax
        self.beta = torch.tensor(args.beta)
        self.trainBatchSize = args.trainBatchSize
        self.neuronMin, self.neuronMax = 0, 1

        # Conv layers parameters
        self.convList = args.convList
        self.FPool = args.FPool
        self.conv = nn.ModuleList(None)
        self.convAccGradients = []
        self.kernel = args.kernel
        self.padding = args.padding
        self.convTau = args.convTau
        self.nbConv = len(self.convList) - 1

        # FC layers parameters
        self.fcList = args.layersList
        self.fc = nn.ModuleList(None)
        self.fcAccGradients = []
        self.fcTau = args.fcTau
        self.nbFC = len(self.fcList)

        self.hasBias = args.hasBias

        if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
            inputSize = 28
        elif args.dataset == 'CIFAR10':
            inputSize = 32

        self.sizeConvPoolTab = [inputSize]
        self.sizeConvTab = [inputSize]

        with torch.no_grad():
            # For conv part
            for i in range(self.nbConv):
                if self.hasBias:
                    self.conv.append(nn.Conv2d(self.convList[i+1], self.convList[i], self.kernel, padding=self.padding, bias=True))
                    torch.nn.init.zeros_(self.conv[-1].bias)

                else:
                    self.conv.append(nn.Conv2d(self.convList[i+1], self.convList[i], self.kernel, padding=self.padding, bias=False))

                for k in range(args.convList[i]):
                    alpha = round((torch.norm(self.conv[-1].weight[k], p=1) / self.conv[-1].weight[k].numel()).item(), 4)
                    self.conv[-1].weight[k] = alpha * torch.sign(self.conv[-1].weight[k])

                self.sizeConvTab.append(int((self.sizeConvPoolTab[i] + 2*self.padding - self.kernel) + 1))
                self.sizeConvPoolTab.append(int((self.sizeConvTab[-1] - self.FPool) / self.FPool + 1))

        self.pool = nn.MaxPool2d(self.FPool, stride=self.FPool, return_indices=True)
        self.unpool = nn.MaxUnpool2d(self.FPool, stride=self.FPool)

        self.sizeConvPoolTab.reverse()
        self.sizeConvTab.reverse()

        self.fcList.append(self.convList[0] * self.sizeConvPoolTab[0]**2)

        with torch.no_grad():
            # For FC part
            for i in range(self.nbFC):
                if self.hasBias:
                    self.fc.append(nn.Linear(self.fcList[i + 1], self.fcList[i], bias=True))
                    torch.nn.init.zeros_(self.fc[-1].bias)

                else:
                    self.fc.append(nn.Linear(self.fcList[i + 1], self.fcList[i], bias=False))

                alpha = round((torch.norm(self.fc[-1].weight, p=1) / self.fc[-1].weight.numel()).item(), 4)
                self.fc[-1].weight.data = alpha * torch.sign(self.fc[-1].weight)

    def initHidden(self, data):
        """Initialize the neurons"""
        state, indices = [], []
        size = data.shape[0]

        for layer in range(self.nbFC):
            state.append(torch.ones(size, self.fcList[layer], requires_grad=False))
            indices.append(None)

        for layer in range(self.nbConv):
            state.append(torch.ones(size, self.convList[layer], self.sizeConvPoolTab[layer], self.sizeConvPoolTab[layer], requires_grad=False))
            indices.append(None)

        return state, indices

    def activ(self, x):
        """Activation function"""
        return (x >= 0.5).int().float()

    def activP(self, x):
        """Derivative of the activation function"""
        return ((x >= 0) & (x <= 1)).int().float()

    def getBinState(self, state):
        """Get the binary state of pre-activations"""
        binState = state.copy()

        for layer in range(len(state) - 1):
            binState[layer] = self.activ(state[layer])

        return binState

    def stepper(self, state, indices, data, target=None, beta=0):
        """Relaxation of the state during free and nudge phases"""
        preAct = state.copy()
        binState = self.getBinState(state)

        #####
        # Last fc layer (output of the network)
        preAct[0] = self.fc[0](binState[1].view(binState[1].shape[0], -1))
        preAct[0] *= self.activP(state[0])

        if beta != 0:
            preAct[0] += beta * (target - state[0])

        #####
        # Middle fc layer
        for layer in range(1, self.nbFC):
            preAct[layer] = self.fc[layer](binState[layer+1].view(binState[layer+1].shape[0], -1))
            preAct[layer] += torch.mm(binState[layer-1], self.fc[layer-1].weight)
            preAct[layer] *= self.activP(state[layer])

        #####
        # Convolutional layers
        # Last conv layer
        pooledState, indice = self.pool(self.conv[0](binState[self.nbFC + 1]))
        indices[self.nbFC] = indice
        preAct[self.nbFC] = pooledState + torch.mm(binState[self.nbFC - 1], self.fc[-1].weight).view(state[self.nbFC].shape)
        preAct[self.nbFC] *= self.activP(state[self.nbFC])

        del pooledState, indice

        #####
        # Middle conv layers
        for layer in range(1, self.nbConv - 1):
            pooledState, indice = self.pool(self.conv[layer](binState[self.nbFC + layer + 1]))
            indices[self.nbFC + layer] = indice

            if indices[self.nbFC + layer - 1] is not None:
                outputSize = [state[self.nbFC + layer - 1].size(0), state[self.nbFC + layer - 1].size(1), self.sizeConvTab[layer - 1], self.sizeConvTab[layer - 1]]

                unpoolState = nn.functional.conv_transpose2d(
                    self.unpool(binState[self.nbFC + layer - 1], indices[self.nbFC + layer - 1], output_size=outputSize),
                    weight=self.conv[layer - 1].weight,
                    padding=self.padding)

            preAct[self.nbFC + layer] = pooledState
            preAct[self.nbFC + layer] += unpoolState
            preAct[self.nbFC + layer] *= self.activP(state[self.nbFC + layer])

            del pooledState, unpoolState, indice, outputSize

        #####
        # First conv layer
        pooledState, indice = self.pool(self.conv[-1](data))
        indices[-1] = indice
        if indices[-2] is not None:
            outputSize = [state[-2].size(0), state[-2].size(1), self.sizeConvTab[-3], self.sizeConvTab[-3]]
            unpoolState = nn.functional.conv_transpose2d(
                self.unpool(binState[-2], indices[-2], output_size=outputSize),
                weight=self.conv[-2].weight,
                padding=self.padding)

        preAct[-1] = self.activP(state[-1]) * (pooledState + unpoolState)

        del pooledState, unpoolState, indice, outputSize

        # Filtering and clamping all states
        for layer in range(len(state)):
            state[layer] = 0.5 * (state[layer] + preAct[layer])
            state[layer] = state[layer].clamp(self.neuronMin, self.neuronMax)

        del preAct, binState

        return state, indices

    def forward(self, state, indices, data, beta=0, target=None):
        """Two state evolution for free and nudged phase"""
        with torch.no_grad():
            # Free phase
            if beta == 0:
                for t in range(self.T):
                    state, indices = self.stepper(state, indices, data)

            # Nudged phase
            else:
                for k in range(self.Kmax):
                    state, indices = self.stepper(state, indices, data, target=target, beta=beta)

        return state, indices
