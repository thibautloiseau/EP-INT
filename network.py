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

        # Initialize the accumulated gradients for the batch + clamping
        self.accGradientsInt = []
        self.clampMom = args.clampMom

        # Initialize the network according to the layersList given with XNOR-net method
        self.W = nn.ModuleList(None)

        with torch.no_grad():
            for i in range(len(self.layersList) - 1):
                if self.hasBias:
                    self.W.extend([nn.Linear(self.layersList[i+1], self.layersList[i], bias=True)])
                    # self.W[-1].bias.data = torch.zeros(size=self.W[-1].bias.data.shape)
                    self.W[-1].bias.data = int(1 / (2 * np.sqrt(self.layersList[i+1])) * self.maxIntState) * torch.sign(self.W[-1].bias)

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
            state.append(torch.zeros(size, self.layersList[layer], requires_grad=False))

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

    def activP(self, x):
        """Derivative of activation function for other layers (integer)"""
        return ((x >= 0) & (x <= self.maxIntState)).int().float()

    def getBinState(self, state):
        """Get the binary activation of the pre-activations"""
        binState = state.copy()

        for layer in range(len(state) - 1):
            # For the first layer (full precision)
            binState[layer] = self.activ(state[layer])

        return binState

    def getBinStateP(self, state):
        """Get the derivative binary activations of the pre-activations"""
        binStateP = state.copy()

        for layer in range(len(state) - 1):
            binStateP[layer] = self.activP(state[layer])

        return binStateP

    def stepper(self, state, target=None, beta=0):
        """Evolution of the state during free phase or nudged phase"""
        preAct = state.copy()
        binState = self.getBinState(state)
        binStateP = self.getBinStateP(state)

        # We compute the pre-activation for each layer
        preAct[0] = binStateP[0] * self.W[0](binState[1])

        if beta != 0:
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
        """Does the two state evolution for free and nudged phase"""
        with torch.no_grad():
            # Free phase
            if beta == 0:
                for i in range(self.T):
                    state = self.stepper(state)

            # Nudged phase
            else:
                for i in range(self.Kmax):
                    state = self.stepper(state, target=target, beta=self.beta)

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
                self.accGradientsInt = [(g.int() + m.int()).clamp(-self.clampMom, self.clampMom - 1) for (g, m) in zip(gradWInt, self.accGradientsInt)]

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

                # Biases updates
                if self.hasBias:
                    self.W[layer].bias += gradBias[layer]

        return noChanges