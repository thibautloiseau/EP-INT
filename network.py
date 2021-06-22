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
        # Right shift for the learning rates of biases
        self.lrBias = [1 / 2**args.lrBias[i] for i in range(len(args.lrBias))]

        # Batch sizes
        self.trainBatchSize = args.trainBatchSize
        self.testBatchSize = args.testBatchSize

        # Scaling factors
        self.alphas = []
        self.learnAlpha = args.learnAlpha
        self.lrAlpha = args.lrAlpha
        self.alphasInt = []

        # Int coding
        self.nbBits = args.nbBits
        self.minInt = -2**(self.nbBits - 1)
        self.maxInt = 2**(self.nbBits - 1) - 1

        # Parameters of BOP
        # Right shift for gammaInt, contribution of gradient and momentum in BOP algorithm
        self.gammaInt = [1 / 2**args.gammaInt[i] for i in range(len(args.gammaInt))]
        self.tauInt = args.tauInt

        # Initialize the accumulated gradients for the batch and the scaling factors
        self.accGradientsInt = []

        # Initialize the network according to the layersList given with XNOR-net method
        self.W = nn.ModuleList(None)

        with torch.no_grad():
            for i in range(len(self.layersList) - 1):
                if self.hasBias:
                    self.W.extend([nn.Linear(self.layersList[i+1], self.layersList[i], bias=True)])
                    self.W[-1].bias.data = torch.zeros(size=self.W[-1].bias.data.shape)

                else:
                    self.W.extend([nn.Linear(self.layersList[i+1], self.layersList[i], bias=False)])

                alphaInt = int(1 / (2 * np.sqrt(self.layersList[i+1])) * self.maxInt)
                self.alphasInt.append(alphaInt)
                self.W[-1].weight.data = alphaInt * torch.sign(self.W[-1].weight)


    def initHidden(self, data):
        """Initialize the neurons"""
        state = []
        size = data.shape[0]

        for layer in range(len(self.layersList)):
            state.append(torch.zeros(size, self.layersList[layer], requires_grad=False))

        # We initialize the input neurons with stochastic binary input data
        # Only one stochastic example at first to see if it works well. Shouldn't be complicated with MNIST (quasi-binarized already)
        randV = torch.rand(size=data.shape)
        stoch = (randV < data).int()

        state[-1] = stoch.float()

        return state

    def activ(self, x):
        """Activation function for other layers (integers)"""
        return (x >= self.maxInt / 2).int().float()

    def activP(self, x):
        """Derivative of activation function for other layers (integer)"""
        return ((x >= 0) & (x <= self.maxInt)).int().float()

    def getBinState(self, state):
        """Get the binary activation of the pre-activations"""
        binState = state.copy()

        for layer in range(len(state) - 1):
            # For the first layer (full precision)
            binState[layer] = self.activ(state[layer])

        # Setting the inputs (full precision) that are not activated (already the case a priori...)
        binState[-1] = state[-1]

        return binState

    def getBinStateP(self, state):
        """Get the derivative binary activations of the pre-activations"""
        binStateP = state.copy()

        for layer in range(len(state) - 1):
            binStateP[layer] = self.activP(state[layer])

        # Setting the inputs (no use...)
        binStateP[-1] = state[-1]

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
            state[layer] = (0.5 * (state[layer] + preAct[layer])).int().float().clamp(0, self.maxInt)

        state[0] = (0.5 * (state[0] + preAct[0])).int().float().clamp(0, self.maxInt)

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
        coefInt = int(1 / (self.beta * self.trainBatchSize) * self.maxInt)
        gradWInt, gradBias, gradAlpha = [], [], []

        with torch.no_grad():
            # We get the binary states of each neuron
            freeBinState, nudgedBinState = self.getBinState(freeState), self.getBinState(nudgedState)

            for layer in range(len(freeState) - 1):
                gradWInt.append(coefInt * (
                        torch.mm(torch.transpose(nudgedBinState[layer], 0, 1), nudgedBinState[layer + 1]) -
                        torch.mm(torch.transpose(freeBinState[layer], 0, 1), freeBinState[layer + 1])
                ))

                if self.hasBias:
                    gradBias.append(coefInt * (nudgedBinState[layer] - freeBinState[layer]).sum(0))

                # if self.learnAlpha:
                #     gradAlpha.append(0.5 * coef * (
                #         torch.diag(torch.mm(nudgedBinState[layer], torch.mm(self.W[layer].weight, nudgedBinState[layer + 1].T))).sum() -
                #         torch.diag(torch.mm(freeBinState[layer], torch.mm(self.W[layer].weight, freeBinState[layer + 1].T))).sum()
                #     ))

            # We initialize the accumulated gradients for first iteration or BOP after first iteration
            if self.accGradientsInt == []:
                self.accGradientsInt = [(self.gammaInt[i] * w).int() for (i, w) in enumerate(gradWInt)]

            else:
                self.accGradientsInt = [torch.add((self.gammaInt[i] * g).int(), ((1 - self.gammaInt[i]) * m).int()) for i, (m, g) in enumerate(zip(self.accGradientsInt, gradWInt))]

            gradWInt = self.accGradientsInt

        return gradWInt, gradBias, gradAlpha

    def updateWeights(self, freeState, nudgedState):
        """Updating parameters after having computed the gradient. We return the number of weight changes in the process"""
        gradWInt, gradBias, gradAlpha = self.computeGradients(freeState, nudgedState)

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
                    self.W[layer].bias += self.lrBias[layer] * gradBias[layer]

                # # Scaling factors updates and weight multiplications
                # if self.learnAlpha:
                #     self.alphas[layer] += self.lrAlpha[layer] * gradAlpha[layer]
                #     self.W[layer].weight.data = self.alphas[layer] * torch.sign(self.W[layer].weight)

        return noChanges
