from main import *
import matplotlib.pyplot as plt


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
# Defining network
# State = preAct
# nbits on which to code the states
nbits = 8

# Parameters
layersList = list(reversed([784, 512, 10]))
T = 10 # steps of free phase

# Defining the weights of the network
W = nn.ModuleList(None)
alphas = []

with torch.no_grad():
    for layer in range(len(layersList) - 1):
        W.extend([nn.Linear(layersList[layer + 1], layersList[layer], bias=False)])
        alphas.append(int((0.5 / np.sqrt(layersList[layer + 1])) * (2**nbits - 1)))

        W[-1].weight.data = (alphas[-1] * torch.sign(W[-1].weight)).int()


def initHidden(data):
    state = []
    size = data.shape[0]

    for layer in range(len(layersList)):
        state.append(torch.zeros(size, layersList[layer], requires_grad=False))

    # We initialize the first neurons with the input data
    state[-1] = torch.mul(data, 2**nbits - 1).float()

    return state


def activ(state):
    return (state >= 2**(nbits - 1)).int()


def activP(state):
    return ((state >= 0 ) & (state <= 2**nbits - 1)).int()


def getAct(state):
    binState = state.copy()

    for layer in range(len(state) - 1):
        binState[layer] = activ(state[layer])

    binState[-1] = state[-1]

    return binState


def stepper(state):
    preAct = state.copy()
    binState = getAct(state)

    # We compute the pre-activation for each layer
    preAct[0] = activP(state[0]) * W[0](binState[1])

    for layer in range(1, len(state) - 1):
        # Previous layer contribution
        preAct[layer] = activP(state[layer]) * W[layer](binState[layer + 1])
        # Next layer contribution
        preAct[layer] += activP(state[layer]) * torch.mm(binState[layer - 1], W[layer - 1].weight)
        # Updating, filtering, and clamping the pre-activations
        state[layer] = (0.5 * (state[layer] + preAct[layer])).int().clamp(0, 2**nbits - 1)

    state[0] = 0.5 * (state[0] + preAct[0]).int().clamp(0, 2**nbits - 1) # Right shift + clamp

    return state


if __name__ == '__main__':
    trainLoader = torch.utils.data.DataLoader(torch.utils.data.Subset(DataSet(args)(), [0]), batch_size=1, shuffle=False)
    _, (data, target) = (next(iter(enumerate(trainLoader))))
    state = initHidden(data)

    with torch.no_grad():
        for i in range(T):
            state = stepper(state)
