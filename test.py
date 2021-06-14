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
x = np.linspace(-1, 1, 2000)
n = 6
y = np.clip(np.floor(2**(n-1) * x), -2**(n-1), 2**(n-1) - 1)
print(y)

plt.plot(x, y)
plt.show()



