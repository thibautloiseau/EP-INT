from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch


class ReshapeTransform:
    """Transform the input in the appropriate way"""
    def __init__(self, newSize):
        self.newSize = newSize

    def __call__(self, img):
        return torch.reshape(img, self.newSize)


class ReshapeTransformTarget:
    """Transform target output in the appropriate way"""
    def __init__(self, noClasses, expandOutput, maxIntState):
        self.noClasses = noClasses
        self.expandOutput = expandOutput
        self.maxIntState = maxIntState

    def __call__(self, target):
        target = torch.tensor(target).unsqueeze(0).unsqueeze(1)
        targetOneHot = torch.zeros((1, self.noClasses))

        return targetOneHot.scatter_(1, target, 1).repeat_interleave(self.expandOutput).squeeze(0) * self.maxIntState


# For fc architecture
class Data_Loader(DataLoader):
    """Create the data_loader for training and for testing"""
    def __init__(self, args):
        self.trainBatchSize = args.trainBatchSize
        self.testBatchSize = args.testBatchSize
        self.fcTransforms = [transforms.ToTensor(), ReshapeTransform((-1,))]
        self.expandOutput = args.expandOutput
        self.dataset = args.dataset
        self.maxIntState = 2**(args.bitsState - 1) - 1

    def __call__(self):
        """We return a tuple with both dataloader for train and test"""
        if self.dataset == 'MNIST':
            return (
                DataLoader(
                    datasets.MNIST(
                        root='./data', train=True, download=True,
                        transform=transforms.Compose(self.fcTransforms),
                        target_transform=ReshapeTransformTarget(10, self.expandOutput, self.maxIntState)

                    ),
                    shuffle=True, batch_size=self.trainBatchSize, num_workers=2, pin_memory=True
                ),

                DataLoader(
                    datasets.MNIST(
                        root='./data', train=False, download=True,
                        transform=transforms.Compose(self.fcTransforms),
                        target_transform=ReshapeTransformTarget(10, self.expandOutput, self.maxIntState)

                    ),
                    shuffle=True, batch_size=self.testBatchSize, num_workers=2, pin_memory=True
                )
            )
        if self.dataset == 'FashionMNIST':
            return (
                DataLoader(
                    datasets.FashionMNIST(
                        root='./data', train=True, download=True,
                        transform=transforms.Compose(self.fcTransforms),
                        target_transform=ReshapeTransformTarget(10, self.expandOutput, self.maxIntState)

                    ),
                    shuffle=True, batch_size=self.trainBatchSize, num_workers=2, pin_memory=True
                ),

                DataLoader(
                    datasets.FashionMNIST(
                        root='./data', train=False, download=True,
                        transform=transforms.Compose(self.fcTransforms),
                        target_transform=ReshapeTransformTarget(10, self.expandOutput, self.maxIntState)

                    ),
                    shuffle=True, batch_size=self.testBatchSize, num_workers=2, pin_memory=True
                )
            )

