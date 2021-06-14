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
    def __init__(self, noClasses, expandOutput):
        self.noClasses = noClasses
        self.expandOutput = expandOutput

    def __call__(self, target):
        target = torch.tensor(target).unsqueeze(0).unsqueeze(1)
        targetOneHot = torch.zeros((1, self.noClasses))

        return targetOneHot.scatter_(1, target, 1).repeat_interleave(self.expandOutput).squeeze(0)


# For fc architecture
class Data_Loader(DataLoader):
    """Create the data_loader for training and for testing"""
    def __init__(self, args):
        self.trainBatchSize = args.trainBatchSize
        self.testBatchSize = args.testBatchSize
        self.fcTransforms = [transforms.ToTensor(), ReshapeTransform((-1,))]
        self.expandOutput = args.expandOutput

    def __call__(self):
        """We return a tuple with both dataloader for train and test"""
        return (
            DataLoader(
                datasets.MNIST(
                    root='./data', train=True, download=True,
                    transform=transforms.Compose(self.fcTransforms),
                    target_transform=ReshapeTransformTarget(10, self.expandOutput)
                ),
                shuffle=True, batch_size=self.trainBatchSize, num_workers=1
            ),

            DataLoader(
                datasets.MNIST(
                    root='./data', train=False, download=True,
                    transform=transforms.Compose(self.fcTransforms),
                    target_transform=ReshapeTransformTarget(10, self.expandOutput)
                ),
                shuffle=True, batch_size=self.testBatchSize, num_workers=1
            )
        )
