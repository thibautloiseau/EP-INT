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


class ReshapeTransformTargetFC:
    """Transform target output in the appropriate way"""
    def __init__(self, noClasses, expandOutput, maxIntState):
        self.noClasses = noClasses
        self.expandOutput = expandOutput
        self.maxIntState = maxIntState

    def __call__(self, target):
        target = torch.tensor(target).unsqueeze(0).unsqueeze(1)
        targetOneHot = torch.zeros((1, self.noClasses))

        return targetOneHot.scatter_(1, target, 1).repeat_interleave(self.expandOutput).squeeze(0) * self.maxIntState


class ReshapeTransformTargetConv:
    """Transform target output in the appropriate way"""
    def __init__(self, noClasses, expandOutput):
        self.noClasses = noClasses
        self.expandOutput = expandOutput

    def __call__(self, target):
        target = torch.tensor(target).unsqueeze(0).unsqueeze(1)
        targetOneHot = torch.zeros((1, self.noClasses))

        return targetOneHot.scatter_(1, target, 1).repeat_interleave(self.expandOutput).squeeze(0)


class Data_Loader(DataLoader):
    """Create the data_loader for training and for testing"""
    def __init__(self, args):
        self.trainBatchSize = args.trainBatchSize
        self.testBatchSize = args.testBatchSize
        self.fcTransforms = [transforms.ToTensor(), ReshapeTransform((-1,))]
        self.expandOutput = args.expandOutput
        self.dataset = args.dataset
        self.maxIntState = 2**(args.bitsState - 1) - 1
        self.archi = args.archi

    def __call__(self):
        """We return a tuple with both dataloader for train and test"""
        if self.archi == 'fc':
            if self.dataset == 'MNIST':
                return (
                    DataLoader(
                        datasets.MNIST(
                            root='./data', train=True, download=True,
                            transform=transforms.Compose(self.fcTransforms),
                            target_transform=ReshapeTransformTargetFC(10, self.expandOutput, self.maxIntState)
                        ),
                        shuffle=True, batch_size=self.trainBatchSize, num_workers=10, pin_memory=True
                    ),

                    DataLoader(
                        datasets.MNIST(
                            root='./data', train=False, download=True,
                            transform=transforms.Compose(self.fcTransforms),
                            target_transform=ReshapeTransformTargetFC(10, self.expandOutput, self.maxIntState)
                        ),
                        shuffle=True, batch_size=self.testBatchSize, num_workers=10, pin_memory=True
                    )
                )

            elif self.dataset == 'FashionMNIST':
                return (
                    DataLoader(
                        datasets.FashionMNIST(
                            root='./data', train=True, download=True,
                            transform=transforms.Compose(self.fcTransforms),
                            target_transform=ReshapeTransformTargetFC(10, self.expandOutput, self.maxIntState)
                        ),
                        shuffle=True, batch_size=self.trainBatchSize, num_workers=10, pin_memory=True
                    ),

                    DataLoader(
                        datasets.FashionMNIST(
                            root='./data', train=False, download=True,
                            transform=transforms.Compose(self.fcTransforms),
                            target_transform=ReshapeTransformTargetFC(10, self.expandOutput, self.maxIntState)
                        ),
                        shuffle=True, batch_size=self.testBatchSize, num_workers=10, pin_memory=True
                    )
                )

        elif self.archi == 'conv':
            if self.dataset == 'MNIST':
                transform = [transforms.ToTensor()]

                return (
                    DataLoader(
                        datasets.MNIST(
                            root='./data', train=True, download=True,
                            transform=transforms.Compose(transform),
                            target_transform=ReshapeTransformTargetConv(10, self.expandOutput)
                        ),
                        batch_size=self.trainBatchSize, shuffle=True, num_workers=10, pin_memory=True
                    ),

                    DataLoader(
                        datasets.MNIST(
                            root='./data', train=False, download=True,
                            transform=transforms.Compose(transform),
                            target_transform=ReshapeTransformTargetConv(10, self.expandOutput)
                        ),
                        batch_size=self.testBatchSize, shuffle=True, num_workers=10, pin_memory=True
                    )
                )

            if self.dataset == 'CIFAR10':
                trainTransforms = [transforms.RandomHorizontalFlip(0.5),
                                   transforms.RandomCrop(size=[32, 32], padding=4, padding_mode='edge'),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

                testTransforms = [transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

                return(
                    DataLoader(
                        datasets.CIFAR10(
                            root='./data', train=True, download=True,
                            transform=transforms.Compose(trainTransforms),
                            target_transform=ReshapeTransformTargetConv(10, self.expandOutput)),
                        batch_size=self.trainBatchSize, shuffle=True, num_workers=10, pin_memory=True
                    ),

                    DataLoader(
                        datasets.CIFAR10(
                            root='./data', train=False, download=True,
                            transform=transforms.Compose(testTransforms),
                            target_transform=ReshapeTransformTargetConv(10, self.expandOutput)),
                        batch_size=self.testBatchSize, shuffle=True, num_workers=10, pin_memory=True
                    )
                )

