import argparse
from data import *
from tools import *
from visualizer import *
from network import *

parser = argparse.ArgumentParser(description='Binary Equilibrium Propagation')

parser.add_argument(
    '--dataset',
    type=str,
    default='MNIST',
    help='Dataset to train the network (default: MNIST, others: CIFAR-10)')
parser.add_argument(
    '--device',
    type=int,
    default=0,
    help='GPU name to use cuda (default = 0)')
parser.add_argument(
    '--archi',
    type=str,
    default='fc',
    help='Architecture of the network (default: fc, others: conv)')
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    default=[784, 4096, 10],
    help='List of layer sizes (default: 2 fc hidden layers (4096))')
parser.add_argument(
    '--trainBatchSize',
    type=int,
    default=64,
    help='Batch size (default=64)')
parser.add_argument(
    '--testBatchSize',
    type=int,
    default=512,
    help='Testing B0atch size (default=512)')
parser.add_argument(
    '--expandOutput',
    type=int,
    default=1,
    help='Quantity by how much we expand the ouput layer)')
parser.add_argument(
    '--T',
    type=int,
    default=250,
    metavar='T',
    help='Number of time steps in the free phase (default: 50)')
parser.add_argument(
    '--Kmax',
    type=int,
    default=10,
    metavar='Kmax',
    help='Number of time steps in the backward pass (default: 10)')
parser.add_argument(
    '--beta',
    type=float,
    default=0.3,
    help='nudging parameter (default: 1)')
parser.add_argument(
    '--randomBeta',
    type=int,
    default=1,
    help='Use random sign of beta for training or fixed >0 sign (default: 1, other: 0)')
parser.add_argument(
    '--gamma',
    nargs='+',
    type=float,
    default=[5e-6, 2e-5, 2e-5],
    help='Low-pass filter constant')
parser.add_argument(
    '--tau',
    nargs='+',
    type=float,
    default=[5e-7, 5e-7, 5e-7],
    help='Thresholds used for the binary optimization in BOP')
# Training settings
parser.add_argument(
    '--hasBias',
    type=int,
    default=1,
    help='Does the network has bias ? (default: 1, other: 0)')
parser.add_argument(
    '--lrBias',
    nargs='+',
    type=float,
    default=[0.025, 0.05, 0.1],
    help='learning rates for bias')
parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    metavar='N',
    help='number of epochs to train (default: 2)')
# Learning the scaling factor
parser.add_argument(
    '--learnAlpha',
    type=int,
    default=1,
    help='Learn the scaling factors or let them fixed (default: 1, other: 0)')
parser.add_argument(
    '--lrAlpha',
    nargs='+',
    type=float,
    default=[1e-2, 1e-2, 1e-2],
    help='learning rates for the scaling factors')
parser.add_argument(
    '--nbBits',
    type=int,
    default=8,
    help='Number of bits for states in int coding')

args = parser.parse_args()

if __name__ == '__main__':
    # We reverse the layersList according to the convention that the output is 0 indexed
    args.layersList.reverse()

    # Initializing the data and the network
    trainLoader, testLoader = Data_Loader(args)()

    net = FCbinWAInt(args)

    # Create visualizer for tensorboard and save training
    visualizer = Visualizer(net, args)
    visualizer.saveHyperParameters()

    if net.cuda:
        net.to(net.device)

    print("Running on " + net.deviceName)

    # Training and testing the network
    for epoch in tqdm(range(args.epochs)):
        print("\nStarting epoch " + str(epoch + 1) + "/" + str(args.epochs))

        # Training
        print("Training")
        nbChanges, aveTrainError, singleTrainError, trainLoss, _ = trainFC(net, trainLoader, args)

        visualizer.addTraining(aveTrainError, singleTrainError, trainLoss, epoch)
        visualizer.addNbChanges(nbChanges, epoch)

        # Testing
        print("Testing")
        aveTestError, singleTestError, testLoss = testFC(net, testLoader, args)
        visualizer.addTesting(aveTestError, singleTestError, testLoss, epoch)

        print("Training loss: " + str(trainLoss))
        print("Average training error: " + str(aveTrainError))
        print("Single training error: " + str(singleTrainError))

        print("Testing loss: " + str(testLoss))
        print("Average testing error: " + str(aveTestError))
        print("Single testing error: " + str(singleTestError))

        # Save checkpoint after epoch
        print("Saving checkpoint")

        torch.save({
            'epoch': epoch,
            'modelStateDict': net.state_dict(),
            'trainLoss': trainLoss,
            'aveTrainError': aveTrainError,
            'testLoss': testLoss,
            'aveTestError': aveTestError,
        }, os.path.join(visualizer.path, 'checkpoint.pt'))

    print("Finished training")


