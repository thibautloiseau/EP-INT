import argparse
from data import *
from tools import *
from visualizer import *
from network import *

parser = argparse.ArgumentParser(description='Binary Equilibrium Propagation')

# For 1 hidden layer with augmented output
# --device 0 --dataset MNIST --archi fc --binarySettings WA --layersList 784 8192 100 --expandOutput 10 --T 20 --Kmax 10
# --beta 2 --randomBeta 1 --gamma 2e-6 2e-6 --tau 2.5e-7 2e-7 --lrBias 1e-7 1e-7 --trainBatchSize 64 --testBatchSize 512
# --epochs 100 --learnAlpha 0

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
    default=[784, 8192, 100],
    help='List of layer sizes (default: 1 fc hidden layers (4096))')
parser.add_argument(
    '--expandOutput',
    type=int,
    default=10,
    help='Quantity by how much we expand the output layer)')
parser.add_argument(
    '--trainBatchSize',
    type=int,
    default=64,
    help='Batch size (default=64)')
parser.add_argument(
    '--testBatchSize',
    type=int,
    default=512,
    help='Testing Batch size (default=512)')
parser.add_argument(
    '--T',
    type=int,
    default=16,
    metavar='T',
    help='Number of time steps in the free phase (default: 50)')
parser.add_argument(
    '--Kmax',
    type=int,
    default=16,
    metavar='Kmax',
    help='Number of time steps in the backward pass (default: 10)')
parser.add_argument(
    '--beta',
    type=float,
    default=2,
    help='Nudging parameter (default: 2)')
parser.add_argument(
    '--randomBeta',
    type=int,
    default=1,
    help='Use random sign of beta for training or fixed >0 sign (default: 1, other: 0)')
parser.add_argument(
    '--tauInt',
    nargs='+',
    type=int,
    default=[12, 12],
    help='Thresholds used for the binary optimization in BOP')
parser.add_argument(
    '--clampMom',
    type=int,
    default=2**9,
    help='Clamp the momentum for each layer')
# Training settings
parser.add_argument(
    '--hasBias',
    type=int,
    default=0,
    help='Does the network has biases ? (default: 1, other: 0)')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    metavar='N',
    help='number of epochs to train (default: 2)')
parser.add_argument(
    '--bitsState',
    type=int,
    default=13,
    help='Number of bits for states in signed int coding')
parser.add_argument(
    '--decay',
    type=int,
    default=1,
    help='Quantity by which we multiply the threshold for BOP')

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
        nbChanges, aveTrainError, singleTrainError, trainLoss, _ = trainFC(net, trainLoader, epoch, args)

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


