import argparse
from data import *
from tools import *
from visualizer import *
from network import *

parser = argparse.ArgumentParser(description='Binary Equilibrium Propagation')

# For 1 hidden layer with augmented output (FC arch)
# --device 0 --dataset MNIST --archi fc --binarySettings WA --layersList 784 8192 100 --expandOutput 10 --T 20 --Kmax 10
# --beta 2 --randomBeta 1 --gamma 2e-6 2e-6 --tau 2.5e-7 2e-7 --lrBias 1e-7 1e-7 --trainBatchSize 64 --testBatchSize 512
# --epochs 100 --learnAlpha 0

# Conv arch
# python main.py --device 0 --dataset mnist --optim ep --archi conv --binary_settings bin_W_N --layersList 700
# --convList 512 256 1 --expand_output 70 --padding 1 --kernelSize 5 --Fpool 3 --activationFun heaviside --T 100
# --Kmax 50 --beta 1 --random_beta 1 --classi_gamma 5e-8 --conv_gamma 5e-8 5e-8 --classi_threshold 8e-8
# --conv_threshold 8e-8 2e-7 --lrBias 2e-6 5e-6 1e-5 --batchSize 64 --test_batchSize 512 --epochs 50 --learnAlpha 0
# --rho_threshold 0.5

parser.add_argument(
    '--dataset',
    type=str,
    default='CIFAR10',
    help='Dataset to train the network (default: MNIST, others: FashionMNIST)')
parser.add_argument(
    '--device',
    type=int,
    default=0,
    help='GPU name to use cuda (default = 0)')
parser.add_argument(
    '--archi',
    type=str,
    default='conv',
    help='Architecture of the network (default: fc, others: conv)')
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    default=[1600],
    help='List of layer sizes (default: 1 fc hidden layers (4096))')
parser.add_argument(
    '--expandOutput',
    type=int,
    default=160,
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
    help='Testing batch size (default=512)')
parser.add_argument(
    '--T',
    type=int,
    default=150,
    metavar='T',
    help='Number of time steps in the free phase (default: 50)')
parser.add_argument(
    '--Kmax',
    type=int,
    default=60,
    metavar='Kmax',
    help='Number of time steps in the backward pass (default: 10)')
parser.add_argument(
    '--beta',
    type=float,
    default=0.01,
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
    default=[1, 1],
    help='Thresholds used for BOP')
parser.add_argument(
    '--bitsMom',
    nargs='+',
    type=int,
    default=7,
    help='Number of bits for the momentum')
# Training settings
parser.add_argument(
    '--hasBias',
    type=int,
    default=1,
    help='Does the network has biases ? (default: 1, other: 0)')
parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    metavar='N',
    help='Number of epochs to train (default: 2)')
parser.add_argument(
    '--bitsState',
    type=int,
    default=10,
    help='Number of bits for states in signed int coding')
parser.add_argument(
    '--bitsBias',
    type=int,
    default=5,
    help='Number of bits for biases in signed int coding')
parser.add_argument(
    '--decay',
    type=int,
    default=1,
    help='Quantity by which we multiply the threshold for BOP after a certain number of epochs')
parser.add_argument(
    '--constNudge',
    type=int,
    default=1,
    help='Apply a constant nudge in nudging phase (default: 0)')
parser.add_argument(
    '--stochInput',
    type=int,
    default=1,
    help='Get stochastic binary inputs')
parser.add_argument(
    '--lrBias',
    nargs='+',
    type=float,
    default=[0, 0, 0],
    help='Learning rates for biases')

# Parameters for conv architecture
parser.add_argument(
    '--convList',
    nargs='+',
    type=int,
    default=[3, 512, 1024],
    help="List of convolutional layers with number of channels (default: )")
parser.add_argument(
    '--kernel',
    type=int,
    default=5,
    help="Kernel size for convolution (default: 5)")
parser.add_argument(
    '--FPool',
    type=int,
    default=3,
    help="Pooling filter size (default: 3)")
parser.add_argument(
    '--padding',
    type=int,
    default=0,
    help="Padding (default: 0)")
parser.add_argument(
    '--convTau',
    nargs='+',
    type=float,
    default=[1e-8, 1e-8],
    help='Thresholds used for the conv part of the conv arch')
parser.add_argument(
    '--convGamma',
    nargs='+',
    type=float,
    default=[1e-8, 1e-8],
    help='Gamma for conv part for BOP')
parser.add_argument(
    '--fcTau',
    nargs='+',
    type=float,
    default=[1e-8],
    help='Thresholds used for the fc of the conv arch')
parser.add_argument(
    '--fcGamma',
    nargs='+',
    type=float,
    default=[1e-8],
    help='Gamma for FC part for BOP')

args = parser.parse_args()

if __name__ == '__main__':
    # We reverse the layersList according to the convention that the output is 0 indexed
    args.layersList.reverse()
    args.convList.reverse()

    # Initializing the data and the network
    trainLoader, testLoader = Data_Loader(args)()

    if args.archi == 'fc':
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

    elif args.archi == 'conv':
        net = ConvWAInt(args)

        if net.cuda:
            net.to(net.device)

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
            aveTrainError, singleTrainError, trainLoss, nbChangesFC, nbChangesConv = trainConv(net, trainLoader, epoch, args)

            visualizer.addTraining(aveTrainError, singleTrainError, trainLoss, epoch)
            visualizer.addNbChanges(nbChangesFC, epoch, nbChangesConv=nbChangesConv)

            # Testing
            print("Testing")
            aveTestError, singleTestError, testLoss = testConv(net, testLoader, args)
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

