from main import args
import threading
import os
import time
import numpy as np

# Defining all hyper-parameters on which to train the network
betaL = [1 + i for i in range(4)]
convGammaL = [[1e-9 * 10**j for i in range(4)] for j in range(5)]
convTauL = [[1e-9 * 10**j for i in range(4)] for j in range(5)]
fcGammaL = [[1e-9 * 10**j for i in range(1)] for j in range(5)]
fcTauL = [[1e-9 * 10**j for i in range(1)] for j in range(5)]

devices = [i for i in range(8)]  # 8 GPUs for Goliath
cmds = []  # We init all the cmds we want to launch

for beta in betaL:
    for i, convGamma in enumerate(convGammaL):
        for j, convTau in enumerate(convTauL):
            args.beta = beta
            args.convGamma = convGamma
            args.convTau = convTau
            args.fcGamma = fcGammaL[i]
            args.fcTau = fcTauL[j]

            cmds.append(''.join(["python main.py --dataset CIFAR10 --archi conv --layersList 100 --convList 3 64 128 256 256 "
                                 "--expandOutput 10 --padding 2 --kernel 5 --FPool 2 --T 150 --Kmax 60 --randomBeta 1 "
                                 "--lrBias 0 0 0 0 0 --trainBatchSize 64 --testBatchSize 512 --epochs 100 --beta ",
                                 str(args.beta), " --convGamma ", str(args.convGamma),
                                 " --convTau ", str(args.convTau), " --fcGamma ", str(args.fcGamma), " --fcTau ",
                                 str(args.fcTau)])
                        .replace(',', '').replace('[', '').replace(']', ''))

class Train(threading.Thread):
    """One run with the cmd specified"""
    def __init__(self, cmd, device):
        threading.Thread.__init__(self)
        self.cmd = cmd + ' --device ' + str(device)
        self.device = device
        self.handled = False

    def run(self):
        os.system(self.cmd)  # Launches the cmd
        time.sleep(1)  # For log files...
        return


# if __name__ == '__main__':
#     # We init the number of cmds that has been launched
#     cmds_launched = 0
#
#     # Launching the first threads
#     threads = [Train(cmds[i], devices[i]) for i in range(8)]  # 8 GPUs for Goliath
#
#     for thread in threads:
#         thread.start()
#         cmds_launched += 1
#
#     running_devices = [thread.device for thread in threads]
#     free_devices = []
#
#     # We enter the while statement until all cmds are launched
#     while cmds_launched != len(cmds):
#         # Checking which threads are running
#         for thread in threads:
#             if not thread.is_alive():
#                 thread.handled = True
#
#         # We get the free devices, threads running and running devices
#         free_devices = [thread.device for thread in threads if thread.handled]
#
#         running_devices = [thread.device for thread in threads if not thread.handled]
#         threads = [thread for thread in threads if not thread.handled]
#
#         # We launch new threads for free devices
#         for device in free_devices:
#             threads.append(Train(cmds[cmds_launched], device))
#             threads[-1].start()
#             cmds_launched += 1




