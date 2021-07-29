from main import args
import threading
import os
import time

# Numbers of devices
no_devices = 2

# Defining all hyper-parameters on which to train the network
convGammaL = [[1e-9 * 10**j for i in range(4)] for j in range(5)]
convTauL = [[1e-9 * 10**j for i in range(4)] for j in range(5)]
fcGammaL = [[1e-9 * 10**j for i in range(1)] for j in range(5)]
fcTauL = [[1e-9 * 10**j for i in range(1)] for j in range(5)]

devices = [i for i in range(8)]  # 8 GPUs for Goliath
cmds = []  # We init all the cmds we want to launch

for i, convGamma in enumerate(convGammaL):
    for j, convTau in enumerate(convTauL):
        args.convGamma = convGamma
        args.convTau = convTau
        args.fcGamma = fcGammaL[i]
        args.fcTau = fcTauL[j]

        cmds.append(''.join(["python main.py --dataset CIFAR10 --archi conv --layersList 800 --convList 3 128 256 512 512 "
                             "--expandOutput 80 --padding 1 --kernel 3 --FPool 2 --hasBias 0 --T 150 --Kmax 60 --randomBeta 1 "
                             "--trainBatchSize 64 --testBatchSize 512 --epochs 25 --beta 1 ",
                             " --convGamma ", str(args.convGamma),
                             " --convTau ", str(args.convTau), " --fcGamma ", str(args.fcGamma), " --fcTau ",
                             str(args.fcTau)])
                    .replace(',', '').replace('[', '').replace(']', ''))

# We reverse for training on several computers
cmds.reverse()
print(len(cmds))

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
#     threads = [Train(cmds[i], devices[i]) for i in range(no_devices)]  # 8 GPUs for Goliath, 2 for Nanoinfer
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




