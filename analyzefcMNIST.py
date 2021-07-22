import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json

runs = []

for path, subdirs, files in os.walk("SAVE-fc-MNIST"):
    for name in files:
        cpath = os.path.join(path, name)
        if 'tfevents' in cpath:
            runs.append(cpath)

acclist = {}

for run in runs:
    event = EventAccumulator(run)
    event.Reload()
    try:
        key = "testing/Average testing error"
        err = event.Scalars(key)
        _, _, acc = zip(*err)
        acclist[run] = list(acc)

    except:
        key = "Average testing error"
        err = event.Scalars(key)
        _, _, acc = zip(*err)
        acclist[run] = list(acc)

# with open('resultsfcMNIST.json', 'w') as file:
#     json.dump(acclist, file, indent=2)

mins = []

for key in acclist.keys():
    mins.append(acclist[key][-1])

mins.sort()

for key, item in acclist.items():
    if item[-1] == mins[2]:
        print(key)

print(mins[2])
