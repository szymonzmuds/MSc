# paths to data for selected models

paths = [
    "../path/to/model/Normal20/",
    "../path/to/model/Normal10/",
    "../path/to/model/Normal06/",
    "../path/to/model/Normal05/",
    "../path/to/model/Normal04/",
    "../path/to/model/Normal02/",
    "../path/to/model/Normal01/"
]

# names for all models
# warning: when creating boxplots, remember to update filter in createBoxPlot fnc in utils.py
# it should contain filter for all models

names = {
    paths[0]: "N 20",
    paths[1]: "N 10",
    paths[2]: "N 06",
    paths[3]: "N 05",
    paths[4]: "N 04",
    paths[5]: "N 02",
    paths[6]: "N 01",
}

scanNum = 4071

namesList = names.values()

colors = [
    'b', 'g', 'r', 'c', 'm', 'y', 'k'
]

kittiClasses = [
             'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
             'person', 'bicyclist', 'motorcyclist',
             'road', 'parking', 'side-walk', 'other ground',
             'building', 'fence', 
             'vegetation', 'trunk', 'terrain',
             'pole', 'traffic-sign'
]

xyDist = [
   0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 
]

generalIouData = 'generaliou.bin'
xyIouData = 'xyiou.bin'
zIouData = 'ziou.bin'

generalAccData = 'generalacc.bin'
xyAccData = 'xyacc.bin'
zAccData = 'zacc.bin'

newShape = {
    generalIouData: (scanNum),
    xyIouData: (scanNum, len(xyDist), len(kittiClasses)),
    zIouData: (scanNum, 16, len(kittiClasses)) # 16 stands for number of z dists
}




