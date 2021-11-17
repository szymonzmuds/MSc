This code is based on https://github.com/mit-han-lab/spvnas
To get more information about network, see original repo (especially closed issues)

Modify files: train.py, core/builder.py

New: all dataLoader (all are based on core/datasets/semantic_kitti.py),
     dockerfile,
     asymptote directory

All our changes are marked with "# TODO" -  hope it will be easy to find
All changes are in made in: train.py, selected data loader and core/builder.py
To see params check all configs files

1) Create environment

    1. create docker image
        open terminal and go to docker directory, then run:
            docker build -t ImageName .

    2. create docker container
            docker run --gpus all --name containerName -it -v /path/to/host/files:/path/container ImageName

    3. start existing container:
            docker start -i containerName

2) Run training process
    Note, that path to dataset (e.g Semantic KITTI) is passed in configs files
    only one param is path to configs
    Dataloader are hardcoded to work on Semantic KITTI (and use 8 sequences as test data)

        python3 train.py configs/semantic_kitti/spvcnn/cr0p5.yaml