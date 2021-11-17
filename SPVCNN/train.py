import argparse
import sys

import random
import numpy as np

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data

from torchpack import distributed as dist
from torchpack.callbacks import (InferenceRunner, MaxSaver,
                                 Saver)
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.trainers import SemanticKITTITrainer
from core.callbacks import MeanIoU
from asymptote import asymptote as asp


# own dataLoaders 
from jumpAndAsymptoteLoader import JumpAsymptDataLoader
from cylinderLoader import CylinderSemanticKITTI
from sphericalLoader import SphericalSemanticKITTI
from differentGridLoader import DiffrentSemanticKITTI
from irregularLoader import IrregularDataLoader

#yaml.load(y, Loader=yaml.UnsafeLoader)


def main() -> None:
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')
    
    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2**32 - 1)
        
    #seed = configs.train.seed + dist.rank() * configs.workers_per_gpu * configs.num_epochs
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # TODO create dataloader
    dataset = builder.make_dataset() # original dataloader
    
    # sperical data loader
    # r = 0.05
    # phi = 0.16 * np.pi / 180
    # theta = 0.8 * np.pi / 180
    # dataset =  SphericalSemanticKITTI(
    #     root=configs.dataset.root, 
    #     num_points=configs.dataset.num_points, 
    #     voxel_size = np.array([r, phi, theta]))

    # Cylinder Data Loader
    # dataset = CylinderSemanticKITTI(
    #     root=configs.dataset.root, 
    #     num_points=configs.dataset.num_points, 
    #     voxel_size = np.array([0.05, 0.005, 0.05]))

    # Different grid by dist data loader
    # dataset = DiffrentSemanticKITTI(
    #     root=configs.dataset.root,
    #     num_points=configs.dataset.num_points, 
    #     voxel_size=configs.dataset.voxel_size)


    # jump or Asymp data loader - modify it inside
    # dataset = jumpAsymptDataLoader(root=configs.dataset.root,
    #                     num_points=configs.dataset.num_points,
    #                     voxel_size=configs.dataset.voxel_size)
    

    # Irregular data loader
    # dataset = IrregularDataLoader(root=configs.dataset.root,
    #                     num_points=configs.dataset.num_points,
    #                     voxel_size=np.array([0.025, 0.025, 0.05]))  
    #                     voxel_size=np.array([0.05, 0.05, 0.025])) 

    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=1,
            rank=0,
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    torch.cuda.set_device(1)
    model = builder.make_model()
    device = torch.device('cuda:1')
    model = model.to(device)

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = SemanticKITTITrainer(model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          num_workers=configs.workers_per_gpu,
                          seed=seed)

    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.num_epochs,
        callbacks=[
            InferenceRunner(
                dataflow[split],
                callbacks=[MeanIoU(
                    name=f'iou/{split}',
                    num_classes=configs.data.num_classes,
                    ignore_label=configs.data.ignore_label
                )])
            for split in ['test']
        ] + [
            MaxSaver('iou/test'),
            Saver(),
        ])



if __name__ == '__main__':
    main()
