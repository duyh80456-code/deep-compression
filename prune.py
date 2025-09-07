import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os
import argparse
import random
import numpy as np

from models import get_model
from utils import *
from tqdm import tqdm

# Import pipeline full từ file thuật toán
from pruners.l1_pruner import TensorToFineTuneReady

################################################################## ARGUMENT PARSING

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 full pipeline pruning")
parser.add_argument("--model", default="resnet18", help="resnet9, resnet18, resnet34, resnet50, wrn_40_2, wrn_16_2, wrn_40_1")
parser.add_argument("--data_loc", default="./data", type=str)
parser.add_argument("--checkpoint", default=None, type=str, help="Pretrained model to start from")
parser.add_argument("--prune_checkpoint", default=None, type=str, help="Where to save pruned models")
parser.add_argument("--n_gpus", default=0, type=int, help="Number of GPUs to use")
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--cutout", action="store_true")

parser.add_argument("--pruning_type", default="unstructured", type=str)
parser.add_argument("--prune_iters", default=100, type=int)
parser.add_argument("--target_prune_rate", default=99, type=int)
parser.add_argument("--finetune_steps", default=100)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--weight_decay", default=0.0005, type=float)

args = parser.parse_args()

################################################################## REPRODUCIBILITY

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

################################################################## MODEL LOADING

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    select_devices(num_gpus_to_use=args.n_gpus)

model = get_model(args.model)

if args.checkpoint is None:
    args.checkpoint = args.model

args.checkpoint = args.checkpoint + "_" + str(args.seed)
model, sd = load_model(model, args.checkpoint)

if args.prune_checkpoint is None:
    args.prune_checkpoint = args.checkpoint + "_fullpipeline_"

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

################################################################## PRUNER: FULL PIPELINE

pruner = TensorToFineTuneReady(
    pruning_type=args.pruning_type,
    init_alpha=None,
    init_delta=None,
    min_distance=0.5
)

################################################################## TRAINING HYPERPARAMETERS

trainloader, testloader = get_cifar_loaders(args.data_loc, cutout=args.cutout)
optimizer = optim.SGD([w for name, w in model.named_parameters() if not "mask" in name],
                      lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-10)
for epoch in range(sd["epoch"]):
    scheduler.step()
for group in optimizer.param_groups:
    group["lr"] = scheduler.get_lr()[0]

################################################################## PRUNING & FINETUNING FULL PIPELINE

prune_rates = np.linspace(0, args.target_prune_rate, args.prune_iters)

for prune_rate in tqdm(prune_rates):
    print(f"\n=== Prune step: {prune_rate:.2f}% ===")

    # Luôn chạy full pipeline
    pruning_results = pruner.prune_and_optimize(model, prune_rate)

    # In chi tiết stats
    if 'apb_stats' in pruning_results:
        print(f"[APB] Sparsity: {pruning_results['apb_stats']['apb_sparsity']:.2%}")
    if 'filtering_stats' in pruning_results:
        fs = pruning_results['filtering_stats']
        print(f"[Layer filtering] Removed {fs['removed_layers']} layers, Remaining {fs['remaining_layers']}")
    if 'final_stats' in pruning_results:
        fs = pruning_results['final_stats']
        print(f"[Final] Model sparsity: {fs['sparsity']:.2%}, Active layers: {fs['active_layers']}/{fs['total_layers']}")

    # Thông tin tổng quan
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum(p.nonzero().size(0) for p in model.parameters())
    sparsity = 100.0 * (1 - nonzero_params / total_params)
    print(f"[INFO] Remaining params: {nonzero_params}/{total_params} ({100 - sparsity:.2f}% kept, {sparsity:.2f}% pruned)")

    # Finetune và lưu checkpoint
    checkpoint = args.prune_checkpoint + f"{prune_rate:.2f}"
    finetune(model, trainloader, criterion, optimizer, args.finetune_steps)
    acc = validate(model, prune_rate, testloader, criterion=criterion, checkpoint=checkpoint)
    print(f"[INFO] Accuracy after prune_rate={prune_rate:.2f}% → {acc:.2f}%\n")
