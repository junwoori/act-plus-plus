# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
    # Optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # Backbone
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--camera_names', default=[], type=list)

    # Transformer
    parser.add_argument('--enc_layers', default=4, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=400, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # Segmentation
    parser.add_argument('--masks', action='store_true')

    # Common args (replicated for internal calls)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--policy_class', type=str)
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--kl_weight', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', type=int)
    parser.add_argument('--vq_dim', type=int)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--action_dim', type=int)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--validate_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--resume_ckpt_path', type=str)
    parser.add_argument('--no_encoder', action='store_true')
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', type=str)
    parser.add_argument('--history_len', type=int)
    parser.add_argument('--future_len', type=int)
    parser.add_argument('--prediction_len', type=int)

    return parser



def build_ACT_model_and_optimizer(args_override):
    # 기본값만 먼저 받아오고
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args, _ = parser.parse_known_args([])  # ⚠️ known_args()로 받고, 충돌 방지

    # 사용자 인자 수동 반영
    for k, v in args_override.items():
        setattr(args, k, v)

    # 디버깅용 출력
    print("args_override keys:", args_override.keys())
    required_keys = ['ckpt_dir', 'policy_class', 'task_name', 'seed', 'num_steps']
    for k in required_keys:
        if not hasattr(args, k):
            raise ValueError(f"Missing required argument: {k}")

    model = build_ACT_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer



from argparse import Namespace

def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args, _ = parser.parse_known_args([])  # 빈 인자 리스트에서 known args만 받아 충돌 방지

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer



