"""
Script de treinamento - Fase 2 (Retomada Épocas 6-10)
Retoma o treinamento do LunaModel a partir do último checkpoint.

Uso: python scripts/run_training_phase2.py
"""

import copy
import math
import random
import sys
import time
import os
from pathlib import Path

# Adicionar src/ ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from luna_data import load_candidates, get_ct, LunaDataset
from model import LunaModel
from training import LunaTrainingDataset, train_one_epoch, validate, compute_metrics


def main():
    # ======================================================
    # CONFIGURACAO
    # ======================================================
    START_EPOCH = 6
    END_EPOCH = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005  # Reduzido para Fine-Tuning
    MOMENTUM = 0.99
    VAL_STRIDE = 10
    RATIO_INT = 2
    NUM_WORKERS = 0 
    PRINT_EVERY = 1 
    
    RESUME_PATH = Path(__file__).parent.parent / "checkpoints" / "luna_model_last.pt"

    AUGMENTATION_DICT = {
        'flip': True,
        'offset': 0.1,
        'scale': 0.2,
        'rotate': True,
        'noise': 25.0,
    }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Diretorio de checkpoints
    ckpt_dir = Path(__file__).parent.parent / "checkpoints"

    print(f"\nConfiguracao (FASE 2):")
    print(f"  Retomando da Epoca: {START_EPOCH}")
    print(f"  Ate a Epoca: {END_EPOCH}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Resume path: {RESUME_PATH}")
    print()

    # ======================================================
    # CARREGAR MODELO E ESTADO
    # ======================================================
    model = LunaModel().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
    )
    loss_fn = nn.CrossEntropyLoss()

    if not RESUME_PATH.exists():
        print(f"ERRO: Checkpoint para retomar nao encontrado em {RESUME_PATH}")
        return

    print(f"Carregando checkpoint: {RESUME_PATH}")
    checkpoint = torch.load(RESUME_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Atualizar LR no otimizador (caso tenha mudado)
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE
        
    history = checkpoint.get('history', {
        'train_loss': [], 'train_f1': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_acc': [],
    })
    best_f1 = checkpoint.get('f1', 0.0)
    print(f"Checkpoint carregado. Melhor F1 anterior: {best_f1:.4f}")

    # ======================================================
    # DATASETS E DATALOADERS
    # ======================================================
    print("Carregando datasets...")
    train_ds = LunaTrainingDataset(
        val_stride=VAL_STRIDE,
        is_val=False,
        ratio_int=RATIO_INT,
        augmentation_dict=AUGMENTATION_DICT,
    )
    val_ds = LunaTrainingDataset(
        val_stride=VAL_STRIDE,
        is_val=True,
        ratio_int=0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ======================================================
    # LOOP DE TREINAMENTO (FASE 2)
    # ======================================================
    total_start = time.time()

    for epoch in range(START_EPOCH, END_EPOCH + 1):
        print(f"{'='*60}")
        print(f"Epoca {epoch}/{END_EPOCH}")
        print(f"{'='*60}")

        train_ds.shuffle_samples()

        # --- Treino ---
        start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device, print_every=PRINT_EVERY,
        )
        train_time = time.time() - start

        print(f"\n  [TRAIN] Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1']:.4f} | "
              f"Time: {train_time:.1f}s")

        # --- Validacao ---
        start = time.time()
        val_metrics = validate(
            model, val_loader, loss_fn,
            device, print_every=PRINT_EVERY,
        )
        val_time = time.time() - start

        print(f"  [VAL]   Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"Time: {val_time:.1f}s")

        # Atualizar historico
        history['train_loss'].append(train_metrics['loss'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Salvar checkpoint se F1 de validacao melhorou
        current_f1 = val_metrics['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'f1': best_f1,
                'history': history,
            }
            ckpt_path = str(ckpt_dir / "luna_model_best.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"  >>> NOVO MELHOR RECORDE! F1={best_f1:.4f} - checkpoint salvo!")

        # Salvar checkpoint da ultima epoca sempre
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'f1': current_f1,
            'history': history,
        }
        ckpt_path = str(ckpt_dir / "luna_model_last.pt")
        torch.save(checkpoint, ckpt_path)
        print()

    total_time = time.time() - total_start
    print(f"{'='*60}")
    print(f"Treinamento Fase 2 concluido!")
    print(f"  Melhor F1 final: {best_f1:.4f}")
    print(f"  Tempo total fase 2: {total_time/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
