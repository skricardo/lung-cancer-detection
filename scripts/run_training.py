"""
Script de treinamento - Fase 1 (5 epocas)
Executa o treinamento completo do LunaModel.

Uso: python scripts/run_training.py
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
    N_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MOMENTUM = 0.99
    VAL_STRIDE = 10
    RATIO_INT = 2
    NUM_WORKERS = 0  # <--- Era 4. Mudado para 0 para evitar travamento (deadlock) no Windows
    PRINT_EVERY = 1  # Printa a cada batch para feedback IMEDIATO

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
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\nConfiguracao:")
    print(f"  Epocas: {N_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Momentum: {MOMENTUM}")
    print(f"  Val stride: {VAL_STRIDE}")
    print(f"  Ratio int: {RATIO_INT}")
    print(f"  Num workers: {NUM_WORKERS}")
    print()

    # ======================================================
    # CRIAR MODELO, OTIMIZADOR, LOSS
    # ======================================================
    model = LunaModel().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
    )
    loss_fn = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo criado: {total_params:,} parametros")

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

    print(f"Train samples por epoca: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print()

    # ======================================================
    # LOOP DE TREINAMENTO
    # ======================================================
    history = {
        'train_loss': [], 'train_f1': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_acc': [],
    }

    best_f1 = 0.0
    total_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        print(f"{'='*60}")
        print(f"Epoca {epoch}/{N_EPOCHS}")
        print(f"{'='*60}")

        # Shuffle a cada epoca
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
              f"Prec: {train_metrics['precision']:.4f} | "
              f"Rec: {train_metrics['recall']:.4f} | "
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
              f"Prec: {val_metrics['precision']:.4f} | "
              f"Rec: {val_metrics['recall']:.4f} | "
              f"Time: {val_time:.1f}s")

        # Salvar historico
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
            print(f"  >>> Novo melhor F1={best_f1:.4f} - checkpoint salvo!")

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
    print(f"Treinamento concluido!")
    print(f"  Melhor F1: {best_f1:.4f}")
    print(f"  Tempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")

    # Resumo
    print("\nResumo do treinamento:")
    for i in range(len(history['train_loss'])):
        print(f"  Epoca {i+1}: "
              f"Train Loss={history['train_loss'][i]:.4f}, "
              f"Val Loss={history['val_loss'][i]:.4f}, "
              f"Train F1={history['train_f1'][i]:.4f}, "
              f"Val F1={history['val_f1'][i]:.4f}")


if __name__ == "__main__":
    main()
