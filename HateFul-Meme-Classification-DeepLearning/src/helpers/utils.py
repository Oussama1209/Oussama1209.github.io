import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from transformers import get_scheduler

def configure_optimizers(params, lr):
    optimizer = torch.optim.AdamW(params, lr)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=3,
        num_training_steps=30 * 133,
    )
    return optimizer, lr_scheduler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5
    return {"Acc": acc, "F1": f1, "Auc": auc}

def plot_training(train_run, val_run, run_path) :
  fig,axes = plt.subplots(2, 2, figsize=(15, 8))
  for i, k in enumerate(train_run.keys()) :
    axes[i//2][i%2].plot(train_run[k], label=f'Training {k}')
    axes[i//2][i%2].plot(val_run[k], label=f'Validation {k}')
    axes[i//2][i%2].set_title(f'Training vs. Validation {k}')
    axes[i//2][i%2].set_xlabel('Epochs')
    axes[i//2][i%2].set_ylabel(k)
    axes[i//2][i%2].legend()

  plt.tight_layout()
  plt.show()
  fig.savefig(run_path/'train_vs_val.png')
