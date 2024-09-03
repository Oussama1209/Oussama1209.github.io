import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import *


MAX_EPOCHS = 100

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_one_epoch(
    device,
    epoch_index,
    max_epochs,
    model,
    training_loader,
    optimizer,
    gradient_clip_val=None,
):
    run = {"Loss": [], "Acc": [], "F1": [], "Auc": []}
    start_time = time.time()

    # Use tqdm for the progress bar
    progress_bar = tqdm(
        enumerate(training_loader),
        total=len(training_loader),
        desc=f"Epoch {epoch_index+1}/{max_epochs}",
    )

    for i, data in progress_bar:
        text, image, label = (
            data["text"].to(device),
            data["image"].to(device),
            data["label"].to(device),
        )

        optimizer.zero_grad()

        preds, loss = model(text, image, label)
        run["Loss"].append(loss.item())

        # compute accuracy, and f1 score using pytorch functions
        metrics = compute_metrics(preds, label)

        for k in metrics:
            run[k].append(metrics[k])

        loss.backward(retain_graph=True)

        if gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        optimizer.step()

        # Update the progress bar with the current loss
        progress_bar.set_postfix(
            loss=run["Loss"][-1],
            acc=run["Acc"][-1],
            f1=run["F1"][-1],
            auc=run["Auc"][-1],
            iters_per_sec=(i + 1) / (time.time() - start_time),
        )

    return run


def train_validate(
    device,
    nb_epochs,
    model,
    training_loader,
    validation_loader,
    optimizer,
    scheduler,
    es_patience,
    es_min_delta,
    gradient_clip_val,
    output_path,
    model_name,
):
    timestamp = datetime.now().strftime("%d_%m_%Y_start_%Hh%Mm")
    run_path = output_path / "runs/{}Model_start_{}".format(model_name, timestamp)
    writer = SummaryWriter(run_path/"tensorboard_logs")

    nb_epochs = min(nb_epochs, MAX_EPOCHS)

    best_vloss = float("inf")
    early_stopper = EarlyStopper(patience=es_patience, min_delta=es_min_delta)
    model = model.to(device)
    train_run = {"Loss": [], "Acc": [], "F1": [], "Auc": []}
    val_run = {"Loss": [], "Acc": [], "F1": [], "Auc": []}

    for epoch in range(nb_epochs):
        model.train()
        train_epoch_run = train_one_epoch(
            device, epoch, nb_epochs, model, training_loader, optimizer, gradient_clip_val
        )

        val_epoch_run = {"Loss": [], "Acc": [], "F1": [], "Auc": []}
        start_time = time.time()
        progress_bar = tqdm(
            enumerate(validation_loader),
            total=len(validation_loader),
            desc="\tValidating",
        )

        model.eval()

        with torch.no_grad():
            for i, vdata in progress_bar:
                vtext, vimage, vlabel = (
                    vdata["text"].to(device),
                    vdata["image"].to(device),
                    vdata["label"].to(device),
                )
                vpreds, vloss = model(vtext, vimage, vlabel)
                val_epoch_run["Loss"].append(vloss.item())
                vmetrics = compute_metrics(vpreds, vlabel)
                for k in vmetrics:
                    val_epoch_run[k].append(vmetrics[k])

                # Update the progress bar with the current loss
                progress_bar.set_postfix(
                    vloss=val_epoch_run["Loss"][-1],
                    vacc=val_epoch_run["Acc"][-1],
                    vf1=val_epoch_run["F1"][-1],
                    vauc=val_epoch_run["Auc"][-1],
                    iters_per_sec=(i + 1) / (time.time() - start_time),
                )

        # average val running values
        for k in val_epoch_run:
            train_run[k].append(np.array(train_epoch_run[k]).mean())
            val_run[k].append(np.array(val_epoch_run[k]).mean())
            # tensorboard logging
            writer.add_scalars(
                "Training vs Validation {} ".format(k),
                {
                    "Train {}".format(k): train_run[k][-1],
                    "Val {}".format(k): val_run[k][-1],
                },
                epoch + 1,
            )

        curr_lr = optimizer.param_groups[0]["lr"]
        # print the epoch, learning rate, average val loss and metrics, avg training loss and metrics
        print(
            "Epoch {} lr: {} Averages : Train Loss: {:.4f} Val Loss: {:.4f} Train Acc: {:.4f} Val Acc: {:.4f} Train Auc: {:.4f} Val Auc: {:.4f} Train F1: {:.4f} Val F1: {:.4f}".format(
                epoch + 1,
                curr_lr,
                train_run["Loss"][-1],
                val_run["Loss"][-1],
                train_run["Acc"][-1],
                val_run["Acc"][-1],
                train_run["Auc"][-1],
                val_run["Auc"][-1],
                train_run["F1"][-1],
                val_run["F1"][-1],
            )
        )

        avg_val_loss = val_run["Loss"][-1]
        scheduler.step(avg_val_loss)

        writer.flush()

        if early_stopper.early_stop(avg_val_loss):
            print("Early stopping activated : epoch {}".format(epoch))
            break

        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            best_model_path = (
                run_path
                / "best_{}Model_start_{}_stoppedAtEpoch{}_lr{}".format(
                    model_name, timestamp, epoch, curr_lr
                )
            )
            print("Best model saved")
            torch.save(model, best_model_path)
    last_model_path = run_path / "last_{}Model_start_{}_stoppedAtEpoch{}_lr{}".format(
        model_name, timestamp, epoch, curr_lr
    )
    torch.save(model, last_model_path)
    return train_run, val_run, run_path, best_model_path, last_model_path


def evaluate(device, model, test_loader, run_path=None):
    model.eval()

    eval_run = {"Loss": [], "Acc": [], "F1": [], "Auc": []}
    start_time = time.time()
    progress_bar = tqdm(
        enumerate(test_loader), total=len(test_loader), desc="\tEvaluating"
    )

    with torch.no_grad():
        for i, data in progress_bar:
            text, image, label = (
                data["text"].to(device),
                data["image"].to(device),
                data["label"].to(device),
            )
            preds, loss = model(text, image, label)
            eval_run["Loss"].append(loss.item())
            metrics = compute_metrics(preds, label)
            for k in metrics:
                eval_run[k].append(metrics[k])

            # Update the progress bar with the current loss
            progress_bar.set_postfix(
                loss=eval_run["Loss"][-1],
                acc=eval_run["Acc"][-1],
                f1=eval_run["F1"][-1],
                auc=eval_run["Auc"][-1],
                iters_per_sec=(i + 1) / (time.time() - start_time),
            )

    for k in eval_run:
        eval_run[k] = np.array(eval_run[k]).mean()

    if run_path is not None:
        json.dump(eval_run, open(run_path/"evaluation.json", 'w' ))
    return eval_run
