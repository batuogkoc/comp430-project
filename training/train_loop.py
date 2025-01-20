from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchmetrics
import wandb
import os
import time
import numpy as np
import torch
from datasets import *
from training.train_helpers import (
    compute_metrics,
    log_metrics_to_wandb,
    log_metrics,
    InplacePrinter,
    RunningAverageLogger,
)
from torch import nn
from torch import optim


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn,
    num_epoch: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    experiment_name: None | str = None,
    printing: bool = True,
    tensorboard_logging: bool = True,
    wandb_logging: bool = True,
    metrics: dict[str, torchmetrics.Metric] = {},
    checkpointing: bool = True,
    device: torch.device = None,
):
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    start_epoch = 0

    tensorboard_writer = (
        SummaryWriter(os.path.join("runs_tensorboard", str(experiment_name)))
        if tensorboard_logging
        else None
    )
    if checkpointing:
        checkpointing_folder = os.path.join("runs_checkpoints", str(experiment_name))
        os.makedirs(checkpointing_folder)
    running_average_training_loss_logger = RunningAverageLogger()
    val_loss_logger = RunningAverageLogger()

    printer = InplacePrinter(2 + len(metrics))

    metrics = {
        metric_name: metric.to(device) for metric_name, metric in metrics.items()
    }

    model = model.to(device)
    for epoch in range(start_epoch, num_epoch):
        if printing:
            printer.reset()
            print("-" * 5 + f"EPOCH: {epoch}" + "-" * 5)
        start = time.time()

        running_average_training_loss_logger.reset()

        for metric in metrics.values():
            metric.reset()

        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_metric_values = {}
            for metric_name, metric in metrics.items():
                epoch_metric_values[metric_name] = metric(y_pred, y).item()

            running_average_training_loss_logger.add_value(loss.item())

            log_metrics(
                None,
                global_step=epoch * len(train_loader) + i,
                prefix="within_epoch_train/",
                extra={
                    "epoch": epoch,
                    "loss": loss.item(),
                    "ratl": running_average_training_loss_logger.get_avg(),
                    "lr": scheduler.get_lr()[0] if scheduler else -1,
                    **epoch_metric_values,
                },
                log_wandb=False,
                tensorboard_writer=tensorboard_writer,
            )
            if i % 1 == 0 and i != 0:
                fraction_done = max(i / len(train_loader), 1e-6)
                time_taken = time.time() - start
                if printing:
                    for metric_name, metric in metrics.items():
                        printer.print(
                            f"{metric_name} : {epoch_metric_values[metric_name]}"
                        )
                    printer.print(
                        f"e: {epoch} | i: {i} | loss: {loss.item():2.3f} | ratl: {running_average_training_loss_logger.get_avg():2.3f}"
                    )
                    printer.print(
                        f"{fraction_done*100:2.2f}% | est time left: {time_taken*(1-fraction_done)/fraction_done:.1f} s | est total: {time_taken/fraction_done:.1f} s"
                    )
        if scheduler:
            scheduler.step()
        total_time_taken = time.time() - start

        log_metrics(
            metrics,
            global_step=epoch,
            prefix="train/",
            extra={
                "epoch": epoch,
                "loss": running_average_training_loss_logger.get_avg(),
                "total_time": total_time_taken,
                "per_epoch_time": total_time_taken / len(train_loader),
                "lr": scheduler.get_lr()[0] if scheduler else -1,
            },
            log_wandb=wandb_logging,
            tensorboard_writer=tensorboard_writer,
        )
        if printing:
            print("--Train--")
            for metric_name, metric in metrics.items():
                print(f"{metric_name} : {metric.compute()}")
        for metric_name, metric in metrics.items():
            metric.reset()

        val_loss_logger.reset()
        model.eval()
        with torch.inference_mode():
            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                val_loss_logger.add_value(loss.item())

                for metric_name, metric in metrics.items():
                    metric(y_pred, y)

        if printing:
            print("--Val--")
            for metric_name, metric in metrics.items():
                print(f"{metric_name} : {metric.compute().item()}")
            print(
                f"train loss: {running_average_training_loss_logger.get_avg()} | val loss: {val_loss_logger.get_avg()}"
            )

        log_metrics(
            metrics,
            global_step=epoch,
            prefix="val/",
            extra={
                "epoch": epoch,
                "loss": val_loss_logger.get_avg(),
            },
            log_wandb=wandb_logging,
            tensorboard_writer=tensorboard_writer,
        )
        if checkpointing:
            torch.save(
                {
                    "model": model,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": {
                        metric_name: metric.compute().item()
                        for metric_name, metric in metrics.items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_loss": running_average_training_loss_logger.get_avg(),
                    "val_loss": val_loss_logger.get_avg(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                },
                os.path.join(checkpointing_folder, f"{epoch}.pt"),
            )

    return metrics
