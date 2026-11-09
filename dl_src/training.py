"""Masked training loop, regularization, checkpointing and learning-curve output."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_


def masked_mse_loss(prediction, target, valid_mask):
    """Mean squared error over valid air cells only, for both velocity components."""
    squared_error = (prediction - target) ** 2
    expanded_mask = valid_mask.expand_as(squared_error)
    valid_count = expanded_mask.sum().clamp_min(1.0)
    return (squared_error * expanded_mask).sum() / valid_count


def _run_validation(model, loader, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for inputs, targets, masks, _, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            prediction = model(inputs)
            loss = masked_mse_loss(prediction, targets, masks)
            total_loss += float(loss.item())
            batches += 1
    return total_loss / max(batches, 1)


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    output_dir,
    model_name,
    epochs=25,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=7,
    gradient_clip=1.0,
    model_config=None,
):
    """Train with AdamW, weight decay, dropout in the model, LR reduction and early stopping."""
    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    figure_dir = output_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []
    checkpoint_path = model_dir / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        batches = 0

        for inputs, targets, masks, _, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = masked_mse_loss(prediction, targets, masks)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()

            total_train_loss += float(loss.item())
            batches += 1

        train_loss = total_train_loss / max(batches, 1)
        validation_loss = _run_validation(model, validation_loader, device)
        scheduler.step(validation_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history_rows.append({
            "epoch": epoch,
            "train_masked_mse": train_loss,
            "validation_masked_mse": validation_loss,
            "learning_rate": current_lr,
        })

        print(
            f"{model_name} | epoch {epoch:02d}/{epochs} | "
            f"train={train_loss:.6f} | validation={validation_loss:.6f} | lr={current_lr:.2e}"
        )

        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "model_config": model_config or {},
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping: validation loss did not improve for {patience} epochs.")
            break

    history = pd.DataFrame(history_rows)
    history_path = output_dir / f"{model_name}_training_history.csv"
    history.to_csv(history_path, index=False)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_masked_mse"], label="Training")
    ax.plot(history["epoch"], history["validation_masked_mse"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE (normalized velocity components)")
    ax.set_title(f"{model_name}: training and validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / f"{model_name}_learning_curve.png", dpi=160)
    plt.show()
    plt.close(fig)

    print(f"Best epoch: {best_epoch}; best validation loss: {best_validation_loss:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    return model, history, checkpoint_path


def get_torch_device():
    """Prefer CUDA in Colab, then Apple Metal (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
