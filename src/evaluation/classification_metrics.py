import torch


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, num_classes=10):
    """
    Val-set evaluation.

    Returns top-1 accuracy (%), top-5 accuracy (%), confusion matrix, and,
    if `criterion` is supplied, the mean val loss (same criterion as train).
    val_loss is essential for overfit detection (train_loss vs val_loss curve).
    """
    model.eval()

    top1 = top5 = total = 0
    loss_sum = 0.0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, labels)
                loss_sum += loss.item() * labels.size(0)

        _, preds = logits.topk(5, dim=1)
        correct = preds.eq(labels.view(-1, 1))
        preds_top1 = preds[:, 0]

        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        for p, t in zip(preds_top1, labels):
            cm[t.item(), p.item()] += 1
        total += labels.size(0)

    return {
        "top1": 100.0 * top1 / total,
        "top5": 100.0 * top5 / total,
        "val_loss": (loss_sum / total) if criterion is not None else None,
        "cm": cm.tolist(),
    }
