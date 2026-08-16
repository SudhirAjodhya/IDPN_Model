import torch
import torch.nn.functional as F

def aux_lung_supervised_loss(pred_heatmap, gt_heatmap, model, images, lung_masks,
                             base_loss_fn, aux_lambda=0.1):
    """
    Computes standard focal loss + auxiliary steering loss on the residual.
    Requires `lung_masks` (B, TARGET_SIZE, TARGET_SIZE) in [0, 1].
    """
    det_loss = base_loss_fn(pred_heatmap, gt_heatmap)

    ldm = model.ldm
    output = ldm(images)
    target_size = output.shape[-1]
    bilinear = F.interpolate(images, size=(target_size, target_size),
                             mode="bilinear", align_corners=False)
    residual = output - bilinear
    res_mag  = residual.abs().squeeze(1)

    # outside mask is heavily penalized (mse to 0)
    # inside mask we just want it free to vary
    mask_outside = 1.0 - lung_masks
    loss_outside = (res_mag * mask_outside).pow(2).mean()

    # optional: small penalty inside to prevent explosion?
    loss_inside = (res_mag * lung_masks).pow(2).mean() * 0.1

    aux_loss = loss_outside + loss_inside
    total_loss = det_loss + aux_lambda * aux_loss

    info = {
        "detection_loss": det_loss.item(),
        "aux_outside":    loss_outside.item(),
        "aux_inside":     loss_inside.item(),
        "aux_total":      aux_loss.item(),
    }
    return total_loss, info
