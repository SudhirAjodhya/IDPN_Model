import pandas as pd
import numpy as np
from torch.utils.data import ConcatDataset

from .jsrt import JSRTDataset
from .node21 import NODE21Dataset
from .vindr import VinDrDataset
from .source_tagged import SourceTaggedDataset, LungMaskDataset

def build_three_way_split(jsrt_csv_path, jsrt_image_dir, node21_csv_path, node21_image_dir,
                          seed=42, test_seed=0, lung_mask_dir=None, target_size=512,
                          augment_train=False, datasets="both", input_size=2048, aug_profile="v3", group_by_image=False, letterbox=False,
                          val_frac=None, vindr_csv_path=None, vindr_image_dir=None,
                          vindr_input_size=None, vindr_negative_ratio=None,
                          vindr_dicom_dir=None):
    # val_frac: if set, overrides the per-dataset val fraction of the (non-test) remainder
    # for BOTH JSRT and NODE21. Default None preserves legacy behavior (NODE21 0.111, JSRT 0.125).
    # Use 0.20 for v2 to bump val ~500 -> ~900 and stabilize the early-stop metric.
    """
    Build train / val / test splits.

    datasets:
      "both"   - legacy JSRT+NODE21 pipeline (Combined = JSRT + NODE21). Preserved for
                 reproducing existing matrix_v1 / n=5 Talebi results.
      "node21" - Node21-only pipeline. JSRT loaders skipped; test_loaders["combined"]
                 is aliased to the Node21 test set so downstream callers (train.py,
                 eval scripts) still find a "combined" key without special-casing.
                 test_loaders["jsrt"] is omitted in this mode.
      "node21_vindr" - Node21 plus VinDr-CXR boxes, added to the TRAIN split only.
                 Validation and test stay Node21-only and byte-identical to the
                 "node21" mode, so checkpoint selection and every reported metric
                 remain measured on the same fixed 517-image / 147-nodule test set.
                 VinDr contributes generic foreground boxes to give the resizer
                 roughly 12x more localised gradient; it is never evaluated on.

    Split ratios per dataset:
      JSRT   - Train 0.70 / Val 0.10 / Test 0.20 (of both nodule and normal strata)
      NODE21 - Train ~0.80 / Val 0.10 / Test 0.10 (of both strata)

    test_seed is FIXED across experiments so the test set is stable; seed varies the
    train/val split within the non-test remainder.
    """
    assert datasets in ("both", "node21", "node21_vindr"), f"unknown datasets={datasets!r}"

    if datasets == "node21_vindr":
        assert vindr_csv_path is not None and vindr_image_dir is not None, \
            "datasets='node21_vindr' requires vindr_csv_path and vindr_image_dir"

    # NODE21 - always built.
    node21_df = pd.read_csv(node21_csv_path, index_col=0)
    pos_mask = node21_df["label"] == 1  # 1 = positive nodule, 0 = negative image

    nodule_df_n21 = node21_df[pos_mask].sample(frac=1, random_state=test_seed)
    normal_df_n21 = node21_df[~pos_mask].sample(frac=1, random_state=test_seed)

    n_nodule_test_n21 = max(1, int(len(nodule_df_n21) * 0.10))
    n_normal_test_n21 = max(1, int(len(normal_df_n21) * 0.10))

    nodule_test_n21 = nodule_df_n21.iloc[:n_nodule_test_n21]
    nodule_rest_n21 = nodule_df_n21.iloc[n_nodule_test_n21:]
    normal_test_n21 = normal_df_n21.iloc[:n_normal_test_n21]
    normal_rest_n21 = normal_df_n21.iloc[n_normal_test_n21:]

    nodule_rest_n21 = nodule_rest_n21.sample(frac=1, random_state=seed)
    normal_rest_n21 = normal_rest_n21.sample(frac=1, random_state=seed)

    _vf_n21 = val_frac if val_frac is not None else 0.111
    n_nodule_val_n21 = max(1, int(len(nodule_rest_n21) * _vf_n21))
    n_normal_val_n21 = max(1, int(len(normal_rest_n21) * _vf_n21))

    nodule_val_n21   = nodule_rest_n21.iloc[:n_nodule_val_n21]
    nodule_train_n21 = nodule_rest_n21.iloc[n_nodule_val_n21:]
    normal_val_n21   = normal_rest_n21.iloc[:n_normal_val_n21]
    normal_train_n21 = normal_rest_n21.iloc[n_normal_val_n21:]

    n21_train      = NODE21Dataset(pd.concat([nodule_train_n21, normal_train_n21]), node21_image_dir, augment=augment_train, input_size=input_size, aug_profile=aug_profile, group_by_image=group_by_image, letterbox=letterbox)
    n21_validation = NODE21Dataset(pd.concat([nodule_val_n21,   normal_val_n21]),   node21_image_dir, augment=False,        input_size=input_size, group_by_image=group_by_image, letterbox=letterbox)
    n21_testing    = NODE21Dataset(pd.concat([nodule_test_n21,  normal_test_n21]),  node21_image_dir, augment=False,        input_size=input_size, group_by_image=group_by_image, letterbox=letterbox)

    n21_train_tagged      = SourceTaggedDataset(n21_train,      "node21")
    n21_validation_tagged = SourceTaggedDataset(n21_validation, "node21")
    n21_test_tagged       = SourceTaggedDataset(n21_testing,    "node21")

    if lung_mask_dir is not None:
        n21_train_tagged      = LungMaskDataset(n21_train_tagged,      "node21", lung_mask_dir, target_size=target_size)
        n21_validation_tagged = LungMaskDataset(n21_validation_tagged, "node21", lung_mask_dir, target_size=target_size)

    if datasets in ("node21", "node21_vindr"):
        # Node21-only: no JSRT loaders built, no leakage risk from JSRT-inside-Node21.
        # "combined" aliased to Node21 test set so train.py's flat_test["combined"] path works.
        train_ds      = n21_train_tagged
        validation_ds = n21_validation_tagged
        test_loaders  = {"combined": n21_test_tagged, "node21": n21_test_tagged}
        split_info    = {
            "train": len(train_ds),
            "val":   len(validation_ds),
            "test_combined": len(n21_test_tagged),
            "test_node21":   len(n21_test_tagged),
            "datasets": datasets,
        }

        # VinDr joins the TRAIN split only. Every VinDr film is used, since none of them
        # are ever validated or tested on, so no VinDr split is needed. Holding val and
        # test at Node21-only is what keeps these runs comparable to the existing grid
        # on the metric axis, and keeps --select-on val_sens selecting on the same
        # quantity it has selected on in every previous run.
        if datasets == "node21_vindr":
            # vindr_input_size lets VinDr enter the model at its native ~2700 px while
            # NODE21 stays at 1024, which hands the resizer a ~12x decimation to target
            # instead of 4.6x. Batches must then be single-source, since default_collate
            # cannot stack two different image sizes - see SingleSourceBatchSampler.
            # Defaults to input_size, which keeps both datasets matched and needs no
            # special sampler.
            effective_vindr_input = vindr_input_size or input_size

            vindr_train = VinDrDataset(
                annotations_csv=vindr_csv_path,
                images_root=vindr_image_dir,
                dicom_root=vindr_dicom_dir,
                augment=augment_train,
                input_size=effective_vindr_input,
                aug_profile=aug_profile,
                letterbox=letterbox,
                negative_ratio=vindr_negative_ratio,
            )
            vindr_train_tagged = SourceTaggedDataset(vindr_train, "vindr")

            train_ds = ConcatDataset([n21_train_tagged, vindr_train_tagged])

            split_info["train"] = len(train_ds)
            split_info["train_node21"] = len(n21_train_tagged)
            split_info["train_vindr"] = len(vindr_train_tagged)
            split_info["vindr_boxes"] = sum(len(boxes) for _, boxes, _ in vindr_train.records)
            split_info["vindr_input_size"] = effective_vindr_input
            split_info["vindr_negative_ratio"] = vindr_negative_ratio
            split_info["mixed_input_sizes"] = effective_vindr_input != input_size

        return train_ds, validation_ds, test_loaders, split_info

    # datasets == "both": legacy JSRT+NODE21 pipeline (unchanged behaviour).
    jsrt_df = pd.read_csv(jsrt_csv_path)

    nodule_df = jsrt_df[jsrt_df["state"] != "non-nodule"].sample(frac=1, random_state=test_seed)
    normal_df = jsrt_df[jsrt_df["state"] == "non-nodule"].sample(frac=1, random_state=test_seed)

    n_nodule_test = int(len(nodule_df) * 0.20)
    n_normal_test = int(len(normal_df) * 0.20)

    nodule_test = nodule_df.iloc[:n_nodule_test]
    nodule_rest = nodule_df.iloc[n_nodule_test:]
    normal_test = normal_df.iloc[:n_normal_test]
    normal_rest = normal_df.iloc[n_normal_test:]

    nodule_rest = nodule_rest.sample(frac=1, random_state=seed)
    normal_rest = normal_rest.sample(frac=1, random_state=seed)

    _vf_jsrt = val_frac if val_frac is not None else 0.125
    n_nodule_validation = max(1, int(len(nodule_rest) * _vf_jsrt))
    n_normal_validation = max(1, int(len(normal_rest) * _vf_jsrt))

    nodule_validation = nodule_rest.iloc[:n_nodule_validation]
    nodule_train      = nodule_rest.iloc[n_nodule_validation:]
    normal_validation = normal_rest.iloc[:n_normal_validation]
    normal_train      = normal_rest.iloc[n_normal_validation:]

    jsrt_train      = JSRTDataset(pd.concat([nodule_train,      normal_train]),      jsrt_image_dir, augment=augment_train, input_size=input_size)
    jsrt_validation = JSRTDataset(pd.concat([nodule_validation, normal_validation]), jsrt_image_dir, augment=False,         input_size=input_size)
    jsrt_testing    = JSRTDataset(pd.concat([nodule_test,       normal_test]),       jsrt_image_dir, augment=False,         input_size=input_size)

    jsrt_train_tagged      = SourceTaggedDataset(jsrt_train,      "jsrt")
    jsrt_validation_tagged = SourceTaggedDataset(jsrt_validation, "jsrt")
    jsrt_test_tagged       = SourceTaggedDataset(jsrt_testing,    "jsrt")

    if lung_mask_dir is not None:
        jsrt_train_tagged      = LungMaskDataset(jsrt_train_tagged,      "jsrt", lung_mask_dir, target_size=target_size)
        jsrt_validation_tagged = LungMaskDataset(jsrt_validation_tagged, "jsrt", lung_mask_dir, target_size=target_size)

    train_ds      = ConcatDataset([jsrt_train_tagged,      n21_train_tagged])
    validation_ds = ConcatDataset([jsrt_validation_tagged, n21_validation_tagged])
    test_combined = ConcatDataset([jsrt_test_tagged,       n21_test_tagged])

    test_loaders = {
        "combined": test_combined,
        "jsrt":     jsrt_test_tagged,
        "node21":   n21_test_tagged,
    }

    split_info = {
        "train":         len(train_ds),
        "val":           len(validation_ds),
        "test_combined": len(test_combined),
        "test_jsrt":     len(jsrt_test_tagged),
        "test_node21":   len(n21_test_tagged),
        "datasets":      "both",
    }

    return train_ds, validation_ds, test_loaders, split_info


def collect_positive_flags(dataset):
    """
    Walk the dataset wrapper chain and return a per-sample nodule flag, without
    decoding a single image. Used to build a class-balanced WeightedRandomSampler.

    Handles ConcatDataset (datasets="both"), SourceTaggedDataset and LungMaskDataset
    by recursing into their children in the same order the flat indices run.

    Args:
        dataset: a training dataset returned by build_three_way_split

    Returns:
        list[bool] of length len(dataset), True where the sample carries a nodule
    """
    # ConcatDataset: flatten children in order so index i of the result matches index i
    if isinstance(dataset, ConcatDataset):
        flags = []

        for part in dataset.datasets:
            flags.extend(collect_positive_flags(part))

        return flags

    # Leaf datasets hold their annotation table on .df - NODE21 marks nodules with
    # label == 1, JSRT marks negatives with state == "non-nodule"
    frame = getattr(dataset, "df", None)

    if frame is not None:
        if "label" in frame.columns: return (frame["label"] == 1).tolist()

        return (frame["state"] != "non-nodule").tolist()

    # VinDr has no annotation frame - it holds (image_id, fused boxes) records, and a
    # film counts as positive when it carries any box. Checked after .df so NODE21 in
    # group_by_image mode, which has both attributes, keeps its existing behaviour.
    records = getattr(dataset, "records", None)

    if records is not None:
        return [len(boxes) > 0 for _, boxes in records]

    # Any other wrapper holds its child on .dataset
    return collect_positive_flags(dataset.dataset)
