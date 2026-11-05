def print_banner(args, split_info, model, device, input_size=2048):
    names = {
        "bilinear":            "Bilinear Baseline",
        "talebi":              "Talebi Resizer (n_blocks=2)",
        "talebi_aux":          "Talebi + Aux Lung Supervision",
        "alpha":               "Simple Residual",
        "gamma":               "Weighted Pooling",
        "delta":               "Learned Kernel",
        "lip":                 "LIP",
        "node21_talebi":       "Talebi (paper-faithful, Node21-only)",
    }
    j = split_info.get("test_jsrt", 0)
    n = split_info.get("test_node21", 0)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 70)
    print(f"Experiment -- {args.variant} ({names.get(args.variant, args.variant)})")
    print(f"Run group: {args.run_group} | Seed: {args.seed} | Test seed: {args.test_seed}")
    print(f"Loss: {args.loss} | Head: {args.head}")
    if args.variant == "talebi_aux":
        print(f"Aux lambda: {args.aux_lambda}")
    print(f"heatmap_size={model.heatmap_size} | "
          f"scale={input_size // model.heatmap_size}")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | "
          f"LR: {args.lr} | Patience: {args.patience} | Grad accum: {args.grad_accum}")
    sources = "NODE21" if split_info.get("datasets") == "node21" else "JSRT + NODE21"
    print(f"Data: {split_info['train']} train / {split_info['val']} val / "
          f"{j+n} test ({sources})")
    print(f"Params: {total_params:,} total, {trainable:,} trainable")
    print("=" * 70)


def print_final_summary(history, flat_test, output_dir):
    combined = flat_test.get("combined", {})
    jsrt     = flat_test.get("jsrt",     {})
    node21   = flat_test.get("node21",   {})
    by_size  = flat_test.get("jsrt_by_size", {})
    print("\n" + "=" * 70)
    print(f"Training complete. Results: {output_dir}")
    print(f"Best val sensitivity: {history['best_val_sensitivity']:.3f} "
          f"(epoch {history['best_val_sensitivity_epoch']})")
    print(f"Best checkpoint -- Sensitivity: {combined.get('sensitivity', 0):.3f}, "
          f"FP/img: {combined.get('fp_per_image', 0):.2f} (combined test)")
    print(f"  JSRT test:   Sens={jsrt.get('sensitivity', 0):.3f}, "
          f"FP/img={jsrt.get('fp_per_image', 0):.2f}")
    if by_size:
        for bn in ("small", "medium", "large"):
            bv = by_size.get(bn, {})
            if bv:
                print(f"    JSRT {bn:6s}: Sens={bv['sensitivity']:.3f} "
                      f"(n={bv['n_nodules']})")
    print(f"  NODE21 test: Sens={node21.get('sensitivity', 0):.3f}, "
          f"FP/img={node21.get('fp_per_image', 0):.2f}")
    print(f"Peak sensitivity during training: {history['peak_sensitivity']:.3f} "
          f"(epoch {history['peak_sensitivity_epoch']})")
    print("=" * 70)
