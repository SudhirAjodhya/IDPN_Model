import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_froc_curve(froc_data, output_dir, variant):
    fig, ax = plt.subplots(figsize=(8, 6))

    for src, label in (("all", "Combined"), ("jsrt", "JSRT"), ("node21", "NODE21")):
        data_points = froc_data.get(src, [])
        
        if data_points:
            ax.plot([point["fp_per_image"] for point in data_points], [point["sensitivity"] for point in data_points], label=label)
    
    ax.set_xlabel("FP / image"); ax.set_ylabel("Sensitivity")
    ax.set_title(f"FROC Curve -- {variant}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / "froc_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
