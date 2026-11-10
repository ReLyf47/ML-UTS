import os
import matplotlib.pyplot as plt

OUTPUT_FIGURES_DIR = "/home/relyf/ML/ML-UTS/outputs/figures"
LAPORAN_FIGURES_DIR = "/home/relyf/ML/ML-UTS/laporan/figures"

def save_figure(fig, filename: str, dpi: int = 300, close_after: bool = True):
    """Save matplotlib figure to both outputs/figures/ and laporan/figures/."""
    os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
    os.makedirs(LAPORAN_FIGURES_DIR, exist_ok=True)
    
    path_out = os.path.join(OUTPUT_FIGURES_DIR, filename)
    path_lap = os.path.join(LAPORAN_FIGURES_DIR, filename)
    
    fig.tight_layout()
    fig.savefig(path_out, dpi=dpi, bbox_inches='tight')
    fig.savefig(path_lap, dpi=dpi, bbox_inches='tight')
    
    if close_after:
        plt.close(fig)
        
    return path_lap
