"""Shared publication style. Titles live in LaTeX captions, never in-figure."""
import matplotlib as mpl
def apply_style():
    mpl.rcParams.update({
        "figure.dpi":200,"savefig.dpi":200,"savefig.bbox":"tight","savefig.pad_inches":0.03,
        "font.family":"serif","font.serif":["DejaVu Serif","Times New Roman"],
        "mathtext.fontset":"dejavuserif","font.size":9,"axes.titlesize":9.5,
        "axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.6,
        "axes.linewidth":0.8,"axes.edgecolor":"#444","axes.labelcolor":"#1a1a1a",
        "text.color":"#1a1a1a","xtick.color":"#444","ytick.color":"#444",
        "axes.grid":True,"grid.color":"#DDD","grid.linewidth":0.6,"grid.alpha":0.9,
        "axes.axisbelow":True,"legend.frameon":False,"figure.facecolor":"white","axes.facecolor":"white",
    })
INK="#1a1a1a"; MUTE="#6b6b6b"; LIGHT="#9aa0a6"
