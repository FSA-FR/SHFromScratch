"""
Visualization.py

FR: Module central pour toutes les fonctions de visualisation du package SHFromScratch.
    Ce module contient TOUTES les fonctions d'affichage et de sauvegarde d'images.
    
    Fonctionnalités principales :
    - Affichage de tableaux 2D (intensité, phase, pentes, erreurs)
    - Sauvegarde d'images avec annotations (PV, RMS, échelles)
    - Visualisation de faisceaux, matrices de microlentilles, tâches d'Airy
    - Comparaison de résultats (réel vs parfait)
    - Génération de graphiques globaux (courbes, histogrammes)
    
    Règle d'or :
    - TOUTE fonction de visualisation DOIT être ici.
    - Aucune fonction de visualisation ne doit exister dans d'autres fichiers.
    - Les fonctions outils doivent être importées depuis MathAndPhysicsTools.py
    
    Chaque image générée aura :
    - Une échelle visuelle
    - Le PV (Peak-to-Valley) et le RMS des valeurs
    - Colormap : "Jet" pour la phase, "hot" pour l'intensité

EN: Central module for all visualization functions in SHFromScratch package.
    This module contains ALL display and image saving functions.
    
    Main features:
    - 2D array display (intensity, phase, slopes, errors)
    - Image saving with annotations (PV, RMS, scales)
    - Beam, microlens array, Airy spot visualization
    - Result comparison (real vs perfect)
    - Global plots generation (curves, histograms)
    
    Golden rule:
    - EVERY visualization function MUST be here.
    - No visualization functions should exist in other files.
    - Utility functions must be imported from MathAndPhysicsTools.py
    
    Each generated image will have:
    - A visual scale
    - PV (Peak-to-Valley) and RMS values
    - Colormap: "Jet" for phase, "hot" for intensity

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - matplotlib
    - MathAndPhysicsTools (pour les fonctions outils)

Sources:
    - "Matplotlib for Python Developers" by S. J. Perlt (2009)
      -> Bonnes pratiques de visualisation scientifique
    - "Visualization with Python" by Z. Pei (2018)
      -> Techniques avancées de visualisation
    - "Scientific Visualization: Python + Matplotlib" by N. Rougier (2016)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
import os
import logging

# Import des fonctions outils depuis MathAndPhysicsTools
from MathAndPhysicsTools import handle_nan, compute_pv_rms


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Visualization")

plt.style.use('seaborn-v0_8')

# Colormaps par défaut
DEFAULT_PHASE_CMAP = 'Jet'
DEFAULT_INTENSITY_CMAP = 'hot'
DEFAULT_SLOPE_CMAP = 'coolwarm'
DEFAULT_ERROR_CMAP = 'RdYlBu'

# Taille des figures
DEFAULT_FIGSIZE = (10, 8)
DEFAULT_DPI = 150

# Police
DEFAULT_FONT_SIZE = 12
plt.rcParams.update({'font.size': DEFAULT_FONT_SIZE})


# =============================================================================
# FONCTION PRINCIPALE: AFFICHAGE 2D AVEC ANNOTATIONS
# =============================================================================

def display_2d_array(data: np.ndarray,
                     title: str = "",
                     save_path: Optional[str] = None,
                     show: bool = False,
                     cmap: str = DEFAULT_INTENSITY_CMAP,
                     xlabel: str = "x",
                     ylabel: str = "y",
                     colorbar_label: str = "Value",
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     figsize: Tuple[int, int] = DEFAULT_FIGSIZE) -> None:
    """
    FR: Affiche un tableau 2D avec annotations (PV, RMS, Mean).
    EN: Displays a 2D array with annotations (PV, RMS, Mean).
    
    Chaque image aura:
    - Une échelle visuelle
    - Le PV et le RMS des valeurs
    - Colormap personnalisable
    
    Args:
        data: Tableau 2D à afficher.
        title: Titre de l'image.
        save_path: Chemin pour sauvegarder.
        show: Afficher l'image.
        cmap: Colormap à utiliser.
        xlabel: Label axe x.
        ylabel: Label axe y.
        colorbar_label: Label de la colorbar.
        vmin: Valeur minimale pour la colorbar.
        vmax: Valeur maximale pour la colorbar.
        figsize: Taille de la figure.
    """
    data_clean = handle_nan(data, method='zero')
    pv, rms = compute_pv_rms(data_clean)
    mean_val = float(np.mean(data_clean))
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data_clean, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    
    plt.colorbar(im, ax=ax, label=colorbar_label)
    
    title_text = title
    if pv != 0 or rms != 0:
        title_text += f"\nPV={pv:.2f}, RMS={rms:.2f}, Mean={mean_val:.2f}"
    ax.set_title(title_text)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
        logger.info(f"Image sauvegardée: {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


# =============================================================================
# FONCTIONS SPÉCIALISÉES
# =============================================================================

def display_phase(data: np.ndarray, **kwargs) -> None:
    """FR: Affiche une carte de phase. EN: Displays a phase map."""
    kwargs.setdefault('cmap', DEFAULT_PHASE_CMAP)
    kwargs.setdefault('colorbar_label', "Phase (nm)")
    display_2d_array(data, **kwargs)


def display_intensity(data: np.ndarray, **kwargs) -> None:
    """FR: Affiche une carte d'intensité. EN: Displays an intensity map."""
    kwargs.setdefault('cmap', DEFAULT_INTENSITY_CMAP)
    kwargs.setdefault('colorbar_label', "Intensité (a.u.)")
    display_2d_array(data, **kwargs)


def display_slopes(slopes_x: np.ndarray, slopes_y: np.ndarray, **kwargs) -> None:
    """FR: Affiche les pentes X et Y. EN: Displays X and Y slopes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    im0 = axes[0].imshow(handle_nan(slopes_x), cmap=DEFAULT_SLOPE_CMAP, origin='lower')
    axes[0].set_title(kwargs.get('title', 'Pentes') + " - X")
    axes[0].set_xlabel(kwargs.get('xlabel', 'x'))
    axes[0].set_ylabel(kwargs.get('ylabel', 'y'))
    plt.colorbar(im0, ax=axes[0], label="Pente X (rad)")
    
    im1 = axes[1].imshow(handle_nan(slopes_y), cmap=DEFAULT_SLOPE_CMAP, origin='lower')
    axes[1].set_title(kwargs.get('title', 'Pentes') + " - Y")
    axes[1].set_xlabel(kwargs.get('xlabel', 'x'))
    axes[1].set_ylabel(kwargs.get('ylabel', 'y'))
    plt.colorbar(im1, ax=axes[1], label="Pente Y (rad)")
    
    save_path = kwargs.get('save_path')
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    
    if kwargs.get('show', False):
        plt.show()
    
    plt.close(fig)


def display_error(data: np.ndarray, **kwargs) -> None:
    """FR: Affiche une carte d'erreur. EN: Displays an error map."""
    kwargs.setdefault('cmap', DEFAULT_ERROR_CMAP)
    kwargs.setdefault('colorbar_label', "Erreur (nm)")
    display_2d_array(data, **kwargs)


def display_airy_spots(image: np.ndarray, centroids: Optional[np.ndarray] = None, **kwargs) -> None:
    """FR: Affiche les tâches d'Airy avec centroïdes. EN: Displays Airy spots with centroids."""
    fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', DEFAULT_FIGSIZE))
    
    im = ax.imshow(handle_nan(image), cmap=DEFAULT_INTENSITY_CMAP, origin='lower')
    plt.colorbar(im, ax=ax, label=kwargs.get('colorbar_label', "Intensité (ADU)"))
    
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=10, label='Centroïdes')
        ax.legend()
    
    ax.set_title(kwargs.get('title', "Tâches d'Airy"))
    ax.set_xlabel(kwargs.get('xlabel', "x (pixels)"))
    ax.set_ylabel(kwargs.get('ylabel', "y (pixels)"))
    
    save_path = kwargs.get('save_path')
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    
    if kwargs.get('show', False):
        plt.show()
    
    plt.close(fig)


# =============================================================================
# FONCTIONS POUR LES GRAPHIQUES GLOBAUX
# =============================================================================

def plot_error_vs_amplitude(amplitudes: List[float],
                           zonal_errors: List[float],
                           modal_errors: List[float],
                           perfect_errors: List[float],
                           error_type: str = 'rms',
                           save_path: Optional[str] = None,
                           show: bool = False) -> None:
    """FR: Trace erreur vs amplitude. EN: Plots error vs amplitude."""
    fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FIGSIZE)
    
    ax.plot(amplitudes, zonal_errors, 'o-', label='Mode Zonal', linewidth=2, markersize=8)
    ax.plot(amplitudes, modal_errors, 's-', label='Mode Modal', linewidth=2, markersize=8)
    ax.plot(amplitudes, perfect_errors, '^-', label='Parfait', linewidth=2, markersize=8)
    
    ax.set_xlabel("Amplitude RMS des aberrations (nm)")
    ax.set_ylabel(f"Erreur {error_type.upper()} moyenne (nm)")
    ax.set_title(f"Erreur {error_type.upper()} en fonction de l'amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)


def plot_zonal_vs_modal(amplitudes: List[float],
                       zonal_errors: List[float],
                       modal_errors: List[float],
                       save_path: Optional[str] = None,
                       show: bool = False) -> None:
    """FR: Compare zonal vs modal. EN: Compares zonal vs modal."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(amplitudes, zonal_errors, 'o-', label='Mode Zonal', linewidth=2)
    axes[0].plot(amplitudes, modal_errors, 's-', label='Mode Modal', linewidth=2)
    axes[0].set_xlabel("Amplitude RMS (nm)")
    axes[0].set_ylabel("Erreur RMS (nm)")
    axes[0].set_title("Comparaison - Erreur RMS")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(amplitudes, zonal_errors, 'o-', label='Mode Zonal', linewidth=2)
    axes[1].plot(amplitudes, modal_errors, 's-', label='Mode Modal', linewidth=2)
    axes[1].set_xlabel("Amplitude RMS (nm)")
    axes[1].set_ylabel("Erreur PV (nm)")
    axes[1].set_title("Comparaison - Erreur PV")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)


def plot_individual_draws(amplitudes: List[float],
                         draws: List[int],
                         zonal_errors: List[float],
                         modal_errors: List[float],
                         perfect_errors: List[float],
                         save_path: Optional[str] = None,
                         show: bool = False) -> None:
    """FR: Trace erreur pour chaque tirage. EN: Plots error for each draw."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    unique_amplitudes = sorted(set(amplitudes))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_amplitudes)))
    
    for i, amp in enumerate(unique_amplitudes):
        amp_indices = [j for j, a in enumerate(amplitudes) if a == amp]
        ax.scatter(amp_indices, [zonal_errors[j] for j in amp_indices],
                  color=colors[i], marker='o', label=f'{amp}nm Zonal' if i == 0 else "", s=50)
        ax.scatter(amp_indices, [modal_errors[j] for j in amp_indices],
                  color=colors[i], marker='s', label=f'{amp}nm Modal' if i == 0 else "", s=50)
        ax.scatter(amp_indices, [perfect_errors[j] for j in amp_indices],
                  color=colors[i], marker='^', label=f'{amp}nm Parfait' if i == 0 else "", s=50)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.set_xlabel("Index du tirage")
    ax.set_ylabel("Erreur RMS (nm)")
    ax.set_title("Comportement de l'erreur pour chaque tirage")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)


def plot_error_histogram(errors: List[float],
                         bins: int = 20,
                         title: str = "Histogramme des erreurs",
                         save_path: Optional[str] = None,
                         show: bool = False) -> None:
    """FR: Trace un histogramme des erreurs. EN: Plots a histogram of errors."""
    fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FIGSIZE)
    ax.hist(errors, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel("Erreur RMS (nm)")
    ax.set_ylabel("Fréquence")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)


def plot_comparison_histogram(zonal_errors: List[float],
                              modal_errors: List[float],
                              perfect_errors: List[float],
                              bins: int = 20,
                              save_path: Optional[str] = None,
                              show: bool = False) -> None:
    """FR: Trace un histogramme comparatif. EN: Plots a comparative histogram."""
    fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FIGSIZE)
    ax.hist(zonal_errors, bins=bins, alpha=0.5, label='Mode Zonal', color='blue')
    ax.hist(modal_errors, bins=bins, alpha=0.5, label='Mode Modal', color='red')
    ax.hist(perfect_errors, bins=bins, alpha=0.5, label='Parfait', color='green')
    ax.set_xlabel("Erreur RMS (nm)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Histogramme comparatif des erreurs RMS")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)


# =============================================================================
# FONCTIONS POUR LA RECONSTRUCTION
# =============================================================================

def display_reconstruction_results(zonal_phase: np.ndarray,
                                  modal_phase: np.ndarray,
                                  perfect_phase: np.ndarray,
                                  title: str = "Résultats de reconstruction",
                                  save_path: Optional[str] = None,
                                  show: bool = False) -> None:
    """FR: Affiche les résultats de reconstruction. EN: Displays reconstruction results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im0 = axes[0].imshow(handle_nan(zonal_phase), cmap=DEFAULT_PHASE_CMAP, origin='lower')
    axes[0].set_title(f"{title} - Zonal")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im0, ax=axes[0], label="Phase (nm)")
    
    im1 = axes[1].imshow(handle_nan(modal_phase), cmap=DEFAULT_PHASE_CMAP, origin='lower')
    axes[1].set_title(f"{title} - Modal")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[1], label="Phase (nm)")
    
    im2 = axes[2].imshow(handle_nan(perfect_phase), cmap=DEFAULT_PHASE_CMAP, origin='lower')
    axes[2].set_title(f"{title} - Parfait")
    axes[2].set_xlabel("x (mm)")
    axes[2].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[2], label="Phase (nm)")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestVisualization:
    """FR: Tests unitaires pour Visualization.py."""

    def test_display_2d_array(self):
        """FR: Test l'affichage d'un tableau 2D."""
        data = np.random.rand(100, 100)
        display_2d_array(data, title="Test", save_path="test_display_2d.png")
        assert os.path.exists("test_display_2d.png")
        os.remove("test_display_2d.png")

    def test_display_phase(self):
        """FR: Test l'affichage d'une phase."""
        phase = np.random.rand(100, 100) * 100
        display_phase(phase, title="Test Phase", save_path="test_phase.png")
        assert os.path.exists("test_phase.png")
        os.remove("test_phase.png")

    def test_plot_error_vs_amplitude(self):
        """FR: Test le tracé erreur vs amplitude."""
        plot_error_vs_amplitude(
            [50, 250, 500, 1000],
            [10, 50, 100, 200],
            [8, 40, 80, 160],
            [2, 10, 20, 40],
            save_path="test_error_vs_amp.png"
        )
        assert os.path.exists("test_error_vs_amp.png")
        os.remove("test_error_vs_amp.png")


if __name__ == "__main__":
    import unittest
    unittest.main()
