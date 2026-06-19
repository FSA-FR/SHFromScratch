"""
Visualization.py
FR: Module regroupant les fonctions d'affichage pour le package SHFromScratch.
    Permet d'afficher des cartes d'intensité, de phase, ou de champ électrique avec des barres de couleur et des annotations PV/RMS.

EN: Module grouping display functions for the SHFromScratch package.
    Allows displaying intensity, phase, or electric field maps with colorbars and PV/RMS annotations.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import logging
from typing import Optional, Tuple
from MathAndPhysicsTools import compute_pv_rms


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_beam_map(
    data: np.ndarray,
    diameter_mm: float,
    title: str = "",
    cmap: str = "viridis",
    show_colorbar: bool = True,
    label: Optional[str] = None,
    **kwargs,
) -> None:
    """
    FR: Affiche une carte 2D (intensité, phase, etc.) avec une échelle et une barre de couleur.
        Annote la barre de couleur avec les valeurs PV et RMS si applicable.

    EN: Displays a 2D map (intensity, phase, etc.) with a scale and colorbar.
        Annotates the colorbar with PV and RMS values if applicable.

    Args:
        data (np.ndarray): Carte 2D à afficher.
        diameter_mm (float): Diamètre du faisceau en mm (pour l'échelle des axes).
        title (str): Titre du graphique.
        cmap (str): Colormap à utiliser (défaut: "viridis").
        show_colorbar (bool): Si True, affiche la barre de couleur (défaut: True).
        label (str, optional): Label pour la barre de couleur. Si None, utilise "Value".
        **kwargs: Arguments supplémentaires pour matplotlib.imshow.

    Raises:
        ImportError: Si matplotlib n'est pas installé.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("Matplotlib is not installed. Cannot plot.")
        raise ImportError("Matplotlib is required for plotting. Install it with: pip install matplotlib")

    # Calculer PV et RMS si les données sont numériques
    pv_rms_str = ""
    if np.issubdtype(data.dtype, np.number):
        pv, rms = compute_pv_rms(data)
        pv_rms_str = f"\nPV: {pv:.2f}, RMS: {rms:.2f}"

    # Afficher la carte
    plt.figure(figsize=(8, 6))
    plt.imshow(
        data,
        cmap=cmap,
        extent=[
            -diameter_mm / 2,
            diameter_mm / 2,
            -diameter_mm / 2,
            diameter_mm / 2,
        ],
        **kwargs,
    )

    if show_colorbar:
        colorbar_label = f"{label}{pv_rms_str}" if label is not None else f"Value{pv_rms_str}"
        plt.colorbar(label=colorbar_label)

    plt.title(title)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.show()


def plot_intensity(
    intensity: np.ndarray,
    diameter_mm: float,
    intensity_unit: str = "a.u.",
    title: str = "Intensity Map",
    **kwargs,
) -> None:
    """
    FR: Affiche une carte d'intensité avec les unités spécifiées.

    EN: Displays an intensity map with the specified units.

    Args:
        intensity (np.ndarray): Carte d'intensité 2D.
        diameter_mm (float): Diamètre du faisceau en mm.
        intensity_unit (str): Unité de l'intensité ("a.u.", "W/m2", "W/cm2").
        title (str): Titre du graphique.
        **kwargs: Arguments supplémentaires pour plot_beam_map.
    """
    label = f"Intensity ({intensity_unit})" if intensity_unit != "a.u." else "Intensity (a.u.)"
    plot_beam_map(
        intensity,
        diameter_mm,
        title=title,
        label=label,
        **kwargs,
    )


def plot_phase(
    phase: np.ndarray,
    diameter_mm: float,
    title: str = "Phase Map",
    **kwargs,
) -> None:
    """
    FR: Affiche une carte de phase en nm avec les valeurs PV et RMS.

    EN: Displays a phase map in nm with PV and RMS values.

    Args:
        phase (np.ndarray): Carte de phase 2D en nm.
        diameter_mm (float): Diamètre du faisceau en mm.
        title (str): Titre du graphique.
        **kwargs: Arguments supplémentaires pour plot_beam_map.
    """
    plot_beam_map(
        phase,
        diameter_mm,
        title=title,
        label="Phase (nm)",
        **kwargs,
    )


def plot_electric_field_amplitude(
    electric_field: np.ndarray,
    diameter_mm: float,
    title: str = "Electric Field Amplitude",
    **kwargs,
) -> None:
    """
    FR: Affiche l'amplitude du champ électrique avec les valeurs PV et RMS.

    EN: Displays the electric field amplitude with PV and RMS values.

    Args:
        electric_field (np.ndarray): Champ électrique complexe 2D.
        diameter_mm (float): Diamètre du faisceau en mm.
        title (str): Titre du graphique.
        **kwargs: Arguments supplémentaires pour plot_beam_map.
    """
    amplitude = np.abs(electric_field)
    plot_beam_map(
        amplitude,
        diameter_mm,
        title=title,
        label="Amplitude (a.u.)",
        **kwargs,
    )


def plot_electric_field_phase(
    electric_field: np.ndarray,
    diameter_mm: float,
    wavelength_nm: float = 633.0,
    title: str = "Electric Field Phase",
    **kwargs,
) -> None:
    """
    FR: Affiche la phase du champ électrique en nm.

    EN: Displays the electric field phase in nm.

    Args:
        electric_field (np.ndarray): Champ électrique complexe 2D.
        diameter_mm (float): Diamètre du faisceau en mm.
        wavelength_nm (float): Longueur d'onde en nm (pour conversion rad → nm).
        title (str): Titre du graphique.
        **kwargs: Arguments supplémentaires pour plot_beam_map.
    """
    from MathAndPhysicsTools import rad_to_nm
    phase_rad = np.angle(electric_field)
    phase_nm = rad_to_nm(phase_rad, wavelength_nm)
    plot_beam_map(
        phase_nm,
        diameter_mm,
        title=title,
        label="Phase (nm)",
        **kwargs,
    )


# =============================================================================
# TESTS UNITAIRES / UNIT TESTS
# =============================================================================

class TestVisualization:
    """
    FR: Classe de tests unitaires pour Visualization.py.
    EN: Unit test class for Visualization.py.
    """

    def test_plot_beam_map(self):
        """Test l'affichage d'une carte 2D."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Mode non-interactif pour les tests
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib not available")

        data = np.random.rand(64, 64) * 100
        diameter_mm = 10.0
        plot_beam_map(data, diameter_mm, title="Test Map", show_colorbar=True)
        plt.close('all')  # Fermer la figure pour éviter les fuites de mémoire

    def test_plot_intensity(self):
        """Test l'affichage d'une carte d'intensité."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib not available")

        intensity = np.random.rand(64, 64)
        diameter_mm = 10.0
        plot_intensity(intensity, diameter_mm, intensity_unit="W/m2")
        plt.close('all')

    def test_plot_phase(self):
        """Test l'affichage d'une carte de phase."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib not available")

        phase = np.random.rand(64, 64) * 100
        diameter_mm = 10.0
        plot_phase(phase, diameter_mm)
        plt.close('all')

    def test_plot_electric_field_amplitude(self):
        """Test l'affichage de l'amplitude du champ électrique."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib not available")

        electric_field = np.random.rand(64, 64) + 1j * np.random.rand(64, 64)
        diameter_mm = 10.0
        plot_electric_field_amplitude(electric_field, diameter_mm)
        plt.close('all')

    def test_plot_electric_field_phase(self):
        """Test l'affichage de la phase du champ électrique."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib not available")

        electric_field = np.random.rand(64, 64) + 1j * np.random.rand(64, 64)
        diameter_mm = 10.0
        plot_electric_field_phase(electric_field, diameter_mm)
        plt.close('all')


if __name__ == "__main__":
    import unittest
    unittest.main()