# -*- coding: utf-8 -*-
"""
Beam Module
-----------

FR: Module pour la génération de faisceaux optiques avec différentes méthodes d'intensité, de phase et de champ électrique.
    Permet la génération de faisceaux aléatoires, paramétrés, ou importés depuis des fichiers.

EN: Module for generating optical beams with different intensity, phase, and electric field methods.
    Allows generation of random, parameterized, or file-imported beams.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
License: MIT
"""

import numpy as np
import logging
import unittest
from typing import Optional, Tuple, Union
from scipy.special import eval_genlaguerre
from MathAndPhysicsTools import (
    create_grid,
    normalize_phase,
    nm_to_rad,
    rad_to_nm,
    generate_zernike_modes,
    generate_legendre_modes,
)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================
# FR: Classe Beam
# EN: Beam Class
# =============================================

class Beam:
    """
    FR: Classe représentant un faisceau optique.
        
    EN: Class representing an optical beam.

    Attributes:
        wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
        diameter_mm (float): Diamètre du faisceau en mm (défaut: 10.0).
        energy (float): Énergie totale du faisceau (défaut: 1.0).
        intensity (np.ndarray): Carte d'intensité 2D (en unités arbitraires).
        phase (np.ndarray): Carte de phase 2D en nm.
        electric_field (np.ndarray): Champ électrique complexe 2D.
        grid_x (np.ndarray): Grille en x en mm.
        grid_y (np.ndarray): Grille en y en mm.
        logger (logging.Logger): Logger pour le débogage.
    """

    def __init__(
        self,
        wavelength_nm: float = 633.0,
        diameter_mm: float = 10.0,
        energy: float = 1.0,
        intensity: Optional[np.ndarray] = None,
        phase: Optional[np.ndarray] = None,
        electric_field: Optional[np.ndarray] = None,
        num_points: int = 512,
    ):
        """
        FR: Initialise un faisceau optique avec les paramètres par défaut.
            
        EN: Initializes an optical beam with default parameters.

        Args:
            wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
            diameter_mm (float): Diamètre du faisceau en mm (défaut: 10.0).
            energy (float): Énergie totale du faisceau (défaut: 1.0).
            intensity (np.ndarray, optional): Carte d'intensité 2D.
            phase (np.ndarray, optional): Carte de phase 2D en nm.
            electric_field (np.ndarray, optional): Champ électrique complexe 2D.
            num_points (int): Nombre de points par dimension pour la grille (défaut: 512).
        """
        self.wavelength_nm = wavelength_nm
        self.diameter_mm = diameter_mm
        self.energy = energy
        self.intensity = intensity
        self.phase = phase
        self.electric_field = electric_field
        self.num_points = num_points
        self.grid_x, self.grid_y = create_grid(diameter_mm, num_points)
        self.logger = logging.getLogger("Beam")
        self.logger.setLevel(logging.INFO)

    # =============================================
    # FR: Méthodes de génération d'intensité
    # EN: Intensity generation methods
    # =============================================

    def generate_intensity(
        self,
        method: str = "gaussian",
        **kwargs,
    ) -> np.ndarray:
        """
        FR: Génère une carte d'intensité selon la méthode spécifiée.
            
        EN: Generates an intensity map according to the specified method.

        Args:
            method (str): Méthode de génération. Options:
                - "random": Intensité aléatoire avec fréquences et amplitudes contrôlées.
                - "gaussian": Intensité gaussienne.
                - "supergaussian": Intensité super-gaussienne.
                - "tophat": Intensité top-hat (uniforme dans un cercle).
                - "from_file": Import depuis un fichier (non implémenté ici).
            **kwargs: Arguments spécifiques à la méthode.

        Returns:
            np.ndarray: Carte d'intensité 2D.

        Raises:
            ValueError: Si la méthode est inconnue.
        """
        if method == "random":
            return self._generate_random_intensity(**kwargs)
        elif method == "gaussian":
            return self._generate_gaussian_intensity(**kwargs)
        elif method == "supergaussian":
            return self._generate_supergaussian_intensity(**kwargs)
        elif method == "tophat":
            return self._generate_tophat_intensity(**kwargs)
        elif method == "from_file":
            raise NotImplementedError("L'import depuis un fichier n'est pas encore implémenté.")
        else:
            raise ValueError(f"Méthode inconnue pour la génération d'intensité: {method}")

    def _generate_random_intensity(
        self,
        min_amplitude: float = 0.1,
        max_amplitude: float = 1.0,
        min_frequency: float = 0.01,
        max_frequency: float = 0.1,
    ) -> np.ndarray:
        """
        FR: Génère une intensité aléatoire avec des fréquences et amplitudes contrôlées.
            Utilise un spectre de Fourier aléatoire pour simuler des variations spatiales.
            
        EN: Generates random intensity with controlled frequencies and amplitudes.
            Uses a random Fourier spectrum to simulate spatial variations.

        Args:
            min_amplitude (float): Amplitude minimale de l'intensité (défaut: 0.1).
            max_amplitude (float): Amplitude maximale de l'intensité (défaut: 1.0).
            min_frequency (float): Fréquence spatiale minimale en 1/mm (défaut: 0.01).
            max_frequency (float): Fréquence spatiale maximale en 1/mm (défaut: 0.1).

        Returns:
            np.ndarray: Carte d'intensité 2D normalisée par l'énergie.
        """
        # Génération d'un spectre de Fourier aléatoire
        shape = (self.num_points, self.num_points)
        spectrum = np.random.uniform(
            low=min_amplitude,
            high=max_amplitude,
            size=shape,
        ) + 1j * np.random.uniform(
            low=min_amplitude,
            high=max_amplitude,
            size=shape,
        )

        # Appliquer un filtre de fréquence
        freq_x = np.fft.fftfreq(shape[0], d=self.diameter_mm / shape[0])
        freq_y = np.fft.fftfreq(shape[1], d=self.diameter_mm / shape[1])
        freq_xx, freq_yy = np.meshgrid(freq_x, freq_y, indexing='ij')
        freq_magnitude = np.sqrt(freq_xx**2 + freq_yy**2)

        # Masque pour les fréquences en dehors de la plage spécifiée
        mask = (freq_magnitude >= min_frequency) & (freq_magnitude <= max_frequency)
        spectrum[~mask] = 0

        # Transformée de Fourier inverse pour obtenir l'intensité spatiale
        intensity = np.abs(np.fft.ifft2(spectrum))**2

        # Normalisation par l'énergie
        intensity = intensity / np.sum(intensity) * self.energy
        return intensity

    def _generate_gaussian_intensity(
        self,
        sigma_mm: float = 2.0,
    ) -> np.ndarray:
        """
        FR: Génère une intensité gaussienne.
            
        EN: Generates a Gaussian intensity.

        Args:
            sigma_mm (float): Écart-type de la gaussienne en mm (défaut: 2.0).

        Returns:
            np.ndarray: Carte d'intensité 2D normalisée par l'énergie.

        Formula:
            I(x,y) = exp(-(x² + y²) / (2σ²))
        """
        intensity = np.exp(-(self.grid_x**2 + self.grid_y**2) / (2 * sigma_mm**2))
        intensity = intensity / np.sum(intensity) * self.energy
        return intensity

    def _generate_supergaussian_intensity(
        self,
        sigma_mm: float = 2.0,
        n: int = 4,
    ) -> np.ndarray:
        """
        FR: Génère une intensité super-gaussienne.
            
        EN: Generates a super-Gaussian intensity.

        Args:
            sigma_mm (float): Écart-type en mm (défaut: 2.0).
            n (int): Ordre de la super-gaussienne (défaut: 4).

        Returns:
            np.ndarray: Carte d'intensité 2D normalisée par l'énergie.

        Formula:
            I(x,y) = exp(-(x² + y²)^n / (2σ²))
        """
        intensity = np.exp(-((self.grid_x**2 + self.grid_y**2) ** n) / (2 * sigma_mm**2))
        intensity = intensity / np.sum(intensity) * self.energy
        return intensity

    def _generate_tophat_intensity(
        self,
        radius_mm: Optional[float] = None,
    ) -> np.ndarray:
        """
        FR: Génère une intensité top-hat (uniforme dans un cercle).
            
        EN: Generates a top-hat intensity (uniform within a circle).

        Args:
            radius_mm (float, optional): Rayon du cercle en mm. Si None, utilise la moitié du diamètre du faisceau.

        Returns:
            np.ndarray: Carte d'intensité 2D normalisée par l'énergie.
        """
        radius_mm = radius_mm if radius_mm is not None else self.diameter_mm / 2
        r = np.sqrt(self.grid_x**2 + self.grid_y**2)
        intensity = np.where(r <= radius_mm, 1.0, 0.0)
        intensity = intensity / np.sum(intensity) * self.energy
        return intensity

    # =============================================
    # FR: Méthodes de génération de phase
    # EN: Phase generation methods
    # =============================================

    def generate_phase(
        self,
        method: str = "random_zernike",
        **kwargs,
    ) -> np.ndarray:
        """
        FR: Génère une carte de phase selon la méthode spécifiée.
            
        EN: Generates a phase map according to the specified method.

        Args:
            method (str): Méthode de génération. Options:
                - "random_zernike": Phase aléatoire comme somme de modes de Zernike.
                - "random_legendre": Phase aléatoire comme somme de modes de Legendre.
                - "random_frequencies": Phase aléatoire avec fréquences et amplitudes contrôlées.
                - "sum_zernike": Somme de modes de Zernike choisis.
                - "sum_legendre": Somme de modes de Legendre choisis.
                - "from_file": Import depuis un fichier (non implémenté ici).
            **kwargs: Arguments spécifiques à la méthode.

        Returns:
            np.ndarray: Carte de phase 2D en nm.

        Raises:
            ValueError: Si la méthode est inconnue.
        """
        if method == "random_zernike":
            return self._generate_random_zernike_phase(**kwargs)
        elif method == "random_legendre":
            return self._generate_random_legendre_phase(**kwargs)
        elif method == "random_frequencies":
            return self._generate_random_frequencies_phase(**kwargs)
        elif method == "sum_zernike":
            return self._generate_sum_zernike_phase(**kwargs)
        elif method == "sum_legendre":
            return self._generate_sum_legendre_phase(**kwargs)
        elif method == "from_file":
            raise NotImplementedError("L'import depuis un fichier n'est pas encore implémenté.")
        else:
            raise ValueError(f"Méthode inconnue pour la génération de phase: {method}")

    def _generate_random_zernike_phase(
        self,
        n_modes: int = 10,
        ordination: str = "Noll",
        max_amplitude_nm: float = 100.0,
        normalization: str = "RMS",
    ) -> np.ndarray:
        """
        FR: Génère une phase aléatoire comme somme de modes de Zernike.
            Chaque mode a une amplitude aléatoire entre -max_amplitude_nm et +max_amplitude_nm.
            
        EN: Generates a random phase as a sum of Zernike modes.
            Each mode has a random amplitude between -max_amplitude_nm and +max_amplitude_nm.

        Args:
            n_modes (int): Nombre de modes de Zernike (défaut: 10).
            ordination (str): Type d'ordination, "Noll" ou "Wyant" (défaut: "Noll").
            max_amplitude_nm (float): Amplitude maximale en nm (défaut: 100.0).
            normalization (str): Normalisation, "PV" ou "RMS" (défaut: "RMS").

        Returns:
            np.ndarray: Carte de phase 2D en nm.
        """
        zernike_modes = generate_zernike_modes(
            n_modes, ordination, grid_x=self.grid_x, grid_y=self.grid_y
        )
        coefficients = np.random.uniform(-max_amplitude_nm, max_amplitude_nm, n_modes)
        phase = np.sum(coefficients[:, np.newaxis, np.newaxis] * zernike_modes, axis=0)
        return normalize_phase(phase, normalization, target_value=1.0, wavelength_nm=self.wavelength_nm)

    def _generate_random_legendre_phase(
        self,
        n_modes: int = 10,
        max_amplitude_nm: float = 100.0,
        normalization: str = "RMS",
    ) -> np.ndarray:
        """
        FR: Génère une phase aléatoire comme somme de modes de Legendre.
            Chaque mode a une amplitude aléatoire entre -max_amplitude_nm et +max_amplitude_nm.
            
        EN: Generates a random phase as a sum of Legendre modes.
            Each mode has a random amplitude between -max_amplitude_nm and +max_amplitude_nm.

        Args:
            n_modes (int): Nombre de modes de Legendre (défaut: 10).
            max_amplitude_nm (float): Amplitude maximale en nm (défaut: 100.0).
            normalization (str): Normalisation, "PV" ou "RMS" (défaut: "RMS").

        Returns:
            np.ndarray: Carte de phase 2D en nm.
        """
        legendre_modes = generate_legendre_modes(n_modes, grid_x=self.grid_x, grid_y=self.grid_y)
        coefficients = np.random.uniform(-max_amplitude_nm, max_amplitude_nm, n_modes)
        phase = np.sum(coefficients[:, np.newaxis, np.newaxis] * legendre_modes, axis=0)
        return normalize_phase(phase, normalization, target_value=1.0, wavelength_nm=self.wavelength_nm)

    def _generate_random_frequencies_phase(
        self,
        min_amplitude: float = 0.1,
        max_amplitude: float = 1.0,
        min_frequency: float = 0.01,
        max_frequency: float = 0.1,
        normalization: str = "RMS",
    ) -> np.ndarray:
        """
        FR: Génère une phase aléatoire avec des fréquences et amplitudes contrôlées.
            Utilise un spectre de Fourier aléatoire pour simuler des variations spatiales.
            
        EN: Generates random phase with controlled frequencies and amplitudes.
            Uses a random Fourier spectrum to simulate spatial variations.

        Args:
            min_amplitude (float): Amplitude minimale en nm (défaut: 0.1).
            max_amplitude (float): Amplitude maximale en nm (défaut: 1.0).
            min_frequency (float): Fréquence spatiale minimale en 1/mm (défaut: 0.01).
            max_frequency (float): Fréquence spatiale maximale en 1/mm (défaut: 0.1).
            normalization (str): Normalisation, "PV" ou "RMS" (défaut: "RMS").

        Returns:
            np.ndarray: Carte de phase 2D en nm.
        """
        # Génération d'un spectre de Fourier aléatoire
        shape = (self.num_points, self.num_points)
        spectrum = np.random.uniform(
            low=min_amplitude,
            high=max_amplitude,
            size=shape,
        ) + 1j * np.random.uniform(
            low=min_amplitude,
            high=max_amplitude,
            size=shape,
        )

        # Appliquer un filtre de fréquence
        freq_x = np.fft.fftfreq(shape[0], d=self.diameter_mm / shape[0])
        freq_y = np.fft.fftfreq(shape[1], d=self.diameter_mm / shape[1])
        freq_xx, freq_yy = np.meshgrid(freq_x, freq_y, indexing='ij')
        freq_magnitude = np.sqrt(freq_xx**2 + freq_yy**2)

        mask = (freq_magnitude >= min_frequency) & (freq_magnitude <= max_frequency)
        spectrum[~mask] = 0

        # Transformée de Fourier inverse pour obtenir la phase spatiale
        phase = np.angle(np.fft.ifft2(spectrum))
        phase = rad_to_nm(phase, self.wavelength_nm)
        return normalize_phase(phase, normalization, target_value=1.0, wavelength_nm=self.wavelength_nm)

    def _generate_sum_zernike_phase(
        self,
        modes: list = None,
        coefficients_nm: Optional[np.ndarray] = None,
        normalization: str = "RMS",
    ) -> np.ndarray:
        """
        FR: Génère une phase comme somme de modes de Zernike choisis.
            
        EN: Generates a phase as a sum of selected Zernike modes.

        Args:
            modes (list): Liste des indices des modes de Zernike à inclure (défaut: [0, 1, 2, 3, 4]).
            coefficients_nm (np.ndarray, optional): Coefficients pour chaque mode en nm. Si None, générés aléatoirement.
            normalization (str): Normalisation, "PV" ou "RMS" (défaut: "RMS").

        Returns:
            np.ndarray: Carte de phase 2D en nm.
        """
        if modes is None:
            modes = [0, 1, 2, 3, 4]
        if coefficients_nm is None:
            coefficients_nm = np.random.uniform(-100.0, 100.0, len(modes))

        zernike_modes = generate_zernike_modes(
            max(modes) + 1, "Noll", grid_x=self.grid_x, grid_y=self.grid_y
        )
        phase = np.sum(
            [coefficients_nm[i] * zernike_modes[modes[i]] for i in range(len(modes))],
            axis=0,
        )
        return normalize_phase(phase, normalization, target_value=1.0, wavelength_nm=self.wavelength_nm)

    def _generate_sum_legendre_phase(
        self,
        modes: list = None,
        coefficients_nm: Optional[np.ndarray] = None,
        normalization: str = "RMS",
    ) -> np.ndarray:
        """
        FR: Génère une phase comme somme de modes de Legendre choisis.
            
        EN: Generates a phase as a sum of selected Legendre modes.

        Args:
            modes (list): Liste des indices des modes de Legendre à inclure (défaut: [0, 1, 2, 3, 4]).
            coefficients_nm (np.ndarray, optional): Coefficients pour chaque mode en nm. Si None, générés aléatoirement.
            normalization (str): Normalisation, "PV" ou "RMS" (défaut: "RMS").

        Returns:
            np.ndarray: Carte de phase 2D en nm.
        """
        if modes is None:
            modes = [0, 1, 2, 3, 4]
        if coefficients_nm is None:
            coefficients_nm = np.random.uniform(-100.0, 100.0, len(modes))

        legendre_modes = generate_legendre_modes(
            max(modes) + 1, grid_x=self.grid_x, grid_y=self.grid_y
        )
        phase = np.sum(
            [coefficients_nm[i] * legendre_modes[modes[i]] for i in range(len(modes))],
            axis=0,
        )
        return normalize_phase(phase, normalization, target_value=1.0, wavelength_nm=self.wavelength_nm)

    # =============================================
    # FR: Méthodes de génération du champ électrique
    # EN: Electric field generation methods
    # =============================================

    def generate_electric_field(
        self,
        intensity: Optional[np.ndarray] = None,
        phase: Optional[np.ndarray] = None,
        method: str = "from_intensity_phase",
        **kwargs,
    ) -> np.ndarray:
        """
        FR: Génère le champ électrique complexe.
            
        EN: Generates the complex electric field.

        Args:
            intensity (np.ndarray, optional): Carte d'intensité 2D. Si None, générée avec self.generate_intensity().
            phase (np.ndarray, optional): Carte de phase 2D en nm. Si None, générée avec self.generate_phase().
            method (str): Méthode de génération. Options:
                - "from_intensity_phase": E = sqrt(I) * exp(1j * phase).
                - "gaussian": Champ électrique gaussien.
                - "laguerre_gauss": Champ électrique Laguerre-Gauss.
                - "plane_wave": Onde plane.
            **kwargs: Arguments spécifiques à la méthode.

        Returns:
            np.ndarray: Champ électrique complexe 2D.

        Raises:
            ValueError: Si la méthode est inconnue.
        """
        if method == "from_intensity_phase":
            return self._generate_electric_field_from_intensity_phase(intensity, phase)
        elif method == "gaussian":
            return self._generate_gaussian_electric_field(**kwargs)
        elif method == "laguerre_gauss":
            return self._generate_laguerre_gauss_electric_field(**kwargs)
        elif method == "plane_wave":
            return self._generate_plane_wave_electric_field(**kwargs)
        else:
            raise ValueError(f"Méthode inconnue pour la génération du champ électrique: {method}")

    def _generate_electric_field_from_intensity_phase(
        self,
        intensity: Optional[np.ndarray] = None,
        phase: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        FR: Génère le champ électrique à partir de l'intensité et de la phase.
            
        EN: Generates the electric field from intensity and phase.

        Args:
            intensity (np.ndarray, optional): Carte d'intensité 2D.
            phase (np.ndarray, optional): Carte de phase 2D en nm.

        Returns:
            np.ndarray: Champ électrique complexe 2D.

        Formula:
            E(x,y) = sqrt(I(x,y)) * exp(1j * phase(x,y) * 2π / wavelength_nm)
        """
        if intensity is None:
            intensity = self.intensity if self.intensity is not None else self.generate_intensity()
        if phase is None:
            phase = self.phase if self.phase is not None else self.generate_phase()

        phase_rad = nm_to_rad(phase, self.wavelength_nm)
        return np.sqrt(intensity) * np.exp(1j * phase_rad)

    def _generate_gaussian_electric_field(
        self,
        sigma_mm: float = 2.0,
        **kwargs,
    ) -> np.ndarray:
        """
        FR: Génère un champ électrique gaussien.
            
        EN: Generates a Gaussian electric field.

        Args:
            sigma_mm (float): Écart-type en mm (défaut: 2.0).
            **kwargs: Arguments supplémentaires pour la phase.

        Returns:
            np.ndarray: Champ électrique complexe 2D.

        Formula:
            E(x,y) = exp(-(x² + y²) / (4σ²)) * exp(1j * k * z)
        """
        amplitude = np.exp(-(self.grid_x**2 + self.grid_y**2) / (4 * sigma_mm**2))
        phase = self.generate_phase(method="random_zernike", **kwargs)
        phase_rad = nm_to_rad(phase, self.wavelength_nm)
        return amplitude * np.exp(1j * phase_rad)

    def _generate_laguerre_gauss_electric_field(
        self,
        p: int = 0,
        l: int = 0,
        sigma_mm: float = 2.0,
        **kwargs,
    ) -> np.ndarray:
        """
        FR: Génère un champ électrique Laguerre-Gauss.
            
        EN: Generates a Laguerre-Gauss electric field.

        Args:
            p (int): Ordre radial (défaut: 0).
            l (int): Ordre azimutal (défaut: 0).
            sigma_mm (float): Écart-type en mm (défaut: 2.0).
            **kwargs: Arguments supplémentaires pour la phase.

        Returns:
            np.ndarray: Champ électrique complexe 2D.

        Formula:
            E(x,y) = L_p^l(r) * exp(-r² / (2σ²)) * exp(1j * l * θ) * exp(1j * k * z)
            où L_p^l est le polynôme de Laguerre généralisé.
        """
        r = np.sqrt(self.grid_x**2 + self.grid_y**2)
        theta = np.arctan2(self.grid_y, self.grid_x)

        # Polynôme de Laguerre généralisé
        laguerre = eval_genlaguerre(p, l, r**2 / sigma_mm**2)

        # Amplitude Laguerre-Gauss
        amplitude = laguerre * np.exp(-r**2 / (2 * sigma_mm**2))

        # Phase
        phase = self.generate_phase(method="random_zernike", **kwargs)
        phase_rad = nm_to_rad(phase, self.wavelength_nm)

        return amplitude * np.exp(1j * l * theta) * np.exp(1j * phase_rad)

    def _generate_plane_wave_electric_field(
        self,
        angle_deg: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
        """
        FR: Génère un champ électrique d'onde plane.
            
        EN: Generates a plane wave electric field.

        Args:
            angle_deg (float): Angle d'inclinaison en degrés (défaut: 0.0).
            **kwargs: Arguments supplémentaires pour la phase.

        Returns:
            np.ndarray: Champ électrique complexe 2D.

        Formula:
            E(x,y) = exp(1j * k * (x * sin(θ) + y * cos(θ)))
        """
        angle_rad = np.deg2rad(angle_deg)
        k = 2 * np.pi / (self.wavelength_nm * 1e-6)  # Nombre d'onde en mm⁻¹
        phase_spatial = k * (self.grid_x * np.sin(angle_rad) + self.grid_y * np.cos(angle_rad))

        # Phase supplémentaire (aberration)
        phase_aberration = self.generate_phase(method="random_zernike", **kwargs)
        phase_aberration_rad = nm_to_rad(phase_aberration, self.wavelength_nm)

        return np.exp(1j * (phase_spatial + phase_aberration_rad))

    # =============================================
    # FR: Méthodes utilitaires
    # EN: Utility methods
    # =============================================

    def compute_pv_rms(self, data: np.ndarray) -> Tuple[float, float]:
        """
        FR: Calcule les valeurs PV (Peak-to-Valley) et RMS (Root Mean Square) d'une carte 2D.
            
        EN: Computes PV (Peak-to-Valley) and RMS (Root Mean Square) values of a 2D map.

        Args:
            data (np.ndarray): Carte 2D (intensité, phase, etc.).

        Returns:
            Tuple[float, float]: (PV, RMS).
        """
        pv = np.max(data) - np.min(data)
        rms = np.sqrt(np.mean(data**2))
        return pv, rms

    def plot(
        self,
        data: np.ndarray,
        title: str = "",
        cmap: str = "viridis",
        show_colorbar: bool = True,
    ):
        """
        FR: Affiche une carte 2D (intensité, phase, etc.) avec une échelle et une barre de couleur.
            
        EN: Displays a 2D map (intensity, phase, etc.) with a scale and colorbar.

        Args:
            data (np.ndarray): Carte 2D à afficher.
            title (str): Titre du graphique.
            cmap (str): Colormap à utiliser (défaut: "viridis").
            show_colorbar (bool): Si True, affiche la barre de couleur (défaut: True).
        """
        import matplotlib.pyplot as plt

        pv, rms = self.compute_pv_rms(data)
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap=cmap, extent=[
            -self.diameter_mm / 2,
            self.diameter_mm / 2,
            -self.diameter_mm / 2,
            self.diameter_mm / 2,
        ])
        if show_colorbar:
            plt.colorbar(label=f"PV: {pv:.2f}, RMS: {rms:.2f}")
        plt.title(title)
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.show()


# =============================================
# FR: Tests unitaires pour la classe Beam
# EN: Unit tests for the Beam class
# =============================================

class TestBeam(unittest.TestCase):
    """FR: Tests unitaires pour la classe Beam."""

    def setUp(self):
        self.beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=10.0,
            energy=1.0,
            num_points=128,  # Réduit pour les tests
        )

    def test_generate_gaussian_intensity(self):
        intensity = self.beam.generate_intensity(method="gaussian", sigma_mm=2.0)
        self.assertEqual(intensity.shape, (128, 128))
        self.assertAlmostEqual(np.sum(intensity), self.beam.energy, places=5)

    def test_generate_tophat_intensity(self):
        intensity = self.beam.generate_intensity(method="tophat")
        self.assertEqual(intensity.shape, (128, 128))
        self.assertAlmostEqual(np.sum(intensity), self.beam.energy, places=5)

    def test_generate_random_zernike_phase(self):
        phase = self.beam.generate_phase(method="random_zernike", n_modes=5, max_amplitude_nm=100.0)
        self.assertEqual(phase.shape, (128, 128))
        pv, rms = self.beam.compute_pv_rms(phase)
        self.assertLessEqual(rms, 100.0)

    def test_generate_random_legendre_phase(self):
        phase = self.beam.generate_phase(method="random_legendre", n_modes=5, max_amplitude_nm=100.0)
        self.assertEqual(phase.shape, (128, 128))

    def test_generate_electric_field_from_intensity_phase(self):
        intensity = self.beam.generate_intensity(method="gaussian")
        phase = self.beam.generate_phase(method="random_zernike", n_modes=5)
        electric_field = self.beam.generate_electric_field(
            intensity=intensity,
            phase=phase,
            method="from_intensity_phase",
        )
        self.assertEqual(electric_field.shape, (128, 128))
        self.assertEqual(electric_field.dtype, np.complex128)

    def test_generate_gaussian_electric_field(self):
        electric_field = self.beam.generate_electric_field(method="gaussian", sigma_mm=2.0)
        self.assertEqual(electric_field.shape, (128, 128))
        self.assertEqual(electric_field.dtype, np.complex128)

    def test_generate_plane_wave_electric_field(self):
        electric_field = self.beam.generate_electric_field(method="plane_wave", angle_deg=10.0)
        self.assertEqual(electric_field.shape, (128, 128))
        self.assertEqual(electric_field.dtype, np.complex128)

    def test_compute_pv_rms(self):
        data = np.random.rand(10, 10) * 100
        pv, rms = self.beam.compute_pv_rms(data)
        self.assertAlmostEqual(pv, np.max(data) - np.min(data), places=5)
        self.assertAlmostEqual(rms, np.sqrt(np.mean(data**2)), places=5)


if __name__ == "__main__":
    unittest.main()
