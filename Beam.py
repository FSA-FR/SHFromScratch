"""
Beam.py
FR: Module pour la génération et la gestion des faisceaux optiques.
    Permet la génération de faisceaux avec différentes méthodes d'intensité, de phase et de champ électrique,
    en supportant les unités d'énergie (J, mJ, a.u.), de puissance (W, mW, a.u.), et d'intensité (W/m², W/cm², a.u.).

EN: Module for generating and managing optical beams.
    Allows generation of beams with different intensity, phase, and electric field methods,
    supporting energy units (J, mJ, a.u.), power units (W, mW, a.u.), and intensity units (W/m², W/cm², a.u.).

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import logging
from typing import Optional, Tuple, Union
from MathAndPhysicsTools import (
    generate_zernike_modes,
    generate_legendre_modes,
    generate_hermite_gauss_modes,
    generate_laguerre_gauss_modes,
    normalize_phase,
    nm_to_rad,
    rad_to_nm,
    create_grid,
    load_data_from_file,
    compute_pv_rms,
    # Conversions d'unités
    J_to_mJ, mJ_to_J,
    W_to_mW, mW_to_W,
    W_m2_to_W_cm2, W_cm2_to_W_m2,
    energy_to_power, power_to_energy,
    power_to_intensity, intensity_to_power,
    get_area_mm2,
)


class Beam:
    """
    FR: Classe représentant un faisceau optique.
        Supporte les unités d'énergie (J, mJ, a.u.), de puissance (W, mW, a.u.), et d'intensité (W/m², W/cm², a.u.).

    EN: Class representing an optical beam.
        Supports energy units (J, mJ, a.u.), power units (W, mW, a.u.), and intensity units (W/m², W/cm², a.u.).

    Attributes:
        wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
        diameter_mm (float): Diamètre du faisceau en mm (défaut: 10.0).
        energy (float): Énergie totale du faisceau (défaut: 1.0).
        energy_unit (str): Unité de l'énergie ("J", "mJ", ou "a.u."). Default: "a.u.".
        power (float): Puissance du faisceau (si spécifiée).
        power_unit (str): Unité de la puissance ("W", "mW", ou "a.u."). Default: "a.u.".
        intensity (np.ndarray): Carte d'intensité 2D.
        intensity_unit (str): Unité de l'intensité ("W/m2", "W/cm2", ou "a.u."). Default: "a.u.".
        phase (np.ndarray): Carte de phase 2D en nm.
        electric_field (np.ndarray): Champ électrique complexe 2D.
        grid_x (np.ndarray): Grille en x en mm.
        grid_y (np.ndarray): Grille en y en mm.
        pulse_duration_s (float): Durée de l'impulsion en secondes (pour conversion énergie/puissance). Default: 1e-9.
        logger (logging.Logger): Logger local pour le débogage.
    """

    def __init__(
        self,
        wavelength_nm: float = 633.0,
        diameter_mm: float = 10.0,
        energy: float = 1.0,
        energy_unit: str = "a.u.",
        power: Optional[float] = None,
        power_unit: str = "a.u.",
        intensity: Optional[np.ndarray] = None,
        intensity_unit: str = "a.u.",
        phase: Optional[np.ndarray] = None,
        electric_field: Optional[np.ndarray] = None,
        pulse_duration_s: float = 1e-9,  # 1 ns par défaut
        num_points: int = 512,
    ):
        """
        FR: Initialise un faisceau optique avec les paramètres par défaut.
            Si `energy` est spécifiée, elle est utilisée pour normaliser l'intensité.
            Si `power` est spécifiée, elle est convertie en énergie en utilisant `pulse_duration_s`.

        EN: Initializes an optical beam with default parameters.
            If `energy` is specified, it is used to normalize the intensity.
            If `power` is specified, it is converted to energy using `pulse_duration_s`.

        Args:
            wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
            diameter_mm (float): Diamètre du faisceau en mm (défaut: 10.0).
            energy (float): Énergie totale du faisceau (défaut: 1.0).
            energy_unit (str): Unité de l'énergie ("J", "mJ", ou "a.u."). Default: "a.u.".
            power (float, optional): Puissance du faisceau. Si spécifiée, remplace `energy`.
            power_unit (str): Unité de la puissance ("W", "mW", ou "a.u."). Default: "a.u.".
            intensity (np.ndarray, optional): Carte d'intensité 2D.
            intensity_unit (str): Unité de l'intensité ("W/m2", "W/cm2", ou "a.u."). Default: "a.u.".
            phase (np.ndarray, optional): Carte de phase 2D en nm.
            electric_field (np.ndarray, optional): Champ électrique complexe 2D.
            pulse_duration_s (float): Durée de l'impulsion en secondes (défaut: 1e-9).
            num_points (int): Nombre de points par dimension pour la grille (défaut: 512).

        Raises:
            ValueError: Si les unités spécifiées sont invalides.
        """
        self.wavelength_nm = wavelength_nm
        self.diameter_mm = diameter_mm
        self.pulse_duration_s = pulse_duration_s
        self.num_points = num_points
        self.grid_x, self.grid_y = create_grid(diameter_mm, num_points)

        # Validation des unités
        if energy_unit not in ["J", "mJ", "a.u."]:
            raise ValueError(f"Unité d'énergie invalide : {energy_unit}. Utilisez 'J', 'mJ', ou 'a.u.'.")
        if power_unit not in ["W", "mW", "a.u."]:
            raise ValueError(f"Unité de puissance invalide : {power_unit}. Utilisez 'W', 'mW', ou 'a.u.'.")
        if intensity_unit not in ["W/m2", "W/cm2", "a.u."]:
            raise ValueError(f"Unité d'intensité invalide : {intensity_unit}. Utilisez 'W/m2', 'W/cm2', ou 'a.u.'.")

        # Configuration des unités
        self.energy_unit = energy_unit
        self.power_unit = power_unit
        self.intensity_unit = intensity_unit

        # Si la puissance est spécifiée, convertir en énergie
        if power is not None:
            if power_unit != "a.u.":
                # Convertir la puissance en énergie (J)
                energy_J = power_to_energy(power, power_unit, pulse_duration_s, "J")
                # Convertir en l'unité d'énergie souhaitée
                if energy_unit == "J":
                    self.energy = energy_J
                elif energy_unit == "mJ":
                    self.energy = J_to_mJ(energy_J)
                else:  # a.u.
                    self.energy = energy_J  # On garde en Joules pour la normalisation
                self.power = power
            else:
                self.energy = energy
                self.power = None
        else:
            self.energy = energy
            self.power = None

        self.intensity = intensity
        self.phase = phase
        self.electric_field = electric_field

        # Configuration du logger local
        self.logger = logging.getLogger("Beam")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Log des paramètres d'initialisation
        energy_log = f"{self.energy} {self.energy_unit}" if self.energy_unit != "a.u." else f"{self.energy} (a.u.)"
        power_log = f", power={self.power} {self.power_unit}" if self.power is not None else ""
        self.logger.info(
            "Beam initialized with wavelength=%.1f nm, diameter=%.1f mm, energy=%s%s, intensity_unit=%s, pulse_duration=%.2e s",
            wavelength_nm, diameter_mm, energy_log, power_log, intensity_unit, pulse_duration_s
        )

    # =========================================================================
    # Méthodes de conversion d'unités / Unit Conversion Methods
    # =========================================================================

    def set_energy(
        self,
        energy: float,
        energy_unit: str = "a.u.",
    ) -> None:
        """
        FR: Définit l'énergie du faisceau et convertit en unités internes.

        EN: Sets the beam energy and converts to internal units.

        Args:
            energy (float): Énergie du faisceau.
            energy_unit (str): Unité de l'énergie ("J", "mJ", ou "a.u."). Default: "a.u.".

        Raises:
            ValueError: Si l'unité est invalide.
        """
        if energy_unit not in ["J", "mJ", "a.u."]:
            raise ValueError(f"Unité d'énergie invalide : {energy_unit}. Utilisez 'J', 'mJ', ou 'a.u.'.")

        self.energy_unit = energy_unit
        if energy_unit == "J":
            self.energy = energy
        elif energy_unit == "mJ":
            self.energy = mJ_to_J(energy)
        else:  # a.u.
            self.energy = energy

        self.logger.info("Energy set to %.4f %s", energy, energy_unit)

    def set_power(
        self,
        power: float,
        power_unit: str = "a.u.",
    ) -> None:
        """
        FR: Définit la puissance du faisceau et convertit en énergie en utilisant pulse_duration_s.

        EN: Sets the beam power and converts to energy using pulse_duration_s.

        Args:
            power (float): Puissance du faisceau.
            power_unit (str): Unité de la puissance ("W", "mW", ou "a.u."). Default: "a.u.".

        Raises:
            ValueError: Si l'unité est invalide.
        """
        if power_unit not in ["W", "mW", "a.u."]:
            raise ValueError(f"Unité de puissance invalide : {power_unit}. Utilisez 'W', 'mW', ou 'a.u.'.")

        self.power_unit = power_unit
        self.power = power

        # Convertir la puissance en énergie
        if power_unit != "a.u.":
            energy_J = power_to_energy(power, power_unit, self.pulse_duration_s, "J")
            if self.energy_unit == "J":
                self.energy = energy_J
            elif self.energy_unit == "mJ":
                self.energy = J_to_mJ(energy_J)
            else:  # a.u.
                self.energy = energy_J
        else:
            self.energy = power  # Cas a.u., on suppose que c'est compatible

        self.logger.info("Power set to %.4f %s, converted to energy %.4f %s",
                         power, power_unit, self.energy, self.energy_unit)

    def set_intensity_unit(self, intensity_unit: str) -> None:
        """
        FR: Définit l'unité d'intensité.

        EN: Sets the intensity unit.

        Args:
            intensity_unit (str): Unité de l'intensité ("W/m2", "W/cm2", ou "a.u.").

        Raises:
            ValueError: Si l'unité est invalide.
        """
        if intensity_unit not in ["W/m2", "W/cm2", "a.u."]:
            raise ValueError(f"Unité d'intensité invalide : {intensity_unit}. Utilisez 'W/m2', 'W/cm2', ou 'a.u.'.")
        self.intensity_unit = intensity_unit
        self.logger.info("Intensity unit set to %s", intensity_unit)

    def get_energy_in_unit(self, target_unit: str = "J") -> float:
        """
        FR: Récupère l'énergie dans l'unité souhaitée.

        EN: Gets the energy in the desired unit.

        Args:
            target_unit (str): Unité cible ("J", "mJ", ou "a.u."). Default: "J".

        Returns:
            float: Énergie dans l'unité cible.

        Raises:
            ValueError: Si l'unité cible est invalide.
        """
        if target_unit not in ["J", "mJ", "a.u."]:
            raise ValueError(f"Unité cible invalide : {target_unit}. Utilisez 'J', 'mJ', ou 'a.u.'.")

        if self.energy_unit == "J":
            energy_J = self.energy
        elif self.energy_unit == "mJ":
            energy_J = mJ_to_J(self.energy)
        else:  # a.u.
            energy_J = self.energy  # On suppose que a.u. = J pour la conversion

        if target_unit == "J":
            return energy_J
        elif target_unit == "mJ":
            return J_to_mJ(energy_J)
        else:  # a.u.
            return self.energy

    def get_power_in_unit(self, target_unit: str = "W") -> float:
        """
        FR: Récupère la puissance dans l'unité souhaitée.

        EN: Gets the power in the desired unit.

        Args:
            target_unit (str): Unité cible ("W", "mW", ou "a.u."). Default: "W".

        Returns:
            float: Puissance dans l'unité cible.

        Raises:
            ValueError: Si l'unité cible est invalide ou si la puissance n'est pas définie.
        """
        if self.power is None:
            # Calculer la puissance à partir de l'énergie
            energy_J = self.get_energy_in_unit("J")
            power_W = energy_J / self.pulse_duration_s
            if target_unit == "W":
                return power_W
            elif target_unit == "mW":
                return W_to_mW(power_W)
            elif target_unit == "a.u.":
                return power_W
            else:
                raise ValueError(f"Unité cible invalide : {target_unit}. Utilisez 'W', 'mW', ou 'a.u.'.")
        else:
            if self.power_unit == "W":
                power_W = self.power
            elif self.power_unit == "mW":
                power_W = mW_to_W(self.power)
            else:  # a.u.
                power_W = self.power

            if target_unit == "W":
                return power_W
            elif target_unit == "mW":
                return W_to_mW(power_W)
            else:  # a.u.
                return self.power

    def get_intensity_in_unit(self, target_unit: str = "W/m2") -> float:
        """
        FR: Récupère l'intensité moyenne dans l'unité souhaitée.

        EN: Gets the average intensity in the desired unit.

        Args:
            target_unit (str): Unité cible ("W/m2", "W/cm2", ou "a.u."). Default: "W/m2".

        Returns:
            float: Intensité moyenne dans l'unité cible.

        Raises:
            ValueError: Si l'unité cible est invalide ou si l'intensité n'est pas définie.
        """
        if self.intensity is None:
            raise ValueError("L'intensité n'est pas définie. Générez-la d'abord avec generate_intensity().")

        if self.intensity_unit == "a.u.":
            # Si l'intensité est en a.u., on la convertit à partir de l'énergie
            energy_J = self.get_energy_in_unit("J")
            power_W = energy_J / self.pulse_duration_s
            intensity_W_m2 = power_to_intensity(power_W, "W", self.diameter_mm, "W/m2")
        elif self.intensity_unit == "W/m2":
            intensity_W_m2 = np.mean(self.intensity)
        elif self.intensity_unit == "W/cm2":
            intensity_W_m2 = W_cm2_to_W_m2(np.mean(self.intensity))
        else:
            raise ValueError(f"Unité d'intensité actuelle invalide : {self.intensity_unit}.")

        if target_unit == "W/m2":
            return intensity_W_m2
        elif target_unit == "W/cm2":
            return W_m2_to_W_cm2(intensity_W_m2)
        else:  # a.u.
            return intensity_W_m2 * get_area_mm2(self.diameter_mm) * 1e-6  # Conversion en a.u.

    # =========================================================================
    # Génération d'Intensité / Intensity Generation
    # =========================================================================

    def generate_intensity(
        self,
        method: str = "gaussian",
        **kwargs,
    ) -> np.ndarray:
        """
        FR: Génère une carte d'intensité selon la méthode spécifiée.
            L'intensité est normalisée selon l'énergie ou la puissance spécifiée.

        EN: Generates an intensity map according to the specified method.
            The intensity is normalized according to the specified energy or power.

        Args:
            method (str): Méthode de génération. Options:
                - "random": Intensité aléatoire avec fréquences et amplitudes contrôlées.
                - "gaussian": Intensité gaussienne.
                - "supergaussian": Intensité super-gaussienne.
                - "tophat": Intensité top-hat (uniforme dans un cercle).
                - "from_file": Import depuis un fichier (txt ou csv).
            **kwargs: Arguments spécifiques à la méthode.

        Returns:
            np.ndarray: Carte d'intensité 2D.

        Raises:
            ValueError: Si la méthode est inconnue.
        """
        self.logger.info("Generating intensity with method: %s", method)
        if method == "random":
            return self._generate_random_intensity(**kwargs)
        elif method == "gaussian":
            return self._generate_gaussian_intensity(**kwargs)
        elif method == "supergaussian":
            return self._generate_supergaussian_intensity(**kwargs)
        elif method == "tophat":
            return self._generate_tophat_intensity(**kwargs)
        elif method == "from_file":
            return self._load_intensity_from_file(**kwargs)
        else:
            raise ValueError(f"Méthode inconnue pour l'intensité : {method}")

    def _normalize_intensity(self, intensity: np.ndarray) -> np.ndarray:
        """
        FR: Normalise l'intensité selon l'énergie ou la puissance.

        EN: Normalizes the intensity according to energy or power.

        Args:
            intensity (np.ndarray): Carte d'intensité non normalisée.

        Returns:
            np.ndarray: Carte d'intensité normalisée.
        """
        if self.energy_unit != "a.u." or self.power_unit != "a.u.":
            # Convertir l'énergie en Joules
            energy_J = self.get_energy_in_unit("J")
            # Calculer la puissance en Watts
            power_W = energy_J / self.pulse_duration_s
            # Calculer l'intensité moyenne en W/m²
            intensity_W_m2 = power_to_intensity(power_W, "W", self.diameter_mm, "W/m2")
            # Calculer la surface en pixels
            area_px = np.pi * (self.diameter_mm / 2)**2  # Approximation de la surface en mm²
            # Normaliser l'intensité pour que la somme corresponde à l'énergie
            # Intensité (W/m²) * surface (m²) * durée (s) = énergie (J)
            surface_m2 = get_area_mm2(self.diameter_mm) * 1e-6  # mm² → m²
            total_energy_J = intensity_W_m2 * surface_m2 * self.pulse_duration_s
            # Normaliser l'intensité pour que la somme = énergie_J
            intensity = intensity / np.sum(intensity) * (total_energy_J / (surface_m2 * self.pulse_duration_s))
        else:
            # Normalisation classique par l'énergie (a.u.)
            intensity = intensity / np.sum(intensity) * self.energy

        return intensity

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
            np.ndarray: Carte d'intensité 2D normalisée.
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

        # Masque pour les fréquences dans la plage spécifiée
        mask = (freq_magnitude >= min_frequency) & (freq_magnitude <= max_frequency)
        spectrum[~mask] = 0

        # Transformée de Fourier inverse pour obtenir l'intensité
        intensity = np.abs(np.fft.ifft2(spectrum))**2

        # Normalisation
        intensity = self._normalize_intensity(intensity)
        self.intensity = intensity
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
            np.ndarray: Carte d'intensité 2D normalisée.

        Formula:
            I(x,y) = exp(-(x² + y²) / (2σ²))
        """
        intensity = np.exp(-(self.grid_x**2 + self.grid_y**2) / (2 * sigma_mm**2))
        intensity = self._normalize_intensity(intensity)
        self.intensity = intensity
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
            np.ndarray: Carte d'intensité 2D normalisée.

        Formula:
            I(x,y) = exp(-(x² + y²)^n / (2σ²))
        """
        intensity = np.exp(-((self.grid_x**2 + self.grid_y**2) ** n) / (2 * sigma_mm**2))
        intensity = self._normalize_intensity(intensity)
        self.intensity = intensity
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
            np.ndarray: Carte d'intensité 2D normalisée.
        """
        radius_mm = radius_mm if radius_mm is not None else self.diameter_mm / 2
        r = np.sqrt(self.grid_x**2 + self.grid_y**2)
        intensity = np.where(r <= radius_mm, 1.0, 0.0)
        intensity = self._normalize_intensity(intensity)
        self.intensity = intensity
        return intensity

    def _load_intensity_from_file(
        self,
        file_path: str,
        delimiter: str = None,
    ) -> np.ndarray:
        """
        FR: Charge une carte d'intensité depuis un fichier (txt ou csv).
            La grille est automatiquement adaptée à la taille du fichier.

        EN: Loads an intensity map from a file (txt or csv).
            The grid is automatically adapted to the file size.

        Args:
            file_path (str): Chemin vers le fichier.
            delimiter (str, optional): Délimiteur pour les fichiers txt/csv. Default: None.

        Returns:
            np.ndarray: Carte d'intensité 2D normalisée.
        """
        data = load_data_from_file(file_path, delimiter)
        if data.shape != (self.num_points, self.num_points):
            self.logger.warning(
                "Resizing intensity map from %s to (%d, %d)",
                data.shape,
                self.num_points,
                self.num_points
            )
            from scipy.interpolate import griddata
            x_flat = np.linspace(-self.diameter_mm / 2, self.diameter_mm / 2, data.shape[1])
            y_flat = np.linspace(-self.diameter_mm / 2, self.diameter_mm / 2, data.shape[0])
            xx_flat, yy_flat = np.meshgrid(x_flat, y_flat, indexing='ij')
            points = np.column_stack((xx_flat.ravel(), yy_flat.ravel()))
            values = data.ravel()
            grid_points = np.column_stack((self.grid_x.ravel(), self.grid_y.ravel()))
            intensity = griddata(points, values, grid_points, method='cubic', fill_value=0.0).reshape(
                self.num_points, self.num_points
            )
        else:
            intensity = data

        # Normalisation
        intensity = self._normalize_intensity(intensity)
        self.intensity = intensity
        return intensity

    # =========================================================================
    # Génération de Phase / Phase Generation
    # =========================================================================

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
                - "from_file": Import depuis un fichier (txt ou csv).
            **kwargs: Arguments spécifiques à la méthode.

        Returns:
            np.ndarray: Carte de phase 2D en nm.

        Raises:
            ValueError: Si la méthode est inconnue.
        """
        self.logger.info("Generating phase with method: %s", method)
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
            return self._load_phase_from_file(**kwargs)
        else:
            raise ValueError(f"Méthode inconnue pour la phase : {method}")

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
            n_modes, ordination, self.grid_x, self.grid_y
        )
        coefficients = np.random.uniform(-max_amplitude_nm, max_amplitude_nm, n_modes)
        phase = np.sum(coefficients[:, np.newaxis, np.newaxis] * zernike_modes, axis=0)
        phase = normalize_phase(phase, normalization, target_value=1.0, wavelength_nm=self.wavelength_nm)
        self.phase = phase
        return phase

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
        legendre_modes = generate_legendre_modes(n_modes, self.grid_x, self.grid_y)
        coefficients = np.random.uniform(-max_amplitude_nm, max_amplitude_nm, n_modes)
        phase = np.sum(coefficients[:, np.newaxis, np.newaxis] * legendre_modes, axis=0)
        phase = normalize_phase(phase, normalization, target_value=1.0, wavelength_nm=self.wavelength_nm)
        self.phase = phase
        return phase

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

        # Transformée de Fourier inverse pour obtenir la phase
        phase = np.angle(np.fft.ifft2(spectrum))
        phase = rad_to_nm(phase, self.wavelength_nm)
        phase = normalize_phase(phase, normalization, target_value=1.0, wavelength_nm=self.wavelength_nm)
        self.phase = phase
        return phase

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
            max(modes) + 1, "Noll", self.grid_x, self.grid_y
        )
        phase = np.sum(
            [coefficients_nm[i] * zernike_modes[modes[i]] for i in range(len(modes))],
            axis=0,
        )
        phase = normalize_phase(phase, normalization, target_value=1.0, wavelength_nm=self.wavelength_nm)
        self.phase = phase
        return phase

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
            max(modes) + 1, self.grid_x, self.grid_y
        )
        phase = np.sum(
            [coefficients_nm[i] * legendre_modes[modes[i]] for i in range(len(modes))],
            axis=0,
        )
        phase = normalize_phase(phase, normalization, target_value=1.0, wavelength_nm=self.wavelength_nm)
        self.phase = phase
        return phase

    def _load_phase_from_file(
        self,
        file_path: str,
        delimiter: str = None,
    ) -> np.ndarray:
        """
        FR: Charge une carte de phase depuis un fichier (txt ou csv).
            La grille est automatiquement adaptée à la taille du fichier.

        EN: Loads a phase map from a file (txt or csv).
            The grid is automatically adapted to the file size.

        Args:
            file_path (str): Chemin vers le fichier.
            delimiter (str, optional): Délimiteur pour les fichiers txt/csv. Default: None.

        Returns:
            np.ndarray: Carte de phase 2D en nm.
        """
        data = load_data_from_file(file_path, delimiter)
        if data.shape != (self.num_points, self.num_points):
            self.logger.warning(
                "Resizing phase map from %s to (%d, %d)",
                data.shape,
                self.num_points,
                self.num_points
            )
            from scipy.interpolate import griddata
            x_flat = np.linspace(-self.diameter_mm / 2, self.diameter_mm / 2, data.shape[1])
            y_flat = np.linspace(-self.diameter_mm / 2, self.diameter_mm / 2, data.shape[0])
            xx_flat, yy_flat = np.meshgrid(x_flat, y_flat, indexing='ij')
            points = np.column_stack((xx_flat.ravel(), yy_flat.ravel()))
            values = data.ravel()
            grid_points = np.column_stack((self.grid_x.ravel(), self.grid_y.ravel()))
            phase = griddata(points, values, grid_points, method='cubic', fill_value=0.0).reshape(
                self.num_points, self.num_points
            )
        else:
            phase = data

        self.phase = phase
        return phase

    # =========================================================================
    # Génération de Champ Électrique / Electric Field Generation
    # =========================================================================

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
                - "hermite_gauss": Champ électrique Hermite-Gauss.
                - "laguerre_gauss": Champ électrique Laguerre-Gauss.
                - "plane_wave": Onde plane.
            **kwargs: Arguments spécifiques à la méthode.

        Returns:
            np.ndarray: Champ électrique complexe 2D.

        Raises:
            ValueError: Si la méthode est inconnue.
        """
        self.logger.info("Generating electric field with method: %s", method)
        if method == "from_intensity_phase":
            return self._generate_electric_field_from_intensity_phase(intensity, phase)
        elif method == "gaussian":
            return self._generate_gaussian_electric_field(**kwargs)
        elif method == "hermite_gauss":
            return self._generate_hermite_gauss_electric_field(**kwargs)
        elif method == "laguerre_gauss":
            return self._generate_laguerre_gauss_electric_field(**kwargs)
        elif method == "plane_wave":
            return self._generate_plane_wave_electric_field(**kwargs)
        else:
            raise ValueError(f"Méthode inconnue pour le champ électrique : {method}")

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
        electric_field = np.sqrt(intensity) * np.exp(1j * phase_rad)
        self.electric_field = electric_field
        return electric_field

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
        electric_field = amplitude * np.exp(1j * phase_rad)
        self.electric_field = electric_field
        return electric_field

    def _generate_hermite_gauss_electric_field(
        self,
        n: int = 0,
        m: int = 0,
        sigma_mm: float = 2.0,
        **kwargs,
    ) -> np.ndarray:
        """
        FR: Génère un champ électrique Hermite-Gauss.

        EN: Generates a Hermite-Gauss electric field.

        Args:
            n (int): Ordre radial (défaut: 0).
            m (int): Ordre azimutal (défaut: 0).
            sigma_mm (float): Écart-type en mm (défaut: 2.0).
            **kwargs: Arguments supplémentaires pour la phase.

        Returns:
            np.ndarray: Champ électrique complexe 2D.

        Formula:
            E(x,y) = H_n(x) * H_m(y) * exp(-(x² + y²) / (2σ²)) * exp(1j * phase)
        """
        hermite_modes = generate_hermite_gauss_modes(1, self.grid_x, self.grid_y, sigma_mm)
        phase = self.generate_phase(method="random_zernike", **kwargs)
        phase_rad = nm_to_rad(phase, self.wavelength_nm)
        electric_field = hermite_modes[0] * np.exp(1j * phase_rad)
        self.electric_field = electric_field
        return electric_field

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
            E(x,y) = L_p^l(r) * exp(-r² / (2σ²)) * exp(1j * l * θ) * exp(1j * phase)
        """
        laguerre_modes = generate_laguerre_gauss_modes(1, self.grid_x, self.grid_y, sigma_mm, p, abs(l))
        r = np.sqrt(self.grid_x**2 + self.grid_y**2)
        theta = np.arctan2(self.grid_y, self.grid_x)
        phase = self.generate_phase(method="random_zernike", **kwargs)
        phase_rad = nm_to_rad(phase, self.wavelength_nm)
        if l >= 0:
            angular = np.exp(1j * l * theta)
        else:
            angular = np.exp(-1j * abs(l) * theta)
        electric_field = laguerre_modes[0] * angular * np.exp(1j * phase_rad)
        self.electric_field = electric_field
        return electric_field

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
            E(x,y) = exp(1j * k * (x * sin(θ) + y * cos(θ))) * exp(1j * phase_aberration)
        """
        angle_rad = np.deg2rad(angle_deg)
        k = 2 * np.pi / (self.wavelength_nm * 1e-6)  # Nombre d'onde en mm⁻¹
        phase_spatial = k * (self.grid_x * np.sin(angle_rad) + self.grid_y * np.cos(angle_rad))

        # Phase supplémentaire (aberration)
        phase_aberration = self.generate_phase(method="random_zernike", **kwargs)
        phase_aberration_rad = nm_to_rad(phase_aberration, self.wavelength_nm)

        electric_field = np.exp(1j * (phase_spatial + phase_aberration_rad))
        self.electric_field = electric_field
        return electric_field

    # =========================================================================
    # Utilitaires / Utilities
    # =========================================================================

    def compute_pv_rms(self, data: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        FR: Calcule les valeurs PV (Peak-to-Valley) et RMS (Root Mean Square) d'une carte 2D.
            Si data est None, utilise self.phase.

        EN: Computes PV (Peak-to-Valley) and RMS (Root Mean Square) values of a 2D map.
            If data is None, uses self.phase.

        Args:
            data (np.ndarray, optional): Carte 2D (intensité, phase, etc.).

        Returns:
            Tuple[float, float]: (PV, RMS).
        """
        if data is None:
            if self.phase is None:
                self.phase = self.generate_phase()
            data = self.phase
        return compute_pv_rms(data)

    def plot(
        self,
        what: str = "intensity",
        **kwargs,
    ):
        """
        FR: Affiche une carte 2D (intensité, phase, etc.) avec une échelle et une barre de couleur.

        EN: Displays a 2D map (intensity, phase, etc.) with a scale and colorbar.

        Args:
            what (str): Ce qu'il faut afficher ("intensity", "phase", "electric_field").
            **kwargs: Arguments pour matplotlib.
        """
        import matplotlib.pyplot as plt

        if what == "intensity":
            if self.intensity is None:
                self.intensity = self.generate_intensity()
            data = self.intensity
            if self.intensity_unit != "a.u.":
                label = f"Intensity ({self.intensity_unit})"
            else:
                label = "Intensity (a.u.)"
            title = "Intensity Map"
        elif what == "phase":
            if self.phase is None:
                self.phase = self.generate_phase()
            data = self.phase
            label = "Phase (nm)"
            title = "Phase Map"
        elif what == "electric_field":
            if self.electric_field is None:
                self.electric_field = self.generate_electric_field()
            data = np.abs(self.electric_field)
            label = "Amplitude (a.u.)"
            title = "Electric Field Amplitude"
        else:
            raise ValueError(f"Unknown plot type: {what}")

        pv, rms = self.compute_pv_rms(data)
        plt.figure(figsize=(8, 6))
        plt.imshow(
            data,
            cmap=kwargs.get("cmap", "viridis"),
            extent=[
                -self.diameter_mm / 2,
                self.diameter_mm / 2,
                -self.diameter_mm / 2,
                self.diameter_mm / 2,
            ],
        )
        plt.colorbar(label=f"{label}\nPV: {pv:.2f}, RMS: {rms:.2f}")
        plt.title(title)
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.show()


# =============================================================================
# TESTS UNITAIRES / UNIT TESTS
# =============================================================================

class TestBeam:
    """
    FR: Classe de tests unitaires pour Beam.py.
    EN: Unit test class for Beam.py.
    """

    def setUp(self):
        self.beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=10.0,
            energy=1.0,
            energy_unit="a.u.",
            num_points=128,  # Réduit pour les tests
        )

    def test_generate_gaussian_intensity(self):
        intensity = self.beam.generate_intensity(method="gaussian", sigma_mm=2.0)
        self.assertEqual(intensity.shape, (128, 128))
        # Vérifier que la somme est proche de l'énergie (en a.u.)
        self.assertAlmostEqual(np.sum(intensity), self.beam.energy, places=5)

    def test_generate_tophat_intensity(self):
        intensity = self.beam.generate_intensity(method="tophat")
        self.assertEqual(intensity.shape, (128, 128))
        self.assertAlmostEqual(np.sum(intensity), self.beam.energy, places=5)

    def test_generate_random_intensity(self):
        intensity = self.beam.generate_intensity(
            method="random",
            min_amplitude=0.1,
            max_amplitude=1.0,
            min_frequency=0.01,
            max_frequency=0.1,
        )
        self.assertEqual(intensity.shape, (128, 128))
        self.assertAlmostEqual(np.sum(intensity), self.beam.energy, places=5)

    def test_generate_random_zernike_phase(self):
        phase = self.beam.generate_phase(method="random_zernike", n_modes=5, max_amplitude_nm=100.0)
        self.assertEqual(phase.shape, (128, 128))
        pv, rms = self.beam.compute_pv_rms(phase)
        self.assertLessEqual(rms, 633.0)

    def test_generate_random_legendre_phase(self):
        phase = self.beam.generate_phase(method="random_legendre", n_modes=5, max_amplitude_nm=100.0)
        self.assertEqual(phase.shape, (128, 128))

    def test_generate_random_frequencies_phase(self):
        phase = self.beam.generate_phase(
            method="random_frequencies",
            min_amplitude=0.1,
            max_amplitude=1.0,
            min_frequency=0.01,
            max_frequency=0.1,
        )
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

    def test_set_energy_J(self):
        self.beam.set_energy(0.001, "J")
        self.assertEqual(self.beam.energy_unit, "J")
        self.assertAlmostEqual(self.beam.energy, 0.001, places=10)

    def test_set_energy_mJ(self):
        self.beam.set_energy(1.0, "mJ")
        self.assertEqual(self.beam.energy_unit, "mJ")
        self.assertAlmostEqual(self.beam.energy, 1.0, places=10)

    def test_set_power_W(self):
        self.beam.set_power(1.0, "W")
        self.assertEqual(self.beam.power_unit, "W")
        self.assertAlmostEqual(self.beam.power, 1.0, places=10)

    def test_get_energy_in_unit(self):
        self.beam.set_energy(1.0, "mJ")
        energy_J = self.beam.get_energy_in_unit("J")
        self.assertAlmostEqual(energy_J, 0.001, places=10)

    def test_get_power_in_unit(self):
        self.beam.set_energy(1.0, "mJ")
        power_W = self.beam.get_power_in_unit("W")
        expected = 1.0 / 1e-9  # 1 mJ / 1 ns = 1e9 W
        self.assertAlmostEqual(power_W, expected, places=5)

    def test_set_intensity_unit(self):
        self.beam.set_intensity_unit("W/m2")
        self.assertEqual(self.beam.intensity_unit, "W/m2")

    def test_energy_to_power_conversion(self):
        # Créer un faisceau avec puissance en W
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=10.0,
            power=1.0,
            power_unit="W",
            pulse_duration_s=1e-3,  # 1 ms
            num_points=128,
        )
        # Vérifier que l'énergie est bien calculée
        energy_J = beam.get_energy_in_unit("J")
        expected_energy_J = 1.0 * 1e-3  # 1 W * 1 ms = 0.001 J
        self.assertAlmostEqual(energy_J, expected_energy_J, places=10)

    def test_power_to_intensity_conversion(self):
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=10.0,
            power=1.0,
            power_unit="W",
            num_points=128,
        )
        intensity_W_m2 = beam.get_intensity_in_unit("W/m2")
        # Surface = π * (5e-3 m)^2 = π * 25e-6 m²
        surface_m2 = np.pi * (5e-3)**2
        expected_intensity = 1.0 / surface_m2
        self.assertAlmostEqual(intensity_W_m2, expected_intensity, places=5)


if __name__ == "__main__":
    import unittest
    unittest.main()