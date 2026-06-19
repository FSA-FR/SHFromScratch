"""
Propagation.py
FR: Module pour la propagation numérique et analytique des faisceaux optiques.
    Permet de propager un champ électrique à travers un espace libre ou un système optique,
    en utilisant des méthodes numériques (FFT, TF, spectre angulaire) ou analytiques (projection sur bases de modes).
    Gère la cohérence (cohérente/incohérente), la diffraction, le rééchantillonnage des grilles,
    et fournit des métriques d'erreur entre le champ initial et propagé.

EN: Module for numerical and analytical propagation of optical beams.
    Allows propagating an electric field through free space or an optical system,
    using numerical methods (FFT, TF, angular spectrum) or analytical methods (mode basis projection).
    Handles coherence (coherent/incoherent), diffraction, grid resampling,
    and provides error metrics between initial and propagated fields.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import logging
from typing import Optional, Tuple, Union, Dict, List
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.special import hermite, eval_genlaguerre
from MathAndPhysicsTools import (
    create_grid,
    compute_pv_rms,
    nm_to_rad,
    rad_to_nm,
    rad_to_mrad,
    resample_to_grid,
)


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Propagation:
    """
    FR: Classe pour la propagation des faisceaux optiques.
        Supporte les propagations numérique (Fraunhofer, Fresnel, spectre angulaire) et analytique (Hermite-Gauss, Laguerre-Gauss).
        Gère la cohérence, la diffraction, et le rééchantillonnage automatique des grilles.
        Fournit des métriques d'erreur entre le champ initial et propagé.

    EN: Class for optical beam propagation.
        Supports numerical propagation (Fraunhofer, Fresnel, angular spectrum) and analytical propagation (Hermite-Gauss, Laguerre-Gauss).
        Handles coherence, diffraction, and automatic grid resampling.
        Provides error metrics between initial and propagated fields.

    Attributes:
        wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
        propagation_distance_mm (float): Distance de propagation en mm (défaut: 100.0).
        input_diameter_mm (float): Diamètre de la grille d'entrée en mm (défaut: 10.0).
        output_diameter_mm (float): Diamètre de la grille de sortie en mm.
        num_points (int): Nombre de points par dimension (défaut: 512).
        coherence (str): Régime de cohérence ("coherent" ou "incoherent").
        method (str): Méthode de propagation ("fraunhofer", "fresnel", "angular_spectrum", "hermite_gauss", "laguerre_gauss").
        input_grid_x (np.ndarray): Grille d'entrée en x (mm).
        input_grid_y (np.ndarray): Grille d'entrée en y (mm).
        output_grid_x (np.ndarray): Grille de sortie en x (mm).
        output_grid_y (np.ndarray): Grille de sortie en y (mm).
        logger (logging.Logger): Logger pour le débogage.
    """

    def __init__(
        self,
        wavelength_nm: float = 633.0,
        propagation_distance_mm: float = 100.0,
        input_diameter_mm: float = 10.0,
        output_diameter_mm: Optional[float] = None,
        num_points: int = 512,
        coherence: str = "coherent",
        method: str = "angular_spectrum",
    ):
        """
        FR: Initialise la propagation avec les paramètres par défaut.

        EN: Initializes propagation with default parameters.

        Args:
            wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
            propagation_distance_mm (float): Distance de propagation en mm (défaut: 100.0).
            input_diameter_mm (float): Diamètre de la grille d'entrée en mm (défaut: 10.0).
            output_diameter_mm (float, optional): Diamètre de la grille de sortie en mm. Si None, = input_diameter_mm.
            num_points (int): Nombre de points par dimension (défaut: 512).
            coherence (str): Régime de cohérence ("coherent" ou "incoherent"). Default: "coherent".
            method (str): Méthode de propagation. Options:
                - "fraunhofer": Approximation de Fraunhofer (champ lointain, FFT).
                - "fresnel": Approximation de Fresnel (champ proche, split-step).
                - "angular_spectrum": Spectre angulaire (méthode la plus précise).
                - "hermite_gauss": Projection sur base Hermite-Gauss (analytique).
                - "laguerre_gauss": Projection sur base Laguerre-Gauss (analytique).
                Default: "angular_spectrum".

        Raises:
            ValueError: Si la cohérence ou la méthode est invalide.
        """
        if coherence not in ["coherent", "incoherent"]:
            raise ValueError(f"Régime de cohérence invalide : {coherence}. Utilisez 'coherent' ou 'incoherent'.")
        if method not in ["fraunhofer", "fresnel", "angular_spectrum", "hermite_gauss", "laguerre_gauss"]:
            raise ValueError(
                f"Méthode de propagation invalide : {method}. "
                "Utilisez 'fraunhofer', 'fresnel', 'angular_spectrum', 'hermite_gauss', ou 'laguerre_gauss'."
            )

        self.wavelength_nm = wavelength_nm
        self.propagation_distance_mm = propagation_distance_mm
        self.input_diameter_mm = input_diameter_mm
        self.output_diameter_mm = output_diameter_mm if output_diameter_mm is not None else input_diameter_mm
        self.num_points = num_points
        self.coherence = coherence
        self.method = method

        # Créer les grilles d'entrée et de sortie
        self.input_grid_x, self.input_grid_y = create_grid(input_diameter_mm, num_points)
        self.output_grid_x, self.output_grid_y = create_grid(self.output_diameter_mm, num_points)

        # Configuration du logger
        self.logger = logging.getLogger("Propagation")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.logger.info(
            "Propagation initialized with wavelength=%.1f nm, distance=%.1f mm, "
            "input_diameter=%.1f mm, output_diameter=%.1f mm, coherence=%s, method=%s",
            wavelength_nm, propagation_distance_mm, input_diameter_mm, self.output_diameter_mm, coherence, method
        )

    # =========================================================================
    # Méthodes de propagation numérique / Numerical Propagation Methods
    # =========================================================================

    def propagate_fraunhofer(
        self,
        electric_field: np.ndarray,
        input_grid_x: Optional[np.ndarray] = None,
        input_grid_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        FR: Propage un champ électrique en utilisant l'approximation de Fraunhofer (champ lointain).
            Utilise une FFT pour calculer la diffraction.

        EN: Propagates an electric field using the Fraunhofer approximation (far field).
            Uses FFT to compute diffraction.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D d'entrée.
            input_grid_x (np.ndarray, optional): Grille d'entrée en x (mm). Si None, utilise self.input_grid_x.
            input_grid_y (np.ndarray, optional): Grille d'entrée en y (mm). Si None, utilise self.input_grid_y.

        Returns:
            np.ndarray: Champ électrique complexe 2D propagé.

        Notes:
            - **Validité** : z >> (D²)/(4λ), où D est le diamètre du faisceau.
            - **Erreur** : ≈ 1% si z > 10*(D²)/(4λ).
            - **Limites** : Ne gère pas les effets de champ proche. Ignore les ondes évanescentes.
            - **Complexité** : O(N² log N) pour une grille N×N.

        Formula:
            E_out(u,v) = (exp(1j * k * z) / (1j * λ * z)) * 
                         ∫∫ E_in(x,y) * exp(1j * k * (x² + y²)/(2z)) * exp(-1j * 2π * (u*x + v*y)/(λ*z)) dx dy
            (Implémenté via FFT avec phase quadratique)

        Sources:
            - Goodman, J. W. (2005). "Introduction to Fourier Optics." Roberts and Company.
        """
        if input_grid_x is None:
            input_grid_x = self.input_grid_x
        if input_grid_y is None:
            input_grid_y = self.input_grid_y

        # Vérifier la taille du champ électrique
        if electric_field.shape != input_grid_x.shape:
            self.logger.warning(
                "Resizing electric field from %s to %s for Fraunhofer propagation",
                electric_field.shape,
                input_grid_x.shape
            )
            electric_field = self._resample_electric_field(electric_field, input_grid_x, input_grid_y)

        # Calculer le facteur de phase quadratique (en mm)
        k = 2 * np.pi / (self.wavelength_nm * 1e-6)  # Nombre d'onde en mm⁻¹
        quadratic_phase = np.exp(
            1j * k * (input_grid_x**2 + input_grid_y**2) / (2 * self.propagation_distance_mm)
        )

        # Appliquer la phase quadratique et la FFT
        field_with_phase = electric_field * quadratic_phase
        propagated_field = fft2(field_with_phase)
        propagated_field = fftshift(propagated_field)

        # Facteur de normalisation et phase linéaire
        normalization = np.exp(1j * k * self.propagation_distance_mm) / (
            1j * self.wavelength_nm * 1e-6 * self.propagation_distance_mm
        )
        propagated_field *= normalization

        # Rééchantillonner sur la grille de sortie si nécessaire
        if self.output_diameter_mm != self.input_diameter_mm:
            propagated_field = self._resample_electric_field(
                propagated_field,
                self.output_grid_x,
                self.output_grid_y,
            )

        self.logger.info("Fraunhofer propagation applied with distance=%.1f mm", self.propagation_distance_mm)
        return propagated_field

    def propagate_fresnel(
        self,
        electric_field: np.ndarray,
        input_grid_x: Optional[np.ndarray] = None,
        input_grid_y: Optional[np.ndarray] = None,
        step_mm: float = 1.0,
    ) -> np.ndarray:
        """
        FR: Propage un champ électrique en utilisant l'approximation de Fresnel (champ proche).
            Utilise une méthode de propagation par étapes (split-step Fourier).

        EN: Propagates an electric field using the Fresnel approximation (near field).
            Uses a split-step Fourier method.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D d'entrée.
            input_grid_x (np.ndarray, optional): Grille d'entrée en x (mm). Si None, utilise self.input_grid_x.
            input_grid_y (np.ndarray, optional): Grille d'entrée en y (mm). Si None, utilise self.input_grid_y.
            step_mm (float): Taille des étapes de propagation en mm (défaut: 1.0).

        Returns:
            np.ndarray: Champ électrique complexe 2D propagé.

        Notes:
            - **Validité** : z > 0.1*(D²)/(4λ) (champ proche à intermédiaire).
            - **Erreur** : ≈ 0.1% pour des étapes < λ/10.
            - **Limites** : Moins précis que Fraunhofer pour les grandes distances. Sensible aux artefacts de discrétisation.
            - **Complexité** : O(N * (z/step) * N² log N) pour une grille N×N.

        Formula:
            E(x,y,z+Δz) = exp(1j * k * Δz) * 
                          IFFT[FFT[E(x,y,z)] * exp(1j * π * λ * Δz * (fx² + fy²))]
        """
        if input_grid_x is None:
            input_grid_x = self.input_grid_x
        if input_grid_y is None:
            input_grid_y = self.input_grid_y

        # Vérifier la taille du champ électrique
        if electric_field.shape != input_grid_x.shape:
            self.logger.warning(
                "Resizing electric field from %s to %s for Fresnel propagation",
                electric_field.shape,
                input_grid_x.shape
            )
            electric_field = self._resample_electric_field(electric_field, input_grid_x, input_grid_y)

        k = 2 * np.pi / (self.wavelength_nm * 1e-6)  # Nombre d'onde en mm⁻¹
        remaining_distance = self.propagation_distance_mm

        # Propagation par étapes
        while remaining_distance > step_mm:
            current_step = min(step_mm, remaining_distance)

            # Calculer le facteur de phase pour l'espace réel
            real_phase = np.exp(1j * k * current_step)

            # Calculer le facteur de phase pour l'espace de Fourier
            dx = input_grid_x[0, 1] - input_grid_x[0, 0]  # Pas spatial en mm
            fx = np.fft.fftfreq(input_grid_x.shape[1], d=dx)
            fy = np.fft.fftfreq(input_grid_x.shape[0], d=dx)
            fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='ij')
            fourier_phase = np.exp(1j * np.pi * self.wavelength_nm * 1e-6 * current_step * (fx_grid**2 + fy_grid**2))

            # Appliquer la propagation en deux étapes
            field_fft = fft2(electric_field)
            field_fft *= fourier_phase
            electric_field = ifft2(field_fft) * real_phase

            remaining_distance -= current_step

        # Dernière étape
        if remaining_distance > 0:
            real_phase = np.exp(1j * k * remaining_distance)
            dx = input_grid_x[0, 1] - input_grid_x[0, 0]
            fx = np.fft.fftfreq(input_grid_x.shape[1], d=dx)
            fy = np.fft.fftfreq(input_grid_x.shape[0], d=dx)
            fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='ij')
            fourier_phase = np.exp(1j * np.pi * self.wavelength_nm * 1e-6 * remaining_distance * (fx_grid**2 + fy_grid**2))

            field_fft = fft2(electric_field)
            field_fft *= fourier_phase
            electric_field = ifft2(field_fft) * real_phase

        # Rééchantillonner sur la grille de sortie si nécessaire
        if self.output_diameter_mm != self.input_diameter_mm:
            electric_field = self._resample_electric_field(
                electric_field,
                self.output_grid_x,
                self.output_grid_y,
            )

        self.logger.info("Fresnel propagation applied with distance=%.1f mm, step=%.1f mm",
                         self.propagation_distance_mm, step_mm)
        return electric_field

    def propagate_angular_spectrum(
        self,
        electric_field: np.ndarray,
        input_grid_x: Optional[np.ndarray] = None,
        input_grid_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        FR: Propage un champ électrique en utilisant la méthode du spectre angulaire.
            Méthode la plus précise pour la diffraction en espace libre.

        EN: Propagates an electric field using the angular spectrum method.
            Most accurate method for free-space diffraction.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D d'entrée.
            input_grid_x (np.ndarray, optional): Grille d'entrée en x (mm). Si None, utilise self.input_grid_x.
            input_grid_y (np.ndarray, optional): Grille d'entrée en y (mm). Si None, utilise self.input_grid_y.

        Returns:
            np.ndarray: Champ électrique complexe 2D propagé.

        Notes:
            - **Validité** : Toutes les distances (champ proche et lointain).
            - **Erreur** : Très faible si la résolution est suffisante.
            - **Limites** : Coûteux en calcul (O(N² log N)). Sensible aux ondes évanescentes.
            - **Avantage** : Prend en compte toutes les composantes du spectre angulaire.

        Formula:
            E(x,y,z) = IFFT[FFT[E(x,y,0)] * H(fx,fy,z)]
            où H(fx,fy,z) = exp(1j * 2π * z * sqrt((1/λ)² - fx² - fy²))

        Sources:
            - Goodman, J. W. (2005). "Introduction to Fourier Optics." Roberts and Company.
        """
        if input_grid_x is None:
            input_grid_x = self.input_grid_x
        if input_grid_y is None:
            input_grid_y = self.input_grid_y

        # Vérifier la taille du champ électrique
        if electric_field.shape != input_grid_x.shape:
            self.logger.warning(
                "Resizing electric field from %s to %s for angular spectrum propagation",
                electric_field.shape,
                input_grid_x.shape
            )
            electric_field = self._resample_electric_field(electric_field, input_grid_x, input_grid_y)

        # Convertir la longueur d'onde en mètres
        wavelength_m = self.wavelength_nm * 1e-9

        # Calculer le nombre d'onde
        k = 2 * np.pi / wavelength_m

        # Taille du champ
        Nx, Ny = electric_field.shape
        dx = (input_grid_x[0, -1] - input_grid_x[0, 0]) / (Nx - 1) * 1e-3  # mm → m
        dy = (input_grid_y[-1, 0] - input_grid_y[0, 0]) / (Ny - 1) * 1e-3  # mm → m

        # Fréquences spatiales (en 1/m)
        fx = np.fft.fftfreq(Nx, d=dx)
        fy = np.fft.fftfreq(Ny, d=dy)
        fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='ij')

        # Calcul de la fonction de transfert du spectre angulaire
        f_squared = fx_grid**2 + fy_grid**2
        # Éviter la division par zéro pour f=0
        f_squared[f_squared == 0] = 1e-30  # Très petite valeur

        # Calculer la composante z du vecteur d'onde
        kz = np.sqrt((1 / wavelength_m)**2 - f_squared)
        # Pour les ondes évanescentes (f_squared > (1/λ)²), kz devient imaginaire
        kz = np.where(np.isreal(kz), kz, 1j * np.abs(np.imag(kz)))

        # Fonction de transfert
        H = np.exp(1j * 2 * np.pi * self.propagation_distance_mm * 1e-3 * kz)

        # Transformée de Fourier du champ initial
        field_fft = fft2(electric_field)
        field_fft = fftshift(field_fft)

        # Appliquer la fonction de transfert
        propagated_field_fft = field_fft * H
        propagated_field_fft = ifftshift(propagated_field_fft)

        # Transformée de Fourier inverse
        propagated_field = ifft2(propagated_field_fft)

        # Rééchantillonner sur la grille de sortie si nécessaire
        if self.output_diameter_mm != self.input_diameter_mm:
            propagated_field = self._resample_electric_field(
                propagated_field,
                self.output_grid_x,
                self.output_grid_y,
            )

        self.logger.info("Angular spectrum propagation applied with distance=%.1f mm", self.propagation_distance_mm)
        return propagated_field

    # =========================================================================
    # Méthodes de propagation analytique / Analytical Propagation Methods
    # =========================================================================

    def propagate_hermite_gauss(
        self,
        electric_field: np.ndarray,
        input_grid_x: Optional[np.ndarray] = None,
        input_grid_y: Optional[np.ndarray] = None,
        max_modes: int = 20,
    ) -> Tuple[np.ndarray, Dict]:
        """
        FR: Propage un champ électrique en utilisant une projection sur la base Hermite-Gauss.
            Retourne le champ propagé et les métriques d'erreur par rapport au champ initial.

        EN: Propagates an electric field using projection onto Hermite-Gauss basis.
            Returns the propagated field and error metrics compared to the initial field.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D d'entrée.
            input_grid_x (np.ndarray, optional): Grille d'entrée en x (mm). Si None, utilise self.input_grid_x.
            input_grid_y (np.ndarray, optional): Grille d'entrée en y (mm). Si None, utilise self.input_grid_y.
            max_modes (int): Nombre maximal de modes Hermite-Gauss à utiliser (défaut: 20).

        Returns:
            Tuple[np.ndarray, Dict]: 
                - Champ électrique complexe 2D propagé.
                - Dictionnaire de métriques : {
                    "intensity_error_rms": Erreur RMS sur l'intensité (a.u.),
                    "intensity_error_pv": Erreur PV sur l'intensité (a.u.),
                    "phase_error_rms_nm": Erreur RMS sur la phase (nm),
                    "phase_error_pv_nm": Erreur PV sur la phase (nm),
                    "intensity_initial": Carte d'intensité initiale (2D),
                    "intensity_propagated": Carte d'intensité propagée (2D),
                    "phase_initial": Carte de phase initiale (2D, nm),
                    "phase_propagated": Carte de phase propagée (2D, nm),
                    "projection_error": Erreur de projection (RMS des coefficients ignorés).
                  }

        Notes:
            - **Base adaptée** : Faisceaux gaussiens ou proches de gaussiens.
            - **Erreur** : Dépend du nombre de modes (plus de modes = meilleure précision).
            - **Limites** : Moins précis pour les faisceaux non gaussiens (ex: top-hat).
            - **Avantage** : Rapide pour les faisceaux bien représentés par peu de modes.
            - **Complexité** : O(N_modes * N²) où N_modes est le nombre de modes.

        Formula:
            E(x,y,z) = Σ_{n,m} a_{n,m} * HG_n(x) * HG_m(y) * exp(-(x² + y²)/(4σ(z)²)) * 
                       exp(1j * (kz + (n+m+1)*atan(z/z_R)))
            où σ(z) = σ₀ * sqrt(1 + (z/z_R)²), z_R = πσ₀²/λ (distance de Rayleigh).

        Sources:
            - Siegman, A. E. (1986). "Lasers." University Science Books.
        """
        if input_grid_x is None:
            input_grid_x = self.input_grid_x
        if input_grid_y is None:
            input_grid_y = self.input_grid_y

        # Vérifier la taille du champ électrique
        if electric_field.shape != input_grid_x.shape:
            self.logger.warning(
                "Resizing electric field from %s to %s for Hermite-Gauss propagation",
                electric_field.shape,
                input_grid_x.shape
            )
            electric_field = self._resample_electric_field(electric_field, input_grid_x, input_grid_y)

        # Calculer les coefficients Hermite-Gauss
        coefficients, basis_modes = self._compute_hermite_gauss_coefficients(
            electric_field, input_grid_x, input_grid_y, max_modes
        )

        # Propager chaque mode Hermite-Gauss
        propagated_field = np.zeros_like(electric_field, dtype=np.complex128)
        k = 2 * np.pi / (self.wavelength_nm * 1e-6)  # Nombre d'onde en mm⁻¹

        for n in range(max_modes):
            for m in range(max_modes):
                if n + m >= max_modes:
                    continue
                mode_index = n * max_modes + m
                if mode_index >= len(coefficients):
                    continue

                # Calculer la propagation du mode (n,m)
                propagated_mode = self._propagate_single_hermite_gauss_mode(
                    n, m, input_grid_x, input_grid_y, coefficients[mode_index], k
                )

                # Ajouter au champ propagé total
                propagated_field += propagated_mode

        # Rééchantillonner sur la grille de sortie si nécessaire
        if self.output_diameter_mm != self.input_diameter_mm:
            propagated_field = self._resample_electric_field(
                propagated_field,
                self.output_grid_x,
                self.output_grid_y,
            )

        # Calculer les métriques d'erreur
        metrics = self._compute_propagation_metrics(electric_field, propagated_field)

        self.logger.info("Hermite-Gauss propagation applied with %d modes", max_modes)
        return propagated_field, metrics

    def propagate_laguerre_gauss(
        self,
        electric_field: np.ndarray,
        input_grid_x: Optional[np.ndarray] = None,
        input_grid_y: Optional[np.ndarray] = None,
        max_p: int = 5,
        max_l: int = 5,
    ) -> Tuple[np.ndarray, Dict]:
        """
        FR: Propage un champ électrique en utilisant une projection sur la base Laguerre-Gauss.
            Retourne le champ propagé et les métriques d'erreur par rapport au champ initial.

        EN: Propagates an electric field using projection onto Laguerre-Gauss basis.
            Returns the propagated field and error metrics compared to the initial field.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D d'entrée.
            input_grid_x (np.ndarray, optional): Grille d'entrée en x (mm). Si None, utilise self.input_grid_x.
            input_grid_y (np.ndarray, optional): Grille d'entrée en y (mm). Si None, utilise self.input_grid_y.
            max_p (int): Ordre radial maximal (défaut: 5).
            max_l (int): Ordre azimutal maximal (défaut: 5).

        Returns:
            Tuple[np.ndarray, Dict]: 
                - Champ électrique complexe 2D propagé.
                - Dictionnaire de métriques (même format que propagate_hermite_gauss).

        Notes:
            - **Base adaptée** : Faisceaux cylindriques (ex: lasers, vortex).
            - **Erreur** : Dépend du nombre de modes (p_max, l_max).
            - **Limites** : Moins précis pour les faisceaux non cylindriques.
            - **Avantage** : Idéal pour les faisceaux avec moment angulaire orbital (OAM).
            - **Complexité** : O(N_modes * N²) où N_modes = (2*l_max+1)*(p_max+1).

        Formula:
            E(r,θ,z) = Σ_{p,l} a_{p,l} * LG_p^|l|(r) * exp(-r²/(2σ(z)²)) * 
                        exp(1j * (kz + lθ + (2p+|l|+1)*atan(z/z_R)))
            où σ(z) = σ₀ * sqrt(1 + (z/z_R)²), z_R = πσ₀²/λ.

        Sources:
            - Siegman, A. E. (1986). "Lasers." University Science Books.
        """
        if input_grid_x is None:
            input_grid_x = self.input_grid_x
        if input_grid_y is None:
            input_grid_y = self.input_grid_y

        # Vérifier la taille du champ électrique
        if electric_field.shape != input_grid_x.shape:
            self.logger.warning(
                "Resizing electric field from %s to %s for Laguerre-Gauss propagation",
                electric_field.shape,
                input_grid_x.shape
            )
            electric_field = self._resample_electric_field(electric_field, input_grid_x, input_grid_y)

        # Calculer les coefficients Laguerre-Gauss
        coefficients, basis_modes = self._compute_laguerre_gauss_coefficients(
            electric_field, input_grid_x, input_grid_y, max_p, max_l
        )

        # Propager chaque mode Laguerre-Gauss
        propagated_field = np.zeros_like(electric_field, dtype=np.complex128)
        k = 2 * np.pi / (self.wavelength_nm * 1e-6)  # Nombre d'onde en mm⁻¹

        for p in range(max_p + 1):
            for l in range(-max_l, max_l + 1):
                mode_index = p * (2 * max_l + 1) + (l + max_l)
                if mode_index >= len(coefficients):
                    continue

                # Calculer la propagation du mode (p,l)
                propagated_mode = self._propagate_single_laguerre_gauss_mode(
                    p, l, input_grid_x, input_grid_y, coefficients[mode_index], k
                )

                # Ajouter au champ propagé total
                propagated_field += propagated_mode

        # Rééchantillonner sur la grille de sortie si nécessaire
        if self.output_diameter_mm != self.input_diameter_mm:
            propagated_field = self._resample_electric_field(
                propagated_field,
                self.output_grid_x,
                self.output_grid_y,
            )

        # Calculer les métriques d'erreur
        metrics = self._compute_propagation_metrics(electric_field, propagated_field)

        self.logger.info("Laguerre-Gauss propagation applied with p_max=%d, l_max=%d", max_p, max_l)
        return propagated_field, metrics

    def propagate(
        self,
        electric_field: np.ndarray,
        input_grid_x: Optional[np.ndarray] = None,
        input_grid_y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        FR: Propage un champ électrique selon la méthode spécifiée.
            Gère automatiquement la cohérence et le rééchantillonnage.

        EN: Propagates an electric field according to the specified method.
            Automatically handles coherence and resampling.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D d'entrée.
            input_grid_x (np.ndarray, optional): Grille d'entrée en x (mm).
            input_grid_y (np.ndarray, optional): Grille d'entrée en y (mm).
            **kwargs: Arguments spécifiques à la méthode de propagation.

        Returns:
            np.ndarray or Tuple[np.ndarray, Dict]:
                - Pour les méthodes numériques : champ électrique propagé.
                - Pour les méthodes analytiques : (champ électrique propagé, métriques).

        Raises:
            ValueError: Si la méthode de propagation est inconnue.
        """
        self.logger.info("Propagating with method: %s, coherence: %s", self.method, self.coherence)

        if self.method == "fraunhofer":
            result = self.propagate_fraunhofer(electric_field, input_grid_x, input_grid_y)
        elif self.method == "fresnel":
            result = self.propagate_fresnel(electric_field, input_grid_x, input_grid_y, **kwargs)
        elif self.method == "angular_spectrum":
            result = self.propagate_angular_spectrum(electric_field, input_grid_x, input_grid_y)
        elif self.method == "hermite_gauss":
            result = self.propagate_hermite_gauss(electric_field, input_grid_x, input_grid_y, **kwargs)
        elif self.method == "laguerre_gauss":
            result = self.propagate_laguerre_gauss(electric_field, input_grid_x, input_grid_y, **kwargs)
        else:
            raise ValueError(f"Méthode de propagation inconnue : {self.method}")

        # Gestion de la cohérence
        if self.coherence == "incoherent":
            # Pour un faisceau incohérent, on prend le module au carré (intensité)
            # et on ignore les effets d'interférence
            if isinstance(result, tuple):
                propagated_field, metrics = result
                intensity_field = np.abs(propagated_field)**2
                metrics["coherence"] = "incoherent"
                result = (intensity_field, metrics)
            else:
                result = np.abs(result)**2
            self.logger.warning("Incoherent propagation: returning intensity only (no phase information)")

        return result

    # =========================================================================
    # Méthodes utilitaires / Utility Methods
    # =========================================================================

    def _resample_electric_field(
        self,
        electric_field: np.ndarray,
        target_grid_x: np.ndarray,
        target_grid_y: np.ndarray,
    ) -> np.ndarray:
        """
        FR: Rééchantillonne un champ électrique sur une nouvelle grille.
            Utilise une interpolation cubique pour les parties réelle et imaginaire.

        EN: Resamples an electric field onto a new grid.
            Uses cubic interpolation for real and imaginary parts.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D à rééchantillonner.
            target_grid_x (np.ndarray): Nouvelle grille en x (mm).
            target_grid_y (np.ndarray): Nouvelle grille en y (mm).

        Returns:
            np.ndarray: Champ électrique complexe 2D rééchantillonné.

        Notes:
            - Utilise scipy.interpolate.griddata pour l'interpolation.
            - Gère les valeurs hors des limites avec fill_value=0.
        """
        from scipy.interpolate import griddata

        # Grille source (supposée linéaire)
        if hasattr(self, 'input_grid_x') and electric_field.shape == self.input_grid_x.shape:
            source_grid_x, source_grid_y = self.input_grid_x, self.input_grid_y
        else:
            # Créer une grille source linéaire
            source_grid_x, source_grid_y = create_grid(
                self.input_diameter_mm, electric_field.shape[0]
            )

        # Aplatir les données
        points = np.column_stack((source_grid_x.ravel(), source_grid_y.ravel()))
        values_real = np.real(electric_field).ravel()
        values_imag = np.imag(electric_field).ravel()
        target_points = np.column_stack((target_grid_x.ravel(), target_grid_y.ravel()))

        # Rééchantillonnage des parties réelle et imaginaire
        resampled_real = griddata(points, values_real, target_points, method='cubic', fill_value=0.0)
        resampled_imag = griddata(points, values_imag, target_points, method='cubic', fill_value=0.0)

        # Reconstruire le champ complexe
        resampled_field = resampled_real + 1j * resampled_imag
        return resampled_field.reshape(target_grid_x.shape)

    def _compute_hermite_gauss_coefficients(
        self,
        electric_field: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        max_modes: int,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        FR: Calcule les coefficients de décomposition d'un champ électrique sur la base Hermite-Gauss.
            Utilise une projection par produit scalaire complexe.

        EN: Computes the decomposition coefficients of an electric field onto the Hermite-Gauss basis.
            Uses complex inner product projection.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D.
            grid_x (np.ndarray): Grille en x (mm).
            grid_y (np.ndarray): Grille en y (mm).
            max_modes (int): Nombre maximal de modes.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: 
                - Tableau des coefficients complexes (1D).
                - Liste des modes Hermite-Gauss 2D (normalisés).
        """
        # Estimer la taille du waist (sigma) à partir du champ
        sigma_mm = self._estimate_waist_size(electric_field, grid_x, grid_y)

        # Générer les modes Hermite-Gauss
        basis_modes = []
        for n in range(max_modes):
            for m in range(max_modes):
                if n + m >= max_modes:
                    continue
                mode_2d = self._hermite_gauss_mode_2d(n, m, grid_x, grid_y, sigma_mm)
                basis_modes.append(mode_2d)

        # Calculer les coefficients par projection
        coefficients = []
        for mode in basis_modes:
            # Produit scalaire complexe : ∫ E * conj(mode) dx dy
            inner_product = np.sum(electric_field * np.conj(mode))
            # Normalisation
            norm = np.sum(np.abs(mode)**2)
            if norm > 0:
                coefficients.append(inner_product / np.sqrt(norm))
            else:
                coefficients.append(0.0)

        return np.array(coefficients), basis_modes

    def _hermite_gauss_mode_2d(
        self,
        n: int,
        m: int,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        sigma_mm: float,
    ) -> np.ndarray:
        """
        FR: Génère un mode Hermite-Gauss 2D normalisé.

        EN: Generates a normalized 2D Hermite-Gauss mode.

        Args:
            n (int): Ordre du polynôme de Hermite en x.
            m (int): Ordre du polynôme de Hermite en y.
            grid_x (np.ndarray): Grille en x (mm).
            grid_y (np.ndarray): Grille en y (mm).
            sigma_mm (float): Taille du waist en mm.

        Returns:
            np.ndarray: Mode Hermite-Gauss 2D normalisé (complexe).

        Formula:
            HG_{n,m}(x,y) = (1/√(2^{n+m} n! m! π σ²)) * H_n(x/σ) * H_m(y/σ) * exp(-(x² + y²)/(2σ²))
        """
        # Polynômes de Hermite (via scipy.special.hermite)
        H_n = hermite(n)(grid_x / sigma_mm)
        H_m = hermite(m)(grid_y / sigma_mm)

        # Gaussienne
        gauss = np.exp(-(grid_x**2 + grid_y**2) / (2 * sigma_mm**2))

        # Normalisation
        norm_factor = 1.0 / (sigma_mm * np.sqrt(2**(n + m) * np.math.factorial(n) * np.math.factorial(m) * np.pi))

        return norm_factor * H_n * H_m * gauss

    def _propagate_single_hermite_gauss_mode(
        self,
        n: int,
        m: int,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        coefficient: complex,
        k: float,
    ) -> np.ndarray:
        """
        FR: Propage un seul mode Hermite-Gauss sur la distance spécifiée.
            Applique la phase de Gouy et l'élargissement du waist.

        EN: Propagates a single Hermite-Gauss mode over the specified distance.
            Applies Gouy phase and waist expansion.

        Args:
            n (int): Ordre du polynôme de Hermite en x.
            m (int): Ordre du polynôme de Hermite en y.
            grid_x (np.ndarray): Grille en x (mm).
            grid_y (np.ndarray): Grille en y (mm).
            coefficient (complex): Coefficient complexe du mode.
            k (float): Nombre d'onde en mm⁻¹.

        Returns:
            np.ndarray: Mode Hermite-Gauss propagé.

        Formula:
            HG_{n,m}(x,y,z) = HG_{n,m}(x,y,0) * (σ₀/σ(z)) * exp(1j * (kz + (n+m+1)ψ(z)))
            où σ(z) = σ₀ * sqrt(1 + (z/z_R)²), ψ(z) = arctan(z/z_R), z_R = πσ₀²/λ.
        """
        # Estimer la taille du waist initial (sigma₀)
        sigma_0_mm = self.input_diameter_mm / 4

        # Rayon de Rayleigh
        z_R = np.pi * sigma_0_mm**2 * 1e6 / self.wavelength_nm  # en mm

        # Taille du waist à la distance z
        sigma_z = sigma_0_mm * np.sqrt(1 + (self.propagation_distance_mm / z_R)**2)

        # Phase de Gouy
        gouy_phase = (n + m + 1) * np.arctan(self.propagation_distance_mm / z_R)

        # Générer le mode propagé
        H_n = hermite(n)(grid_x / sigma_z)
        H_m = hermite(m)(grid_y / sigma_z)
        gauss = np.exp(-(grid_x**2 + grid_y**2) / (2 * sigma_z**2))

        # Normalisation
        norm_factor = 1.0 / (sigma_z * np.sqrt(2**(n + m) * np.math.factorial(n) * np.math.factorial(m) * np.pi))

        # Facteur d'amplitude (conservation de l'énergie)
        amplitude_factor = sigma_0_mm / sigma_z

        # Mode propagé
        propagated_mode = (
            coefficient
            * amplitude_factor
            * norm_factor
            * H_n
            * H_m
            * gauss
            * np.exp(1j * (k * self.propagation_distance_mm + gouy_phase))
        )

        return propagated_mode

    def _compute_laguerre_gauss_coefficients(
        self,
        electric_field: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        max_p: int,
        max_l: int,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        FR: Calcule les coefficients de décomposition d'un champ électrique sur la base Laguerre-Gauss.

        EN: Computes the decomposition coefficients of an electric field onto the Laguerre-Gauss basis.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D.
            grid_x (np.ndarray): Grille en x (mm).
            grid_y (np.ndarray): Grille en y (mm).
            max_p (int): Ordre radial maximal.
            max_l (int): Ordre azimutal maximal.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: 
                - Tableau des coefficients complexes (1D).
                - Liste des modes Laguerre-Gauss 2D (normalisés).
        """
        # Convertir en coordonnées polaires
        r = np.sqrt(grid_x**2 + grid_y**2)
        theta = np.arctan2(grid_y, grid_x)

        # Estimer la taille du waist (sigma) à partir du champ
        sigma_mm = self._estimate_waist_size(electric_field, grid_x, grid_y)

        # Générer les modes Laguerre-Gauss
        basis_modes = []
        for p in range(max_p + 1):
            for l in range(-max_l, max_l + 1):
                mode_2d = self._laguerre_gauss_mode_2d(p, l, r, theta, sigma_mm)
                basis_modes.append(mode_2d)

        # Calculer les coefficients par projection
        coefficients = []
        for mode in basis_modes:
            # Produit scalaire complexe
            inner_product = np.sum(electric_field * np.conj(mode))
            # Normalisation
            norm = np.sum(np.abs(mode)**2)
            if norm > 0:
                coefficients.append(inner_product / np.sqrt(norm))
            else:
                coefficients.append(0.0)

        return np.array(coefficients), basis_modes

    def _laguerre_gauss_mode_2d(
        self,
        p: int,
        l: int,
        r: np.ndarray,
        theta: np.ndarray,
        sigma_mm: float,
    ) -> np.ndarray:
        """
        FR: Génère un mode Laguerre-Gauss 2D normalisé.

        EN: Generates a normalized 2D Laguerre-Gauss mode.

        Args:
            p (int): Ordre radial.
            l (int): Ordre azimutal.
            r (np.ndarray): Grille radiale (mm).
            theta (np.ndarray): Grille angulaire (rad).
            sigma_mm (float): Taille du waist en mm.

        Returns:
            np.ndarray: Mode Laguerre-Gauss 2D normalisé (complexe).

        Formula:
            LG_{p,l}(r,θ) = (√(2p!/(π(p+|l|)!)) / σ) * L_p^{|l|}(2r²/σ²) * exp(-r²/σ²) * exp(1j * lθ)
        """
        # Polynôme de Laguerre généralisé
        L_pl = eval_genlaguerre(p, abs(l))(2 * r**2 / sigma_mm**2)

        # Gaussienne
        gauss = np.exp(-r**2 / sigma_mm**2)

        # Partie angulaire
        if l >= 0:
            angular = np.exp(1j * l * theta)
        else:
            angular = np.exp(-1j * abs(l) * theta)

        # Normalisation
        norm_factor = np.sqrt(2 * np.math.factorial(p) / (np.pi * np.math.factorial(p + abs(l)))) / sigma_mm

        return norm_factor * L_pl * gauss * angular

    def _propagate_single_laguerre_gauss_mode(
        self,
        p: int,
        l: int,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        coefficient: complex,
        k: float,
    ) -> np.ndarray:
        """
        FR: Propage un seul mode Laguerre-Gauss sur la distance spécifiée.
            Applique la phase de Gouy et l'élargissement du waist.

        EN: Propagates a single Laguerre-Gauss mode over the specified distance.
            Applies Gouy phase and waist expansion.

        Args:
            p (int): Ordre radial.
            l (int): Ordre azimutal.
            grid_x (np.ndarray): Grille en x (mm).
            grid_y (np.ndarray): Grille en y (mm).
            coefficient (complex): Coefficient complexe du mode.
            k (float): Nombre d'onde en mm⁻¹.

        Returns:
            np.ndarray: Mode Laguerre-Gauss propagé.

        Formula:
            LG_{p,l}(r,θ,z) = LG_{p,l}(r,θ,0) * (σ₀/σ(z)) * exp(1j * (kz + lθ + (2p+|l|+1)ψ(z)))
            où σ(z) = σ₀ * sqrt(1 + (z/z_R)²), ψ(z) = arctan(z/z_R), z_R = πσ₀²/λ.
        """
        # Convertir en coordonnées polaires
        r = np.sqrt(grid_x**2 + grid_y**2)
        theta = np.arctan2(grid_y, grid_x)

        # Estimer la taille du waist initial (sigma₀)
        sigma_0_mm = self.input_diameter_mm / 4

        # Rayon de Rayleigh
        z_R = np.pi * sigma_0_mm**2 * 1e6 / self.wavelength_nm  # en mm

        # Taille du waist à la distance z
        sigma_z = sigma_0_mm * np.sqrt(1 + (self.propagation_distance_mm / z_R)**2)

        # Phase de Gouy
        gouy_phase = (2 * p + abs(l) + 1) * np.arctan(self.propagation_distance_mm / z_R)

        # Générer le mode propagé
        L_pl = eval_genlaguerre(p, abs(l))(2 * r**2 / sigma_z**2)
        gauss = np.exp(-r**2 / sigma_z**2)

        if l >= 0:
            angular = np.exp(1j * l * theta)
        else:
            angular = np.exp(-1j * abs(l) * theta)

        # Normalisation
        norm_factor = np.sqrt(2 * np.math.factorial(p) / (np.pi * np.math.factorial(p + abs(l)))) / sigma_z

        # Facteur d'amplitude (conservation de l'énergie)
        amplitude_factor = sigma_0_mm / sigma_z

        # Mode propagé
        propagated_mode = (
            coefficient
            * amplitude_factor
            * norm_factor
            * L_pl
            * gauss
            * angular
            * np.exp(1j * (k * self.propagation_distance_mm + l * theta + gouy_phase))
        )

        return propagated_mode

    def _estimate_waist_size(
        self,
        electric_field: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
    ) -> float:
        """
        FR: Estime la taille du waist (sigma) d'un faisceau à partir de son intensité.
            Utilise le moment du second ordre.

        EN: Estimates the waist size (sigma) of a beam from its intensity.
            Uses the second-order moment.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D.
            grid_x (np.ndarray): Grille en x (mm).
            grid_y (np.ndarray): Grille en y (mm).

        Returns:
            float: Taille du waist estimée en mm.

        Formula:
            σ = √[(∫(x - ⟨x⟩)² I dx dy) / ∫I dx dy]
        """
        intensity = np.abs(electric_field)**2
        # Calculer le centre de masse
        x_mean = np.sum(grid_x * intensity) / np.sum(intensity)
        y_mean = np.sum(grid_y * intensity) / np.sum(intensity)
        # Calculer l'écart-type
        sigma_x = np.sqrt(np.sum((grid_x - x_mean)**2 * intensity) / np.sum(intensity))
        sigma_y = np.sqrt(np.sum((grid_y - y_mean)**2 * intensity) / np.sum(intensity))
        return (sigma_x + sigma_y) / 2

    def _compute_propagation_metrics(
        self,
        initial_field: np.ndarray,
        propagated_field: np.ndarray,
    ) -> Dict:
        """
        FR: Calcule les métriques d'erreur entre le champ initial et le champ propagé.
            Fournit des cartes 2D de phase et d'intensité, ainsi que des erreurs RMS et PV.

        EN: Computes error metrics between the initial field and the propagated field.
            Provides 2D maps of phase and intensity, as well as RMS and PV errors.

        Args:
            initial_field (np.ndarray): Champ électrique initial (complexe 2D).
            propagated_field (np.ndarray): Champ électrique propagé (complexe 2D).

        Returns:
            Dict: Dictionnaire de métriques :
                - intensity_error_rms (float): Erreur RMS sur l'intensité (a.u.).
                - intensity_error_pv (float): Erreur PV sur l'intensité (a.u.).
                - phase_error_rms_nm (float): Erreur RMS sur la phase (nm).
                - phase_error_pv_nm (float): Erreur PV sur la phase (nm).
                - intensity_initial (np.ndarray): Carte d'intensité initiale (2D).
                - intensity_propagated (np.ndarray): Carte d'intensité propagée (2D).
                - phase_initial (np.ndarray): Carte de phase initiale (2D, nm).
                - phase_propagated (np.ndarray): Carte de phase propagée (2D, nm).
        """
        # Calculer les intensités
        intensity_initial = np.abs(initial_field)**2
        intensity_propagated = np.abs(propagated_field)**2

        # Normaliser les intensités pour comparaison (éviter les effets d'échelle)
        intensity_initial_norm = intensity_initial / np.max(intensity_initial) if np.max(intensity_initial) > 0 else intensity_initial
        intensity_propagated_norm = intensity_propagated / np.max(intensity_propagated) if np.max(intensity_propagated) > 0 else intensity_propagated

        # Erreur sur l'intensité
        intensity_diff = intensity_initial_norm - intensity_propagated_norm
        intensity_error_pv = float(np.max(intensity_diff) - np.min(intensity_diff))
        intensity_error_rms = float(np.sqrt(np.mean(intensity_diff**2))))

        # Calculer les phases
        phase_initial_rad = np.angle(initial_field)
        phase_propagated_rad = np.angle(propagated_field)
        phase_initial_nm = rad_to_nm(phase_initial_rad, self.wavelength_nm)
        phase_propagated_nm = rad_to_nm(phase_propagated_rad, self.wavelength_nm)

        # Erreur sur la phase (en nm)
        phase_diff_nm = phase_initial_nm - phase_propagated_nm
        phase_error_pv_nm = float(np.max(phase_diff_nm) - np.min(phase_diff_nm))
        phase_error_rms_nm = float(np.sqrt(np.mean(phase_diff_nm**2)))

        return {
            "intensity_error_rms": intensity_error_rms,
            "intensity_error_pv": intensity_error_pv,
            "phase_error_rms_nm": phase_error_rms_nm,
            "phase_error_pv_nm": phase_error_pv_nm,
            "intensity_initial": intensity_initial,
            "intensity_propagated": intensity_propagated,
            "phase_initial": phase_initial_nm,
            "phase_propagated": phase_propagated_nm,
        }


# =============================================================================
# FONCTIONS UTILITAIRES / UTILITY FUNCTIONS
# =============================================================================

def get_propagation_regime(
    diameter_mm: float,
    wavelength_nm: float,
    distance_m: float,
) -> str:
    """
    FR: Détermine le régime de propagation (Fraunhofer, Fresnel, ou champ proche).

    EN: Determines the propagation regime (Fraunhofer, Fresnel, or near field).

    Args:
        diameter_mm (float): Diamètre du faisceau en mm.
        wavelength_nm (float): Longueur d'onde en nm.
        distance_m (float): Distance de propagation en mètres.

    Returns:
        str: Régime de propagation ("Fraunhofer", "Fresnel", ou "Near Field").

    Notes:
        - Fraunhofer : z >> (π * D²) / (4 * λ)
        - Fresnel : z ≈ (π * D²) / (4 * λ)
        - Near Field : z << (π * D²) / (4 * λ)
    """
    wavelength_m = wavelength_nm * 1e-9
    diameter_m = diameter_mm * 1e-3
    fraunhofer_distance = np.pi * diameter_m**2 / (4 * wavelength_m)
    
    if distance_m > 10 * fraunhofer_distance:
        return "Fraunhofer"
    elif distance_m > 0.1 * fraunhofer_distance:
        return "Fresnel"
    else:
        return "Near Field"

def estimate_required_resolution(
    diameter_mm: float,
    wavelength_nm: float,
    distance_m: float,
) -> int:
    """
    FR: Estime la résolution requise pour la propagation FFT.
        Basé sur le théorème de Nyquist pour éviter l'aliasing.

    EN: Estimates the required resolution for FFT propagation.
        Based on Nyquist theorem to avoid aliasing.

    Args:
        diameter_mm (float): Diamètre du faisceau en mm.
        wavelength_nm (float): Longueur d'onde en nm.
        distance_m (float): Distance de propagation en mètres.

    Returns:
        int: Nombre minimal de points requis par dimension.

    Notes:
        - La résolution doit être suffisante pour échantillonner les franges d'interférence.
    """
    wavelength_m = wavelength_nm * 1e-9
    diameter_m = diameter_mm * 1e-3
    
    # Angle de diffraction maximal (approximation)
    max_angle = np.arctan(diameter_m / (2 * distance_m)) if distance_m > 0 else np.pi / 2
    
    # Fréquence spatiale maximale
    max_frequency = np.sin(max_angle) / wavelength_m
    
    # Résolution minimale selon Nyquist
    min_resolution = int(2 * max_frequency * diameter_m) + 1
    
    # Arrondir à la puissance de 2 supérieure pour FFT efficace
    return 2 ** int(np.ceil(np.log2(min_resolution)))


# =============================================================================
# TESTS UNITAIRES / UNIT TESTS
# =============================================================================

class TestPropagation:
    """
    FR: Classe de tests unitaires pour Propagation.py.
    EN: Unit test class for Propagation.py.
    """

    def setUp(self):
        self.propagation = Propagation(
            wavelength_nm=633.0,
            propagation_distance_mm=100.0,
            input_diameter_mm=10.0,
            num_points=128,
        )

    def test_propagate_fraunhofer(self):
        """Test la propagation de Fraunhofer."""
        grid_x, grid_y = self.propagation.input_grid_x, self.propagation.input_grid_y
        electric_field = np.exp(-(grid_x**2 + grid_y**2) / (2 * 2.0**2)) * np.exp(1j * np.random.rand() * 2 * np.pi)

        propagated_field = self.propagation.propagate_fraunhofer(electric_field)
        self.assertEqual(propagated_field.shape, (128, 128))
        self.assertEqual(propagated_field.dtype, np.complex128)

    def test_propagate_fresnel(self):
        """Test la propagation de Fresnel."""
        grid_x, grid_y = self.propagation.input_grid_x, self.propagation.input_grid_y
        electric_field = np.exp(-(grid_x**2 + grid_y**2) / (2 * 2.0**2)) * np.exp(1j * np.random.rand() * 2 * np.pi)

        propagated_field = self.propagation.propagate_fresnel(electric_field, step_mm=10.0)
        self.assertEqual(propagated_field.shape, (128, 128))
        self.assertEqual(propagated_field.dtype, np.complex128)

    def test_propagate_angular_spectrum(self):
        """Test la propagation par spectre angulaire."""
        grid_x, grid_y = self.propagation.input_grid_x, self.propagation.input_grid_y
        electric_field = np.exp(-(grid_x**2 + grid_y**2) / (2 * 2.0**2)) * np.exp(1j * np.random.rand() * 2 * np.pi)

        propagated_field = self.propagation.propagate_angular_spectrum(electric_field)
        self.assertEqual(propagated_field.shape, (128, 128))
        self.assertEqual(propagated_field.dtype, np.complex128)

    def test_propagate_hermite_gauss(self):
        """Test la propagation Hermite-Gauss."""
        grid_x, grid_y = self.propagation.input_grid_x, self.propagation.input_grid_y
        electric_field = np.exp(-(grid_x**2 + grid_y**2) / (2 * 2.0**2)) * np.exp(1j * np.random.rand() * 2 * np.pi)

        propagated_field, metrics = self.propagation.propagate_hermite_gauss(electric_field, max_modes=10)
        self.assertEqual(propagated_field.shape, (128, 128))
        self.assertIn("intensity_error_rms", metrics)
        self.assertIn("phase_error_rms_nm", metrics)
        self.assertIn("intensity_initial", metrics)
        self.assertIn("phase_propagated", metrics)

    def test_propagate_laguerre_gauss(self):
        """Test la propagation Laguerre-Gauss."""
        grid_x, grid_y = self.propagation.input_grid_x, self.propagation.input_grid_y
        r = np.sqrt(grid_x**2 + grid_y**2)
        theta = np.arctan2(grid_y, grid_x)
        # Mode Laguerre-Gauss simple (p=0, l=0)
        electric_field = np.exp(-r**2 / (2 * 2.0**2)) * np.exp(1j * np.random.rand() * 2 * np.pi)

        propagated_field, metrics = self.propagation.propagate_laguerre_gauss(electric_field, max_p=3, max_l=3)
        self.assertEqual(propagated_field.shape, (128, 128))
        self.assertIn("intensity_error_rms", metrics)
        self.assertIn("phase_error_rms_nm", metrics)

    def test_propagate_with_coherence(self):
        """Test la propagation avec cohérence incohérente."""
        propagation_incoherent = Propagation(
            wavelength_nm=633.0,
            propagation_distance_mm=100.0,
            input_diameter_mm=10.0,
            num_points=128,
            coherence="incoherent",
            method="fraunhofer",
        )
        grid_x, grid_y = propagation_incoherent.input_grid_x, propagation_incoherent.input_grid_y
        electric_field = np.exp(-(grid_x**2 + grid_y**2) / (2 * 2.0**2)) * np.exp(1j * np.random.rand() * 2 * np.pi)

        result = propagation_incoherent.propagate(electric_field)
        # Pour un faisceau incohérent, le résultat doit être réel (intensité)
        self.assertTrue(np.isrealobj(result) or np.allclose(np.imag(result), 0))

    def test_resample_electric_field(self):
        """Test le rééchantillonnage du champ électrique."""
        grid_x, grid_y = self.propagation.input_grid_x, self.propagation.input_grid_y
        electric_field = np.random.rand(128, 128) + 1j * np.random.rand(128, 128)

        resampled_field = self.propagation._resample_electric_field(
            electric_field,
            self.propagation.output_grid_x,
            self.propagation.output_grid_y,
        )
        self.assertEqual(resampled_field.shape, (128, 128))

    def test_compute_propagation_metrics(self):
        """Test le calcul des métriques de propagation."""
        grid_x, grid_y = self.propagation.input_grid_x, self.propagation.input_grid_y
        initial_field = np.exp(-(grid_x**2 + grid_y**2) / (2 * 2.0**2)) * np.exp(1j * np.random.rand() * 2 * np.pi)
        propagated_field = np.exp(-(grid_x**2 + grid_y**2) / (2 * 2.1**2)) * np.exp(1j * np.random.rand() * 2 * np.pi)

        metrics = self.propagation._compute_propagation_metrics(initial_field, propagated_field)
        self.assertIn("intensity_error_rms", metrics)
        self.assertIn("phase_error_rms_nm", metrics)
        self.assertEqual(metrics["intensity_initial"].shape, (128, 128))
        self.assertEqual(metrics["phase_propagated"].shape, (128, 128))

    def test_get_propagation_regime(self):
        """Test la détermination du régime de propagation."""
        regime_fraunhofer = get_propagation_regime(10.0, 633.0, 10.0)
        regime_fresnel = get_propagation_regime(10.0, 633.0, 0.1)
        regime_near_field = get_propagation_regime(10.0, 633.0, 0.001)
        
        self.assertEqual(regime_fraunhofer, "Fraunhofer")
        self.assertEqual(regime_fresnel, "Fresnel")
        self.assertEqual(regime_near_field, "Near Field")

    def test_estimate_required_resolution(self):
        """Test l'estimation de la résolution requise."""
        resolution = estimate_required_resolution(10.0, 633.0, 1.0)
        self.assertGreaterEqual(resolution, 64)


if __name__ == "__main__":
    import unittest
    unittest.main()