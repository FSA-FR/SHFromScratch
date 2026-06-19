"""
Propagation.py
FR: Module pour la propagation numérique et analytique des faisceaux optiques.
    Permet de propager un champ électrique selon différentes méthodes :
    - Propagation numérique (FFT) pour les régimes de Fraunhofer, Fresnel, et Gauss.
    - Propagation analytique en projetant sur des bases de modes (Hermite-Gauss, Laguerre-Gauss).
    Gère le rééchantillonnage des grilles, la cohérence (cohérente/incohérente), et la diffraction.

EN: Module for numerical and analytical propagation of optical beams.
    Allows propagating an electric field using different methods:
    - Numerical propagation (FFT) for Fraunhofer, Fresnel, and Gaussian regimes.
    - Analytical propagation by projecting onto mode bases (Hermite-Gauss, Laguerre-Gauss).
    Handles grid resampling, coherence (coherent/incoherent), and diffraction.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import logging
from typing import Optional, Tuple, Union
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from MathAndPhysicsTools import (
    create_grid,
    generate_hermite_gauss_modes,
    generate_laguerre_gauss_modes,
    nm_to_rad,
    rad_to_nm,
    resample_to_grid,
)


# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Propagation")


# =============================================================================
# 1. PROPAGATION NUMÉRIQUE / NUMERICAL PROPAGATION
# =============================================================================

def propagate_fraunhofer(
    electric_field: np.ndarray,
    wavelength_nm: float,
    propagation_distance_m: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    output_grid_x: Optional[np.ndarray] = None,
    output_grid_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    FR: Propagation en régime de Fraunhofer (champ lointain) utilisant la FFT.
        Approximation valable quand : z >> (π * D²) / (4 * λ), où D est le diamètre du faisceau.

    EN: Fraunhofer regime propagation (far field) using FFT.
        Valid when: z >> (π * D²) / (4 * λ), where D is the beam diameter.

    Args:
        electric_field (np.ndarray): Champ électrique complexe 2D à propager.
        wavelength_nm (float): Longueur d'onde en nm.
        propagation_distance_m (float): Distance de propagation en mètres.
        grid_x (np.ndarray): Grille x en mm correspondante au champ d'entrée.
        grid_y (np.ndarray): Grille y en mm correspondante au champ d'entrée.
        output_grid_x (np.ndarray, optional): Grille x de sortie en mm. Si None, même taille que grid_x.
        output_grid_y (np.ndarray, optional): Grille y de sortie en mm. Si None, même taille que grid_y.

    Returns:
        np.ndarray: Champ électrique propagé (complexe 2D).

    Notes:
        - Cette méthode ignore les effets de diffraction proches.
        - La résolution de la grille de sortie dépend de la taille de la FFT.

    Sources:
        - Goodman, J. W. (2005). "Introduction to Fourier Optics." Roberts and Company.
    """
    # Convertir la longueur d'onde en mètres
    wavelength_m = wavelength_nm * 1e-9

    # Calculer le nombre d'onde
    k = 2 * np.pi / wavelength_m

    # Taille du champ
    Nx, Ny = electric_field.shape
    dx = (grid_x[0, -1] - grid_x[0, 0]) / (Nx - 1) * 1e-3  # mm → m
    dy = (grid_y[-1, 0] - grid_y[0, 0]) / (Ny - 1) * 1e-3  # mm → m

    # Fréquences spatiales (en 1/m)
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='ij')

    # Calcul de la phase de propagation en champ lointain
    H = np.exp(1j * np.pi * propagation_distance_m * wavelength_m * (fx_grid**2 + fy_grid**2))

    # Transformée de Fourier du champ initial
    field_fft = fft2(electric_field)
    field_fft = fftshift(field_fft)

    # Appliquer la fonction de transfert
    propagated_field_fft = field_fft * H
    propagated_field_fft = ifftshift(propagated_field_fft)

    # Transformée de Fourier inverse
    propagated_field = ifft2(propagated_field_fft)

    # Rééchantillonnage si une grille de sortie est spécifiée
    if output_grid_x is not None and output_grid_y is not None:
        propagated_field = resample_to_grid(
            np.abs(propagated_field),
            (output_grid_x.shape[0], output_grid_x.shape[1]),
            grid_x, grid_y,
            output_grid_x, output_grid_y,
        )
        # Appliquer la phase (uniforme en Fraunhofer)
        propagated_field = propagated_field * np.exp(1j * np.angle(propagated_field))

    return propagated_field


def propagate_fresnel(
    electric_field: np.ndarray,
    wavelength_nm: float,
    propagation_distance_m: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    output_grid_x: Optional[np.ndarray] = None,
    output_grid_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    FR: Propagation en régime de Fresnel (champ proche) utilisant la FFT.
        Approximation valable pour des distances intermédiaires.

    EN: Fresnel regime propagation (near field) using FFT.
        Valid for intermediate distances.

    Args:
        electric_field (np.ndarray): Champ électrique complexe 2D à propager.
        wavelength_nm (float): Longueur d'onde en nm.
        propagation_distance_m (float): Distance de propagation en mètres.
        grid_x (np.ndarray): Grille x en mm correspondante au champ d'entrée.
        grid_y (np.ndarray): Grille y en mm correspondante au champ d'entrée.
        output_grid_x (np.ndarray, optional): Grille x de sortie en mm. Si None, même taille que grid_x.
        output_grid_y (np.ndarray, optional): Grille y de sortie en mm. Si None, même taille que grid_y.

    Returns:
        np.ndarray: Champ électrique propagé (complexe 2D).

    Notes:
        - Cette méthode utilise l'approximation de Fresnel : exp(1j * k * z) * exp(1j * k * (x² + y²) / (2z)).
        - Plus précise que Fraunhofer pour les distances intermédiaires.

    Sources:
        - Goodman, J. W. (2005). "Introduction to Fourier Optics." Roberts and Company.
    """
    # Convertir la longueur d'onde en mètres
    wavelength_m = wavelength_nm * 1e-9

    # Calculer le nombre d'onde
    k = 2 * np.pi / wavelength_m

    # Taille du champ
    Nx, Ny = electric_field.shape
    dx = (grid_x[0, -1] - grid_x[0, 0]) / (Nx - 1) * 1e-3  # mm → m
    dy = (grid_y[-1, 0] - grid_y[0, 0]) / (Ny - 1) * 1e-3  # mm → m

    # Fréquences spatiales (en 1/m)
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='ij')

    # Calcul de la fonction de transfert de Fresnel
    H = np.exp(1j * np.pi * propagation_distance_m * wavelength_m * (fx_grid**2 + fy_grid**2))
    H *= np.exp(1j * k * propagation_distance_m)

    # Transformée de Fourier du champ initial
    field_fft = fft2(electric_field)
    field_fft = fftshift(field_fft)

    # Appliquer la fonction de transfert
    propagated_field_fft = field_fft * H
    propagated_field_fft = ifftshift(propagated_field_fft)

    # Transformée de Fourier inverse
    propagated_field = ifft2(propagated_field_fft)

    # Rééchantillonnage si une grille de sortie est spécifiée
    if output_grid_x is not None and output_grid_y is not None:
        propagated_field = resample_to_grid(
            propagated_field,
            (output_grid_x.shape[0], output_grid_x.shape[1]),
            grid_x, grid_y,
            output_grid_x, output_grid_y,
        )

    return propagated_field


def propagate_angular_spectrum(
    electric_field: np.ndarray,
    wavelength_nm: float,
    propagation_distance_m: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    output_grid_x: Optional[np.ndarray] = None,
    output_grid_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    FR: Propagation par la méthode du spectre angulaire (méthode la plus précise pour la diffraction).
        Utilise la FFT pour calculer la propagation exacte en espace libre.

    EN: Angular spectrum propagation (most accurate method for diffraction).
        Uses FFT to compute exact free-space propagation.

    Args:
        electric_field (np.ndarray): Champ électrique complexe 2D à propager.
        wavelength_nm (float): Longueur d'onde en nm.
        propagation_distance_m (float): Distance de propagation en mètres.
        grid_x (np.ndarray): Grille x en mm correspondante au champ d'entrée.
        grid_y (np.ndarray): Grille y en mm correspondante au champ d'entrée.
        output_grid_x (np.ndarray, optional): Grille x de sortie en mm. Si None, même taille que grid_x.
        output_grid_y (np.ndarray, optional): Grille y de sortie en mm. Si None, même taille que grid_y.

    Returns:
        np.ndarray: Champ électrique propagé (complexe 2D).

    Notes:
        - Cette méthode est la plus précise mais aussi la plus coûteuse en calcul.
        - Valable pour toutes les distances (champ proche et lointain).

    Sources:
        - Goodman, J. W. (2005). "Introduction to Fourier Optics." Roberts and Company.
    """
    # Convertir la longueur d'onde en mètres
    wavelength_m = wavelength_nm * 1e-9

    # Calculer le nombre d'onde
    k = 2 * np.pi / wavelength_m

    # Taille du champ
    Nx, Ny = electric_field.shape
    dx = (grid_x[0, -1] - grid_x[0, 0]) / (Nx - 1) * 1e-3  # mm → m
    dy = (grid_y[-1, 0] - grid_y[0, 0]) / (Ny - 1) * 1e-3  # mm → m

    # Fréquences spatiales (en 1/m)
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='ij')

    # Calcul de la fonction de transfert du spectre angulaire
    f_squared = fx_grid**2 + fy_grid**2
    # Éviter la division par zéro pour f=0
    f_squared[f_squared == 0] = 1e-30  # Très petite valeur pour éviter NaN
    
    sqrt_term = np.sqrt(1 / wavelength_m**2 - f_squared)
    # Pour les fréquences spatiales trop élevées (évanescentes), on utilise une onde évanescente
    sqrt_term[np.imag(sqrt_term) != 0] = 1j * np.abs(np.imag(sqrt_term[np.imag(sqrt_term) != 0]))
    
    H = np.exp(1j * 2 * np.pi * propagation_distance_m * sqrt_term)

    # Transformée de Fourier du champ initial
    field_fft = fft2(electric_field)
    field_fft = fftshift(field_fft)

    # Appliquer la fonction de transfert
    propagated_field_fft = field_fft * H
    propagated_field_fft = ifftshift(propagated_field_fft)

    # Transformée de Fourier inverse
    propagated_field = ifft2(propagated_field_fft)

    # Rééchantillonnage si une grille de sortie est spécifiée
    if output_grid_x is not None and output_grid_y is not None:
        propagated_field = resample_to_grid(
            propagated_field,
            (output_grid_x.shape[0], output_grid_x.shape[1]),
            grid_x, grid_y,
            output_grid_x, output_grid_y,
        )

    return propagated_field


# =============================================================================
# 2. PROPAGATION ANALYTIQUE / ANALYTICAL PROPAGATION
# =============================================================================

def project_onto_hermite_gauss(
    electric_field: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    wavelength_nm: float,
    n_modes: int = 10,
    sigma_mm: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FR: Projette un champ électrique sur une base de modes Hermite-Gauss.
        Retourne les coefficients et les modes.

    EN: Projects an electric field onto a Hermite-Gauss mode basis.
        Returns the coefficients and modes.

    Args:
        electric_field (np.ndarray): Champ électrique complexe 2D à projeter.
        grid_x (np.ndarray): Grille x en mm.
        grid_y (np.ndarray): Grille y en mm.
        wavelength_nm (float): Longueur d'onde en nm.
        n_modes (int): Nombre de modes à utiliser (défaut: 10).
        sigma_mm (float): Écart-type des modes en mm (défaut: 2.0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (coefficients, modes), où coefficients est un tableau 1D des coefficients complexes,
                                       et modes est un tableau 3D des modes Hermite-Gauss.

    Notes:
        - Les modes sont normalisés pour que l'intégrale de |mode|² = 1.
        - La projection utilise le produit scalaire complexe.

    Sources:
        - Siegman, A. E. (1986). "Lasers." University Science Books.
    """
    # Générer les modes Hermite-Gauss
    modes = generate_hermite_gauss_modes(n_modes, grid_x, grid_y, sigma_mm)
    
    # Normaliser les modes
    for i in range(modes.shape[0]):
        norm = np.sqrt(np.sum(np.abs(modes[i])**2))
        modes[i] = modes[i] / norm
    
    # Calculer les coefficients de projection
    coefficients = np.zeros(n_modes, dtype=np.complex128)
    for i in range(n_modes):
        # Produit scalaire : ∫ E * conj(mode_i) dx dy
        coefficients[i] = np.sum(electric_field * np.conj(modes[i]))
    
    return coefficients, modes


def project_onto_laguerre_gauss(
    electric_field: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    wavelength_nm: float,
    n_modes: int = 10,
    sigma_mm: float = 2.0,
    p_max: int = 5,
    l_max: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FR: Projette un champ électrique sur une base de modes Laguerre-Gauss.
        Retourne les coefficients et les modes.

    EN: Projects an electric field onto a Laguerre-Gauss mode basis.
        Returns the coefficients and modes.

    Args:
        electric_field (np.ndarray): Champ électrique complexe 2D à projeter.
        grid_x (np.ndarray): Grille x en mm.
        grid_y (np.ndarray): Grille y en mm.
        wavelength_nm (float): Longueur d'onde en nm.
        n_modes (int): Nombre de modes à utiliser (défaut: 10).
        sigma_mm (float): Écart-type des modes en mm (défaut: 2.0).
        p_max (int): Ordre radial maximal (défaut: 5).
        l_max (int): Ordre azimutal maximal (défaut: 5).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (coefficients, modes), où coefficients est un tableau 1D des coefficients complexes,
                                       et modes est un tableau 3D des modes Laguerre-Gauss.

    Notes:
        - Les modes sont normalisés pour que l'intégrale de |mode|² = 1.
        - La projection utilise le produit scalaire complexe.

    Sources:
        - Siegman, A. E. (1986). "Lasers." University Science Books.
    """
    # Générer les modes Laguerre-Gauss
    modes = generate_laguerre_gauss_modes(n_modes, grid_x, grid_y, sigma_mm, p_max, l_max)
    
    # Normaliser les modes
    for i in range(modes.shape[0]):
        norm = np.sqrt(np.sum(np.abs(modes[i])**2))
        modes[i] = modes[i] / norm
    
    # Calculer les coefficients de projection
    coefficients = np.zeros(n_modes, dtype=np.complex128)
    for i in range(n_modes):
        coefficients[i] = np.sum(electric_field * np.conj(modes[i]))
    
    return coefficients, modes


def propagate_analytical_hermite_gauss(
    coefficients: np.ndarray,
    modes: np.ndarray,
    propagation_distance_m: float,
    wavelength_nm: float,
    n: int = 0,
    m: int = 0,
    sigma_mm: float = 2.0,
) -> np.ndarray:
    """
    FR: Propage analytiquement un champ projeté sur des modes Hermite-Gauss.
        Chaque mode se propage indépendamment selon sa propre phase de Gouy.

    EN: Analytically propagates a field projected onto Hermite-Gauss modes.
        Each mode propagates independently according to its own Gouy phase.

    Args:
        coefficients (np.ndarray): Coefficients complexes de la projection.
        modes (np.ndarray): Modes Hermite-Gauss 3D (n_modes, height, width).
        propagation_distance_m (float): Distance de propagation en mètres.
        wavelength_nm (float): Longueur d'onde en nm.
        n (int): Ordre radial du mode dominant (défaut: 0).
        m (int): Ordre azimutal du mode dominant (défaut: 0).
        sigma_mm (float): Écart-type des modes en mm (défaut: 2.0).

    Returns:
        np.ndarray: Champ électrique propagé (complexe 2D).

    Notes:
        - La phase de Gouy pour les modes Hermite-Gauss est : ψ(z) = (n + m + 1) * arctan(z / z_R),
          où z_R = π * σ₀² / λ est la distance de Rayleigh.
        - L'amplitude change aussi selon : 1 / sqrt(1 + (z / z_R)²).

    Sources:
        - Siegman, A. E. (1986). "Lasers." University Science Books.
    """
    # Convertir la longueur d'onde en mètres
    wavelength_m = wavelength_nm * 1e-9
    sigma_m = sigma_mm * 1e-3
    
    # Distance de Rayleigh
    z_R = np.pi * sigma_m**2 / wavelength_m
    
    # Calculer la phase de Gouy et l'amplitude pour chaque mode
    propagated_field = np.zeros_like(modes[0], dtype=np.complex128)
    for i, (coeff, mode) in enumerate(zip(coefficients, modes)):
        # Phase de Gouy
        gouy_phase = (i + 1) * np.arctan(propagation_distance_m / z_R)
        
        # Facteur d'amplitude
        amplitude_factor = 1 / np.sqrt(1 + (propagation_distance_m / z_R)**2)
        
        # Facteur de phase
        phase_factor = np.exp(1j * (2 * np.pi * propagation_distance_m / wavelength_m + gouy_phase))
        
        # Propager le mode
        propagated_mode = mode * amplitude_factor * phase_factor
        propagated_field += coeff * propagated_mode
    
    return propagated_field


def propagate_analytical_laguerre_gauss(
    coefficients: np.ndarray,
    modes: np.ndarray,
    propagation_distance_m: float,
    wavelength_nm: float,
    p: int = 0,
    l: int = 0,
    sigma_mm: float = 2.0,
) -> np.ndarray:
    """
    FR: Propage analytiquement un champ projeté sur des modes Laguerre-Gauss.
        Chaque mode se propage indépendamment selon sa propre phase de Gouy.

    EN: Analytically propagates a field projected onto Laguerre-Gauss modes.
        Each mode propagates independently according to its own Gouy phase.

    Args:
        coefficients (np.ndarray): Coefficients complexes de la projection.
        modes (np.ndarray): Modes Laguerre-Gauss 3D (n_modes, height, width).
        propagation_distance_m (float): Distance de propagation en mètres.
        wavelength_nm (float): Longueur d'onde en nm.
        p (int): Ordre radial du mode dominant (défaut: 0).
        l (int): Ordre azimutal du mode dominant (défaut: 0).
        sigma_mm (float): Écart-type des modes en mm (défaut: 2.0).

    Returns:
        np.ndarray: Champ électrique propagé (complexe 2D).

    Notes:
        - La phase de Gouy pour les modes Laguerre-Gauss est : ψ(z) = (2p + |l| + 1) * arctan(z / z_R).
        - L'amplitude change aussi selon : 1 / sqrt(1 + (z / z_R)²).

    Sources:
        - Siegman, A. E. (1986). "Lasers." University Science Books.
    """
    # Convertir la longueur d'onde en mètres
    wavelength_m = wavelength_nm * 1e-9
    sigma_m = sigma_mm * 1e-3
    
    # Distance de Rayleigh
    z_R = np.pi * sigma_m**2 / wavelength_m
    
    # Calculer la phase de Gouy et l'amplitude pour chaque mode
    propagated_field = np.zeros_like(modes[0], dtype=np.complex128)
    for i, (coeff, mode) in enumerate(zip(coefficients, modes)):
        # Phase de Gouy (approximation pour Laguerre-Gauss)
        gouy_phase = (2 * p + abs(l) + 1) * np.arctan(propagation_distance_m / z_R)
        
        # Facteur d'amplitude
        amplitude_factor = 1 / np.sqrt(1 + (propagation_distance_m / z_R)**2)
        
        # Facteur de phase
        phase_factor = np.exp(1j * (2 * np.pi * propagation_distance_m / wavelength_m + gouy_phase))
        
        # Propager le mode
        propagated_mode = mode * amplitude_factor * phase_factor
        propagated_field += coeff * propagated_mode
    
    return propagated_field


# =============================================================================
# 3. CLASSE PROPAGATOR / PROPAGATOR CLASS
# =============================================================================

class Propagator:
    """
    FR: Classe pour la propagation des faisceaux optiques.
        Gère la propagation numérique (FFT) et analytique (bases de modes).

    EN: Class for propagating optical beams.
        Handles numerical (FFT) and analytical (mode bases) propagation.

    Attributes:
        wavelength_nm (float): Longueur d'onde en nm.
        coherence (str): Régime de cohérence ("coherent" ou "incoherent").
        logger (logging.Logger): Logger pour le débogage.
    """

    def __init__(
        self,
        wavelength_nm: float = 633.0,
        coherence: str = "coherent",
    ):
        """
        FR: Initialise le propagateur avec une longueur d'onde et un régime de cohérence.

        EN: Initializes the propagator with a wavelength and coherence regime.

        Args:
            wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
            coherence (str): Régime de cohérence ("coherent" ou "incoherent"). Default: "coherent".

        Raises:
            ValueError: Si la cohérence est invalide.
        """
        if coherence not in ["coherent", "incoherent"]:
            raise ValueError(f"Régime de cohérence invalide : {coherence}. Utilisez 'coherent' ou 'incoherent'.")
        
        self.wavelength_nm = wavelength_nm
        self.coherence = coherence
        
        # Configuration du logger
        self.logger = logging.getLogger("Propagator")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.info("Propagator initialized with wavelength=%.1f nm, coherence=%s", wavelength_nm, coherence)

    def set_coherence(self, coherence: str) -> None:
        """
        FR: Définit le régime de cohérence.

        EN: Sets the coherence regime.

        Args:
            coherence (str): Régime de cohérence ("coherent" ou "incoherent").

        Raises:
            ValueError: Si la cohérence est invalide.
        """
        if coherence not in ["coherent", "incoherent"]:
            raise ValueError(f"Régime de cohérence invalide : {coherence}. Utilisez 'coherent' ou 'incoherent'.")
        self.coherence = coherence
        self.logger.info("Coherence set to %s", coherence)

    def propagate(
        self,
        electric_field: np.ndarray,
        propagation_distance_m: float,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        method: str = "angular_spectrum",
        output_grid_x: Optional[np.ndarray] = None,
        output_grid_y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        FR: Propage un champ électrique selon la méthode spécifiée.
            Gère automatiquement le régime de cohérence.

        EN: Propagates an electric field according to the specified method.
            Automatically handles the coherence regime.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D à propager.
            propagation_distance_m (float): Distance de propagation en mètres.
            grid_x (np.ndarray): Grille x en mm correspondante au champ d'entrée.
            grid_y (np.ndarray): Grille y en mm correspondante au champ d'entrée.
            method (str): Méthode de propagation. Options:
                - "fraunhofer": Régime de Fraunhofer (champ lointain).
                - "fresnel": Régime de Fresnel (champ proche).
                - "angular_spectrum": Spectre angulaire (méthode la plus précise).
                - "hermite_gauss": Propagation analytique avec modes Hermite-Gauss.
                - "laguerre_gauss": Propagation analytique avec modes Laguerre-Gauss.
            output_grid_x (np.ndarray, optional): Grille x de sortie en mm.
            output_grid_y (np.ndarray, optional): Grille y de sortie en mm.
            **kwargs: Arguments spécifiques à la méthode.

        Returns:
            np.ndarray: Champ électrique propagé (complexe 2D).

        Raises:
            ValueError: Si la méthode est inconnue.
        """
        self.logger.info("Propagating field with method: %s, distance: %.2f m, coherence: %s",
                         method, propagation_distance_m, self.coherence)

        if method == "fraunhofer":
            propagated_field = propagate_fraunhofer(
                electric_field, self.wavelength_nm, propagation_distance_m,
                grid_x, grid_y, output_grid_x, output_grid_y
            )
        elif method == "fresnel":
            propagated_field = propagate_fresnel(
                electric_field, self.wavelength_nm, propagation_distance_m,
                grid_x, grid_y, output_grid_x, output_grid_y
            )
        elif method == "angular_spectrum":
            propagated_field = propagate_angular_spectrum(
                electric_field, self.wavelength_nm, propagation_distance_m,
                grid_x, grid_y, output_grid_x, output_grid_y
            )
        elif method == "hermite_gauss":
            # Projeter sur les modes Hermite-Gauss
            coefficients, modes = project_onto_hermite_gauss(
                electric_field, grid_x, grid_y, self.wavelength_nm, **kwargs
            )
            propagated_field = propagate_analytical_hermite_gauss(
                coefficients, modes, propagation_distance_m, self.wavelength_nm, **kwargs
            )
        elif method == "laguerre_gauss":
            # Projeter sur les modes Laguerre-Gauss
            coefficients, modes = project_onto_laguerre_gauss(
                electric_field, grid_x, grid_y, self.wavelength_nm, **kwargs
            )
            propagated_field = propagate_analytical_laguerre_gauss(
                coefficients, modes, propagation_distance_m, self.wavelength_nm, **kwargs
            )
        else:
            raise ValueError(f"Méthode de propagation inconnue : {method}")

        # Gestion de la cohérence
        if self.coherence == "incoherent":
            # Pour un faisceau incohérent, on prend le module au carré (intensité)
            # et on ignore les effets d'interférence
            propagated_field = np.abs(propagated_field)
            self.logger.warning("Incoherent propagation: returning intensity only (no phase information)")

        return propagated_field

    def propagate_multiple_distances(
        self,
        electric_field: np.ndarray,
        distances_m: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        method: str = "angular_spectrum",
        **kwargs,
    ) -> list:
        """
        FR: Propage un champ électrique à plusieurs distances.

        EN: Propagates an electric field to multiple distances.

        Args:
            electric_field (np.ndarray): Champ électrique complexe 2D à propager.
            distances_m (np.ndarray): Tableau de distances de propagation en mètres.
            grid_x (np.ndarray): Grille x en mm correspondante au champ d'entrée.
            grid_y (np.ndarray): Grille y en mm correspondante au champ d'entrée.
            method (str): Méthode de propagation (défaut: "angular_spectrum").
            **kwargs: Arguments spécifiques à la méthode.

        Returns:
            list: Liste des champs électriques propagés (un par distance).
        """
        propagated_fields = []
        for distance in distances_m:
            propagated_field = self.propagate(
                electric_field, distance, grid_x, grid_y, method, **kwargs
            )
            propagated_fields.append(propagated_field)
        return propagated_fields


# =============================================================================
# 4. FONCTIONS UTILITAIRES / UTILITY FUNCTIONS
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
# 5. TESTS UNITAIRES / UNIT TESTS
# =============================================================================

class TestPropagation:
    """
    FR: Classe de tests unitaires pour Propagation.py.
    EN: Unit test class for Propagation.py.
    """

    def setUp(self):
        self.wavelength_nm = 633.0
        self.diameter_mm = 10.0
        self.grid_x, self.grid_y = create_grid(self.diameter_mm, 128)
        self.electric_field = np.random.rand(128, 128) + 1j * np.random.rand(128, 128)

    def test_propagate_fraunhofer(self):
        propagated = propagate_fraunhofer(
            self.electric_field, self.wavelength_nm, 1.0, self.grid_x, self.grid_y
        )
        self.assertEqual(propagated.shape, (128, 128))
        self.assertEqual(propagated.dtype, np.complex128)

    def test_propagate_fresnel(self):
        propagated = propagate_fresnel(
            self.electric_field, self.wavelength_nm, 0.1, self.grid_x, self.grid_y
        )
        self.assertEqual(propagated.shape, (128, 128))
        self.assertEqual(propagated.dtype, np.complex128)

    def test_propagate_angular_spectrum(self):
        propagated = propagate_angular_spectrum(
            self.electric_field, self.wavelength_nm, 0.01, self.grid_x, self.grid_y
        )
        self.assertEqual(propagated.shape, (128, 128))
        self.assertEqual(propagated.dtype, np.complex128)

    def test_project_onto_hermite_gauss(self):
        coefficients, modes = project_onto_hermite_gauss(
            self.electric_field, self.grid_x, self.grid_y, self.wavelength_nm, n_modes=5
        )
        self.assertEqual(len(coefficients), 5)
        self.assertEqual(modes.shape[0], 5)

    def test_project_onto_laguerre_gauss(self):
        coefficients, modes = project_onto_laguerre_gauss(
            self.electric_field, self.grid_x, self.grid_y, self.wavelength_nm, n_modes=5
        )
        self.assertEqual(len(coefficients), 5)
        self.assertEqual(modes.shape[0], 5)

    def test_propagate_analytical_hermite_gauss(self):
        coefficients = np.random.rand(5) + 1j * np.random.rand(5)
        modes = np.random.rand(5, 128, 128) + 1j * np.random.rand(5, 128, 128)
        propagated = propagate_analytical_hermite_gauss(
            coefficients, modes, 0.1, self.wavelength_nm
        )
        self.assertEqual(propagated.shape, (128, 128))
        self.assertEqual(propagated.dtype, np.complex128)

    def test_propagate_analytical_laguerre_gauss(self):
        coefficients = np.random.rand(5) + 1j * np.random.rand(5)
        modes = np.random.rand(5, 128, 128) + 1j * np.random.rand(5, 128, 128)
        propagated = propagate_analytical_laguerre_gauss(
            coefficients, modes, 0.1, self.wavelength_nm
        )
        self.assertEqual(propagated.shape, (128, 128))
        self.assertEqual(propagated.dtype, np.complex128)

    def test_propagator_class(self):
        propagator = Propagator(wavelength_nm=self.wavelength_nm, coherence="coherent")
        propagated = propagator.propagate(
            self.electric_field, 0.1, self.grid_x, self.grid_y, method="fresnel"
        )
        self.assertEqual(propagated.shape, (128, 128))
        self.assertEqual(propagated.dtype, np.complex128)

    def test_propagator_incoherent(self):
        propagator = Propagator(wavelength_nm=self.wavelength_nm, coherence="incoherent")
        propagated = propagator.propagate(
            self.electric_field, 0.1, self.grid_x, self.grid_y, method="fresnel"
        )
        # Pour incohérent, le résultat doit être réel (intensité)
        self.assertEqual(propagated.shape, (128, 128))
        self.assertTrue(np.isrealobj(propagated))

    def test_get_propagation_regime(self):
        # Fraunhofer
        regime = get_propagation_regime(10.0, 633.0, 100.0)
        self.assertEqual(regime, "Fraunhofer")
        
        # Fresnel
        regime = get_propagation_regime(10.0, 633.0, 0.1)
        self.assertEqual(regime, "Fresnel")
        
        # Near Field
        regime = get_propagation_regime(10.0, 633.0, 0.001)
        self.assertEqual(regime, "Near Field")

    def test_estimate_required_resolution(self):
        resolution = estimate_required_resolution(10.0, 633.0, 1.0)
        self.assertGreaterEqual(resolution, 128)  # Doit être au moins 128


if __name__ == "__main__":
    import unittest
    unittest.main()