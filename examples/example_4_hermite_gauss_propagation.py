"""
Example 4: Hermite-Gauss Beam Propagation
FR: Exemple de propagation analytique d'un faisceau en utilisant la base Hermite-Gauss.
    Démonstration de la décomposition modale, de la propagation analytique,
    et de la comparaison avec les méthodes numériques.

EN: Example of analytical propagation of a beam using Hermite-Gauss basis.
    Demonstrates modal decomposition, analytical propagation,
    and comparison with numerical methods.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Beam import Beam
from Propagation import Propagation
from Visualization import plot_intensity, plot_phase, plot_beam_map


def run_hermite_gauss_example():
    """FR: Exécute l'exemple de propagation Hermite-Gauss."""
    print("\n" + "="*80)
    print("Example 4: Hermite-Gauss Beam Propagation")
    print("="*80)
    
    # =========================================================================
    # 1. Génération du faisceau initial
    # =========================================================================
    print("\n--- Génération du faisceau initial ---")
    
    wavelength_nm = 633.0
    diameter_mm = 10.0
    energy = 1.0
    num_points = 256
    
    # Créer le faisceau
    beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=diameter_mm,
        energy=energy,
        num_points=num_points,
        coherence="coherent"
    )
    
    # Générer un champ électrique gaussien (qui est déjà bien représenté par Hermite-Gauss)
    electric_field = beam.generate_electric_field(method="gaussian", sigma_mm=2.0)
    
    # Extraire l'intensité et la phase initiales
    intensity_initial = beam.compute_intensity_from_electric_field(electric_field)
    phase_initial = beam.extract_phase_from_electric_field(electric_field)
    
    pv_initial, rms_initial = beam.compute_pv_rms(phase_initial)
    print(f"Faisceau initial: PV={pv_initial:.2f} nm, RMS={rms_initial:.2f} nm")
    
    # =========================================================================
    # 2. Propagation analytique avec Hermite-Gauss
    # =========================================================================
    print("\n--- Propagation analytique Hermite-Gauss ---")
    
    propagation_distance_mm = 500.0
    max_modes_list = [5, 10, 20, 30]
    
    for max_modes in max_modes_list:
        print(f"\nNombre de modes: {max_modes}")
        
        propagator_hg = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=propagation_distance_mm,
            input_diameter_mm=diameter_mm,
            num_points=num_points,
            coherence="coherent",
            method="hermite_gauss"
        )
        
        # Propage avec décomposition modale
        propagated_field_hg, metrics = propagator_hg.propagate(
            electric_field,
            max_modes=max_modes
        )
        
        # Afficher les métriques
        print(f"  Erreur RMS intensité: {metrics['intensity_error_rms']:.6f}")
        print(f"  Erreur RMS phase: {metrics['phase_error_rms_nm']:.4f} nm")
        print(f"  Énergie initiale: {np.sum(intensity_initial):.4f}")
        print(f"  Énergie propagée: {np.sum(metrics['intensity_propagated']):.4f}")
        
        # Visualisation pour max_modes=10
        if max_modes == 10:
            plot_intensity(
                metrics['intensity_propagated'],
                diameter_mm,
                title=f"Hermite-Gauss Propagation ({max_modes} modes) - Intensity"
            )
            plot_phase(
                metrics['phase_propagated'],
                diameter_mm,
                title=f"Hermite-Gauss Propagation ({max_modes} modes) - Phase"
            )
            plt.savefig('examples/output/example4_hermite_gauss_10_modes.png', dpi=150, bbox_inches='tight')
            plt.close('all')
    
    # =========================================================================
    # 3. Comparaison avec les méthodes numériques
    # =========================================================================
    print("\n--- Comparaison Hermite-Gauss vs Méthodes Numériques ---")
    
    # Méthodes numériques à comparer
    numerical_methods = ["angular_spectrum", "fraunhofer"]
    
    # Propagation Hermite-Gauss de référence (avec beaucoup de modes)
    propagator_hg_ref = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=propagation_distance_mm,
        input_diameter_mm=diameter_mm,
        num_points=num_points,
        method="hermite_gauss"
    )
    propagated_field_hg_ref, metrics_hg_ref = propagator_hg_ref.propagate(
        electric_field,
        max_modes=30
    )
    
    # Comparaison avec chaque méthode numérique
    for method in numerical_methods:
        print(f"\nComparaison avec {method}:")
        
        propagator_num = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=propagation_distance_mm,
            input_diameter_mm=diameter_mm,
            num_points=num_points,
            method=method
        )
        
        propagated_field_num = propagator_num.propagate(electric_field)
        
        # Calculer les différences
        intensity_hg = beam.compute_intensity_from_electric_field(propagated_field_hg_ref)
        intensity_num = beam.compute_intensity_from_electric_field(propagated_field_num)
        
        # Normaliser pour comparaison
        intensity_hg_norm = intensity_hg / np.max(intensity_hg)
        intensity_num_norm = intensity_num / np.max(intensity_num)
        
        # Calculer les métriques de différence
        intensity_diff = intensity_hg_norm - intensity_num_norm
        rms_error = np.sqrt(np.mean(intensity_diff**2))
        max_error = np.max(np.abs(intensity_diff))
        
        print(f"  Erreur RMS: {rms_error:.6f}")
        print(f"  Erreur maximale: {max_error:.6f}")
        
        # Visualisation de la comparaison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        im1 = axes[0].imshow(
            intensity_hg_norm,
            extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
            cmap='viridis'
        )
        axes[0].set_title(f"Hermite-Gauss (30 modes)")
        axes[0].set_xlabel("x (mm)")
        axes[0].set_ylabel("y (mm)")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(
            intensity_num_norm,
            extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
            cmap='viridis'
        )
        axes[1].set_title(f"{method.capitalize()}")
        axes[1].set_xlabel("x (mm)")
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(
            intensity_diff,
            extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
            cmap='coolwarm'
        )
        axes[2].set_title(f"Difference (HG - {method})")
        axes[2].set_xlabel("x (mm)")
        plt.colorbar(im3, ax=axes[2], label="Error")
        
        plt.tight_layout()
        plt.savefig(f'examples/output/example4_comparison_hg_{method}.png', dpi=150, bbox_inches='tight')
        plt.close('all')
    
    # =========================================================================
    # 4. Test avec différents types de faisceaux
    # =========================================================================
    print("\n--- Test avec différents types de faisceaux ---")
    
    beam_types = [
        ("gaussian", {"sigma_mm": 2.0}),
        ("supergaussian", {"sigma_mm": 2.0, "n": 4}),
        ("tophat", {})
    ]
    
    for beam_type, params in beam_types:
        print(f"\nType de faisceau: {beam_type}")
        
        # Générer le faisceau
        beam_test = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=diameter_mm,
            energy=energy,
            num_points=128
        )
        
        intensity_test = beam_test.generate_intensity(method=beam_type, **params)
        electric_field_test = beam_test.generate_electric_field(
            intensity=intensity_test,
            phase=np.zeros_like(intensity_test),  # Phase uniforme
            method="from_intensity_phase"
        )
        
        # Propager avec Hermite-Gauss
        propagator_test = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=propagation_distance_mm,
            input_diameter_mm=diameter_mm,
            num_points=128,
            method="hermite_gauss"
        )
        
        propagated_field_test, metrics_test = propagator_test.propagate(
            electric_field_test,
            max_modes=20
        )
        
        print(f"  Erreur RMS intensité: {metrics_test['intensity_error_rms']:.6f}")
        print(f"  Erreur RMS phase: {metrics_test['phase_error_rms_nm']:.4f} nm")
    
    # =========================================================================
    # 5. Visualisation des modes Hermite-Gauss individuels
    # =========================================================================
    print("\n--- Visualisation des modes Hermite-Gauss individuels ---")
    
    # Créer une grille pour la visualisation
    grid_x, grid_y = beam.grid_x, beam.grid_y
    sigma_mm = 2.0
    
    # Visualiser les premiers modes Hermite-Gauss
    modes_to_plot = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    
    for idx, (n, m) in enumerate(modes_to_plot):
        # Générer le mode
        from scipy.special import hermite
        H_n = hermite(n)(grid_x / sigma_mm)
        H_m = hermite(m)(grid_y / sigma_mm)
        gauss = np.exp(-(grid_x**2 + grid_y**2) / (2 * sigma_mm**2))
        norm_factor = 1.0 / (sigma_mm * np.sqrt(2**(n + m) * math.factorial(n) * math.factorial(m) * np.pi))
        mode = norm_factor * H_n * H_m * gauss
        
        # Afficher le mode
        im = axes_flat[idx].imshow(
            mode,
            extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
            cmap='coolwarm'
        )
        axes_flat[idx].set_title(f"Mode HG ({n},{m})")
        axes_flat[idx].set_xlabel("x (mm)")
        axes_flat[idx].set_ylabel("y (mm)")
        plt.colorbar(im, ax=axes_flat[idx])
    
    plt.tight_layout()
    plt.savefig('examples/output/example4_hermite_gauss_modes.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    print("\n" + "="*80)
    print("Example 4 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    import os
    import math  # Pour math.factorial
    os.makedirs('examples/output', exist_ok=True)
    run_hermite_gauss_example()
