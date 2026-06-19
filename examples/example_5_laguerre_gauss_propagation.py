"""
Example 5: Laguerre-Gauss Beam Propagation
FR: Exemple de propagation analytique d'un faisceau en utilisant la base Laguerre-Gauss.
    Démonstration de la décomposition modale pour les faisceaux cylindriques,
    de la propagation analytique, et de la visualisation des modes.

EN: Example of analytical propagation of a beam using Laguerre-Gauss basis.
    Demonstrates modal decomposition for cylindrical beams,
    analytical propagation, and mode visualization.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import math  # Pour math.factorial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Beam import Beam
from Propagation import Propagation
from Visualization import plot_intensity, plot_phase, plot_beam_map


def run_laguerre_gauss_example():
    """FR: Exécute l'exemple de propagation Laguerre-Gauss."""
    print("\n" + "="*80)
    print("Example 5: Laguerre-Gauss Beam Propagation")
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
    
    # Générer un champ électrique gaussien (bien représenté par Laguerre-Gauss)
    electric_field = beam.generate_electric_field(method="gaussian", sigma_mm=2.0)
    
    # Extraire l'intensité et la phase initiales
    intensity_initial = beam.compute_intensity_from_electric_field(electric_field)
    phase_initial = beam.extract_phase_from_electric_field(electric_field)
    
    pv_initial, rms_initial = beam.compute_pv_rms(phase_initial)
    print(f"Faisceau initial: PV={pv_initial:.2f} nm, RMS={rms_initial:.2f} nm")
    
    # =========================================================================
    # 2. Propagation analytique avec Laguerre-Gauss
    # =========================================================================
    print("\n--- Propagation analytique Laguerre-Gauss ---")
    
    propagation_distance_mm = 500.0
    
    # Test avec différentes combinaisons de p et l
    p_max_list = [3, 5, 7]
    l_max_list = [3, 5, 7]
    
    for p_max in p_max_list:
        for l_max in l_max_list:
            print(f"\n p_max={p_max}, l_max={l_max}")
            
            propagator_lg = Propagation(
                wavelength_nm=wavelength_nm,
                propagation_distance_mm=propagation_distance_mm,
                input_diameter_mm=diameter_mm,
                num_points=num_points,
                coherence="coherent",
                method="laguerre_gauss"
            )
            
            # Propage avec décomposition modale
            propagated_field_lg, metrics = propagator_lg.propagate(
                electric_field,
                max_p=p_max,
                max_l=l_max
            )
            
            # Afficher les métriques
            print(f"  Erreur RMS intensité: {metrics['intensity_error_rms']:.6f}")
            print(f"  Erreur RMS phase: {metrics['phase_error_rms_nm']:.4f} nm")
            print(f"  Nombre de modes: {(2*l_max+1)*(p_max+1)}")
            
            # Visualisation pour p_max=5, l_max=5
            if p_max == 5 and l_max == 5:
                plot_intensity(
                    metrics['intensity_propagated'],
                    diameter_mm,
                    title=f"Laguerre-Gauss Propagation (p_max={p_max}, l_max={l_max}) - Intensity"
                )
                plot_phase(
                    metrics['phase_propagated'],
                    diameter_mm,
                    title=f"Laguerre-Gauss Propagation (p_max={p_max}, l_max={l_max}) - Phase"
                )
                plt.savefig('examples/output/example5_laguerre_gauss_5x5.png', dpi=150, bbox_inches='tight')
                plt.close('all')
    
    # =========================================================================
    # 3. Faisceau avec moment angulaire orbital (OAM)
    # =========================================================================
    print("\n--- Faisceau avec OAM (l ≠ 0) ---")
    
    # Créer un faisceau avec OAM (mode Laguerre-Gauss pur)
    grid_x, grid_y = beam.grid_x, beam.grid_y
    r = np.sqrt(grid_x**2 + grid_y**2)
    theta = np.arctan2(grid_y, grid_x)
    sigma_mm = 2.0
    
    # Mode LG avec l = 1 (OAM)
    from scipy.special import eval_genlaguerre
    p = 0
    l = 1
    L_pl = eval_genlaguerre(p, abs(l), 2 * r**2 / sigma_mm**2)
    gauss = np.exp(-r**2 / sigma_mm**2)
    angular = np.exp(1j * l * theta)
    norm_factor = np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + abs(l)))) / sigma_mm
    lg_mode = norm_factor * L_pl * gauss * angular
    
    # Créer un champ électrique avec ce mode
    electric_field_oam = lg_mode * np.sqrt(energy / np.sum(np.abs(lg_mode)**2))
    
    # Propager
    propagator_oam = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=propagation_distance_mm,
        input_diameter_mm=diameter_mm,
        num_points=num_points,
        method="laguerre_gauss"
    )
    
    propagated_field_oam, metrics_oam = propagator_oam.propagate(
        electric_field_oam,
        max_p=3,
        max_l=3
    )
    
    print(f"Faisceau OAM (l={l}):")
    print(f"  Erreur RMS intensité: {metrics_oam['intensity_error_rms']:.6f}")
    print(f"  Erreur RMS phase: {metrics_oam['phase_error_rms_nm']:.4f} nm")
    
    # Visualisation
    plot_intensity(
        metrics_oam['intensity_initial'],
        diameter_mm,
        title=f"OAM Beam (l={l}) - Initial Intensity"
    )
    plot_phase(
        metrics_oam['phase_initial'],
        diameter_mm,
        title=f"OAM Beam (l={l}) - Initial Phase"
    )
    plt.savefig('examples/output/example5_oam_initial.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    plot_intensity(
        metrics_oam['intensity_propagated'],
        diameter_mm,
        title=f"OAM Beam (l={l}) - Propagated Intensity"
    )
    plot_phase(
        metrics_oam['phase_propagated'],
        diameter_mm,
        title=f"OAM Beam (l={l}) - Propagated Phase"
    )
    plt.savefig('examples/output/example5_oam_propagated.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 4. Comparaison avec les méthodes numériques
    # =========================================================================
    print("\n--- Comparaison Laguerre-Gauss vs Méthodes Numériques ---")
    
    # Propagation Laguerre-Gauss de référence
    propagator_lg_ref = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=propagation_distance_mm,
        input_diameter_mm=diameter_mm,
        num_points=num_points,
        method="laguerre_gauss"
    )
    propagated_field_lg_ref, metrics_lg_ref = propagator_lg_ref.propagate(
        electric_field,
        max_p=7,
        max_l=7
    )
    
    # Comparaison avec angular_spectrum
    propagator_num = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=propagation_distance_mm,
        input_diameter_mm=diameter_mm,
        num_points=num_points,
        method="angular_spectrum"
    )
    
    propagated_field_num = propagator_num.propagate(electric_field)
    
    # Calculer les différences
    intensity_lg = beam.compute_intensity_from_electric_field(propagated_field_lg_ref)
    intensity_num = beam.compute_intensity_from_electric_field(propagated_field_num)
    
    # Normaliser
    intensity_lg_norm = intensity_lg / np.max(intensity_lg)
    intensity_num_norm = intensity_num / np.max(intensity_num)
    
    intensity_diff = intensity_lg_norm - intensity_num_norm
    rms_error = np.sqrt(np.mean(intensity_diff**2))
    max_error = np.max(np.abs(intensity_diff))
    
    print(f"Comparaison avec angular_spectrum:")
    print(f"  Erreur RMS: {rms_error:.6f}")
    print(f"  Erreur maximale: {max_error:.6f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im1 = axes[0].imshow(
        intensity_lg_norm,
        extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
        cmap='viridis'
    )
    axes[0].set_title("Laguerre-Gauss (p_max=7, l_max=7)")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(
        intensity_num_norm,
        extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
        cmap='viridis'
    )
    axes[1].set_title("Angular Spectrum")
    axes[1].set_xlabel("x (mm)")
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(
        intensity_diff,
        extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
        cmap='coolwarm'
    )
    axes[2].set_title("Difference (LG - AS)")
    axes[2].set_xlabel("x (mm)")
    plt.colorbar(im3, ax=axes[2], label="Error")
    
    plt.tight_layout()
    plt.savefig('examples/output/example5_comparison_lg_as.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 5. Visualisation des modes Laguerre-Gauss individuels
    # =========================================================================
    print("\n--- Visualisation des modes Laguerre-Gauss individuels ---")
    
    # Modes à visualiser
    modes_to_plot = [(0, 0), (0, 1), (0, -1), (1, 0), (0, 2), (0, -2)]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    
    for idx, (p, l) in enumerate(modes_to_plot):
        # Générer le mode
        L_pl = eval_genlaguerre(p, abs(l), 2 * r**2 / sigma_mm**2)
        gauss = np.exp(-r**2 / sigma_mm**2)
        angular = np.exp(1j * l * theta) if l >= 0 else np.exp(-1j * abs(l) * theta)
        norm_factor = np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + abs(l)))) / sigma_mm
        mode = norm_factor * L_pl * gauss * angular
        
        # Afficher la partie réelle (ou le module pour les modes complexes)
        if l != 0:
            # Pour les modes avec OAM, afficher le module
            mode_display = np.abs(mode)
            cmap = 'viridis'
            label = "Amplitude"
        else:
            # Pour les modes réels, afficher la valeur réelle
            mode_display = np.real(mode)
            cmap = 'coolwarm'
            label = "Value"
        
        im = axes_flat[idx].imshow(
            mode_display,
            extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
            cmap=cmap
        )
        axes_flat[idx].set_title(f"Mode LG (p={p}, l={l})")
        axes_flat[idx].set_xlabel("x (mm)")
        axes_flat[idx].set_ylabel("y (mm)")
        plt.colorbar(im, ax=axes_flat[idx], label=label)
    
    plt.tight_layout()
    plt.savefig('examples/output/example5_laguerre_gauss_modes.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    print("\n" + "="*80)
    print("Example 5 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('examples/output', exist_ok=True)
    run_laguerre_gauss_example()
