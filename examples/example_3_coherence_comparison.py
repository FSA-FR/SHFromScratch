"""
Example 3: Coherence Comparison
FR: Comparaison entre propagation cohérente et incohérente.
    Démonstration de l'impact du régime de cohérence sur les résultats de propagation.

EN: Comparison between coherent and incoherent propagation.
    Demonstrates the impact of coherence regime on propagation results.

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


def run_coherence_comparison_example():
    """FR: Exécute la comparaison entre propagation cohérente et incohérente."""
    print("\n" + "="*80)
    print("Example 3: Coherence Comparison")
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
        coherence="coherent"  # On commence avec cohérent
    )
    
    # Générer un champ électrique avec phase aléatoire
    intensity = beam.generate_intensity(method="gaussian", sigma_mm=2.0)
    phase = beam.generate_phase(method="random_zernike", n_modes=10, max_amplitude_nm=50.0)
    electric_field = beam.generate_electric_field(
        intensity=intensity,
        phase=phase,
        method="from_intensity_phase"
    )
    
    print(f"Faisceau initial: PV phase={beam.compute_pv_rms(phase)[0]:.2f} nm")
    
    # =========================================================================
    # 2. Propagation cohérente
    # =========================================================================
    print("\n--- Propagation COHÉRENTE ---")
    
    propagation_distance_mm = 500.0
    
    propagator_coherent = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=propagation_distance_mm,
        input_diameter_mm=diameter_mm,
        num_points=num_points,
        coherence="coherent",
        method="angular_spectrum"
    )
    
    # Propage le champ électrique
    propagated_field_coherent = propagator_coherent.propagate(electric_field)
    
    # Extraire l'intensité et la phase
    intensity_coherent = beam.compute_intensity_from_electric_field(propagated_field_coherent)
    phase_coherent = beam.extract_phase_from_electric_field(propagated_field_coherent)
    
    pv_coherent, rms_coherent = beam.compute_pv_rms(phase_coherent)
    print(f"Résultat cohérent:")
    print(f"  Intensité: somme={np.sum(intensity_coherent):.4f} (a.u.)")
    print(f"  Phase: PV={pv_coherent:.2f} nm, RMS={rms_coherent:.2f} nm")
    print(f"  Type du résultat: {propagated_field_coherent.dtype} (complexe)")
    
    # Visualisation
    plot_intensity(
        intensity_coherent,
        diameter_mm,
        title="Coherent Propagation - Intensity"
    )
    plot_phase(
        phase_coherent,
        diameter_mm,
        title="Coherent Propagation - Phase"
    )
    plt.savefig('examples/output/example3_coherent_propagation.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 3. Propagation incohérente
    # =========================================================================
    print("\n--- Propagation INCOHÉRENTE ---")
    
    propagator_incoherent = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=propagation_distance_mm,
        input_diameter_mm=diameter_mm,
        num_points=num_points,
        coherence="incoherent",
        method="angular_spectrum"
    )
    
    # Propage le champ électrique (retourne l'intensité)
    propagated_intensity_incoherent = propagator_incoherent.propagate(electric_field)
    
    # Pour incohérent, le résultat est déjà une intensité
    if isinstance(propagated_intensity_incoherent, np.ndarray):
        intensity_incoherent = propagated_intensity_incoherent
    else:
        # Si c'est un tuple (champ, métriques), prendre l'intensité
        intensity_incoherent = np.abs(propagated_intensity_incoherent)**2
    
    print(f"Résultat incohérent:")
    print(f"  Intensité: somme={np.sum(intensity_incoherent):.4f} (a.u.)")
    print(f"  Type du résultat: réel (intensité seulement)")
    
    # Visualisation
    plot_intensity(
        intensity_incoherent,
        diameter_mm,
        title="Incoherent Propagation - Intensity"
    )
    plt.savefig('examples/output/example3_incoherent_propagation.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 4. Comparaison directe
    # =========================================================================
    print("\n--- Comparaison cohérent vs incohérent ---")
    
    # Calculer les différences
    intensity_diff = intensity_coherent - intensity_incoherent
    
    # Métriques de comparaison
    max_diff = np.max(np.abs(intensity_diff))
    mean_diff = np.mean(np.abs(intensity_diff))
    rms_diff = np.sqrt(np.mean(intensity_diff**2))
    
    print(f"Différence entre cohérent et incohérent:")
    print(f"  Différence maximale: {max_diff:.4f}")
    print(f"  Différence moyenne: {mean_diff:.4f}")
    print(f"  Différence RMS: {rms_diff:.4f}")
    
    # Visualisation de la différence
    plot_beam_map(
        intensity_diff,
        diameter_mm,
        title="Difference: Coherent - Incoherent Intensity",
        label="Difference (a.u.)",
        cmap="coolwarm"
    )
    plt.savefig('examples/output/example3_coherence_difference.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 5. Test avec faisceau uniformément illuminé
    # =========================================================================
    print("\n--- Test avec faisceau top-hat (uniforme) ---")
    
    # Générer un faisceau top-hat
    beam_tophat = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=diameter_mm,
        energy=energy,
        num_points=num_points
    )
    
    intensity_tophat = beam_tophat.generate_intensity(method="tophat")
    electric_field_tophat = beam_tophat.generate_electric_field(
        intensity=intensity_tophat,
        phase=np.zeros_like(intensity_tophat),  # Phase uniforme
        method="from_intensity_phase"
    )
    
    # Propager en cohérent
    propagator_coherent_th = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=propagation_distance_mm,
        input_diameter_mm=diameter_mm,
        num_points=num_points,
        coherence="coherent",
        method="fraunhofer"
    )
    propagated_field_coherent_th = propagator_coherent_th.propagate(electric_field_tophat)
    intensity_coherent_th = beam_tophat.compute_intensity_from_electric_field(propagated_field_coherent_th)
    
    # Propager en incohérent
    propagator_incoherent_th = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=propagation_distance_mm,
        input_diameter_mm=diameter_mm,
        num_points=num_points,
        coherence="incoherent",
        method="fraunhofer"
    )
    intensity_incoherent_th = propagator_incoherent_th.propagate(electric_field_tophat)
    
    print(f"Faisceau top-hat:")
    print(f"  Cohérent: somme intensité propagée = {np.sum(intensity_coherent_th):.4f}")
    print(f"  Incohérent: somme intensité propagée = {np.sum(intensity_incoherent_th):.4f}")
    print(f"  Ratio incohérent/cohérent = {np.sum(intensity_incoherent_th)/np.sum(intensity_coherent_th):.4f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        intensity_coherent_th,
        extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
        cmap='viridis'
    )
    axes[0].set_title("Top-Hat Beam - Coherent Propagation")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Intensity (a.u.)")
    
    im2 = axes[1].imshow(
        intensity_incoherent_th,
        extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
        cmap='viridis'
    )
    axes[1].set_title("Top-Hat Beam - Incoherent Propagation")
    axes[1].set_xlabel("x (mm)")
    plt.colorbar(im2, ax=axes[1], label="Intensity (a.u.)")
    
    plt.tight_layout()
    plt.savefig('examples/output/example3_tophat_coherence_comparison.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    print("\n" + "="*80)
    print("Example 3 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('examples/output', exist_ok=True)
    run_coherence_comparison_example()
