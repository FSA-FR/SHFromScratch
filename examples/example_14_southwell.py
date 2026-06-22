"""
Example 14: Southwell.py - Phase Reconstruction from Slopes
FR: Exemple complet d'utilisation du module Southwell.py.
    Démonstration de :
    - Reconstruction de phase à partir des pentes locales (Shack-Hartmann)
    - Comparaison de 16 algorithmes différents :
        * Southwell (original, régularisé, pondéré, itératif)
        * Moindres carrés (standard, pondéré, Tikhonov)
        * Modal (Zernike, Legendre, personnalisé)
        * Autres (Hudgin, Fried, Gendron, Poyneer, Wallner, Roddier)
    - Intégration avec Shack_Hartmann.py pour obtenir les pentes
    - Visualisation des phases reconstruites
    - Comparaison des performances (PV, RMS, temps de calcul)
    - Sélection du meilleur algorithme
    - Export des résultats

    Unités :
    - Phase : nm (nanomètres)
    - Pentes : rad (radians)

EN: Complete example of using Southwell.py module.
    Demonstrates:
    - Phase reconstruction from local slopes (Shack-Hartmann)
    - Comparison of 16 different algorithms:
        * Southwell (original, regularized, weighted, iterative)
        * Least squares (standard, weighted, Tikhonov)
        * Modal (Zernike, Legendre, custom)
        * Others (Hudgin, Fried, Gendron, Poyneer, Wallner, Roddier)
    - Integration with Shack_Hartmann.py to get slopes
    - Visualization of reconstructed phases
    - Performance comparison (PV, RMS, computation time)
    - Best algorithm selection
    - Results export

    Units:
    - Phase: nm (nanometers)
    - Slopes: rad (radians)

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - matplotlib
    - Southwell.py
    - Shack_Hartmann.py
    - Beam.py
    - Microstructure.py
    - Camera.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

from Southwell import (
    SouthwellReconstructor, create_southwell_reconstructor,
    ReconstructionAlgorithm, ReconstructionResult
)
from Shack_Hartmann import create_shack_hartmann, CentroidAlgorithm
from Beam import Beam
from Optiques import WaveFrontError


def run_southwell_example():
    """
    FR: Exécute l'exemple Southwell.
        Démonstration complète des fonctionnalités du module.
    """
    print("\n" + "="*80)
    print("Example 14: Southwell.py - Phase Reconstruction from Slopes")
    print("="*80)
    
    # Paramètres globaux
    wavelength_nm = 633.0
    output_dir = 'examples/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # 1. CRÉATION D'UN SYSTÈME SHACK-HARTMANN POUR OBTENIR DES PENTES
    # =========================================================================
    print("\n--- 1. Génération des pentes avec Shack-Hartmann ---")
    
    # 1.1 Créer un système Shack-Hartmann
    print("\n1.1 Création du système Shack-Hartmann")
    sh = create_shack_hartmann(
        name="Southwell_Input",
        num_microlenses_x=8,
        num_microlenses_y=8,
        microlens_pitch_mm=0.5,
        focal_length_mm=20.0,
        camera_num_pixels_x=256,
        camera_num_pixels_y=256,
        camera_pixel_size_um=10.0,  # 10 µm
        wavelength_nm=wavelength_nm,
        centroid_algorithm=CentroidAlgorithm.WEIGHTED_CENTROID,
        display=False
    )
    
    print(f"  {sh}")
    
    # 1.2 Créer un faisceau avec des aberrations connues
    print("\n1.2 Création d'un faisceau avec aberrations Zernike")
    beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        num_points=256
    )
    
    electric_field = beam.generate_electric_field(method="gaussian", sigma_mm=1.0)
    
    # Ajouter des aberrations (coefficient en nm)
    wfe = WaveFrontError(
        zernike_coefficients={
            (2, 0): 10.0,   # Defocus (Z4)
            (2, 2): 5.0,    # Astigmatisme (Z5)
            (3, 1): 3.0,    # Coma (Z7)
            (3, 3): 2.0,    # Coma secondaire (Z8)
            (4, 0): 1.0     # Spherical (Z11)
        },
        seed=42
    )
    
    # Appliquer les aberrations au faisceau
    x = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
    y = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
    X, Y = np.meshgrid(x, y)
    
    phase_map_nm = wfe.generate_phase_map(X, Y, beam.wavelength_nm)
    phase_rad = phase_map_nm * 2 * np.pi / beam.wavelength_nm
    
    aberrated_field = np.abs(electric_field) * np.exp(1j * (np.angle(electric_field) + phase_rad))
    beam.electric_field = aberrated_field
    beam.intensity = beam.compute_intensity_from_electric_field(aberrated_field)
    beam.phase = beam.extract_phase_from_electric_field(aberrated_field)
    
    # Statistiques du faisceau
    pv_beam, rms_beam = beam.compute_pv_rms(beam.phase)
    print(f"  Faisceau avec aberrations: PV={pv_beam:.2f} nm, RMS={rms_beam:.2f} nm")
    print(f"  Aberrations: {wfe.zernike_coefficients}")
    
    # 1.3 Simuler pour obtenir les pentes
    print("\n1.3 Simulation Shack-Hartmann pour obtenir les pentes")
    sh.simulate(beam)
    
    slopes_x, slopes_y = sh.get_slope_maps()
    
    # Statistiques des pentes
    stats = sh.get_slope_statistics()
    print(f"  Pentes X: PV={stats['slopes_x']['pv']:.6f} rad, RMS={stats['slopes_x']['rms']:.6f} rad")
    print(f"  Pentes Y: PV={stats['slopes_y']['pv']:.6f} rad, RMS={stats['slopes_y']['rms']:.6f} rad")
    
    # Sauvegarder les pentes pour référence
    np.save(os.path.join(output_dir, "slopes_x.npy"), slopes_x)
    np.save(os.path.join(output_dir, "slopes_y.npy"), slopes_y)
    print(f"  ✓ Pentes sauvegardées dans {output_dir}/")
    
    # =========================================================================
    # 2. RECONSTRUCTION AVEC SOUTHWELL (ALGORITHME PAR DÉFAUT)
    # =========================================================================
    print("\n--- 2. Reconstruction avec Southwell (par défaut) ---")
    
    # 2.1 Créer un reconstructeur
    print("\n2.1 Création du reconstructeur")
    reconstructor = create_southwell_reconstructor(
        name="Southwell_Default",
        wavelength_nm=wavelength_nm,
        pixel_size_mm=sh.camera.pixel_width_mm
    )
    
    print(f"  {reconstructor}")
    
    # 2.2 Reconstruire avec l'algorithme par défaut (Southwell)
    print("\n2.2 Reconstruction avec Southwell original")
    result_southwell = reconstructor.reconstruct(slopes_x, slopes_y)
    
    print(f"  Résultat:")
    print(f"    PV: {result_southwell.pv:.2f} nm")
    print(f"    RMS: {result_southwell.rms:.2f} nm")
    print(f"    Temps: {result_southwell.computation_time:.4f} s")
    print(f"    Succès: {result_southwell.success}")
    
    # 2.3 Visualiser la phase reconstruite
    print("\n2.3 Visualisation de la phase reconstruite")
    plt.figure(figsize=(10, 8))
    plt.imshow(result_southwell.phase, cmap='Jet')
    plt.title(f"Phase reconstruite - Southwell\nPV={result_southwell.pv:.2f} nm, RMS={result_southwell.rms:.2f} nm")
    plt.colorbar(label="Phase (nm)")
    plt.savefig(os.path.join(output_dir, "southwell_default.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: southwell_default.png")
    
    # =========================================================================
    # 3. COMPARAISON DE TOUS LES ALGORITHMES
    # =========================================================================
    print("\n--- 3. Comparaison de tous les algorithmes ---")
    
    # 3.1 Liste de tous les algorithmes
    print("\n3.1 Liste des algorithmes à tester")
    all_algorithms = list(ReconstructionAlgorithm)
    print(f"  Nombre d'algorithmes: {len(all_algorithms)}")
    for algo in all_algorithms:
        print(f"    - {algo.value}")
    
    # 3.2 Exécuter tous les algorithmes
    print("\n3.2 Exécution de tous les algorithmes...")
    results = reconstructor.compare_algorithms(slopes_x, slopes_y)
    
    # 3.3 Tableau comparatif
    print("\n3.3 Tableau comparatif des algorithmes")
    print("\n{:<25} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Algorithme", "PV (nm)", "RMS (nm)", "Temps (s)", "Succès", "Erreur"
    ))
    print("-" * 85)
    
    comparison_data = []
    for algo in all_algorithms:
        result = results[algo.value]
        status = "✓" if result.success else "✗"
        error = "-" if result.success else result.error[:20]
        
        print("{:<25} {:<10.2f} {:<10.2f} {:<10.4f} {:<10} {:<10}".format(
            algo.value,
            result.pv,
            result.rms,
            result.computation_time,
            status,
            error
        ))
        
        comparison_data.append({
            'algorithm': algo.value,
            'pv': result.pv,
            'rms': result.rms,
            'time': result.computation_time,
            'success': result.success,
            'error': result.error
        })
    
    # 3.4 Visualisation comparative
    print("\n3.4 Visualisation comparative")
    
    # Sélectionner quelques algorithmes clés pour la visualisation
    key_algorithms = [
        ReconstructionAlgorithm.SOUTHWELL,
        ReconstructionAlgorithm.SOUTHWELL_REGULARIZED,
        ReconstructionAlgorithm.LEAST_SQUARES,
        ReconstructionAlgorithm.TIKHONOV,
        ReconstructionAlgorithm.MODAL_ZERNIKE,
        ReconstructionAlgorithm.POYNEER
    ]
    
    key_results = {algo.value: results[algo.value] for algo in key_algorithms if algo.value in results}
    
    # Créer une grille de visualisation
    n_algos = len(key_results)
    n_cols = 3
    n_rows = (n_algos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for idx, (algo_name, result) in enumerate(key_results.items()):
        ax = axes[idx]
        im = ax.imshow(result.phase, cmap='Jet')
        ax.set_title(f"{algo_name}\nPV={result.pv:.2f}nm, RMS={result.rms:.2f}nm")
        plt.colorbar(im, ax=ax)
    
    # Masquer les axes inutilisés
    for idx in range(n_algos, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "southwell_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: southwell_comparison.png")
    
    # 3.5 Graphique des performances
    print("\n3.5 Graphique des performances")
    
    # Extraire les données
    algo_names = [r['algorithm'] for r in comparison_data if r['success']]
    pvs = [r['pv'] for r in comparison_data if r['success']]
    rmss = [r['rms'] for r in comparison_data if r['success']]
    times = [r['time'] for r in comparison_data if r['success']]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PV
    axes[0].bar(algo_names, pvs)
    axes[0].set_title("PV (nm) par algorithme")
    axes[0].set_xticklabels(algo_names, rotation=45, ha='right')
    axes[0].set_ylabel("PV (nm)")
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # RMS
    axes[1].bar(algo_names, rmss)
    axes[1].set_title("RMS (nm) par algorithme")
    axes[1].set_xticklabels(algo_names, rotation=45, ha='right')
    axes[1].set_ylabel("RMS (nm)")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Temps
    axes[2].bar(algo_names, times)
    axes[2].set_title("Temps (s) par algorithme")
    axes[2].set_xticklabels(algo_names, rotation=45, ha='right')
    axes[2].set_ylabel("Temps (s)")
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "southwell_performance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: southwell_performance.png")
    
    # =========================================================================
    # 4. SÉLECTION DU MEILLEUR ALGORITHME
    # =========================================================================
    print("\n--- 4. Sélection du meilleur algorithme ---")
    
    # 4.1 Meilleur selon PV
    print("\n4.1 Meilleur algorithme selon PV (plus petit PV)")
    best_algo_pv, best_result_pv = reconstructor.get_best_algorithm(
        slopes_x, slopes_y, metric="pv"
    )
    print(f"  Meilleur: {best_algo_pv}")
    print(f"    PV: {best_result_pv.pv:.2f} nm")
    print(f"    RMS: {best_result_pv.rms:.2f} nm")
    print(f"    Temps: {best_result_pv.computation_time:.4f} s")
    
    # 4.2 Meilleur selon RMS
    print("\n4.2 Meilleur algorithme selon RMS (plus petit RMS)")
    best_algo_rms, best_result_rms = reconstructor.get_best_algorithm(
        slopes_x, slopes_y, metric="rms"
    )
    print(f"  Meilleur: {best_algo_rms}")
    print(f"    PV: {best_result_rms.pv:.2f} nm")
    print(f"    RMS: {best_result_rms.rms:.2f} nm")
    print(f"    Temps: {best_result_rms.computation_time:.4f} s")
    
    # 4.3 Meilleur selon le temps
    print("\n4.3 Meilleur algorithme selon le temps (plus rapide)")
    best_algo_time, best_result_time = reconstructor.get_best_algorithm(
        slopes_x, slopes_y, metric="time"
    )
    print(f"  Meilleur: {best_algo_time}")
    print(f"    PV: {best_result_time.pv:.2f} nm")
    print(f"    RMS: {best_result_time.rms:.2f} nm")
    print(f"    Temps: {best_result_time.computation_time:.4f} s")
    
    # =========================================================================
    # 5. RECONSTRUCTION MODALE AVEC ZERNIKE
    # =========================================================================
    print("\n--- 5. Reconstruction modale avec Zernike ---")
    
    # 5.1 Reconstruction avec différents degrés maximaux
    print("\n5.1 Influence du degré maximal des polynômes de Zernike")
    
    degrees = [3, 5, 7, 10, 15]
    zernike_results = []
    
    for degree in degrees:
        result = reconstructor.reconstruct(
            slopes_x, slopes_y,
            algorithm=ReconstructionAlgorithm.MODAL_ZERNIKE,
            max_zernike_degree=degree
        )
        zernike_results.append(result)
        print(f"  Degré {degree}: PV={result.pv:.2f} nm, RMS={result.rms:.2f} nm, "
              f"Temps={result.computation_time:.4f} s")
    
    # 5.2 Visualisation de l'influence du degré
    print("\n5.2 Visualisation de l'influence du degré")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PV et RMS en fonction du degré
    degrees_array = np.array(degrees)
    pvs_array = np.array([r.pv for r in zernike_results])
    rmss_array = np.array([r.rms for r in zernike_results])
    times_array = np.array([r.computation_time for r in zernike_results])
    
    axes[0].plot(degrees_array, pvs_array, 'b-o', label='PV')
    axes[0].plot(degrees_array, rmss_array, 'r-o', label='RMS')
    axes[0].set_xlabel("Degré maximal des polynômes de Zernike")
    axes[0].set_ylabel("Valeur (nm)")
    axes[0].set_title("PV et RMS en fonction du degré Zernike")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(degrees_array, times_array, 'g-o')
    axes[1].set_xlabel("Degré maximal des polynômes de Zernike")
    axes[1].set_ylabel("Temps (s)")
    axes[1].set_title("Temps de calcul en fonction du degré Zernike")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zernike_degree_influence.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: zernike_degree_influence.png")
    
    # 5.3 Comparaison avec la phase vraie
    print("\n5.3 Comparaison avec la phase vraie")
    
    # La phase vraie est celle du faisceau (en nm)
    # Mais elle est définie sur une grille différente
    # On va interpoler pour comparer
    
    # Phase reconstruite (meilleur Zernike)
    best_zernike_result = zernike_results[np.argmin([r.rms for r in zernike_results])]
    reconstructed_phase = best_zernike_result.phase
    
    # Phase vraie (interpolée sur la grille des pentes)
    # Les pentes sont sur la grille de la caméra (256x256)
    # La phase vraie est sur la grille du faisceau (256x256)
    # Elles devraient avoir la même taille
    
    if reconstructed_phase.shape == beam.phase.shape:
        phase_diff = reconstructed_phase - beam.phase
        
        pv_diff = np.max(np.abs(phase_diff))
        rms_diff = np.std(phase_diff)
        mean_diff = np.mean(phase_diff)
        
        print(f"  Différence avec la phase vraie:")
        print(f"    PV: {pv_diff:.2f} nm")
        print(f"    RMS: {rms_diff:.2f} nm")
        print(f"    Moyenne: {mean_diff:.2f} nm")
        
        # Visualisation
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        im1 = axes[0].imshow(beam.phase, cmap='Jet')
        axes[0].set_title(f"Phase vraie\nPV={pv_beam:.2f} nm, RMS={rms_beam:.2f} nm")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(reconstructed_phase, cmap='Jet')
        axes[1].set_title(f"Phase reconstruite (Zernike)\nPV={best_zernike_result.pv:.2f} nm, RMS={best_zernike_result.rms:.2f} nm")
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(phase_diff, cmap='Jet')
        axes[2].set_title(f"Différence\nPV={pv_diff:.2f} nm, RMS={rms_diff:.2f} nm")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "phase_comparison_true_vs_reconstructed.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Image sauvegardée: phase_comparison_true_vs_reconstructed.png")
    else:
        print(f"  ⚠️  Les grilles ont des tailles différentes, comparaison impossible")
    
    # =========================================================================
    # 6. RECONSTRUCTION AVEC DIFFÉRENTS NIVEAUX DE BRUIT
    # =========================================================================
    print("\n--- 6. Reconstruction avec différents niveaux de bruit ---")
    
    # 6.1 Ajouter du bruit aux pentes
    print("\n6.1 Ajout de bruit aux pentes")
    
    noise_levels = [0.0, 0.01, 0.05, 0.1]  # rad (RMS)
    noise_results = {}
    
    for noise_level in noise_levels:
        # Ajouter du bruit gaussien
        noisy_slopes_x = slopes_x + np.random.normal(0, noise_level, slopes_x.shape)
        noisy_slopes_y = slopes_y + np.random.normal(0, noise_level, slopes_y.shape)
        
        # Reconstruire avec Southwell régularisé
        result = reconstructor.reconstruct(
            noisy_slopes_x, noisy_slopes_y,
            algorithm=ReconstructionAlgorithm.SOUTHWELL_REGULARIZED,
            alpha=0.01
        )
        
        noise_results[noise_level] = result
        print(f"  Bruit {noise_level:.2f} rad: PV={result.pv:.2f} nm, RMS={result.rms:.2f} nm")
    
    # 6.2 Visualisation de l'effet du bruit
    print("\n6.2 Visualisation de l'effet du bruit")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PV et RMS en fonction du bruit
    noise_array = np.array(noise_levels)
    pvs_noise = np.array([r.pv for r in noise_results.values()])
    rmss_noise = np.array([r.rms for r in noise_results.values()])
    
    axes[0].plot(noise_array, pvs_noise, 'b-o', label='PV')
    axes[0].plot(noise_array, rmss_noise, 'r-o', label='RMS')
    axes[0].set_xlabel("Niveau de bruit (rad RMS)")
    axes[0].set_ylabel("Valeur (nm)")
    axes[0].set_title("PV et RMS en fonction du niveau de bruit")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Temps en fonction du bruit
    times_noise = np.array([r.computation_time for r in noise_results.values()])
    axes[1].plot(noise_array, times_noise, 'g-o')
    axes[1].set_xlabel("Niveau de bruit (rad RMS)")
    axes[1].set_ylabel("Temps (s)")
    axes[1].set_title("Temps de calcul en fonction du niveau de bruit")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_effect.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: noise_effect.png")
    
    # =========================================================================
    # 7. RECONSTRUCTION AVEC PENTES DE SHACK-HARTMANN RÉEL
    # =========================================================================
    print("\n--- 7. Reconstruction avec pentes de Shack-Hartmann réel ---")
    
    # 7.1 Utiliser les pentes d'un vrai système Shack-Hartmann
    print("\n7.1 Utilisation des pentes d'un système Shack-Hartmann réel")
    
    # On a déjà les pentes de sh (slopes_x, slopes_y)
    # Utiliser différents algorithmes
    
    real_algorithms = [
        ReconstructionAlgorithm.SOUTHWELL,
        ReconstructionAlgorithm.SOUTHWELL_REGULARIZED,
        ReconstructionAlgorithm.LEAST_SQUARES,
        ReconstructionAlgorithm.TIKHONOV,
        ReconstructionAlgorithm.MODAL_ZERNIKE
    ]
    
    real_results = reconstructor.compare_algorithms(
        slopes_x, slopes_y,
        algorithms=real_algorithms
    )
    
    # 7.2 Tableau des résultats
    print("\n7.2 Résultats avec pentes réelles")
    print("\n{:<25} {:<10} {:<10} {:<10}".format(
        "Algorithme", "PV (nm)", "RMS (nm)", "Temps (s)"
    ))
    print("-" * 55)
    
    for algo in real_algorithms:
        result = real_results[algo.value]
        print("{:<25} {:<10.2f} {:<10.2f} {:<10.4f}".format(
            algo.value,
            result.pv,
            result.rms,
            result.computation_time
        ))
    
    # 7.3 Visualisation
    print("\n7.3 Visualisation des résultats")
    
    fig, axes = plt.subplots(1, len(real_algorithms), figsize=(5 * len(real_algorithms), 6))
    
    for idx, algo in enumerate(real_algorithms):
        result = real_results[algo.value]
        im = axes[idx].imshow(result.phase, cmap='Jet')
        axes[idx].set_title(f"{algo.value}\nPV={result.pv:.2f}nm")
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "real_slopes_reconstruction.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: real_slopes_reconstruction.png")
    
    # =========================================================================
    # 8. EXPORT DES RÉSULTATS
    # =========================================================================
    print("\n--- 8. Export des résultats ---")
    
    # 8.1 Export complet
    print("\n8.1 Export complet des résultats")
    
    export_data = {
        'metadata': {
            'date': str(datetime.datetime.now()),
            'wavelength_nm': wavelength_nm,
            'shack_hartmann_config': {
                'num_microlenses_x': sh.microlens_array.num_elements_x,
                'num_microlenses_y': sh.microlens_array.num_elements_y,
                'focal_length_mm': sh.focal_length_mm,
                'camera_pixels': sh.camera.specifications.num_pixels_x
            },
            'beam_aberrations': wfe.zernike_coefficients
        },
        'comparison': comparison_data,
        'zernike_degree_influence': [
            {
                'degree': degree,
                'pv': result.pv,
                'rms': result.rms,
                'time': result.computation_time
            }
            for degree, result in zip(degrees, zernike_results)
        ],
        'noise_effect': [
            {
                'noise_level': noise_level,
                'pv': result.pv,
                'rms': result.rms,
                'time': result.computation_time
            }
            for noise_level, result in noise_results.items()
        ],
        'best_algorithms': {
            'by_pv': best_algo_pv,
            'by_rms': best_algo_rms,
            'by_time': best_algo_time
        }
    }
    
    export_file = os.path.join(output_dir, "southwell_results.json")
    with open(export_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"  ✓ Données exportées: {export_file}")
    
    # 8.2 Export des phases reconstruites
    print("\n8.2 Export des phases reconstruites")
    
    for algo in all_algorithms:
        result = results[algo.value]
        if result.success:
            np.save(os.path.join(output_dir, f"phase_{algo.value}.npy"), result.phase)
    
    print(f"  ✓ Phases sauvegardées dans {output_dir}/")
    
    # =========================================================================
    # 9. FIN DE L'EXEMPLE
    # =========================================================================
    print("\n" + "="*80)
    print("Example 14 terminé avec succès !")
    print(f"Les résultats ont été sauvegardés dans {output_dir}/")
    print("="*80)
    print("\nRésumé des fonctionnalités démontrées:")
    print("  ✓ Reconstruction de phase à partir des pentes locales")
    print("  ✓ Comparaison de 16 algorithmes différents:")
    print("    - Southwell (original, régularisé, pondéré, itératif)")
    print("    - Moindres carrés (standard, pondéré, Tikhonov)")
    print("    - Modal (Zernike, Legendre, personnalisé)")
    print("    - Autres (Hudgin, Fried, Gendron, Poyneer, Wallner, Roddier)")
    print("  ✓ Intégration avec Shack_Hartmann.py")
    print("  ✓ Visualisation des phases reconstruites")
    print("  ✓ Comparaison des performances (PV, RMS, temps)")
    print("  ✓ Sélection du meilleur algorithme")
    print("  ✓ Influence du degré des polynômes de Zernike")
    print("  ✓ Effet du bruit sur la reconstruction")
    print("  ✓ Comparaison avec la phase vraie")
    print("  ✓ Export des résultats en JSON et NPY")
    print("="*80)


if __name__ == "__main__":
    import datetime
    run_southwell_example()
