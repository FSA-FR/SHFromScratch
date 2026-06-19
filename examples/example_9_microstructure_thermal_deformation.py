"""
Example 9: Microstructure Thermal Deformation
FR: Exemple de simulation de la déformation thermique des microstructures optiques.
    Démonstration de l'utilisation des méthodes de Material_Behaviour.py pour simuler :
    - La dilatation des matrices de microlentilles
    - La déformation des systèmes optiques complets
    - L'impact sur les positions et distances focales

EN: Example of thermal deformation simulation for optical microstructures.
    Demonstrates the use of Material_Behaviour.py methods to simulate:
    - Thermal expansion of microlens arrays
    - Deformation of complete optical systems
    - Impact on positions and focal lengths

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Material_Behaviour import MaterialBehaviour, MicrostructureType


def run_microstructure_thermal_deformation_example():
    """FR: Exécute l'exemple de déformation thermique des microstructures."""
    print("\n" + "="*80)
    print("Example 9: Microstructure Thermal Deformation")
    print("="*80)
    
    # =========================================================================
    # 1. Déformation d'une matrice de microlentilles
    # =========================================================================
    print("\n--- Déformation d'une matrice de microlentilles ---")
    
    # Paramètres de la matrice de microlentilles
    material_name = "Fused_Silica"
    pitch_mm = 0.5  # Pas entre les microlentilles
    num_elements = 11  # Nombre de microlentilles (11x11)
    temperature_C = 100.0  # Température finale
    reference_temperature_C = 20.0  # Température de référence
    
    temperature_K = temperature_C + 273.15
    reference_temperature_K = reference_temperature_C + 273.15
    
    # Créer l'objet MaterialBehaviour
    material = MaterialBehaviour(material_name)
    
    # Calculer la déformation
    deformation = material.get_microstructure_deformation(
        microstructure_type=MicrostructureType.MICROLENS_ARRAY,
        pitch_mm=pitch_mm,
        num_elements=num_elements,
        temperature_K=temperature_K,
        reference_temperature_K=reference_temperature_K,
    )
    
    print(f"Matrice de microlentilles en {material_name}:")
    print(f"  Pas initial: {deformation['initial_pitch_mm']:.4f} mm")
    print(f"  Nouveau pas: {deformation['new_pitch_mm']:.4f} mm")
    print(f"  Variation du pas: {deformation['delta_pitch_mm']:+.6f} mm")
    print(f"  Déplacement maximal: {deformation['max_displacement_mm']:+.6f} mm")
    
    # Visualisation de la déformation
    plt.figure(figsize=(14, 8))
    
    # Tracer les positions initiales et finales
    x_initial = deformation['initial_positions_mm']
    x_final = deformation['new_positions_mm']
    
    plt.subplot(2, 1, 1)
    plt.plot(x_initial, np.zeros_like(x_initial), 'bo-', label='Positions initiales', markersize=8)
    plt.plot(x_final, np.zeros_like(x_final), 'ro-', label='Positions finales', markersize=8)
    plt.xlabel("Position (mm)")
    plt.ylabel("Y (mm)")
    plt.title(f"Déformation d'une matrice de {num_elements} microlentilles ({material_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='k', linewidth=0.5)
    
    # Tracer le décalage
    plt.subplot(2, 1, 2)
    displacement = x_final - x_initial
    plt.plot(x_initial, displacement, 'go-', label='Déplacement', markersize=4)
    plt.xlabel("Position initiale (mm)")
    plt.ylabel("Déplacement (mm)")
    plt.title("Déplacement de chaque microlentille")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('examples/output/example9_microlens_array_deformation.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 2. Comparaison entre différents matériaux de microstructure
    # =========================================================================
    print("\n--- Comparaison entre différents matériaux de microstructure ---")
    
    materials = ["Fused_Silica", "BK7", "SF5"]
    pitch_mm = 0.5
    num_elements = 5
    
    plt.figure(figsize=(12, 8))
    
    for material_name in materials:
        material = MaterialBehaviour(material_name)
        deformation = material.get_microstructure_deformation(
            microstructure_type=MicrostructureType.MICROLENS_ARRAY,
            pitch_mm=pitch_mm,
            num_elements=num_elements,
            temperature_K=temperature_K,
            reference_temperature_K=reference_temperature_K,
        )
        
        print(f"\n  {material_name}:")
        print(f"    Δ pas: {deformation['delta_pitch_mm']:+.6f} mm")
        print(f"    Déplacement maximal: {deformation['max_displacement_mm']:+.6f} mm")
        
        # Tracer le décalage pour le matériau
        x_initial = deformation['initial_positions_mm']
        displacement = deformation['new_positions_mm'] - x_initial
        plt.plot(x_initial, displacement, 'o-', label=material_name, markersize=6)
    
    plt.xlabel("Position initiale (mm)")
    plt.ylabel("Déplacement (mm)")
    plt.title(f"Comparaison des déformations pour différents matériaux (ΔT = {temperature_C - reference_temperature_C}°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.savefig('examples/output/example9_material_comparison_deformation.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 3. Déformation avec support différent
    # =========================================================================
    print("\n--- Déformation avec support différent ---")
    
    # Matrice de microlentilles en Fused Silica sur différents supports
    microstructure_material = "Fused_Silica"
    support_materials = ["Aluminum", "Steel", "Invar"]
    
    plt.figure(figsize=(14, 8))
    
    for i, support_material in enumerate(support_materials):
        material = MaterialBehaviour(microstructure_material)
        deformation = material.get_microstructure_deformation(
            microstructure_type=MicrostructureType.MICROLENS_ARRAY,
            pitch_mm=pitch_mm,
            num_elements=num_elements,
            temperature_K=temperature_K,
            reference_temperature_K=reference_temperature_K,
            support_material_name=support_material,
        )
        
        print(f"\n  Matrice en {microstructure_material} sur support {support_material}:")
        print(f"    Δ pas: {deformation['delta_pitch_mm']:+.6f} mm")
        print(f"    Déplacement maximal: {deformation['max_displacement_mm']:+.6f} mm")
        
        # Tracer le décalage
        x_initial = deformation['initial_positions_mm']
        displacement = deformation['new_positions_mm'] - x_initial
        plt.subplot(2, 2, i+1)
        plt.plot(x_initial, displacement, 'o-', label=support_material, markersize=6)
        plt.xlabel("Position initiale (mm)")
        plt.ylabel("Déplacement (mm)")
        plt.title(f"Support: {support_material}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('examples/output/example9_support_material_comparison.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 4. Déformation d'un système optique complet
    # =========================================================================
    print("\n--- Déformation d'un système optique complet ---")
    
    # Définir un système optique : Shack-Hartmann simplifié
    optical_system = [
        {
            "name": "Matrice de microlentilles",
            "material": "Fused_Silica",
            "position_mm": 0.0,
            "focal_length_mm": 10.0,  # Distance focale de chaque microlentille
            "diameter_mm": 0.4,       # Diamètre de chaque microlentille
            "thickness_mm": 2.0,      # Épaisseur de la matrice
        },
        {
            "name": "Lentille de champ",
            "material": "BK7",
            "position_mm": 20.0,
            "focal_length_mm": 50.0,
            "diameter_mm": 25.0,
            "thickness_mm": 5.0,
        },
        {
            "name": "Capteur",
            "material": "Silicon",
            "position_mm": 25.0,
            "thickness_mm": 0.5,
        },
    ]
    
    # Calculer la déformation du système
    # On utilise le matériau du premier élément pour la simulation
    first_material = MaterialBehaviour(optical_system[0]["material"])
    deformed_system = first_material.get_optical_system_deformation(
        optical_system,
        temperature_K=temperature_K,
        reference_temperature_K=reference_temperature_K,
        support_material_name="Aluminum",  # Supposons que tout est monté sur de l'aluminium
    )
    
    print("\nSystème optique initial (à 20°C) :")
    for element in optical_system:
        print(f"  {element['name']} ({element['material']}):")
        print(f"    Position: {element['position_mm']:.2f} mm")
        if 'focal_length_mm' in element:
            print(f"    Distance focale: {element['focal_length_mm']:.2f} mm")
        if 'diameter_mm' in element:
            print(f"    Diamètre: {element['diameter_mm']:.2f} mm")
        if 'thickness_mm' in element:
            print(f"    Épaisseur: {element['thickness_mm']:.2f} mm")
    
    print("\nSystème optique déformé (à 100°C) :")
    for element in deformed_system:
        print(f"  {element['name']} ({element['material']}):")
        print(f"    Position initiale: {element['initial_position_mm']:.2f} mm")
        print(f"    Nouvelle position: {element['new_position_mm']:.2f} mm")
        print(f"    Décalage: {element['delta_position_mm']:+.4f} mm")
        if 'initial_focal_length_mm' in element:
            print(f"    Distance focale initiale: {element['initial_focal_length_mm']:.2f} mm")
            print(f"    Nouvelle distance focale: {element['new_focal_length_mm']:.2f} mm")
            print(f"    Variation de la distance focale: {element['delta_focal_length_mm']:+.4f} mm")
        if 'initial_diameter_mm' in element:
            print(f"    Diamètre initial: {element['initial_diameter_mm']:.2f} mm")
            print(f"    Nouveau diamètre: {element['new_diameter_mm']:.2f} mm")
            print(f"    Variation du diamètre: {element['delta_diameter_mm']:+.4f} mm")
        if 'initial_thickness_mm' in element:
            print(f"    Épaisseur initiale: {element['initial_thickness_mm']:.2f} mm")
            print(f"    Nouvelle épaisseur: {element['new_thickness_mm']:.2f} mm")
            print(f"    Variation de l'épaisseur: {element['delta_thickness_mm']:+.4f} mm")
    
    # Visualisation du système optique
    plt.figure(figsize=(14, 8))
    
    # Positions initiales
    initial_positions = [elem['position_mm'] for elem in optical_system]
    initial_names = [elem['name'] for elem in optical_system]
    
    # Nouvelles positions
    new_positions = [elem['new_position_mm'] for elem in deformed_system]
    new_names = [elem['name'] for elem in deformed_system]
    
    # Tracer les positions
    plt.subplot(2, 1, 1)
    for i, (pos, name) in enumerate(zip(initial_positions, initial_names)):
        plt.plot(pos, 0, 'bo', markersize=10, label=f"{name} (initial)" if i == 0 else "")
        plt.text(pos, 0.1, name, ha='center', va='bottom')
    plt.xlabel("Position (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Système optique initial (20°C)")
    plt.axhline(0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 1.0)
    
    plt.subplot(2, 1, 2)
    for i, (pos, name) in enumerate(zip(new_positions, new_names)):
        plt.plot(pos, 0, 'ro', markersize=10, label=f"{name} (final)" if i == 0 else "")
        plt.text(pos, 0.1, name, ha='center', va='bottom')
    plt.xlabel("Position (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Système optique déformé (100°C)")
    plt.axhline(0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig('examples/output/example9_optical_system_deformation.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 5. Impact sur la qualité optique
    # =========================================================================
    print("\n--- Impact sur la qualité optique ---")
    
    # Considérons une matrice de microlentilles pour un Shack-Hartmann
    # La déformation du pas affecte la mesure du front d'onde
    
    # Paramètres
    pitch_mm = 0.5
    num_elements = 11
    focal_length_mm = 10.0  # Distance focale des microlentilles
    wavelength_nm = 633.0
    
    # Matériau de la matrice
    matrix_material = MaterialBehaviour("Fused_Silica")
    
    # Calculer la déformation à différentes températures
    temperatures_C = [20.0, 50.0, 100.0, 150.0]
    pitches = []
    focal_lengths = []
    
    for temp_C in temperatures_C:
        temp_K = temp_C + 273.15
        
        # Déformation de la matrice
        deformation = matrix_material.get_microstructure_deformation(
            microstructure_type=MicrostructureType.MICROLENS_ARRAY,
            pitch_mm=pitch_mm,
            num_elements=num_elements,
            temperature_K=temp_K,
            reference_temperature_K=293.15,
        )
        
        # Variation de la distance focale
        delta_f = matrix_material.get_focal_length_variation(
            focal_length_mm, temp_K, 293.15, wavelength_nm
        )
        
        pitches.append(deformation['new_pitch_mm'])
        focal_lengths.append(focal_length_mm + delta_f)
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(temperatures_C, pitches, 'bo-', label='Pas de la matrice', markersize=8)
    plt.xlabel("Température (°C)")
    plt.ylabel("Pas (mm)")
    plt.title("Variation du pas de la matrice de microlentilles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(temperatures_C, focal_lengths, 'ro-', label='Distance focale', markersize=8)
    plt.xlabel("Température (°C)")
    plt.ylabel("Distance focale (mm)")
    plt.title("Variation de la distance focale des microlentilles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('examples/output/example9_optical_quality_impact.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # Afficher les valeurs
    print("\nImpact sur la qualité optique :")
    for temp_C, pitch, f in zip(temperatures_C, pitches, focal_lengths):
        delta_pitch = pitch - pitch_mm
        delta_f = f - focal_length_mm
        print(f"  À {temp_C}°C:")
        print(f"    Pas: {pitch:.6f} mm (Δ = {delta_pitch:+.6f} mm)")
        print(f"    Distance focale: {f:.4f} mm (Δ = {delta_f:+.4f} mm)")
    
    # =========================================================================
    # 6. Comparaison des matériaux pour les microstructures
    # =========================================================================
    print("\n--- Comparaison des matériaux pour les microstructures ---")
    
    # Matériaux à comparer
    materials = ["Fused_Silica", "BK7", "SF5"]
    
    # Calculer la déformation pour chaque matériau
    data = {
        "Matériau": [],
        "CTE (ppm/°C)": [],
        "Δ pas (µm)": [],
        "Δ f (µm)": [],
        "Stabilité": [],
    }
    
    for material_name in materials:
        material = MaterialBehaviour(material_name)
        
        # Déformation de la matrice
        deformation = material.get_microstructure_deformation(
            microstructure_type=MicrostructureType.MICROLENS_ARRAY,
            pitch_mm=0.5,
            num_elements=5,
            temperature_K=373.15,  # 100°C
            reference_temperature_K=293.15,  # 20°C
        )
        
        # Variation de la distance focale
        delta_f = material.get_focal_length_variation(
            10.0, 373.15, 293.15, wavelength_nm
        )
        
        data["Matériau"].append(material_name)
        data["CTE (ppm/°C)"].append(f"{material.get_thermal_expansion_coefficient() * 1e6:.2f}")
        data["Δ pas (µm)"].append(f"{deformation['delta_pitch_mm'] * 1000:+.2f}")
        data["Δ f (µm)"].append(f"{delta_f * 1000:+.2f}")
        
        # Évaluer la stabilité (plus le CTE est faible, plus c'est stable)
        cte = material.get_thermal_expansion_coefficient()
        if cte < 1e-6:
            stability = "Excellente"
        elif cte < 5e-6:
            stability = "Bonne"
        elif cte < 10e-6:
            stability = "Moyenne"
        else:
            stability = "Faible"
        data["Stabilité"].append(stability)
    
    # Afficher le tableau
    print("\nTableau comparatif des matériaux pour les microstructures :")
    print("-" * 80)
    print(f"{|'Matériau':<15}|{'CTE (ppm/°C)':>15}|{'Δ pas (µm)':>12}|{'Δ f (µm)':>12}|{'Stabilité':>12}|")
    print("-" * 80)
    for i in range(len(data["Matériau"])):
        print(f"{|data['Matériau'][i]:<15}|{data['CTE (ppm/°C)'][i]:>15}|{data['Δ pas (µm)'][i]:>12}|{data['Δ f (µm)'][i]:>12}|{data['Stabilité'][i]:>12}|")
    print("-" * 80)
    
    print("\n" + "="*80)
    print("Example 9 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('examples/output', exist_ok=True)
    run_microstructure_thermal_deformation_example()
