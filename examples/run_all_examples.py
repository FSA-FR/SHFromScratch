"""
run_all_examples.py
FR: Script pour exécuter tous les exemples du package SHFromScratch.
    Exécute séquentiellement tous les exemples dans le répertoire examples/
    et génère un rapport de succès/échec.

EN: Script to run all examples in the SHFromScratch package.
    Sequentially executes all examples in the examples/ directory
    and generates a success/failure report.

Usage:
    python examples/run_all_examples.py

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import os
import sys
import traceback
import time
from typing import Dict, List, Tuple


# Liste des exemples à exécuter
EXAMPLES = [
    "example_1_gaussian_beam_propagation.py",
    "example_2_zernike_beam_propagation.py",
    "example_3_coherence_comparison.py",
    "example_4_hermite_gauss_propagation.py",
    "example_5_laguerre_gauss_propagation.py",
    "example_6_resampling_test.py",
    "example_7_normalization_test.py",
    "example_8_material_behaviour.py",
    "example_9_microstructure_thermal_deformation.py",
]

# Couleurs pour l'affichage
COLORS = {
    "HEADER": '\033[95m',
    "OKBLUE": '\033[94m',
    "OKGREEN": '\033[92m',
    "WARNING": '\033[93m',
    "FAIL": '\033[91m',
    "ENDC": '\033[0m',
    "BOLD": '\033[1m',
    "UNDERLINE": '\033[4m',
}


def print_color(color: str, message: str) -> None:
    """Affiche un message avec une couleur spécifique."""
    if os.getenv('NO_COLOR'):
        print(message)
    else:
        print(f"{COLORS.get(color, '')}{message}{COLORS['ENDC']}")


def run_example(example_name: str) -> Tuple[bool, float, str]:
    """
    Exécute un exemple et retourne son statut.
    
    Args:
        example_name: Nom du fichier de l'exemple
        
    Returns:
        Tuple[bool, float, str]: (succès, temps d'exécution, message d'erreur)
    """
    example_path = os.path.join("examples", example_name)
    
    # Vérifier que le fichier existe
    if not os.path.exists(example_path):
        return False, 0.0, f"Fichier non trouvé: {example_path}"
    
    # Changer de répertoire pour les imports
    original_dir = os.getcwd()
    examples_dir = os.path.dirname(os.path.abspath(example_path))
    os.chdir(examples_dir)
    
    try:
        start_time = time.time()
        
        # Exécuter l'exemple
        with open(example_name, 'r') as f:
            code = f.read()
        
        # Extraire le nom de la fonction principale
        exec_globals = {}
        exec(code, exec_globals)
        
        # Trouver et exécuter la fonction run_*
        run_function = None
        for name, obj in exec_globals.items():
            if name.startswith('run_') and callable(obj):
                run_function = obj
                break
        
        if run_function is None:
            return False, 0.0, "Aucune fonction run_* trouvée dans l'exemple"
        
        # Exécuter la fonction
        run_function()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return True, execution_time, ""
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return False, execution_time, error_msg
        
    finally:
        os.chdir(original_dir)


def main():
    """Fonction principale pour exécuter tous les exemples."""
    print_color("HEADER", "\n" + "="*80)
    print_color("HEADER", "EXÉCUTION DE TOUS LES EXEMPLES SHFromScratch")
    print_color("HEADER", "="*80 + "\n")
    
    # Créer le répertoire de sortie
    output_dir = os.path.join("examples", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialisation
    results = []
    total_time = 0.0
    success_count = 0
    failure_count = 0
    
    # Exécuter chaque exemple
    for example in EXAMPLES:
        print_color("BOLD", f"\n[{success_count + failure_count + 1}/{len(EXAMPLES)}] {example}")
        print("-" * 80)
        
        success, exec_time, error_msg = run_example(example)
        total_time += exec_time
        
        if success:
            success_count += 1
            print_color("OKGREEN", f"✅ SUCCESS")
            print_color("OKBLUE", f"   Temps d'exécution: {exec_time:.2f} secondes")
        else:
            failure_count += 1
            print_color("FAIL", f"❌ FAILED")
            print_color("FAIL", f"   Temps d'exécution: {exec_time:.2f} secondes")
            print_color("FAIL", f"   Erreur: {error_msg}")
        
        results.append((example, success, exec_time, error_msg))
    
    # Résumé
    print_color("HEADER", "\n" + "="*80)
    print_color("HEADER", "RÉSUMÉ")
    print_color("HEADER", "="*80)
    
    print_color("OKBLUE", f"\nExemples exécutés: {len(EXAMPLES)}")
    print_color("OKGREEN", f"  ✅ Réussis: {success_count}")
    print_color("FAIL", f"  ❌ Échoués: {failure_count}")
    print_color("OKBLUE", f"\nTemps total: {total_time:.2f} secondes")
    print_color("OKBLUE", f"Temps moyen par exemple: {total_time/len(EXAMPLES):.2f} secondes")
    
    # Détails des échecs
    if failure_count > 0:
        print_color("WARNING", "\nDétails des échecs:")
        for example, success, exec_time, error_msg in results:
            if not success:
                print_color("FAIL", f"\n  {example}:")
                print_color("FAIL", f"    {error_msg}")
    
    # Message final
    print_color("HEADER", "\n" + "="*80)
    if failure_count == 0:
        print_color("OKGREEN", "✅ TOUS LES EXEMPLES ONT RÉUSSI !")
    else:
        print_color("WARNING", f"⚠️  {success_count}/{len(EXAMPLES)} exemples ont réussi")
    print_color("HEADER", "="*80 + "\n")
    
    # Retourner le code de sortie
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
