"""
Script pour tester tous les exemples et vérifier qu'il n'y a pas d'erreur.
FR: Ce script exécute tous les exemples créés et vérifie qu'ils fonctionnent
    correctement sans erreur.

EN: Script to test all examples and verify they run without errors.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import sys
import traceback
import os
import time
from datetime import datetime


def test_example(example_name: str, example_function: callable) -> bool:
    """
    FR: Teste un exemple et retourne True si succès, False sinon.
    
    EN: Tests an example and returns True if successful, False otherwise.
    
    Args:
        example_name: str - Nom de l'exemple
        example_function: callable - Fonction à exécuter
    
    Returns:
        bool - True si succès, False sinon
    """
    print(f"\n{'='*80}")
    print(f"Test de {example_name}...")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        example_function()
        elapsed_time = time.time() - start_time
        print(f"✅ {example_name} : SUCCÈS ({elapsed_time:.2f}s)")
        return True
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {example_name} : ÉCHEC ({elapsed_time:.2f}s)")
        print(f"Erreur: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """FR: Fonction principale."""
    print("\n" + "="*80)
    print("TEST DE TOUS LES EXEMPLES")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"NumPy: {__import__('numpy').__version__}")
    
    # Liste des exemples à tester
    examples = []
    
    # 1. Example 10: Optiques
    try:
        from example_10_optics import run_optics_example
        examples.append(("Example 10: Optiques", run_optics_example))
    except ImportError as e:
        print(f"⚠️  Example 10: Optiques - Module non disponible: {e}")
    
    # 2. Example 11: Microstructure
    try:
        from example_11_microstructure import run_microstructure_example
        examples.append(("Example 11: Microstructure", run_microstructure_example))
    except ImportError as e:
        print(f"⚠️  Example 11: Microstructure - Module non disponible: {e}")
    
    # 3. Example 12: Camera
    try:
        from example_12_camera import run_camera_example
        examples.append(("Example 12: Camera", run_camera_example))
    except ImportError as e:
        print(f"⚠️  Example 12: Camera - Module non disponible: {e}")
    
    # 4. Example 13: Shack-Hartmann
    try:
        from example_13_shack_hartmann import run_shack_hartmann_example
        examples.append(("Example 13: Shack-Hartmann", run_shack_hartmann_example))
    except ImportError as e:
        print(f"⚠️  Example 13: Shack-Hartmann - Module non disponible: {e}")
    
    # 5. Génération des tâches d'Airy
    try:
        from generate_airy_spots_image import generate_airy_spots_image
        examples.append(("Génération tâches d'Airy", generate_airy_spots_image))
    except ImportError as e:
        print(f"⚠️  Génération tâches d'Airy - Module non disponible: {e}")
    
    # 6. Southwell (à créer)
    try:
        from example_14_southwell import run_southwell_example
        examples.append(("Example 14: Southwell", run_southwell_example))
    except ImportError as e:
        print(f"⚠️  Example 14: Southwell - Module non disponible: {e}")
    
    # Résumé
    print(f"\n{'='*80}")
    print("RÉSUMÉ DES TESTS")
    print(f"{'='*80}")
    print(f"Nombre d'exemples à tester: {len(examples)}")
    
    # Exécuter les tests
    results = []
    for name, func in examples:
        success = test_example(name, func)
        results.append((name, success))
    
    # Statistiques
    successful = sum(1 for _, s in results if s)
    failed = len(results) - successful
    
    print(f"\n{'='*80}")
    print("STATISTIQUES")
    print(f"{'='*80}")
    print(f"Réussis: {successful}/{len(results)}")
    print(f"Échoués: {failed}/{len(results)}")
    print(f"Taux de succès: {(successful/len(results)*100):.1f}%")
    
    # Détails
    if failed > 0:
        print(f"\n{'='*80}")
        print("EXEMPLES ÉCHOUÉS")
        print(f"{'='*80}")
        for name, success in results:
            if not success:
                print(f"  ❌ {name}")
    
    # Retourner le code de sortie
    if failed > 0:
        print(f"\n❌ {failed} exemple(s) ont échoué.")
        return 1
    else:
        print(f"\n✅ Tous les exemples ont réussi !")
        return 0


if __name__ == "__main__":
    sys.exit(main())
