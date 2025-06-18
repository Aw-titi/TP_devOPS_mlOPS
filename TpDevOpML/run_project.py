"""
Démonstration de MLflow Projects avec régression étudiant 📚
Exécute plusieurs expériences avec différents hyperparamètres
"""
import subprocess
import sys
import os

print("🚀 Démonstration MLflow Projects - Student Regression")

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
train_script = os.path.join(current_dir, "train_project.py")

print("\n1️⃣ Lancement avec paramètres par défaut (15 arbres, profondeur 3)...")

cmd1 = [sys.executable, train_script, "--n_estimators", "15", "--max_depth", "3"]
result1 = subprocess.run(cmd1, capture_output=True, text=True)

if result1.returncode == 0:
    print("✅ Expérience 1 terminée avec succès")
else:
    print(f"❌ Erreur dans l'expérience 1:\n{result1.stderr}")

print("\n2️⃣ Lancement avec paramètres personnalisés (20 arbres, profondeur 5)...")

cmd2 = [sys.executable, train_script, "--n_estimators", "20", "--max_depth", "5"]
result2 = subprocess.run(cmd2, capture_output=True, text=True)

if result2.returncode == 0:
    print("✅ Expérience 2 terminée avec succès")
else:
    print(f"❌ Erreur dans l'expérience 2:\n{result2.stderr}")

print("\n3️⃣ Lancement avec paramètres avancés (50 arbres, profondeur 10)...")

cmd3 = [sys.executable, train_script, "--n_estimators", "50", "--max_depth", "10"]
result3 = subprocess.run(cmd3, capture_output=True, text=True)

if result3.returncode == 0:
    print("✅ Expérience 3 terminée avec succès")
else:
    print(f"❌ Erreur dans l'expérience 3:\n{result3.stderr}")
