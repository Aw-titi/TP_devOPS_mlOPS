"""
Enregistrer le meilleur modèle de régression dans le Registry MLflow
Basé sur l'expérience student-score-regression-project 📚
"""
import mlflow

# Nom du modèle à enregistrer dans le registry
model_name = "student-score-regressor"

print("1️⃣ Recherche du meilleur modèle (R²)...")

# Chercher l'expérience
experiment = mlflow.get_experiment_by_name("student-score-regression-project")

if experiment:
    # Trouver le run avec le meilleur R²
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.R2 DESC"],
        max_results=1
    )
    
    if not runs.empty:
        best_run = runs.iloc[0]
        run_id = best_run.run_id
        r2_score_val = best_run['metrics.R2']
        
        print(f"✅ Meilleur run trouvé (R²: {r2_score_val:.2%})")

        print("\n2️⃣ Enregistrement dans le Model Registry...")
        model_uri = f"runs:/{run_id}/student_model"

        # Enregistrement dans le registry
        result = mlflow.register_model(model_uri, model_name)

        print(f"✅ Modèle enregistré dans le registry MLflow:")
        print(f"   - Nom: {model_name}")
        print(f"   - Version: {result.version}")
        print(f"🔍 Voir le registry: lancez `mlflow ui` puis ouvrez http://localhost:5000/#/models")
    else:
        print("❌ Aucun run trouvé. Veuillez d'abord entraîner un modèle.")
else:
    print("❌ Expérience 'student-score-regression-project' non trouvée.")
