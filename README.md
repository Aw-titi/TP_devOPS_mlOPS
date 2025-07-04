README - Projet DevOps & MLOps 

🎯 Objectif du projet
Ce projet a pour but de démontrer l'automatisation complète d’un pipeline DevOps et MLOps.
Il repose sur l’orchestration d’outils d’infrastructure (Terraform, Ansible), de gestion de conteneurs (Docker), d’apprentissage automatique (Python, MLflow) et d’exposition d’API (FastAPI ou Node.js).

L’objectif est d’automatiser l’entraînement d’un modèle de Machine Learning, son déploiement dans un conteneur, et la mise à disposition d’un point d’accès pour l’interroger à distance.

🧱 Architecture du projet
- Terraform : Provisionnement d’une instance EC2 sur AWS
- Ansible : Configuration de l’environnement (Docker, Nginx, Python)
- Python : Scripts pour entraîner, sauvegarder et servir un modèle ML
- FastAPI ou Node.js : API web exposant les prédictions du modèle
- Docker : Conteneurisation de l’API
- run_all.sh : Script principal lançant le pipeline complet
  
📁 Structure des fichiers

devops/
├── terraform/
├── main.tf                  
├── _credentials/            
├── aws_learner_lab_credentials         # À remplir avec vos propres identifiants AWS
├── labsuser.pem                                     # À remplacer par votre clé privée SSH
├── ansible
├── playbook.yml             
├── inventory.ini            

ml/
├── train_project.py         
├── train_with_tracking.py   
├── register_model.py        
├── load_model.py            
├── api.py                   
├── Dockerfile               
├── requirements.txt         

api/
├── index.js                 
├── Dockerfile               
├── package.json             

run_all.sh                                                              # Script global d’exécution

🔐 Personnalisation de votre environnement AWS
1. Remplacez le fichier `aws_learner_lab_credentials` avec vos identifiants temporaires AWS.
2. Remplacez le fichier `labsuser.pem` par votre propre clé privée depuis AWS.
   Assurez-vous de restreindre ses permissions avec `chmod 400`.
3. Ces fichiers ne doivent jamais être versionnés sur un dépôt public.

🚀 Lancement du projet
Rendez exécutable le script global :
```bash
chmod +x run_all.sh
./run_all.sh
```
Ce script réalise :
- Le déploiement de l’infrastructure
- La configuration du serveur
- L’exécution du modèle
- Le lancement de l’API

⚙️ Structure du projet
Le projet est organisé autour de trois grandes briques :
- DevOps : Provisionnement de l’infrastructure, configuration d’EC2, déploiement Docker, Ansible.
- MLOps : Entraînement de modèles, suivi via MLflow, gestion des dépendances.
- API : Exposition des résultats via une API Node.js et FastAPI.

🔨 Infrastructure & DevOps
L’infrastructure est provisionnée via Terraform (OpenTofu) en utilisant des credentials AWS. Une instance EC2 est créée automatiquement, sécurisée et configurée avec Ansible. Docker est installé et permet le déploiement des services ML et API dans des conteneurs isolés. L’accès se fait avec une clé SSH `myKey.pem`. Un fichier d’inventaire Ansible permet la gestion distante.

🤖 Entraînement & Suivi ML (MLOps)
Les modèles sont entraînés à l’aide de `train_project.py` et `train_with_tracking.py`. Les expériences sont suivies avec MLflow, avec tracking local. L’intégration est faite pour automatiser :
- L’entraînement du modèle
- Le logging des paramètres/metrics
- L’enregistrement du modèle dans un répertoire `mlruns`
- Le déploiement du modèle via `load_model.py` et `run_project.py`.
  
🔄 API & Services
Deux APIs sont proposées :
- Une API Node.js exposée sur le port 3000 (fichier `index.js`) pour tester des endpoints REST.
- Une API FastAPI exposée sur le port 5000 pour charger et interroger le modèle ML.
Les dépendances sont décrites dans `requirements.txt` et `package.json`.

 ![image](https://github.com/user-attachments/assets/8b56c625-21b4-4d2f-bfdb-9183d655a3f4)

🧪 Vérifications
- Accédez à http://<IP_EC2>:3000 pour l’API Node.js
- Utilisez curl ou Postman pour tester les prédictions
- Vérifiez les logs Docker avec `docker ps` et `docker logs`

📦 Technologies utilisées
Terraform, Ansible, AWS EC2, Docker, Python, MLflow, scikit-learn, FastAPI, Node.js, Bash
