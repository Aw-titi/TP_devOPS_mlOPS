#!/bin/bash
set -e

echo "Commencement du projet devops, mlops"

echo "Étape 1 : Initialisation de Terraform"
cd devops/terraform
terraform init
terraform plan
terraform apply -auto-approve

EC2_IP=$(terraform output -raw public_ip)
echo "Adresse IP publique de l'EC2 : $EC2_IP"
cd ../..

echo "Étape 2 : Mise à jour du fichier inventory.ini pour Ansible"
cat > devops/ansible/inventory.ini <<EOF
[webservers]
ec2-instance-1 ansible_host=$EC2_IP

[webservers:vars]
ansible_user=ubuntu
ansible_ssh_private_key_file=~/.ssh/myKey.pem
ansible_ssh_common_args='-o StrictHostKeyChecking=no'
ansible_python_interpreter=/usr/bin/python3
EOF

echo "Étape 3 : Exécution du playbook Ansible"
cd devops/ansible
ansible-playbook -i inventory.ini playbook.yml
cd ../..

echo "Étape 4 : Transfert des dossiers ml/ et api/ vers l'EC2"
scp -i ~/.ssh/labsuser.pem -r ml ubuntu@$EC2_IP:~/
scp -i ~/.ssh/labsuser.pem -r api ubuntu@$EC2_IP:~/

echo "Étape 5 : Construction et lancement du conteneur ML sur l'EC2"
ssh -i ~/.ssh/labsuser.pem ubuntu@$EC2_IP <<EOF
  cd ml
  docker build -t mlapp .
  docker run -d --name ml_container -p 5000:5000 mlapp
EOF

echo "Étape 6 : Construction et lancement du conteneur API sur l'EC2"
ssh -i ~/.ssh/labsuser.pem ubuntu@$EC2_IP <<EOF
  cd api
  docker build -t apiapp .
  docker run -d --name api_container -p 3000:3000 apiapp
EOF

echo ""
echo "Déploiement terminé."
echo "API accessible à l'adresse : http://$EC2_IP:3000"
echo "Service ML accessible à l'adresse : http://$EC2_IP:5000/predict"
