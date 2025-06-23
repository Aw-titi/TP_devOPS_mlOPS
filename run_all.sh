set -e

echo "Commencement"

echo "Terraform"
cd devops/terraform
terraform init
terraform plan
terraform apply -auto-approve

EC2_IP=$(terraform output -raw public_ip)
echo "Adresse IP publique de l'EC2 : $EC2_IP"
cd ../..

echo "Mise à jour de inventory.ini"
cat > devops/ansible/inventory.ini <<EOF
[webservers]
ec2-instance-1 ansible_host=$EC2_IP

[webservers:vars]
ansible_user=ubuntu
ansible_ssh_private_key_file=~/.ssh/myKey.pem
ansible_ssh_common_args='-o StrictHostKeyChecking=no'
ansible_python_interpreter=/usr/bin/python3
EOF

echo "Playbook Ansible"
cd devops/ansible
ansible-playbook -i inventory.ini playbook.yml
cd ../..

echo "Transfère ML et API"
scp -i ~/.ssh/labsuser.pem -r ml ubuntu@$EC2_IP:~/
scp -i ~/.ssh/labsuser.pem -r api ubuntu@$EC2_IP:~/

echo "ML"
ssh -i ~/.ssh/labsuser.pem ubuntu@$EC2_IP <<EOF
  cd ml
  docker build -t mlapp .
  docker run -d --name ml_container -p 8000:8000 mlapp
EOF

echo "API"
ssh -i ~/.ssh/labsuser.pem ubuntu@$EC2_IP <<EOF
  cd api
  docker build -t apiapp .
  docker run -d --name api_container -p 3000:3000 apiapp
EOF

echo ""
echo "Déploiement terminé."
echo "API: http://$EC2_IP:3000"
echo "ML: http://$EC2_IP:8000/predict"

