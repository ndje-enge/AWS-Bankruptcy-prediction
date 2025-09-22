import boto3
import json
import zipfile
import os
from datetime import datetime

class LambdaAPIDeployment:
    def __init__(self, region='eu-west-3'):
        """Initialise le déploiement Lambda et API Gateway"""
        self.region = region
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.api_gateway_client = boto3.client('apigateway', region_name=region)
        self.iam_client = boto3.client('iam', region_name=region)
        
    def create_lambda_role(self):
        """Crée le rôle IAM pour la Lambda function"""
        print("Création du rôle IAM pour Lambda...")
        
        role_name = 'bankruptcy-prediction-lambda-role'
        
        # Politique de confiance pour Lambda
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Politique d'autorisation pour SageMaker
        sagemaker_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:InvokeEndpoint"
                    ],
                    "Resource": f"arn:aws:sagemaker:{self.region}:*:endpoint/bankruptcy-predictor-optimized-compatible-*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "arn:aws:logs:*:*:*"
                }
            ]
        }
        
        try:
            # Créer le rôle
            role_response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Rôle pour la Lambda function de prédiction de faillite'
            )
            
            # Attacher la politique SageMaker
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName='SageMakerInvokePolicy',
                PolicyDocument=json.dumps(sagemaker_policy)
            )
            
            # Attacher la politique de base Lambda
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            role_arn = role_response['Role']['Arn']
            print(f"Rôle créé: {role_arn}")
            
            # Attendre que le rôle soit prêt
            import time
            time.sleep(10)
            
            return role_arn
            
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            print(f"Rôle {role_name} existe déjà")
            return f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/{role_name}"
        except Exception as e:
            print(f"Erreur lors de la création du rôle: {str(e)}")
            return None
    
    def create_lambda_package(self):
        """Crée le package ZIP pour la Lambda function"""
        print("Création du package Lambda...")
        
        package_name = 'lambda_bankruptcy_prediction.zip'
        
        with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Ajouter le code Lambda
            zip_file.write('lambda_function.py', 'lambda_function.py')
            
            # Ajouter le fichier de configuration si il existe
            try:
                zip_file.write('config.py', 'config.py')
            except FileNotFoundError:
                pass  # Le fichier config.py n'existe pas, ce n'est pas grave
            
            # Ajouter les dépendances si nécessaire
            # (boto3 est déjà disponible dans l'environnement Lambda)
        
        print(f"Package créé: {package_name}")
        return package_name
    
    def deploy_lambda_function(self, role_arn):
        """Déploie la Lambda function"""
        print("Déploiement de la Lambda function...")
        
        function_name = 'bankruptcy-prediction-api'
        
        # Créer le package
        package_name = self.create_lambda_package()
        
        try:
            # Lire le package
            with open(package_name, 'rb') as zip_file:
                zip_content = zip_file.read()
            
            # Créer ou mettre à jour la fonction
            try:
                # Mettre à jour la fonction existante
                response = self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_content
                )
                
                # Attendre que la mise à jour du code soit terminée
                import time
                time.sleep(5)
                
                # Mettre à jour les variables d'environnement
                self.lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Environment={
                        'Variables': {
                            'SAGEMAKER_ENDPOINT_NAME': 'bankruptcy-predictor-optimized-compatible-20250922-143917'
                        }
                    }
                )
                
                print(f"Lambda function mise à jour: {function_name}")
            except self.lambda_client.exceptions.ResourceNotFoundException:
                # Créer une nouvelle fonction
                response = self.lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.9',
                    Role=role_arn,
                    Handler='lambda_function.lambda_handler',
                    Code={'ZipFile': zip_content},
                    Description='API Lambda pour la prédiction de faillite d\'entreprises',
                    Timeout=30,
                    MemorySize=256,
                    Environment={
                        'Variables': {
                            'SAGEMAKER_ENDPOINT_NAME': 'bankruptcy-predictor-optimized-compatible-20250922-143917'
                        }
                    }
                )
                print(f"Lambda function créée: {function_name}")
            
            function_arn = response['FunctionArn']
            print(f"ARN de la fonction: {function_arn}")
            
            return function_arn
            
        except Exception as e:
            print(f"Erreur lors du déploiement Lambda: {str(e)}")
            return None
    
    def create_api_gateway(self, lambda_arn):
        """Crée l'API Gateway"""
        print("Création de l'API Gateway...")
        
        api_name = 'bankruptcy-prediction-api'
        
        try:
            # Créer l'API REST
            api_response = self.api_gateway_client.create_rest_api(
                name=api_name,
                description='API REST pour la prédiction de faillite d\'entreprises',
                endpointConfiguration={'types': ['REGIONAL']}
            )
            
            api_id = api_response['id']
            print(f"API Gateway créé: {api_id}")
            
            # Obtenir la racine de l'API
            root_response = self.api_gateway_client.get_resources(restApiId=api_id)
            root_id = root_response['items'][0]['id']
            
            # Créer la ressource /predict
            resource_response = self.api_gateway_client.create_resource(
                restApiId=api_id,
                parentId=root_id,
                pathPart='predict'
            )
            
            resource_id = resource_response['id']
            print(f"Ressource /predict créée: {resource_id}")
            
            # Créer la méthode POST
            method_response = self.api_gateway_client.put_method(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='POST',
                authorizationType='NONE'
            )
            print("Méthode POST créée")
            
            # Créer la méthode OPTIONS pour CORS
            self.api_gateway_client.put_method(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='OPTIONS',
                authorizationType='NONE'
            )
            print("Méthode OPTIONS créée")
            
            # Configurer l'intégration Lambda pour POST
            integration_response = self.api_gateway_client.put_integration(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='POST',
                type='AWS_PROXY',
                integrationHttpMethod='POST',
                uri=f"arn:aws:apigateway:{self.region}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"
            )
            print("Intégration Lambda configurée pour POST")
            
            # Configurer l'intégration Lambda pour OPTIONS
            self.api_gateway_client.put_integration(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='OPTIONS',
                type='MOCK',
                requestTemplates={'application/json': '{"statusCode": 200}'}
            )
            print("Intégration OPTIONS configurée")
            
            # Configurer les réponses pour OPTIONS
            self.api_gateway_client.put_method_response(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='OPTIONS',
                statusCode='200',
                responseParameters={
                    'method.response.header.Access-Control-Allow-Origin': True,
                    'method.response.header.Access-Control-Allow-Headers': True,
                    'method.response.header.Access-Control-Allow-Methods': True
                }
            )
            
            # Configurer les réponses d'intégration pour OPTIONS
            self.api_gateway_client.put_integration_response(
                restApiId=api_id,
                resourceId=resource_id,
                httpMethod='OPTIONS',
                statusCode='200',
                responseParameters={
                    'method.response.header.Access-Control-Allow-Origin': "'*'",
                    'method.response.header.Access-Control-Allow-Headers': "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
                    'method.response.header.Access-Control-Allow-Methods': "'POST,OPTIONS'"
                }
            )
            
            # Déployer l'API
            deployment_response = self.api_gateway_client.create_deployment(
                restApiId=api_id,
                stageName='prod',
                description=f'Déploiement du {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )
            print("API déployée en production")
            
            # Donner la permission à API Gateway d'invoquer Lambda
            try:
                # Obtenir l'account ID
                sts_client = boto3.client('sts', region_name=self.region)
                account_id = sts_client.get_caller_identity()['Account']
                
                self.lambda_client.add_permission(
                    FunctionName=lambda_arn,
                    StatementId='api-gateway-invoke',
                    Action='lambda:InvokeFunction',
                    Principal='apigateway.amazonaws.com',
                    SourceArn=f"arn:aws:execute-api:{self.region}:{account_id}:{api_id}/*/*"
                )
                print("Permission API Gateway accordée")
            except self.lambda_client.exceptions.ResourceConflictException:
                print("Permission API Gateway déjà accordée")
            
            # URL de l'API
            api_url = f"https://{api_id}.execute-api.{self.region}.amazonaws.com/prod/predict"
            print(f"URL de l'API: {api_url}")
            
            return api_url
            
        except Exception as e:
            print(f"Erreur lors de la création de l'API Gateway: {str(e)}")
            return None
    
    def run_deployment(self):
        """Exécute le déploiement complet"""
        print("DÉPLOIEMENT LAMBDA + API GATEWAY")
        print("=" * 50)
        
        try:
            # 1. Créer le rôle IAM
            role_arn = self.create_lambda_role()
            if not role_arn:
                return None
            
            # 2. Déployer la Lambda function
            lambda_arn = self.deploy_lambda_function(role_arn)
            if not lambda_arn:
                return None
            
            # 3. Créer l'API Gateway
            api_url = self.create_api_gateway(lambda_arn)
            if not api_url:
                return None
            
            print("\n" + "=" * 50)
            print("DÉPLOIEMENT TERMINÉ AVEC SUCCÈS!")
            print("=" * 50)
            print(f"URL de l'API: {api_url}")
            print(f"Lambda ARN: {lambda_arn}")
            print(f"Région: {self.region}")
            
            return api_url
            
        except Exception as e:
            print(f"Erreur lors du déploiement: {str(e)}")
            return None

def main():
    """Fonction principale"""
    deployer = LambdaAPIDeployment()
    api_url = deployer.run_deployment()
    
    if api_url:
        print(f"\nUtilisation de l'API:")
        print(f"curl -X POST {api_url} \\")
        print(f"  -H 'Content-Type: application/json' \\")
        print(f"  -d '{{\"data\": [0.1, 0.2, ...]}}'")
        print(f"\nExemple de requête:")
        print(f"curl -X POST {api_url} \\")
        print(f"  -H 'Content-Type: application/json' \\")
        print(f"  -d '{{\"data\": {[0.1] * 50}}}'")
    else:
        print("Le déploiement a échoué")

if __name__ == "__main__":
    main()
