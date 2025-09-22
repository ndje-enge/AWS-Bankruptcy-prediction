import boto3
import argparse
import sys
import os
from datetime import datetime

# Charger la configuration
try:
    from config import load_config
    config = load_config()
except ImportError:
    # Fallback si config.py n'existe pas
    config = {
        'SAGEMAKER_ENDPOINT_NAME': os.environ.get('SAGEMAKER_ENDPOINT_NAME'),
        'LAMBDA_FUNCTION_NAME': os.environ.get('LAMBDA_FUNCTION_NAME', 'bankruptcy-prediction-api'),
        'API_GATEWAY_ID': os.environ.get('API_GATEWAY_ID'),
        'AWS_REGION': os.environ.get('AWS_REGION', 'eu-west-3')
    }

class ProjectManager:
    def __init__(self, region=None):
        self.region = region or config['AWS_REGION']
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        self.lambda_client = boto3.client('lambda', region_name=self.region)
        self.api_gateway_client = boto3.client('apigateway', region_name=self.region)
        
        # Noms des ressources déployées
        self.sagemaker_endpoint = self.get_active_endpoint()
        self.lambda_function = config['LAMBDA_FUNCTION_NAME']
        self.api_gateway_id = config['API_GATEWAY_ID']
        self.api_url = f'https://{self.api_gateway_id}.execute-api.{self.region}.amazonaws.com/prod/predict'
    
    def get_active_endpoint(self):
        """Trouve automatiquement l'endpoint SageMaker actif"""
        # D'abord, essayer de charger depuis le fichier sauvegardé
        saved_endpoint = self.load_endpoint_name()
        if saved_endpoint:
            try:
                response = self.sagemaker_client.describe_endpoint(EndpointName=saved_endpoint)
                if response['EndpointStatus'] == 'InService':
                    return saved_endpoint
            except:
                pass
        
        # Sinon, chercher dans la liste des endpoints
        try:
            response = self.sagemaker_client.list_endpoints()
            for endpoint in response['Endpoints']:
                if 'bankruptcy-predictor' in endpoint['EndpointName']:
                    return endpoint['EndpointName']
            return None
        except Exception as e:
            print(f"Erreur lors de la découverte de l'endpoint: {e}")
            return None
    
    def save_endpoint_name(self, endpoint_name):
        """Sauvegarde le nom de l'endpoint dans un fichier"""
        try:
            with open('.endpoint_name', 'w') as f:
                f.write(endpoint_name)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'endpoint: {e}")
    
    def load_endpoint_name(self):
        """Charge le nom de l'endpoint depuis le fichier"""
        try:
            with open('.endpoint_name', 'r') as f:
                return f.read().strip()
        except:
            return None

    def status(self):
        """Affiche le statut de toutes les ressources"""
        print("STATUT DU PROJET BANKRUPTCY PREDICTION")
        print("=" * 60)
        
        # Vérifier SageMaker Endpoint
        if self.sagemaker_endpoint:
            try:
                response = self.sagemaker_client.describe_endpoint(EndpointName=self.sagemaker_endpoint)
                status = response['EndpointStatus']
                print(f"SageMaker Endpoint: {status}")
                if status == 'InService':
                    print(f"   Endpoint actif: {self.sagemaker_endpoint}")
                else:
                    print(f"   Endpoint inactif: {status}")
            except Exception as e:
                print(f"SageMaker Endpoint: Erreur ({str(e)})")
        else:
            print("SageMaker Endpoint: Aucun endpoint trouvé")
            print("   Utilisez 'python3 manage_project.py resume' pour créer un endpoint")
        
        # Vérifier Lambda Function
        try:
            response = self.lambda_client.get_function(FunctionName=self.lambda_function)
            state = response['Configuration']['State']
            print(f"Lambda Function: {state}")
            if state == 'Active':
                print(f"   Lambda active: {self.lambda_function}")
            else:
                print(f"   Lambda inactive: {state}")
        except Exception as e:
            print(f"Lambda Function: Non trouvée ({str(e)})")
        
        # Vérifier API Gateway
        try:
            response = self.api_gateway_client.get_rest_api(restApiId=self.api_gateway_id)
            print(f"API Gateway: Actif")
            print(f"   API active: {self.api_gateway_id}")
            print(f"   URL: {self.api_url}")
        except Exception as e:
            print(f"API Gateway: Non trouvé ({str(e)})")
        
        print("\n" + "=" * 60)

    def pause(self):
        """Met en pause les ressources (supprime l'endpoint SageMaker)"""
        print("MISE EN PAUSE DU PROJET")
        print("=" * 40)
        
        try:
            # Supprimer l'endpoint SageMaker
            print(f"Suppression de l'endpoint SageMaker: {self.sagemaker_endpoint}")
            self.sagemaker_client.delete_endpoint(EndpointName=self.sagemaker_endpoint)
            print("Endpoint SageMaker supprimé")
            
            # Supprimer la configuration d'endpoint
            try:
                self.sagemaker_client.delete_endpoint_config(
                    EndpointConfigName=self.sagemaker_endpoint
                )
                print("Configuration d'endpoint supprimée")
            except:
                pass
            
            # Supprimer le fichier d'endpoint
            try:
                import os
                if os.path.exists('.endpoint_name'):
                    os.remove('.endpoint_name')
            except:
                pass
            
            print("\nPROJET EN PAUSE")
            print("   - Lambda Function: Conservée")
            print("   - API Gateway: Conservé")
            print("   - SageMaker Endpoint: Supprimé (économies)")
            print("\nPour relancer: python3 manage_project.py resume")
            
        except Exception as e:
            print(f"Erreur lors de la mise en pause: {str(e)}")

    def resume(self):
        """Relance les ressources (recrée l'endpoint SageMaker)"""
        print("RELANCE DU PROJET")
        print("=" * 40)
        
        try:
            # Vérifier si le modèle existe
            model_name = config['SAGEMAKER_MODEL_NAME']
            try:
                self.sagemaker_client.describe_model(ModelName=model_name)
                print(f"Modèle trouvé: {model_name}")
            except:
                print(f"Modèle non trouvé: {model_name}")
                print("   Veuillez d'abord déployer le modèle avec: python3 deploy.py")
                return
            
            # Créer la configuration d'endpoint (nom court pour éviter les limites)
            endpoint_config_name = f"bankruptcy-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            print(f"Création de la configuration d'endpoint: {endpoint_config_name}")
            
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.t2.medium'
                }]
            )
            print("Configuration d'endpoint créée")
            
            # Créer l'endpoint avec un nom unique
            endpoint_name = f"bankruptcy-predictor-optimized-compatible-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            print(f"Création de l'endpoint: {endpoint_name}")
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print("Endpoint créé")
            print("Attente de la disponibilité de l'endpoint...")
            
            # Attendre que l'endpoint soit prêt
            waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=endpoint_name)
            
            # Sauvegarder le nom de l'endpoint
            self.save_endpoint_name(endpoint_name)
            
            # Mettre à jour la Lambda function avec le nouveau endpoint
            print("Mise à jour de la Lambda function...")
            self.lambda_client.update_function_configuration(
                FunctionName=self.lambda_function,
                Environment={
                    'Variables': {
                        'SAGEMAKER_ENDPOINT_NAME': endpoint_name
                    }
                }
            )
            print("Lambda function mise à jour")
            
            print("\nPROJET RELANCÉ")
            print(f"   - SageMaker Endpoint: {endpoint_name}")
            print("   - Lambda Function: Active (mise à jour)")
            print("   - API Gateway: Actif")
            print(f"   - URL API: {self.api_url}")
            print(f"\nEndpoint sauvegardé pour la prochaine utilisation")
            
        except Exception as e:
            print(f"Erreur lors de la relance: {str(e)}")

    def test(self):
        """Teste l'API déployée"""
        print("TEST DE L'API")
        print("=" * 30)
        
        try:
            import requests
            
            # Données de test
            test_data = {
                "data": [0.1] * 50  # 50 features avec valeur 0.1
            }
            
            print(f"Test de l'API: {self.api_url}")
            response = requests.post(
                self.api_url,
                json=test_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("API fonctionnelle")
                print(f"   Prédiction: {result.get('prediction')}")
                print(f"   Probabilité: {result.get('probability')}")
                print(f"   Niveau de risque: {result.get('risk_level')}")
            else:
                print(f"Erreur API: {response.status_code}")
                print(f"   Réponse: {response.text}")
                
        except Exception as e:
            print(f"Erreur lors du test: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Gestionnaire du projet Bankruptcy Prediction')
    parser.add_argument('action', choices=['status', 'pause', 'resume', 'test'], 
                       help='Action à effectuer')
    parser.add_argument('--region', default='eu-west-3', 
                       help='Région AWS (défaut: eu-west-3)')
    
    args = parser.parse_args()
    
    manager = ProjectManager(region=args.region)
    
    if args.action == 'status':
        manager.status()
    elif args.action == 'pause':
        manager.pause()
    elif args.action == 'resume':
        manager.resume()
    elif args.action == 'test':
        manager.test()

if __name__ == "__main__":
    main()
