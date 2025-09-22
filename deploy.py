import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel
import os
import json
import time
from datetime import datetime
import joblib
import tarfile
import shutil

try:
    from config import load_config
    config = load_config()
except ImportError:
    # Fallback si config.py n'existe pas
    config = {
        'SAGEMAKER_ROLE_ARN': os.environ.get('SAGEMAKER_ROLE_ARN'),
        'S3_BUCKET_NAME': os.environ.get('S3_BUCKET_NAME'),
        'AWS_REGION': os.environ.get('AWS_REGION', 'eu-west-3')
    }



class OptimizedCompatibleDeployment:
    def __init__(self, region='eu-west-3'):
        self.region = region
        self.session = sagemaker.Session(boto3.Session(region_name=region))
        self.role_arn = config['SAGEMAKER_ROLE_ARN']
        self.bucket_name =  config['S3_BUCKET_NAME'] 
        
    def prepare_optimized_model(self):
        print("Préparation du modèle MLPClassifier optimisé...")
        
        # Vérifier que le modèle optimisé existe
        model_file = 'models/MLPClassifier_optimized.pkl'
        if not os.path.exists(model_file):
            print("Modèle optimisé non trouvé. Exécutez d'abord train_model.py")
            return None
        
        # Créer le dossier du modèle
        model_dir = 'optimized_compatible_package'
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        
        # Copier tous les fichiers nécessaires
        files_to_copy = [
            ('models/MLPClassifier_optimized.pkl', 'MLPClassifier_optimized.pkl'),
            ('models/scaler_optimized.pkl', 'scaler_optimized.pkl'),
            ('models/selected_features_optimized.pkl', 'selected_features_optimized.pkl'),
            ('inference.py', 'inference.py')
        ]
        
        for src, dst in files_to_copy:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(model_dir, dst))
                print(f"Copié: {src} -> {dst}")
            else:
                print(f"Fichier manquant: {src}")
                return None
        
        # Créer l'archive
        model_tar = 'optimized_compatible_model.tar.gz'
        with tarfile.open(model_tar, 'w:gz') as tar:
            tar.add(model_dir, arcname='.')
        
        shutil.rmtree(model_dir)
        print(f"Modèle optimisé préparé: {model_tar}")
        return model_tar
    
    def upload_model_to_s3(self, model_tar):
        print("Téléchargement du modèle vers S3...")
        
        s3_key = f"models/{model_tar}"
        s3_client = boto3.client('s3', region_name=self.region)
        
        try:
            s3_client.upload_file(model_tar, self.bucket_name, s3_key)
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            print(f"Modèle téléchargé: {s3_uri}")
            return s3_uri
        except Exception as e:
            print(f"Erreur lors du téléchargement: {str(e)}")
            return None
    
    def deploy_model(self, model_s3_uri):
        print("Déploiement du modèle optimisé sur SageMaker...")
        
        try:
            # Créer le modèle SageMaker
            model_name = f"bankruptcy-model-optimized-compatible-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            sklearn_model = SKLearnModel(
                model_data=model_s3_uri,
                role=self.role_arn,
                entry_point='inference.py',
                framework_version='1.0-1',
                py_version='py3',
                sagemaker_session=self.session,
                name=model_name
            )
            
            # Déployer le endpoint
            endpoint_name = f"bankruptcy-predictor-optimized-compatible-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            predictor = sklearn_model.deploy(
                initial_instance_count=1,
                instance_type='ml.t2.medium',
                endpoint_name=endpoint_name
            )
            
            print(f"Modèle optimisé déployé: {endpoint_name}")
            return endpoint_name, predictor
            
        except Exception as e:
            print(f"Erreur lors du déploiement: {str(e)}")
            return None, None
    
    def test_deployed_model(self, endpoint_name):
        print("Test du modèle optimisé déployé...")
        
        try:
            # Créer un client SageMaker Runtime
            runtime = boto3.client('sagemaker-runtime', region_name=self.region)
            
            # Données de test (50 features)
            import numpy as np
            test_data = list(np.random.randn(50))
            
            # Faire la prédiction
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(test_data)
            )
            
            # Lire la réponse
            result = json.loads(response['Body'].read().decode())
            print("Test réussi!")
            print(f"Résultat: {json.dumps(result, indent=2)}")
            return True
            
        except Exception as e:
            print(f"Test échoué: {str(e)}")
            return False
    
    def run_optimized_deployment(self):
        print("DÉMARRAGE DU DÉPLOIEMENT DU MODÈLE OPTIMISÉ COMPATIBLE")
        print("=" * 70)
        print("Utilisation du modèle MLPClassifier optimisé (ROC-AUC: 0.9936)")
        print("=" * 70)
        
        try:
            # 1. Préparer le modèle optimisé
            model_tar = self.prepare_optimized_model()
            if not model_tar:
                return None
            
            # 2. Télécharger vers S3
            model_s3_uri = self.upload_model_to_s3(model_tar)
            if not model_s3_uri:
                return None
            
            # 3. Déployer
            endpoint_name, predictor = self.deploy_model(model_s3_uri)
            if not endpoint_name:
                return None
            
            # 4. Tester
            if self.test_deployed_model(endpoint_name):
                print("Test du modèle réussi!")
            else:
                print("Test du modèle échoué, mais le endpoint est déployé")
            
            print("\n" + "=" * 70)
            print("DÉPLOIEMENT DU MODÈLE OPTIMISÉ TERMINÉ AVEC SUCCÈS!")
            print("=" * 70)
            print(f"Endpoint déployé: {endpoint_name}")
            print(f"Modèle: MLPClassifier finetuné")
            print(f"Features: 50 (sélectionnées)")
            print(f"Région: {self.region}")
            print(f"Instance: ml.t2.medium")
            
            return endpoint_name
            
        except Exception as e:
            print(f"Erreur lors du déploiement: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    deployer = OptimizedCompatibleDeployment()
    endpoint_name = deployer.run_optimized_deployment()

if __name__ == "__main__":
    main()
