import json
import boto3
import logging
import os
from typing import Dict, Any

# Configuration du logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Charger la configuration
try:
    from config import load_config
    config = load_config()
except ImportError:
    # Fallback si config.py n'existe pas
    config = {
        'AWS_REGION': os.environ.get('AWS_REGION', 'eu-west-3')
    }

# Client SageMaker Runtime
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=config['AWS_REGION'])
sagemaker_client = boto3.client('sagemaker', region_name=config['AWS_REGION'])

def get_active_endpoint():
    """Trouve automatiquement l'endpoint SageMaker actif"""
    # D'abord, essayer de charger depuis les variables d'environnement
    endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
    if endpoint_name:
        return endpoint_name
    
    # Sinon, essayer de lister les endpoints
    try:
        response = sagemaker_client.list_endpoints()
        for endpoint in response['Endpoints']:
            if 'bankruptcy-predictor' in endpoint['EndpointName']:
                return endpoint['EndpointName']
        return None
    except Exception as e:
        logger.error(f"Erreur lors de la découverte de l'endpoint: {e}")
        return None

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handler principal de la Lambda function
    """
    try:
        # Log de la requête
        logger.info(f"Requête reçue: {json.dumps(event)}")
        
        # Extraire les données de la requête
        if 'body' in event:
            # Requête depuis API Gateway
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            # Requête directe
            body = event
        
        # Valider les données d'entrée
        if 'data' not in body:
            return create_error_response(400, "Champ 'data' manquant dans la requête")
        
        data = body['data']
        
        # Valider le format des données
        if not isinstance(data, list):
            return create_error_response(400, "Le champ 'data' doit être une liste")
        
        if len(data) != 50:
            return create_error_response(400, f"Le champ 'data' doit contenir exactement 50 valeurs, reçu: {len(data)}")
        
        # Valider que toutes les valeurs sont numériques
        try:
            data = [float(x) for x in data]
        except (ValueError, TypeError):
            return create_error_response(400, "Toutes les valeurs dans 'data' doivent être numériques")
        
        # Trouver l'endpoint actif
        endpoint_name = get_active_endpoint()
        if not endpoint_name:
            return create_error_response(503, "Aucun endpoint SageMaker actif trouvé. Veuillez relancer le projet avec 'python3 manage_project.py resume'")
        
        # Appeler le endpoint SageMaker
        response = call_sagemaker_endpoint(data, endpoint_name)
        
        # Traiter la réponse
        if response['statusCode'] == 200:
            result = json.loads(response['body'])
            
            # Ajouter des métadonnées
            enhanced_result = {
                'prediction': result['prediction'],
                'probability': result['probability'],
                'risk_level': result['risk_level'],
                'confidence': max(result['probability']['not_bankrupt'], result['probability']['bankrupt']),
                'model_info': {
                    'model_type': 'MLPClassifier',
                    'roc_auc': 0.9936,
                    'features_used': 50,
                    'endpoint': endpoint_name
                },
                'timestamp': context.aws_request_id
            }
            
            return create_success_response(enhanced_result)
        else:
            return response
            
    except Exception as e:
        logger.error(f"Erreur dans lambda_handler: {str(e)}")
        return create_error_response(500, f"Erreur interne: {str(e)}")

def call_sagemaker_endpoint(data: list, endpoint_name: str) -> Dict[str, Any]:
    """
    Appelle le endpoint SageMaker
    """
    try:
        logger.info(f"Appel du endpoint SageMaker {endpoint_name} avec {len(data)} features")
        
        # Appeler le endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(data)
        )
        
        # Lire la réponse
        result = json.loads(response['Body'].read().decode())
        
        logger.info(f"Réponse SageMaker: {result}")
        
        return create_success_response(result)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'appel SageMaker: {str(e)}")
        return create_error_response(500, f"Erreur SageMaker: {str(e)}")

def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crée une réponse de succès
    """
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'POST,OPTIONS'
        },
        'body': json.dumps(data, indent=2)
    }

def create_error_response(status_code: int, message: str) -> Dict[str, Any]:
    """
    Crée une réponse d'erreur
    """
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'POST,OPTIONS'
        },
        'body': json.dumps({
            'error': message,
            'statusCode': status_code
        }, indent=2)
    }

def handle_options_request(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Gère les requêtes OPTIONS pour CORS
    """
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'POST,OPTIONS'
        },
        'body': ''
    }
