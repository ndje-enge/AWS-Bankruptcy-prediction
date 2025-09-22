import json
import joblib
import numpy as np
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMLPPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.model_loaded = False
    
    def model_fn(self, model_dir):
        try:
            logger.info(f"Chargement du modèle depuis: {model_dir}")
            
            # Charger le modèle
            model_path = os.path.join(model_dir, 'MLPClassifier_optimized.pkl')
            self.model = joblib.load(model_path)
            logger.info("Modèle MLPClassifier chargé avec succès")
            
            # Charger le scaler
            scaler_path = os.path.join(model_dir, 'scaler_optimized.pkl')
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler chargé avec succès")
            
            # Charger les features sélectionnées
            features_path = os.path.join(model_dir, 'selected_features_optimized.pkl')
            self.selected_features = joblib.load(features_path)
            logger.info(f"Features sélectionnées chargées: {len(self.selected_features)}")
            
            self.model_loaded = True
            logger.info("Tous les artefacts chargés avec succès")
            
            return self
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise e
    
    def input_fn(self, request_body, content_type):
        """Parse les données d'entrée"""
        try:
            if content_type == 'application/json':
                # Charger les données JSON
                data = json.loads(request_body)
                
                # Convertir en numpy array
                if isinstance(data, list):
                    # Données sous forme de liste
                    input_data = np.array(data, dtype=np.float32)
                elif isinstance(data, dict):
                    # Données sous forme de dictionnaire
                    if 'data' in data:
                        input_data = np.array(data['data'], dtype=np.float32)
                    else:
                        # Prendre toutes les valeurs du dictionnaire
                        input_data = np.array(list(data.values()), dtype=np.float32)
                else:
                    raise ValueError("Format de données non supporté")
                
                # Vérifier le nombre de features
                expected_features = len(self.selected_features)
                if len(input_data) != expected_features:
                    logger.warning(f"Nombre de features reçu: {len(input_data)}, attendu: {expected_features}")
                    # Ajuster si nécessaire
                    if len(input_data) > expected_features:
                        input_data = input_data[:expected_features]
                    else:
                        # Compléter avec des zéros
                        padding = np.zeros(expected_features - len(input_data))
                        input_data = np.concatenate([input_data, padding])
                
                logger.info(f"Données d'entrée traitées: shape {input_data.shape}")
                return input_data
                
            else:
                raise ValueError(f"Content type non supporté: {content_type}")
                
        except Exception as e:
            logger.error(f"Erreur lors du parsing des données d'entrée: {str(e)}")
            raise e
    
    def predict_fn(self, input_data, model=None):
        """Effectue la prédiction"""
        try:
            if not self.model_loaded:
                raise ValueError("Modèle non chargé")
            
            # Reshape pour une seule prédiction
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            # Normaliser les données
            input_scaled = self.scaler.transform(input_data)
            
            # Faire la prédiction
            prediction = self.model.predict(input_scaled)[0]
            prediction_proba = self.model.predict_proba(input_scaled)[0]
            
            # Préparer la réponse
            result = {
                'prediction': int(prediction),
                'probability': {
                    'not_bankrupt': float(prediction_proba[0]),
                    'bankrupt': float(prediction_proba[1])
                },
                'risk_level': 'high' if prediction_proba[1] > 0.7 else 'medium' if prediction_proba[1] > 0.3 else 'low'
            }
            
            logger.info(f"Prédiction effectuée: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise e
    
    def output_fn(self, prediction, accept):
        """Formate la sortie"""
        try:
            if accept == 'application/json':
                return json.dumps(prediction), 'application/json'
            else:
                return json.dumps(prediction), 'application/json'
                
        except Exception as e:
            logger.error(f"Erreur lors du formatage de la sortie: {str(e)}")
            raise e

# Instance globale du prédicteur
predictor = OptimizedMLPPredictor()

def model_fn(model_dir):
    """Fonction de chargement du modèle pour SageMaker"""
    return predictor.model_fn(model_dir)

def input_fn(request_body, content_type):
    """Fonction de parsing des données d'entrée pour SageMaker"""
    return predictor.input_fn(request_body, content_type)

def predict_fn(input_data, model=None):
    """Fonction de prédiction pour SageMaker"""
    return predictor.predict_fn(input_data, model)

def output_fn(prediction, accept):
    """Fonction de formatage de la sortie pour SageMaker"""
    return predictor.output_fn(prediction, accept)
