import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.combine import SMOTETomek
import joblib
import os

def load_and_prepare_data():
    print("Chargement des données...")
    
    # Charger les données
    df = pd.read_csv('data.csv')
    
    # Séparer les features et la target
    X = df.drop('Bankrupt?', axis=1)
    y = df['Bankrupt?']
    
    print(f"Données chargées: {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"Distribution des classes: {y.value_counts().to_dict()}")
    
    return X, y

def optimize_features(X, y):
    print("Optimisation des features...")
    
    # Sélection des meilleures features
    selector = SelectKBest(score_func=f_classif, k=50)  # Garder 50 meilleures features
    X_selected = selector.fit_transform(X, y)
    
    # Récupérer les noms des features sélectionnées
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Features sélectionnées: {len(selected_features)}")
    print(f"Features: {selected_features[:10]}...")  # Afficher les 10 premières
    
    return X_selected, selected_features

def balance_data(X, y):
    print("Équilibrage des données...")
    
    # Appliquer SMOTE-Tomek
    smote_tomek = SMOTETomek(random_state=42)
    X_balanced, y_balanced = smote_tomek.fit_resample(X, y)
    
    print(f"Données équilibrées: {X_balanced.shape[0]} échantillons")
    print(f"Nouvelle distribution: {np.bincount(y_balanced)}")
    
    return X_balanced, y_balanced

def train_mlp_model(X, y):
    print("Entraînement du modèle MLPClassifier optimisé...")
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normaliser les données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraîner le modèle MLPClassifier avec des paramètres optimisés
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Architecture optimisée
        activation='relu',
        solver='adam',
        alpha=0.001,  # Régularisation L2
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    # Entraîner le modèle
    mlp.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = mlp.predict(X_test_scaled)
    y_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
    
    # Métriques
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Modèle entraîné avec succès!")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(f"Accuracy: {mlp.score(X_test_scaled, y_test):.4f}")
    
    return mlp, scaler, auc_score, X_test_scaled, y_test, y_pred_proba

def save_model_and_artifacts(model, scaler, selected_features, auc_score):
    print("Sauvegarde du modèle et des artefacts...")
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = 'models/MLPClassifier_optimized.pkl'
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé: {model_path}")
    
    # Sauvegarder le scaler
    scaler_path = 'models/scaler_optimized.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler sauvegardé: {scaler_path}")
    
    # Sauvegarder les features sélectionnées
    features_path = 'models/selected_features_optimized.pkl'
    joblib.dump(selected_features, features_path)
    print(f"Features sauvegardées: {features_path}")
    
    # Sauvegarder les métriques
    metrics = {
        'roc_auc': auc_score,
        'model_type': 'MLPClassifier',
        'n_features': len(selected_features),
        'features': selected_features
    }
    metrics_path = 'models/model_metrics_optimized.pkl'
    joblib.dump(metrics, metrics_path)
    print(f"Métriques sauvegardées: {metrics_path}")
    
    return model_path, scaler_path, features_path, metrics_path

def main():
    print("ENTRAÎNEMENT D'UN MODÈLE COMPATIBLE SAGEMAKER")
    print("=" * 60)
    
    try:
        # 1. Charger et préparer les données
        X, y = load_and_prepare_data()
        
        # 2. Optimiser les features
        X_selected, selected_features = optimize_features(X, y)
        
        # 3. Équilibrer les données
        X_balanced, y_balanced = balance_data(X_selected, y)
        
        # 4. Entraîner le modèle
        model, scaler, auc_score, X_test, y_test, y_pred_proba = train_mlp_model(X_balanced, y_balanced)
        
        # 5. Sauvegarder les artefacts
        model_path, scaler_path, features_path, metrics_path = save_model_and_artifacts(
            model, scaler, selected_features, auc_score
        )
        
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        print(f"Modèle: MLPClassifier optimisé")
        print(f"ROC-AUC Score: {auc_score:.4f}")
        print(f"Nombre de features: {len(selected_features)}")
        print(f"Fichiers créés:")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")
        print(f"  - {features_path}")
        print(f"  - {metrics_path}")
        
        return True
        
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nLe modèle compatible est prêt pour le déploiement SageMaker!")
    else:
        print("\nL'entraînement a échoué.")