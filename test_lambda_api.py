import requests
import json
import numpy as np
import argparse
import time

def test_lambda_api(api_url, test_cases=None):
    """Teste l'API Lambda"""
    
    print(f"TEST DE L'API LAMBDA: {api_url}")
    print("=" * 60)
    
    if test_cases is None:
        test_cases = [
            {
                "name": "Données simples",
                "data": [0.1] * 50
            },
            {
                "name": "Entreprise saine",
                "data": [
                    0.15, 0.12, 0.18, 0.20, 0.16, 0.14, 0.17, 0.19, 0.13, 0.15,
                    0.11, 0.16, 0.18, 0.14, 0.12, 0.17, 0.19, 0.15, 0.13, 0.16,
                    0.18, 0.14, 0.12, 0.17, 0.19, 0.15, 0.13, 0.16, 0.18, 0.14,
                    0.12, 0.17, 0.19, 0.15, 0.13, 0.16, 0.18, 0.14, 0.12, 0.17,
                    0.19, 0.15, 0.13, 0.16, 0.18, 0.14, 0.12, 0.17, 0.19, 0.15
                ]
            },
            {
                "name": "Entreprise à risque",
                "data": [
                    -0.20, -0.15, -0.18, -0.12, -0.08, -0.15, -0.10, -0.05, -0.12, -0.08,
                    -0.06, -0.14, -0.09, -0.04, -0.11, -0.07, -0.05, -0.13, -0.08, -0.03,
                    -0.10, -0.06, -0.04, -0.12, -0.07, -0.03, -0.09, -0.05, -0.02, -0.08,
                    -0.04, -0.01, -0.07, -0.03, -0.01, -0.06, -0.02, -0.01, -0.05, -0.01,
                    -0.01, -0.04, -0.01, -0.01, -0.03, -0.01, -0.01, -0.02, -0.01, -0.01
                ]
            },
            {
                "name": "Données aléatoires",
                "data": list(np.random.randn(50))
            }
        ]
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ Test: {test_case['name']}")
        print("-" * 40)
        
        payload = {
            "data": test_case['data']
        }
        
        try:
            # Mesurer le temps de réponse
            start_time = time.time()
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"Succès! Temps de réponse: {response_time:.2f}s")
                print(f"Prédiction: {result['prediction']}")
                print(f"Probabilité: {result['probability']}")
                print(f"Niveau de risque: {result['risk_level']}")
                print(f"Confiance: {result['confidence']:.4f}")
                
                results.append({
                    'test': test_case['name'],
                    'success': True,
                    'response_time': response_time,
                    'prediction': result['prediction'],
                    'risk_level': result['risk_level'],
                    'confidence': result['confidence']
                })
            else:
                print(f"Erreur HTTP {response.status_code}")
                print(f"Réponse: {response.text}")
                
                results.append({
                    'test': test_case['name'],
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'response': response.text
                })
                
        except requests.exceptions.Timeout:
            print("Timeout de la requête")
            results.append({
                'test': test_case['name'],
                'success': False,
                'error': 'Timeout'
            })
        except requests.exceptions.RequestException as e:
            print(f"Erreur de requête: {str(e)}")
            results.append({
                'test': test_case['name'],
                'success': False,
                'error': str(e)
            })
        except Exception as e:
            print(f"Erreur inattendue: {str(e)}")
            results.append({
                'test': test_case['name'],
                'success': False,
                'error': str(e)
            })
    
    # Résumé des tests
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"Tests réussis: {len(successful_tests)}/{len(results)}")
    print(f"Tests échoués: {len(failed_tests)}/{len(results)}")
    
    if successful_tests:
        avg_response_time = sum(r['response_time'] for r in successful_tests) / len(successful_tests)
        print(f"Temps de réponse moyen: {avg_response_time:.2f}s")
        
        predictions = [r['prediction'] for r in successful_tests]
        print(f"Prédictions: {predictions}")
        
        risk_levels = [r['risk_level'] for r in successful_tests]
        print(f"Niveaux de risque: {risk_levels}")
    
    if failed_tests:
        print(f"\nTests échoués:")
        for test in failed_tests:
            print(f"  - {test['test']}: {test['error']}")
    
    return len(failed_tests) == 0

def test_cors(api_url):
    """Teste les en-têtes CORS"""
    print(f"\nTest des en-têtes CORS...")
    
    try:
        # Test OPTIONS
        response = requests.options(api_url, timeout=10)
        
        if response.status_code == 200:
            print("Méthode OPTIONS fonctionne")
            
            cors_headers = {
                'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
            }
            
            print(f"En-têtes CORS: {cors_headers}")
            return True
        else:
            print(f"Méthode OPTIONS échouée: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Erreur lors du test CORS: {str(e)}")
        return False

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Tester l\'API Lambda + API Gateway')
    parser.add_argument('--api-url', type=str, required=True, help='URL de l\'API Gateway')
    parser.add_argument('--test-cors', action='store_true', help='Tester les en-têtes CORS')
    
    args = parser.parse_args()
    
    # Test principal
    success = test_lambda_api(args.api_url)
    
    # Test CORS si demandé
    if args.test_cors:
        cors_success = test_cors(args.api_url)
        success = success and cors_success
    
    if success:
        print("\nTOUS LES TESTS ONT RÉUSSI!")
        print("L'API Lambda + API Gateway fonctionne correctement")
        print("Prêt pour la production!")
    else:
        print("\nCertains tests ont échoué")
        print("Vérifiez la configuration de l'API")

if __name__ == "__main__":
    main()
