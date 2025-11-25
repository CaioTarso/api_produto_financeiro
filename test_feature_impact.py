#!/usr/bin/env python3
import requests
import json

def test_scenario(scenario_name, payload):
    """Testa um cenário e retorna as features TE"""
    resp = requests.post('http://localhost:8000/recomendar?debug=true', json=payload)
    data = resp.json()
    
    features = data['debug']['processed_features_all']
    
    # Extrair features TE
    te_features = {k: v for k, v in features.items() if k.startswith('TE_')}
    
    print(f"\n{'='*60}")
    print(f"Cenário: {scenario_name}")
    print(f"{'='*60}")
    print(f"Idade: {payload['idade']}, Renda: {payload['renda']}, Segmento: {payload['segmento']}")
    print(f"Canal: {payload['canal']}, Província: {payload['provincia']}")
    
    # Mostrar algumas features chave
    print(f"\nFeatures Numéricas:")
    print(f"  renda_log: {features['renda_log']:.4f}")
    print(f"  idade_categoria_encoded: {features['idade_categoria_encoded']:.4f}")
    print(f"  renda_categoria_encoded: {features['renda_categoria_encoded']:.4f}")
    
    print(f"\nAlgumas Features TE (primeiras 10):")
    for k, v in list(te_features.items())[:10]:
        print(f"  {k}: {v:.4f}")
    
    print(f"\nTop 3 Produtos:")
    for i, r in enumerate(data['ranking'][:3], 1):
        print(f"  {i}. {r['produto']}: {r['prob']:.6f}")
    
    return te_features, data['ranking']

# Cenário 1: Jovem, baixa renda
scenario1 = {
    "genero": "M",
    "idade": 22,
    "antiguidade": 6,
    "renda": 40000,
    "segmento": "universitario",
    "provincia": "MADRID",
    "canal": "KHL",
    "produtos": []
}

# Cenário 2: Sênior, alta renda
scenario2 = {
    "genero": "M",
    "idade": 60,
    "antiguidade": 200,
    "renda": 500000,
    "segmento": "top",
    "provincia": "MADRID",
    "canal": "KHL",
    "produtos": []
}

# Cenário 3: Mudança de canal
scenario3 = {
    "genero": "M",
    "idade": 35,
    "antiguidade": 48,
    "renda": 85000,
    "segmento": "particulares",
    "provincia": "MADRID",
    "canal": "KAT",  # Canal diferente
    "produtos": []
}

# Cenário 4: Mudança de província
scenario4 = {
    "genero": "M",
    "idade": 35,
    "antiguidade": 48,
    "renda": 85000,
    "segmento": "particulares",
    "provincia": "BARCELONA",  # Província diferente
    "canal": "KHL",
    "produtos": []
}

te1, rank1 = test_scenario("Jovem Universitário", scenario1)
te2, rank2 = test_scenario("Sênior Top", scenario2)
te3, rank3 = test_scenario("Canal KAT", scenario3)
te4, rank4 = test_scenario("Província Barcelona", scenario4)

print(f"\n{'='*60}")
print("ANÁLISE COMPARATIVA")
print(f"{'='*60}")

# Comparar TE entre cenários
print("\nComparando TE_canal entre cenários:")
te_canal_keys = [k for k in te1.keys() if 'canal' in k.lower()]
for key in te_canal_keys[:5]:
    print(f"  {key}:")
    print(f"    Cenário 1: {te1.get(key, 0):.4f}")
    print(f"    Cenário 2: {te2.get(key, 0):.4f}")
    print(f"    Cenário 3 (KAT): {te3.get(key, 0):.4f}")

print("\nComparando TE_segmento entre cenários:")
te_seg_keys = [k for k in te1.keys() if 'segmento_marketing' in k.lower()]
for key in te_seg_keys[:3]:
    print(f"  {key}:")
    print(f"    Cenário 1 (universitario): {te1.get(key, 0):.4f}")
    print(f"    Cenário 2 (top): {te2.get(key, 0):.4f}")
