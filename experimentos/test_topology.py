import pytest
import numpy as np
from topology import generate_topology

def test_generate_topology_seed_42_structure():
    # ==========================================
    # 1. Preparação e Execução
    # ==========================================
    rng = np.random.default_rng(seed=42)
    topo = generate_topology(rng=rng)
    layers = topo['layers']
    
    # ==========================================
    # 2. Verificação das Regras (Tabela 5 + Pilhas)
    # ==========================================
    
    # A. Car Dumpers (3 a 7)
    cd_layer = layers["0"]
    assert cd_layer["name"] == "Car Dumpers"
    assert 3 <= len(cd_layer["machines"]) <= 7
    assert cd_layer["machines"][0] == "CD1"

    # B. Camadas de Conveyors e Distribuição de Máquinas
    # O total de chaves em 'layers' é: 1 (CD) + num_cb_layers + 1 (R)
    num_cb_layers = len(layers) - 2
    assert 2 <= num_cb_layers <= 3
    
    total_cbs_gerados = 0
    for i in range(1, num_cb_layers + 1):
        cb_layer = layers[str(i)]
        num_maquinas_nesta_camada = len(cb_layer["machines"])
        
        # Regra estrutural: Nenhuma camada intermediária pode ficar vazia
        assert num_maquinas_nesta_camada >= 1
        assert cb_layer["name"] == f"Conveyors - Stage {i}"
        
        total_cbs_gerados += num_maquinas_nesta_camada

    # O TOTAL de correias no porto deve ser entre 3 e 7
    assert 3 <= total_cbs_gerados <= 7

    # C. Stackers / Reclaimers (3 a 7)
    last_layer_idx = str(num_cb_layers + 1)
    r_layer = layers[last_layer_idx]
    
    assert r_layer["name"] == "Stackers / Reclaimers"
    assert 3 <= len(r_layer["machines"]) <= 7
    assert r_layer["machines"][0] == "R1"

    # D. Destino Final (Distribuição Normal: Média 20, Desvio 5)
    dest = topo["destination"]
    assert dest["name"] == "Stockpiles"
    # Garante que gerou um número positivo de pilhas (evita quebra por cauda negativa da curva normal)
    assert len(dest["stacks"]) > 0
    assert dest["stacks"][0] == "S01"


def test_generate_topology_defaults():
    # ==========================================
    # Teste de Estresse Estocástico
    # ==========================================
    # Roda sem seed várias vezes para garantir a estabilidade da distribuição
    for _ in range(10):
        topo = generate_topology()
        layers = topo['layers']
        dest = topo["destination"]
        
        num_cb_layers = len(layers) - 2
        total_cbs = sum(len(layers[str(i)]["machines"]) for i in range(1, num_cb_layers + 1))
        
        # Verificações de limites da Cristiane
        assert 3 <= len(layers["0"]["machines"]) <= 7
        assert 3 <= total_cbs <= 7
        assert 3 <= len(layers[str(num_cb_layers + 1)]["machines"]) <= 7
        
        # Verificação das pilhas (sempre deve haver pilhas)
        assert len(dest["stacks"]) > 0
