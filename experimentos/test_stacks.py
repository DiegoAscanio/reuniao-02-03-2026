import pytest
import numpy as np
from stacks import generate_stacks

@pytest.fixture
def mock_data():
    topology = {
        'layers': {
            '0': {'machines': ['CD1', 'CD2']}
        },
        'destination': {
            'stacks': ['S01', 'S02']
        }
    }
    
    # S01 é o destino de 2 rotas. S02 de apenas 1.
    routes = {
        10: ['CD1', 'CB1', 'S01'],
        20: ['CD2', 'CB1', 'S01'],
        30: ['CD1', 'CB2', 'S02']
    }
    
    machines = {
        'CD1': {'processing_rate': 1/45},
        'CD2': {'processing_rate': 1/45},
        'CB1': {'processing_rate': 1/50}, # Mais lenta (0.02), será o gargalo de S01
        'CB2': {'processing_rate': 1/40}, # Mais rápida (0.025), não será gargalo frente aos CDs
    }
    
    return topology, routes, machines

def test_generate_stacks_structure(mock_data):
    topo, routes, machines = mock_data
    rng = np.random.default_rng(seed=42)
    
    stacks = generate_stacks(topo, routes, machines, rng=rng)
    
    # 1. Verifica presença das chaves e remoção do initial_volume
    assert 'S01' in stacks
    assert 'capacity' in stacks['S01']
    assert 'product' in stacks['S01']
    assert 'eligible_routes' in stacks['S01']
    assert 'initial_volume' not in stacks['S01']
    
    # 2. Verifica a consistência dimensional
    assert isinstance(stacks['S01']['capacity'], int)
    assert stacks['S01']['capacity'] > 0

def test_generate_stacks_custom_props(mock_data):
    topo, routes, machines = mock_data
    rng = np.random.default_rng(seed=42)
    
    # Propriedades customizadas para travar os resultados estocásticos
    custom_props = {
        'hz': 43200, 
        'utilization_factor': 0.8, 
        'range_num_products': (1, 1), # Força a criação de EXATAMENTE 1 produto
        'avg_number_of_piles': 2,
        'lot_sizes': [100], 
        'lot_probs': [1.0], 
        'lt': 1.0, 
        'cap_limiter': 0.8
    }
    
    stacks = generate_stacks(topo, routes, machines, rng=rng, props=custom_props)
    
    # Como travamos o range_num_products, ambas as pilhas DEVEM armazenar "P1"
    assert stacks['S01']['product'] == 'P1'
    assert stacks['S02']['product'] == 'P1'

def test_generate_stacks_proportional_capacity(mock_data):
    topo, routes, machines = mock_data
    rng = np.random.default_rng(seed=42)
    
    stacks = generate_stacks(topo, routes, machines, rng=rng)
    
    # S01 recebe 2 rotas. S02 recebe 1 rota.
    # Pela fórmula do código, S01 deve ter uma capacidade calculada maior.
    assert stacks['S01']['capacity'] > stacks['S02']['capacity']
    
    # Verifica o mapeamento exato de strings nas rotas elegíveis
    assert set(stacks['S01']['eligible_routes']) == {'10', '20'}
    assert set(stacks['S02']['eligible_routes']) == {'30'}

def test_generate_stacks_empty_routes(mock_data):
    topo, _, machines = mock_data
    empty_routes = {} # Simulando um pipeline desconectado
    
    # Valida se a proteção dispara o erro correto
    with pytest.raises(ValueError, match="Nenhuma rota alcança as pilhas"):
        generate_stacks(topo, empty_routes, machines)
