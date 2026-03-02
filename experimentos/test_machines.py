import pdb

import pytest
import numpy as np
from machines import (
    calculate_in_degrees, 
    get_base_setup_time, 
    _setup_time, 
    generate_machines
)

@pytest.fixture
def mock_setup_config():
    return {'CD': 0.0, 'CB': 900.0, 'R': 1800.0}

@pytest.fixture
def mock_topo_pipeline_and_routes():
    topo = {
        'layers': {
            '0': {'machines': ['CD1', 'CD2'], 'double_cap_prob_base': 0.0},
            '1': {'machines': ['CB1_1'], 'double_cap_prob_base': 0.2},
            '2': {'machines': ['R1'], 'double_cap_prob_base': 0.0}
        }
    }
    # CB1_1 recebe de dois CDs (In-degree = 2, é um gargalo)
    pipe = {
        'CD1': ['CB1_1'],
        'CD2': ['CB1_1'],
        'CB1_1': ['R1'],
        'R1': []
    }
    routes = {
        10: ['CD1', 'CB1_1', 'R1'], # Rota 10 passa pelo CD1
        20: ['CD2', 'CB1_1', 'R1']  # Rota 20 passa pelo CD2
    }
    
    return topo, pipe, routes

def test_calculate_in_degrees(mock_topo_pipeline_and_routes):
    _, pipe, _ = mock_topo_pipeline_and_routes
    in_degrees = calculate_in_degrees(pipe)
    
    assert in_degrees['CD1'] == 0
    assert in_degrees['CD2'] == 0
    assert in_degrees['CB1_1'] == 2 # Gargalo corretamente identificado
    assert in_degrees['R1'] == 1

def test_get_base_setup_time(mock_setup_config):
    assert get_base_setup_time('CD1', mock_setup_config) == 0.0
    assert get_base_setup_time('CB2_3', mock_setup_config) == 900.0
    assert get_base_setup_time('R5', mock_setup_config) == 1800.0
    assert get_base_setup_time('UNKNOWN_MACHINE', mock_setup_config) == 0.0

def test_setup_time_logic(mock_setup_config):
    served_routes = {10, 20}
    inf_val = 2**20 - 1
    
    # Caso 1: Mesma rota, e a máquina atende essa rota -> Setup 0
    assert _setup_time(10, 10, served_routes, 'CB1_1', mock_setup_config, inf_val) == 0
    
    # Caso 2: Rotas diferentes, mas ambas passam pela máquina -> Setup Base
    assert _setup_time(10, 20, served_routes, 'CB1_1', mock_setup_config, inf_val) == 900.0
    
    # Caso 3: Transição para uma rota que a máquina NÃO atende -> Penalidade Big-M (inf)
    assert _setup_time(10, 99, served_routes, 'CB1_1', mock_setup_config, inf_val) == inf_val
    assert _setup_time(99, 10, served_routes, 'CB1_1', mock_setup_config, inf_val) == inf_val

def test_generate_machines_integration(mock_topo_pipeline_and_routes, mock_setup_config):
    topo, pipe, routes = mock_topo_pipeline_and_routes
    rng = np.random.default_rng(seed=42)
    inf_val = 2**20 - 1
    
    machines = generate_machines(
        topology=topo,
        pipeline_graph=pipe,
        routes_dict=routes,
        rng=rng,
        alpha_input_degree=0.15,
        base_processing_rate=0.0222,
        setup_config=mock_setup_config,
        inf=inf_val
    )
    
    # 1. Checa a presença e tipos estruturais
    assert 'CD1' in machines
    assert 'CB1_1' in machines
    
    cb = machines['CB1_1']
    assert isinstance(cb['processing_rate'], float)
    assert isinstance(cb['double_capacity'], bool)
    assert isinstance(cb['served_routes'], list)
    assert isinstance(cb['setup_times'], np.ndarray)
    
    # 2. Valida o filtro de rotas servidas
    # CB1_1 atende as rotas 10 e 20
    assert set(cb['served_routes']) == {10, 20}
    
    # CD1 atende APENAS a rota 10
    assert set(machines['CD1']['served_routes']) == {10}
    
    # 3. Valida as dimensões da Matriz Numpy e os valores Big-M
    # A matriz deve ser len(routes) x len(routes) -> 2x2
    assert cb['setup_times'].shape == (2, 2)
    
    # Para o CD1 (que só atende a rota 10):
    # setup(10, 10) = 0
    # setup(10, 20) = inf (pois CD1 não atende a rota 20)
    # setup(20, 10) = inf
    # setup(20, 20) = inf (embora r1==r2, r1 não está em served_routes para o CD1)
    
    cd1_matrix = machines['CD1']['setup_times']
    # Como as rotas no dict são [10, 20], a matriz segue essa ordem de indexação
    assert cd1_matrix[0, 0] == 0       # 10 -> 10
    assert cd1_matrix[0, 1] == inf_val # 10 -> 20
    assert cd1_matrix[1, 0] == inf_val # 20 -> 10
    assert cd1_matrix[1, 1] == inf_val # 20 -> 20
