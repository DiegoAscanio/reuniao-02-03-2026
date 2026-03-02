import numpy as np

import numpy as np

def generate_topology(
    rng=None,
    cds_info={
        'range': (3, 8),
        'properties': {
            'same_layer_interconnection_prob': 0.0
        }
    },
    cbs_info={
        'total_range': (3, 8),
        'layer_range': (2, 4), # Note: (2, 4) in numpy integers is 2 to 3 inclusive
        'properties': {
            'same_layer_interconnection_prob': 0.3,
            'double_cap_prob_base': 0.2,
            'next_layer_connection_to_next_index_prob': 0.0
        }
    },
    stackers_info={
        'range': (3, 8),
        'properties': {
            'same_layer_interconnection_prob': 0.0
        }
    },
    stacks_info={
        'mean': 20,
        'std_dev': 5,
        'properties': {
            'coverage_strategy': 'proportional_split',
            'adjacent_machine_overlap_prob': 0.85
        }
    }
) -> dict:
    '''
    Gera uma configuração de topologia aleatória baseada nos critérios da Tabela 5.
    Totalmente parametrizada via dicionários para facilitar a criação de novos cenários.
    '''
    if rng is None:
        rng = np.random.default_rng()
    
    # 1. Sorteio das quantidades base usando os parâmetros injetados
    n_cds = rng.integers(cds_info['range'][0], cds_info['range'][1])
    total_cbs = rng.integers(cbs_info['total_range'][0], cbs_info['total_range'][1])
    num_cb_layers = rng.integers(cbs_info['layer_range'][0], cbs_info['layer_range'][1])
    n_rs = rng.integers(stackers_info['range'][0], stackers_info['range'][1])
    
    n_stacks = int(
        rng.normal(
            loc=stacks_info['mean'],
            scale=stacks_info['std_dev']
        )
    )
    # Proteção: Garante pelo menos 1 pilha
    n_stacks = max(n_stacks, 1)
    
    # Proteção: Garante que o número de camadas não seja maior que o total de CBs
    num_cb_layers = min(num_cb_layers, total_cbs)
    
    # 2. Distribui o total de CBs pelas camadas
    cbs_per_layer = [1] * num_cb_layers
    remaining_cbs = total_cbs - num_cb_layers
    
    for _ in range(remaining_cbs):
        idx = rng.integers(0, num_cb_layers)
        cbs_per_layer[idx] += 1

    layers = {}
    
    # Camada 0: Car Dumpers
    layers["0"] = {
        "name": "Car Dumpers",
        "machines": [f"CD{i+1}" for i in range(n_cds)],
        **cds_info['properties'] # Desempacota as propriedades dinamicamente
    }
    
    # Camadas 1 a N: Conveyor Belts
    for i in range(1, num_cb_layers + 1):
        n_cbs_in_this_layer = cbs_per_layer[i-1] 
        
        layers[str(i)] = {
            "name": f"Conveyors - Stage {i}",
            "machines": [f"CB{i}_{j+1}" for j in range(n_cbs_in_this_layer)],
            **cbs_info['properties'] # Desempacota as propriedades dinamicamente (inclui double_cap)
        }
    
    # Última Camada de Máquinas: Stackers / Reclaimers
    last_layer_idx = num_cb_layers + 1
    layers[str(last_layer_idx)] = {
        "name": "Stackers / Reclaimers",
        "machines": [f"R{i+1}" for i in range(n_rs)],
        **stackers_info['properties']
    }
    
    # Destino Final: Pilhas
    destination = {
        "name": "Stockpiles",
        "stacks": [f"S{i+1:02d}" for i in range(n_stacks)],
        **stacks_info['properties']
    }
    
    return {"layers": layers, "destination": destination}
