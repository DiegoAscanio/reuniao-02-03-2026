import numpy as np
from utils import compute_lambda

def generate_stacks(
    topology: dict,
    routes_dict: dict,
    machines_dict: dict,
    rng=None,
    props: dict = None # type: ignore
) -> dict:
    '''
    Estágio 5: Instancia as pilhas (Stockpiles) do porto.
    Totalmente parametrizado por um dicionário de propriedades (props).
    Calcula a capacidade em VAGÕES baseada na vazão esperada (Lambda)
    e distribui essa capacidade proporcionalmente às rotas.
    '''
    if rng is None:
        rng = np.random.default_rng()
        
    # Dicionário padrão caso não seja fornecido na assinatura
    if props is None:
        props = {
            'hz': 43200, 
            'utilization_factor': 0.8, 
            'range_num_products': (2, 20), 
            'lot_sizes': [100, 50, 25], 
            'lot_probs': [0.81, 0.17, 0.02], 
            'lt': 1.11, 
            'cap_limiter': 0.8
        }

    # Desempacotamento de parâmetros principais
    hz = props['hz']
    utilization_factor = props['utilization_factor']
    cap_limiter = props['cap_limiter']
    lot_sizes = props['lot_sizes']
    lot_probs = props['lot_probs']
    lt = props['lt']

    stack_names = topology['destination']['stacks']
    stacks_quantity = len(stack_names)
    
    # 1. Cálculo da Esperança Matemática de Vagões
    w_avg = sum(size * prob for size, prob in zip(lot_sizes, lot_probs))
    
    cds_count = len(topology['layers']["0"]['machines'])
    
    # 2. Identificação do Gargalo e Cálculo do p_bar (Corrigido dimensionalmente)
    route_bottlenecks = [
        min(machines_dict[m]['processing_rate'] for m in path[:-1])
        for path in routes_dict.values() if len(path) > 1
    ]
    
    avg_bottleneck_rate = float(np.mean(route_bottlenecks)) if route_bottlenecks else (1/45)
    
    p_bar = sum(
        (size / avg_bottleneck_rate) * prob 
        for size, prob in zip(lot_sizes, lot_probs)
    )
    
    # 3. Taxa esperada de chegadas e Capacidade do Porto
    lambda_rate = compute_lambda(u=utilization_factor, cds=cds_count, p_bar=p_bar, lt=lt)
    expected_total_wagons = lambda_rate * hz * lt * w_avg
    
    stockyard_capacity = np.floor(cap_limiter * expected_total_wagons)
    
    # 4. Catálogo de Produtos do Porto
    # O rng.integers(low, high) é exclusivo no high, por isso somamos 1
    min_prod, max_prod = props['range_num_products']
    num_products = rng.integers(min_prod, max_prod + 1)
    
    products_catalog = [f"P{i+1}" for i in range(num_products)]
    
    # 5. Cálculo das Razões de Aparição (Proxy via Rotas para distribuir capacidade)
    route_counts = {s: 0 for s in stack_names}
    for r_id, path in routes_dict.items():
        s = path[-1] 
        if s in route_counts:
            route_counts[s] += 1
            
    total_routes = sum(route_counts.values())
    if total_routes == 0:
        raise ValueError('Nenhuma rota alcança as pilhas. Topologia/Pipeline inválidos.')

    # 6. Construção do dicionário final de pilhas
    stacks = {}
    for s in stack_names:
        # Sorteia qual minério esta pilha vai estocar
        product = rng.choice(products_catalog)
        
        eligible_routes = [
            str(r_id) for r_id, path in routes_dict.items() if path[-1] == s
        ]
        
        ratio = route_counts[s] / total_routes
        
        capacity = int(np.floor(
            0.5 * (1.0 / stacks_quantity) * stockyard_capacity + 
            0.5 * ratio * stockyard_capacity
        ))

        capacity = max(1, capacity)
        
        stacks[s] = {
            'capacity': capacity,
            'product': str(product),
            'eligible_routes': eligible_routes
        }
        
    return stacks
