import numpy as np
from utils import (
    compute_lambda, 
    calculate_p_bar, 
    sample_lambda_from_poisson,
    extract_product_catalog,
    filter_eligible_stockpiles,
    calculate_route_processing_rate
)

def generate_train_arrivals(total_trains: int, hz: int, rng=None) -> list:
    '''
    Gera uma lista cronológica de tempos de chegada de trens (em segundos).
    Baseado na propriedade de que chegadas condicionadas ao total N num 
    Processo de Poisson seguem uma distribuição Uniforme(0, hz) ordenada.
    '''
    if rng is None:
        rng = np.random.default_rng()
        
    if total_trains <= 0:
        return []
        
    # Sorteia N tempos aleatórios dentro do horizonte e os ordena cronologicamente
    arrivals = rng.uniform(0, hz, total_trains)
    
    return np.sort(arrivals).tolist()

def generate_lots(
    routes : dict,
    machines: dict,
    stacks: dict,
    rng = None,
    props: dict = None # type: ignore
) -> dict:
    '''
        Estágio 6: Gera a demanda estocástica do porto (Lotes).
        Amarrando Poisson, fracionamento de vagões e matching espacial de pilhas.
    '''
    if rng is None:
        rng = np.random.default_rng()

    if props is None:
        props = {
            'hz': 43200,
            'utilization_factor': 0.8,
            'lot_sizes': [100, 50, 25],
            'lot_probs': [0.81, 0.17, 0.02],
            'lot_sizes_and_frequencies': {
                100: 90,
                50: 19,
                25: 2
            },
            'lt': 1.11,
            'initial_maneuver_time': 1800
        }

    # 1. Preparação dos dados para a Teoria das Filas
    hz = props['hz']
    u = props['utilization_factor']
    lt = props['lt']
    cds = sum(1 for m in machines.keys() if str(m).startswith('CD')) # Contagem de car dumpers
    lot_sizes_and_frequencies = props['lot_sizes_and_frequencies']

    # 2. Cálculo de p_bar (taxa média de processamento)
    p_bar = calculate_p_bar(routes, machines, lot_sizes_and_frequencies)

    # 3. Cálculo da taxa de chegada (lambda) usando a fórmula presente
    #    no artigo da cristiane: lam = (u * cds) / (p_bar * lt)
    lam = compute_lambda(u, cds, p_bar, lt)

    # 4. Cálculo do número total de trens que chegarão no horizonte (hz)
    total_trains = sample_lambda_from_poisson(lam,  rng = rng, hz = hz) # type: ignore

    # 5. Geração dos tempos de chegada dos trens
    train_arrivals = generate_train_arrivals(total_trains, hz, rng)

    # 6. Geração dos lotes para cada trem baseado na distribuição de fracionamento
    lot_sizes = props['lot_sizes']
    lot_probs = props['lot_probs']
    initial_maneuver_time = props['initial_maneuver_time']
    product_catalog = extract_product_catalog(stacks) # type: ignore
    lots_products = [ rng.choice(product_catalog) for _ in range(total_trains) ] # type: ignore

    # 6.1 Define o dicionário de lotes

    lots = {
        i: {
            'arrival_time': arrival_time + initial_maneuver_time,
            'wagons': rng.choice(lot_sizes, p=lot_probs),
            'product': lots_products[i],
            'eligible_stockpiles': filter_eligible_stockpiles(stacks, lots_products[i])
        } for i, arrival_time in enumerate(train_arrivals)
    }

    # 6.2 A partir das eligible stockpiles, vamos definir as rotas possíveis
    #     para cada lote
    for l in lots:
        eligible_stockpiles = lots[l]['eligible_stockpiles']
        possible_routes = []
        for sp in eligible_stockpiles:
            for route in routes:
                _dest = routes[route][-1] # Destino da rota
                if _dest == sp:
                    possible_routes.append(route)
        lots[l]['eligible_routes'] = possible_routes

    # 6.3 Para cada rota elegível, vamos calcular o tempo de processamento
    #     considerando a taxa de processamento da rota e a quantidade de vagões
    #     do lote. O tempo de processamento é dado por: 
    #     processing_time = wagons / processing_rate
    for l in lots:
        eligible_routes = lots[l]['eligible_routes']
        route_processing_times = {}
        for r in eligible_routes:
            processing_rate = calculate_route_processing_rate(
                routes[r], machines
            )
            processing_time = lots[l]['wagons'] / processing_rate
            route_processing_times[r] = processing_time
        lots[l]['processing_times'] = route_processing_times

    return lots
