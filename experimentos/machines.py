import numpy as np

def calculate_in_degrees(pipeline_graph: dict) -> dict:
    '''
    Varre o pipeline (lista de adjacências direcional source -> targets)
    e conta quantas arestas chegam em cada máquina (in-degree).
    '''
    in_degrees = {node: 0 for node in pipeline_graph.keys()}
    
    for source, targets in pipeline_graph.items():
        for target in targets:
            if target not in in_degrees:
                in_degrees[target] = 0
            in_degrees[target] += 1
            
    return in_degrees

def get_base_setup_time(machine_id: str, setup_config: dict) -> float:
    '''
    Identifica o prefixo da máquina (CD, CB, R) e retorna o seu tempo
    de setup padrão de acordo com a configuração injetada.
    '''
    if machine_id.startswith('CD'):
        return setup_config.get('CD', 0.0)
    elif machine_id.startswith('CB'):
        return setup_config.get('CB', 0.0)
    elif machine_id.startswith('R'):
        return setup_config.get('R', 0.0)
    return 0.0

def _setup_time(
    r1, r2, served_routes, m, setup_config, inf = 2**20 - 1
):
    if r1 == r2 and r1 in served_routes:
        return 0
    elif r1 in served_routes and r2 in served_routes:
        return get_base_setup_time(m, setup_config)
    else:
        return inf


def generate_machines(
    topology: dict, 
    pipeline_graph: dict, 
    routes_dict: dict,  # <-- NOVO PARÂMETRO
    rng=None, 
    alpha_input_degree: float = 0.15, 
    base_processing_rate: float = 0.0222,
    setup_config: dict = None, # type: ignore
    inf : int = 2**20 - 1
) -> dict:
    '''
    Estágio 4: Instancia as máquinas do porto.
    Gera taxas de processamento gaussianas, define capacidade dupla com 
    base no in-degree (gargalos), atribui tempos de setup e filtra quais
    rotas do sistema passam por cada máquina.
    '''

    if rng is None:
        rng = np.random.default_rng()
        
    if setup_config is None:
        setup_config = {'CD': 1800.0, 'CB': 900, 'R': 900} # Tempos de setup padrão (em segundos) para cada tipo de máquina.
        
    in_degrees = calculate_in_degrees(pipeline_graph)
    machines = {}
    
    for layer_id, layer_data in topology['layers'].items():
        p_base = layer_data.get('double_cap_prob_base', 0.0)
        
        for m in layer_data['machines']:
            in_degree = in_degrees.get(m, 0)
            extra_inputs = max(0, in_degree - 1)
            final_prob = min(1.0, p_base + (alpha_input_degree * extra_inputs))
            
            rate = rng.normal(loc=base_processing_rate, scale=base_processing_rate / 5)
            rate = max(0.0001, rate)
            
            # --- O FILTRO DE ROTAS AQUI ---
            # Para cada (ID da Rota, Caminho) no dicionário, se a máquina 
            # estiver no Caminho, guardamos o ID da Rota.
            served_routes = set([
                route_id for route_id, path in routes_dict.items() if m in path
            ])
            
            machines[m] = {
                'processing_rate': rate,
                'double_capacity': bool(rng.random() < final_prob),
                'served_routes': list(served_routes), # Atributo preenchido!
                'setup_times': np.array([
                    [_setup_time(r1, r2, served_routes, m, setup_config, inf) for r2 in routes_dict ] for r1 in routes_dict
                ]), # Matriz de tempos de setup entre as rotas servidas por esta máquina
                'in_degree': in_degree
            }
            
    return machines
