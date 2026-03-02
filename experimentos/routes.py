from scipy.stats import qmc

def get_car_dumpers(pipeline: dict) -> list:
    '''Filtra e retorna apenas as chaves que representam Car Dumpers.'''
    return list(filter(lambda k: str(k).startswith('CD'), pipeline.keys()))

def backtrack(graph: dict, node: object, current_path: list, routes: list):
    '''
    Implementação revisada, recursiva e sem variáveis globais.
    Encontra todas as rotas de um nó até os nós folha (len == 0).
    '''
    current_path.append(node)
    if len(graph.get(node, [])) == 0:
        routes.append(current_path.copy())
    else:
        for next_node in graph[node]:
            if next_node not in current_path:
                backtrack(graph, next_node, current_path, routes)
    current_path.pop()

def generate_routes(pipeline: dict, strategy=None, strategy_args=None) -> dict:
    '''
    Gera todas as rotas possíveis a partir dos CDs encontrados no pipeline.
    
    Argumentos:
        pipeline (dict): Grafo do porto.
        strategy (callable): Função responsável por extrair as rotas de um nó.
                             Se None, utiliza o backtrack padrão.
        strategy_args (dict): Dicionário de argumentos nomeados (kwargs) 
                              para repassar à estratégia.
    Retornos:
        dict: Dicionário onde a chave é um int (0 a N) e o valor é a rota (list).
    '''
    if strategy_args is None:
        strategy_args = {}

    # Wrapper padrão para o backtrack (já que ele muta listas em vez de retornar)
    if strategy is None:
        def default_backtrack_strategy(graph, start_node, **kwargs):
            routes_found = []
            backtrack(graph, start_node, [], routes_found)
            return routes_found
        strategy = default_backtrack_strategy

    cds = get_car_dumpers(pipeline)
    all_routes_list = []

    for cd in cds:
        # A estratégia chamada deve sempre retornar uma lista de rotas
        cd_routes = strategy(pipeline, cd, **strategy_args)
        all_routes_list.extend(cd_routes)

    # Constrói o dicionário indexado por inteiros
    return {i: all_routes_list[i] for i in range(len(all_routes_list))}

import math
from scipy.stats import qmc

import math
import numpy as np
from scipy.stats import qmc

def sobol_prune_routes(
    routes_dict: dict, 
    target_n: int, 
    stacks: list, 
    sampler: qmc.Sobol = None, # type: ignore
    rng = None
) -> dict:
    """
    Reduz a dimensionalidade do dicionário de rotas usando a sequência de Sobol.
    Garante que toda pilha do porto seja atendida por ao menos uma rota,
    escolhida de forma estocástica entre as elegíveis.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    total_routes = len(routes_dict)
    
    # 1. Trava de segurança: target_n não pode ser menor que o número de pilhas
    target_n = max(target_n, len(stacks))
    
    if target_n >= total_routes:
        return routes_dict.copy()
        
    if sampler is None:
        sampler = qmc.Sobol(d=1, scramble=True)
        
    original_keys = list(routes_dict.keys())
    selected_indices = []
    seen = set()
    
    # 2. Mapeamento Reverso: Quais índices de rota chegam em cada pilha?
    stack_to_route_indices = {s: [] for s in stacks}
    
    for idx, k in enumerate(original_keys):
        route = routes_dict[k]
        dest_stack = route[-1] # Assume que o último nó da rota é a pilha
        
        if dest_stack in stack_to_route_indices:
            stack_to_route_indices[dest_stack].append(idx)
            
    # 3. Pré-alocação Garantida Estocástica: Separa 1 rota aleatória para cada pilha
    for s, indices in stack_to_route_indices.items():
        if not indices:
            raise ValueError(f"Falha na Topologia: Nenhuma rota alcança a pilha {s}.")
        
        # O PULO DO GATO ESTÁ AQUI: Sorteia uniformemente uma rota em vez de pegar a [0]
        chosen_idx = int(rng.choice(indices))
        
        if chosen_idx not in seen:
            seen.add(chosen_idx)
            selected_indices.append(chosen_idx)
            
    # 4. Preenchimento Sobol: Completa as vagas restantes até o target_n
    m = math.ceil(math.log2(target_n))
    
    while len(selected_indices) < target_n:
        sobol_points = sampler.random_base2(m=m)
        
        for p in sobol_points:
            idx = int(p[0] * total_routes)
            
            if idx not in seen:
                seen.add(idx)
                selected_indices.append(idx)
                
            if len(selected_indices) == target_n:
                break
                
    # 5. Reconstrói o dicionário podado (ordenando os índices para consistência)
    pruned_routes = {
        i: routes_dict[original_keys[idx]] 
        for i, idx in enumerate(sorted(selected_indices))
    }
    
    return pruned_routes
