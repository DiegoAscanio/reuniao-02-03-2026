from typing import Any
import numpy as np

def restore_numeric_keys(obj):
    """
    Percorre recursivamente um dicionário/lista carregado de um JSON
    e converte chaves de string puramente numéricas de volta para int.
    Isso é vital para solvers MIP que dependem de indexação estrita de tuplas.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Se a chave for string e contiver apenas dígitos, vira int
            new_key = int(k) if isinstance(k, str) and k.isdigit() else k
            new_dict[new_key] = restore_numeric_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [restore_numeric_keys(item) for item in obj]
    else:
        return obj

def filter_eligible_stockpiles(stacks_dict: dict, product : Any) -> list:
    '''
    Varre o dicionário de pilhas gerado e retorna uma lista de IDs das pilhas
    que estão configuradas para operar com o produto especificado.

    Argumentos:
        stacks_dict: Dicionário contendo as informações das pilhas.
        product: O produto para o qual queremos encontrar pilhas elegíveis.

    Retorna:
        Uma lista de IDs de pilhas que operam com o produto especificado.
    '''
    if not stacks_dict:
        return []

    eligible_stockpiles = [
        stack_id for stack_id, data in stacks_dict.items() 
        if data.get('product') == product
    ]

    return eligible_stockpiles

def extract_product_catalog(stacks_dict: dict) -> list:
    '''
    Varre o dicionário de pilhas gerado e extrai a lista única
    de todos os produtos que o porto está configurado para operar.
    '''
    if not stacks_dict:
        return []

    # Usa set comprehension para extrair valores únicos e depois ordena
    unique_products = {data['product'] for data in stacks_dict.values()}

    return sorted(list(unique_products))

def sample_lambda_from_poisson(lambda_rate: float, rng=None, hz: int = 43200) -> int:
    '''
    Realiza a amostragem da quantidade total de trens a chegar no porto 
    a partir de uma distribuição de Poisson.

    Argumentos:
        lambda_rate: Taxa contínua de chegada de trens (trens por segundo).
        rng: Instância de numpy.random.Generator para controle de semente. 
             Se None, instancia um novo gerador aleatório.
        hz: Horizonte de planejamento em segundos (default: 43200, equivalentes a 12h).

    Retorna:
        Um número inteiro representando a quantidade sorteada de trens para o horizonte.
    '''
    # Injeção de dependência para instanciar o gerador caso não seja fornecido
    if rng is None:
        rng = np.random.default_rng()
        
    # Esperança matemática de chegadas (trens/segundo * segundos)
    expected_arrivals = lambda_rate * hz
    
    # Proteção de contorno: a distribuição de Poisson requer que lam >= 0
    if expected_arrivals <= 0:
        return 0
        
    # Extrai uma amostra inteira da distribuição
    return int(rng.poisson(lam=expected_arrivals))

def compute_lambda(u: float, cds: int, p_bar: float, lt: float) -> float:
    '''
    Calcula a taxa de chegada (lambda) de trens por unidade de tempo 
    baseado na Teoria das Filas.

    Argumentos:
        u: Fator de utilização alvo (ex: 0.8 para 80% de saturação).
        cds: Número de servidores paralelos (viradores de vagão / car dumpers).
        p_bar: Tempo médio ponderado de processamento de um lote.
        lt: Número médio estatístico de lotes por trem.

    Retorna:
        A taxa lambda (float). Retorna 0.0 se p_bar ou lt forem nulos para evitar divisão por zero.
    '''
    # Proteção contra ZeroDivisionError caso a topologia venha vazia
    if p_bar <= 0 or lt <= 0:
        return 0.0
        
    numerador = u * cds
    denominador = p_bar * lt
    
    return float(numerador / denominador)

def calculate_lt(train_compositions: dict) -> float:
    '''
    Calcula o número médio de lotes por trem (lt) baseado nas frequências
    de fracionamento dos trens.

    Argumentos:
        train_compositions: Dict no formato {quantidade_de_lotes_no_trem: frequencia_de_trens}.
                            Ex: {1: 90, 2: 9, 3: 1} representa 90 trens com 1 lote,
                            9 trens com 2 lotes e 1 trem com 3 lotes.
    '''
    # Soma total de trens (denominador)
    total_trains = sum(train_compositions.values())
    if total_trains == 0:
        return 0.0

    # Soma total de lotes físicos gerados (numerador)
    total_lots = sum(qtd_lotes * freq for qtd_lotes, freq in train_compositions.items())

    # Média estatística (Esperança de lotes por trem)
    return float(total_lots / total_trains)

def calculate_p_bar(routes_dict: dict, machines_dict: dict, lot_sizes_and_frequencies: dict) -> float:
    '''
    Calcula o tempo médio ponderado de processamento (p_bar) em segundos,
    baseado em um dicionário de frequências de tamanhos de lotes e
    na capacidade da máquina gargalo de cada rota possível.
    
    Argumentos:
        routes_dict: Dicionário de rotas do pipeline.
        machines_dict: Dicionário contendo as capacidades das máquinas.
        lot_sizes_and_frequencies: Dict no formato {tamanho_vagões: frequencia_absoluta}.
                                   Ex: {100: 90, 50: 19, 25: 2}
    '''
    total_frequency = sum(lot_sizes_and_frequencies.values())
    if total_frequency == 0:
        return 0.0

    # Inicializa um dicionário para guardar os tempos calculados para cada tamanho de lote
    # Ex: {100: [], 50: [], 25: []}
    tempos_por_tamanho = {size: [] for size in lot_sizes_and_frequencies.keys()}

    # 1. Calcula os tempos para todos os cenários em todas as rotas
    for r_id, path in routes_dict.items():
        maquinas_rota = path[:-1] # Ignora a pilha (último elemento)
        
        if not maquinas_rota:
            continue
            
        gargalo_rota = min(machines_dict[m]['processing_rate'] for m in maquinas_rota)
        
        # Popula as listas de tempo dividindo o tamanho do lote pela vazão do gargalo
        for size in lot_sizes_and_frequencies.keys():
            tempos_por_tamanho[size].append(size / gargalo_rota)

    # Prevenção: se não houver rotas válidas computadas
    if not tempos_por_tamanho[list(lot_sizes_and_frequencies.keys())[0]]:
        return 0.0

    # 2. Consolida as médias e aplica a ponderação
    p_bar_sec = 0.0
    for size, frequency in lot_sizes_and_frequencies.items():
        # Média simples do tempo desse tamanho específico de lote em todas as rotas
        p_s = np.mean(tempos_por_tamanho[size])
        
        # Soma ponderada
        peso = frequency / total_frequency
        p_bar_sec += peso * p_s

    return float(p_bar_sec)

def calculate_route_processing_rate(route: list, machines_dict: dict) -> float:
    '''
    Calcula a taxa de processamento de uma rota específica, 
    considerando a capacidade da máquina gargalo.
    Argumentos:
        route: Lista de IDs de máquinas que compõem a rota (ex: ['M1', 'M2', 'CD1']).
        machines_dict: Dicionário contendo as capacidades das máquinas.
    Retorna:
        A taxa de processamento da rota (float), que é a capacidade do gargalo.
    '''
    maquinas_rota = route[:-1] # Ignora a pilha (último elemento)
    if not maquinas_rota:
        return 0.0
    return min(
        machines_dict[m]['processing_rate'] for m in maquinas_rota
    )
