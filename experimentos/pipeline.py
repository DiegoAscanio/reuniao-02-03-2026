import numpy as np

def balanced_bottleneck(current_layer: dict, next_layer: dict) -> dict:
    '''
    Distribui as máquinas da camada atual entre as máquinas da próxima camada
    de forma igualitária (n para 1), evitando sobrecarga em uma única esteira.
    Ideal para o início do pipeline (CDs -> Stage 1).
    '''
    curr_machines = current_layer['machines']
    next_machines = next_layer['machines']
    num_next = len(next_machines)
    
    connections = {}
    
    # Determina o tamanho do grupo (chunk)
    # Ex: 7 máquinas / 3 destinos = 2.33...
    for i, m_curr in enumerate(curr_machines):
        # A mágica está aqui: o índice do destino é a divisão inteira 
        # do índice atual pelo 'fator de agrupamento'
        # Ou mais simples: i % num_next (round-robin) ou i // (total/next)
        
        # Usaremos a lógica de fatiamento proporcional:
        target_idx = min(i // (max(1, len(curr_machines) // num_next)), num_next - 1)
        connections[m_curr] = [next_machines[target_idx]]
        
    return connections

def proportional_rescue(connections: dict, current_layer: dict, next_layer: dict) -> dict:
    '''
    Distribui as máquinas da próxima camada em blocos contíguos para as máquinas
    da camada atual, respeitando a proximidade física e evitando cruzamentos.
    '''
    curr_machines = current_layer['machines']
    next_machines = next_layer['machines']

    num_curr = len(curr_machines)
    num_next = len(next_machines)

    # Limpa as conexões iniciais do bottleneck para refazer de forma proporcional
    for m in curr_machines:
        connections[m] = []

    # Cálculo das fatias (mesma lógica das pilhas)
    chunk_size = num_next // num_curr
    remainder = num_next % num_curr

    cursor = 0
    for i, m_curr in enumerate(curr_machines):
        # Distribui o resto entre as primeiras máquinas
        tamanho_fatia = chunk_size + (1 if i < remainder else 0)
        fatia = next_machines[cursor : cursor + tamanho_fatia]

        for m_next in fatia:
            if m_next not in connections[m_curr]:
                connections[m_curr].append(m_next)

        cursor += tamanho_fatia

    return connections

def solve_bottleneck(current_layer: dict, next_layer: dict) -> dict:
    '''
    Gera as conexões base de avanço entre duas camadas adjacentes, 
    aplicando a regra de afunilamento (Teto de Índice) quando a próxima 
    camada possui menos máquinas que a camada atual.

    Argumentos:
        current_layer: Dicionário contendo os dados da camada atual.
        next_layer: Dicionário contendo os dados da próxima camada.

    Retorna:
        Um dicionário mapeando cada máquina da camada atual para uma 
        lista contendo sua máquina de destino na próxima camada.
    '''
    connections = {}
    curr_machines = current_layer['machines']
    next_machines = next_layer['machines']
    
    for i, m_curr in enumerate(curr_machines):
        # Aplica o Teto de Índice para evitar IndexError e forçar o afunilamento
        target_idx = min(i, len(next_machines) - 1)
        
        # Cria a chave e adiciona o destino numa lista (formato de grafo)
        connections[m_curr] = [next_machines[target_idx]]
        
    return connections

def ensure_rescue(connections: dict, current_layer: dict, next_layer: dict) -> dict:
    '''
    Verifica se a próxima camada é maior que a atual (expansão).
    Se for, a última máquina da camada atual "adota" (conecta-se a)
    todas as máquinas órfãs da próxima camada.

    Argumentos:
        connections: O dicionário de conexões prévio (ex: gerado por solve_bottleneck).
        current_layer: Dicionário da camada de origem.
        next_layer: Dicionário da camada de destino.

    Retorna:
        O dicionário de conexões atualizado com os resgates aplicados.
    '''
    curr_machines = current_layer['machines']
    next_machines = next_layer['machines']

    # Verifica se há expansão (próxima camada é maior que a atual)
    if len(next_machines) > len(curr_machines):
        # A última máquina da camada atual é a responsável pelo leque
        last_curr_machine = curr_machines[-1]

        # Pega todas as máquinas da próxima camada que ficaram de fora da regra do "Teto"
        # Isso é feito fatiando a lista a partir do tamanho da camada atual
        orphans = next_machines[len(curr_machines):]

        # Adiciona as órfãs na lista de conexões da última máquina
        connections[last_curr_machine].extend(orphans)

    return connections

def apply_horizontal_connections(connections: dict, current_layer: dict, rng = None) -> dict:
    '''
    Itera sobre as máquinas da camada atual e, com base em uma probabilidade,
    cria conexões bidirecionais (horizontais) entre máquinas adjacentes (i e i+1).
    Garante que não haja arestas duplicadas.
    '''
    if rng is None:
        rng = np.random.default_rng()

    machines = current_layer['machines']
    prob = current_layer.get('same_layer_interconnection_prob', 0.0)

    if prob > 0.0:
        for i in range(len(machines) - 1):
            if rng.random() < prob:
                m1 = machines[i]
                m2 = machines[i + 1]

                # Inicializa as listas caso a máquina ainda não esteja no dicionário
                if m1 not in connections:
                    connections[m1] = []
                if m2 not in connections:
                    connections[m2] = []

                # Estratégia 2: Append seguro (Evita duplicatas mantendo a lista nativa)
                if m2 not in connections[m1]:
                    connections[m1].append(m2)
                if m1 not in connections[m2]:
                    connections[m2].append(m1)

    return connections

def apply_diagonal_connections(connections: dict, current_layer: dict, next_layer: dict, rng=None) -> dict:
    '''
    Avalia a probabilidade de uma máquina i da camada atual se conectar
    diagonalmente à máquina i+1 da próxima camada.

    Argumentos:
        connections: O dicionário de conexões atual.
        current_layer: Dicionário da camada de origem.
        next_layer: Dicionário da camada de destino.
        rng: Instância opcional do gerador numpy.

    Retorna:
        O dicionário de conexões atualizado.
    '''
    if rng is None:
        rng = np.random.default_rng()

    prob = current_layer.get('next_layer_connection_to_next_index_prob', 0.0)

    if prob > 0.0:
        curr_machines = current_layer['machines']
        next_machines = next_layer['machines']

        for i, m_curr in enumerate(curr_machines):
            # O índice de avanço padrão (respeitando o teto de afunilamento)
            base_idx = min(i, len(next_machines) - 1)

            # O alvo diagonal é sempre a máquina imediatamente à direita do índice base
            diagonal_idx = base_idx + 1

            # Só rola o dado se a máquina diagonal realmente existir na próxima camada
            if diagonal_idx < len(next_machines):
                if rng.random() < prob:
                    m_target = next_machines[diagonal_idx]

                    # Inicialização por segurança
                    if m_curr not in connections:
                        connections[m_curr] = []

                    # Append seguro para evitar arestas duplicadas
                    if m_target not in connections[m_curr]:
                        connections[m_curr].append(m_target)

    return connections

def apply_destination_coverage(pipeline: dict, last_layer: dict, destination: dict, rng=None) -> dict:
    if rng is None:
        rng = np.random.default_rng()

    machines = last_layer['machines']
    stacks = destination['stacks']
    prob_overlap = destination.get('adjacent_machine_overlap_prob', 0.0)

    num_m = len(machines)
    num_s = len(stacks)

    # Dicionário auxiliar para guardar as pilhas "nativas" de cada máquina
    native_assignments = {m: [] for m in machines}

    # BIFURCAÇÃO DA REGRA DE DISTRIBUIÇÃO
    if num_s >= num_m:
        # CENÁRIO 1: Expansão (Mais pilhas do que máquinas)
        # Fatiamos as pilhas e entregamos blocos para as máquinas
        chunk_size = num_s // num_m
        remainder = num_s % num_m

        cursor = 0
        for i, m in enumerate(machines):
            tamanho_fatia = chunk_size + (1 if i < remainder else 0)
            fatia = stacks[cursor : cursor + tamanho_fatia]

            for s in fatia:
                if s not in pipeline[m]:
                    pipeline[m].append(s)
                native_assignments[m].append(s)
            cursor += tamanho_fatia
    else:
        # CENÁRIO 2: Afunilamento (Mais máquinas do que pilhas)
        # Sorte do seu bug! Aqui distribuímos as máquinas proporcionalmente
        # para garantir que cada máquina alcance exata 1 pilha "nativa".
        for i, m in enumerate(machines):
            # Mapeamento proporcional: ex: 3 máq e 2 pilhas -> i*2//3
            # M0 -> S0, M1 -> S0, M2 -> S1
            stack_idx = min(i * num_s // num_m, num_s - 1)
            s = stacks[stack_idx]

            if s not in pipeline[m]:
                pipeline[m].append(s)
            native_assignments[m].append(s)

    # Aplica o overlap usando as atribuições nativas como referência
    if prob_overlap > 0.0:
        for i in range(num_m - 1):
            m_curr = machines[i]
            m_next = machines[i+1]

            # Proteção: Garante que as listas tenham itens antes de tentar pegar [0] ou [-1]
            if native_assignments[m_next] and native_assignments[m_curr]:

                # Máquina i tenta alcançar a primeira pilha nativa da i+1
                if rng.random() < prob_overlap:
                    target = native_assignments[m_next][0]
                    if target not in pipeline[m_curr]:
                        pipeline[m_curr].append(target)

                # Máquina i+1 tenta alcançar a última pilha nativa da i
                if rng.random() < prob_overlap:
                    target = native_assignments[m_curr][-1]
                    if target not in pipeline[m_next]:
                        pipeline[m_next].append(target)

    return pipeline

def build_pipeline_graph(topology: dict, rng=None) -> dict:
    '''
    Orquestra a construção do pipeline completo integrando as funções
    de avanço, resgate, conexões horizontais e destino.
    '''
    if rng is None:
        rng = np.random.default_rng()

    pipeline = {}
    layers = topology['layers']
    dest = topology['destination']

    # Ordena as chaves das camadas numericamente
    layer_keys = sorted(layers.keys(), key=int)
    num_layers = len(layer_keys)

    # 1. Inicializa todas as máquinas e pilhas no dicionário
    for k in layer_keys:
        for m in layers[k]['machines']:
            pipeline[m] = []
    for s in dest['stacks']:
        pipeline[s] = []

    # INICIO DO PIPELINE
    # Passo 1: Conexão da Primeira Camada (CDs) para a Segunda Camada (Stage 1)
    current_layer = layers[layer_keys[0]]
    next_layer = layers[layer_keys[1]]
    # A. avanço balanceado (n para 1) para evitar sobrecarga na primeira etapa
    connections = balanced_bottleneck(current_layer, next_layer)
    # B. resgate de máquinas órfãs (expansão) se a próxima camada for maior que a atual
    connections = ensure_rescue(connections, current_layer, next_layer)
    # C. conexões horizontais (mesma camada) para a primeira camada, se aplicável
    connections = apply_horizontal_connections(connections, current_layer, rng)
    # D. conexões diagonais (se existir o parâmetro na camada)
    if 'next_layer_connection_to_next_index_prob' in current_layer:
        connections = apply_diagonal_connections(connections, current_layer, next_layer, rng)
    # Integra as conexões da primeira camada ao pipeline geral
    pipeline |= connections

    # MIOLO DO PIPELINE
    # Passo 2. Iterar entre as camadas de conveyors (1 até a antepenúltima)
    for i in range(1, num_layers - 2):
        curr_layer = layers[layer_keys[i]]
        next_layer = layers[layer_keys[i+1]]

        # Passo A: Avanço Base e Afunilamento
        connections = solve_bottleneck(curr_layer, next_layer)

        # Passo B: Resgate de Máquinas Órfãs (Expansão)
        connections = ensure_rescue(connections, curr_layer, next_layer)

        # Passo C: Conexões Horizontais (Mesma camada)
        connections = apply_horizontal_connections(connections, curr_layer, rng)

        # Passo D: Conexões Diagonais (Se existir o parâmetro na camada)
        if 'next_layer_connection_to_next_index_prob' in curr_layer:
             connections = apply_diagonal_connections(connections, curr_layer, next_layer, rng)
        # Integra as conexões da camada atual ao pipeline geral
        pipeline |= connections

    # FIM DO PIPELINE
    # Passo 3. Conexão da Última camada de conveyors (penúltima de máquinas)
    #          para a camada de stackers (última)
    penultimate_m_layer_idx = num_layers - 2
    last_m_layer_idx = num_layers - 1
    curr_layer = layers[
        layer_keys[penultimate_m_layer_idx]
    ]
    next_layer = layers[
        layer_keys[last_m_layer_idx]
    ]
    # A. avanço base e afunilamento
    connections = solve_bottleneck(curr_layer, next_layer)
    # B. resgate de máquinas órfãs (expansão) se a próxima camada for maior
    #    que a atual com balanceamento proporcional para evitar sobrecarga
    #    em uma única esteira
    connections = proportional_rescue(connections, curr_layer, next_layer)
    # C. conexões horizontais (mesma camada) para a última camada de conveyors, se aplicável
    connections = apply_horizontal_connections(connections, curr_layer, rng)
    # D. conexões diagonais (se existir o parâmetro na camada)
    if 'next_layer_connection_to_next_index_prob' in curr_layer:
        connections = apply_diagonal_connections(connections, curr_layer, next_layer, rng)
    # Integra as conexões da última camada de conveyors ao pipeline geral
    pipeline |= connections

    # Passo 4. Conexão com a Última Camada (Destino/Pilhas)
    last_layer = layers[layer_keys[-1]]
    pipeline = apply_destination_coverage(pipeline, last_layer, dest, rng)

    # Passo 5. Ordenar as listas de conexões para consistência (opcional, mas útil para testes)
    for key in pipeline:
        pipeline[key] = sorted(pipeline[key])

    return pipeline
