import pdb
import pytest
import numpy as np

from pipeline import (
    solve_bottleneck,
    ensure_rescue,
    apply_horizontal_connections,
    apply_diagonal_connections,
    apply_destination_coverage,
    build_pipeline_graph,
    balanced_bottleneck,
    proportional_rescue
)

def test_proportional_rescue_spatial():
    # Arrange: 2 CBs -> 5 Rs
    curr_layer = {"machines": ["CB1", "CB2"]}
    next_layer = {"machines": ["R1", "R2", "R3", "R4", "R5"]}

    # connections iniciais (1-para-1)
    connections = {"CB1": ["R1"], "CB2": ["R2"]}

    # Act
    # A função deve dividir as 5 máquinas entre as 2 CBs de forma contígua
    resultado = proportional_rescue(connections, curr_layer, next_layer)

    # Assert:
    # CB1 deve pegar o bloco inicial: R1, R2, R3
    # CB2 deve pegar o bloco final: R4, R5
    assert "R3" in resultado["CB1"]
    assert "R4" in resultado["CB2"]
    assert "R5" in resultado["CB2"]

    # E o mais importante: R5 NÃO pode estar na CB1 (evita o espaguete)
    assert "R5" not in resultado["CB1"]

def test_balanced_bottleneck_distribution():
    # Arrange: 7 CDs -> 3 CBs
    curr_layer = {"machines": ["CD1", "CD2", "CD3", "CD4", "CD5", "CD6", "CD7"]}
    next_layer = {"machines": ["CB1", "CB2", "CB3"]}

    # Act
    connections = balanced_bottleneck(curr_layer, next_layer)

    # Assert:
    # CD1, CD2 -> CB1
    # CD3, CD4 -> CB2
    # CD5, CD6, CD7 -> CB3 (ou similar)
    assert connections["CD1"] == ["CB1"]
    assert connections["CD3"] == ["CB2"]
    assert connections["CD7"] == ["CB3"]

    # Verifica se todas as CBs estão sendo alimentadas (anti-gargalo)
    all_targets = [dest[0] for dest in connections.values()]
    assert all_targets.count("CB1") >= 2
    assert all_targets.count("CB2") >= 2
    assert all_targets.count("CB3") >= 2

def test_apply_destination_coverage_with_overlap():
    # Arrange
    last_layer = {"machines": ["R1", "R2", "R3"]}
    destination = {
        "stacks": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "adjacent_machine_overlap_prob": 0.5
    }

    # Inicializamos o pipeline com as chaves necessárias
    pipeline = {m: [] for m in last_layer["machines"]}
    for s in destination["stacks"]:
        pipeline[s] = []

    # Seed 42 para previsibilidade
    # 1º sorteio (R1 p/ S3): 0.77 > 0.5 (FALHA)
    # 2º sorteio (R2 p/ S2): 0.43 < 0.5 (SUCESSO!)
    # 3º sorteio (R2 p/ S5): 0.85 > 0.5 (FALHA)
    # 4º sorteio (R3 p/ S4): 0.69 > 0.5 (FALHA)
    rng = np.random.default_rng(seed=42)

    # Act
    resultado = apply_destination_coverage(pipeline, last_layer, destination, rng)

    # Assert
    # R1 deve ter apenas suas originais
    assert resultado["R1"] == ["S1", "S2"]

    # R2 deve ter suas originais (S3, S4) + a invasão em S2 (da R1)
    assert "S2" in resultado["R2"]
    assert "S3" in resultado["R2"]
    assert "S4" in resultado["R2"]

    # R3 deve ter apenas suas originais
    assert resultado["R3"] == ["S5", "S6"]

def test_build_pipeline_graph_integration():
    # Arrange
    topology = {
        "layers": {
            "0": {"machines": ["CD1"], "same_layer_interconnection_prob": 0.0},
            "1": {"machines": ["CB1", "CB2"], "same_layer_interconnection_prob": 0.0}
        },
        "destination": {
            "stacks": ["S1", "S2"],
            "adjacent_machine_overlap_prob": 0.0
        }
    }
    rng = np.random.default_rng(seed=42)

    # Act
    pipeline = build_pipeline_graph(topology, rng)

    # Assert
    # 1. CD1 deve conectar em CB1 e resgatar CB2 (Expansão)
    assert "CB1" in pipeline["CD1"]
    assert "CB2" in pipeline["CD1"]

    # 2. CB1 e CB2 devem conectar nas pilhas (Destino)
    assert "S1" in pipeline["CB1"]
    assert "S2" in pipeline["CB2"]

    # 3. As pilhas devem ser chaves terminais []
    assert pipeline["S1"] == []
    assert pipeline["S2"] == []

def test_apply_destination_coverage_equality():
    # 4 máquinas para 8 pilhas -> Exatamente 2 pilhas por máquina
    last_layer = {"machines": ["R1", "R2", "R3", "R4"]}
    destination = {
        "stacks": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
        "adjacent_machine_overlap_prob": 0.0 # Zero para testar apenas a divisão base
    }

    conexoes = {m: [] for m in last_layer["machines"]}
    for s in destination["stacks"]: conexoes[s] = []

    resultado = apply_destination_coverage(conexoes, last_layer, destination)

    # Verificação de igualdade (Anti-Gargalo)
    assert len(resultado["R1"]) == 2
    assert len(resultado["R4"]) == 2
    assert resultado["R1"] == ["S1", "S2"]
    assert resultado["R4"] == ["S7", "S8"]

def test_apply_diagonal_connections():
    # ==========================================
    # 1. Preparação (Arrange)
    # ==========================================
    layer_3 = {
        "name": "Camada Origem",
        "machines": ["A1", "A2", "A3"],
        "next_layer_connection_to_next_index_prob": 0.5
    }

    layer_4 = {
        "name": "Camada Destino",
        "machines": ["B1", "B2", "B3", "B4"]
    }

    # Simulamos o estado do dicionário após o solve_bottleneck e ensure_rescue
    # A1 -> B1
    # A2 -> B2
    # A3 -> B3 (teto) + B4 (resgate de órfã)
    conexoes_iniciais = {
        "A1": ["B1"],
        "A2": ["B2"],
        "A3": ["B3", "B4"]
    }

    rng = np.random.default_rng(seed=42)

    # ==========================================
    # 2. Execução (Act)
    # ==========================================
    resultado = apply_diagonal_connections(conexoes_iniciais, layer_3, layer_4, rng)

    # ==========================================
    # 3. Verificação (Assert)
    # ==========================================
    # Como funciona a lógica do i -> i+1:
    # A1 (idx 0) tenta B2 (idx 1). Sorteio: 0.77 > 0.5 (Falha)
    # A2 (idx 1) tenta B3 (idx 2). Sorteio: 0.43 < 0.5 (Passa!) -> Injeta B3
    # A3 (idx 2) tenta B4 (idx 3). Sorteio: 0.85 > 0.5 (Falha)

    expected = {
        "A1": ["B1"],
        "A2": ["B2", "B3"], # <-- A conexão diagonal estocástica apareceu aqui!
        "A3": ["B3", "B4"]
    }

    assert resultado == expected, f"Falha na conexão diagonal. Retornou: {resultado}"

def test_apply_horizontal_connections():
    # ==========================================
    # 1. Preparação (Arrange)
    # ==========================================
    layer_2 = {
        "name": "Conveyors - Stage 2",
        "machines": ["CB3", "CB4", "CB5"],
        "same_layer_interconnection_prob": 0.5
    }

    # Simulamos o dicionário de conexões já com as chaves inicializadas
    # (podem estar vazias ou já com conexões para a próxima camada, não importa)
    conexoes_iniciais = {
        "CB3": [],
        "CB4": [],
        "CB5": []
    }

    # Travamos o gerador estocástico para garantir previsibilidade
    rng = np.random.default_rng(seed=42)

    # ==========================================
    # 2. Execução (Act)
    # ==========================================
    resultado = apply_horizontal_connections(conexoes_iniciais, layer_2, rng)

    # ==========================================
    # 3. Verificação (Assert)
    # ==========================================
    # Com a seed 42:
    # CB3 <-> CB4: Sorteio 0.77 (Falha, pois 0.77 > 0.5)
    # CB4 <-> CB5: Sorteio 0.43 (Passa, pois 0.43 < 0.5)
    expected = {
        "CB3": [],
        "CB4": ["CB5"],
        "CB5": ["CB4"]
    }

    assert resultado == expected, f"Falha na conexão horizontal. Retornou: {resultado}"

def test_ensure_rescue_no_action_needed():
    # ==========================================
    # CASO 1: Afunilamento (Não deve alterar nada)
    # ==========================================
    layer_0 = {"machines": ["CD1", "CD2", "CD3"]}
    layer_1 = {"machines": ["CB1", "CB2"]}
    
    # Simulamos o que a função anterior gerou
    conexoes_iniciais = solve_bottleneck(layer_0, layer_1)
    
    # Executamos o resgate
    resultado = ensure_rescue(conexoes_iniciais, layer_0, layer_1)
    
    # A expectativa é que o CD3 continue morrendo no CB2, sem adicionar nada novo
    expected = {
        "CD1": ["CB1"],
        "CD2": ["CB2"],
        "CD3": ["CB2"]
    }
    
    assert resultado == expected, "Falha no Caso 1: O resgate alterou conexões onde não havia órfãs."


def test_ensure_rescue_fan_out():
    # ==========================================
    # CASO 2: Expansão (Resgate Obrigatório)
    # ==========================================
    layer_1 = {"machines": ["CB1", "CB2"]}
    layer_2 = {"machines": ["CB3", "CB4", "CB5"]}
    
    # Simulamos a passagem pela função anterior
    # CB1 vai para CB3, CB2 vai para CB4. O CB5 ficou órfão.
    conexoes_iniciais = solve_bottleneck(layer_1, layer_2)
    
    # Executamos o resgate
    resultado = ensure_rescue(conexoes_iniciais, layer_1, layer_2)
    
    # A expectativa é que a ÚLTIMA máquina da camada atual (CB2) abra um leque
    # para abraçar todas as órfãs da próxima camada (CB5).
    expected = {
        "CB1": ["CB3"],
        "CB2": ["CB4", "CB5"]
    }
    
    assert resultado == expected, f"Falha no Caso 2: Resgate não ocorreu. Retornou: {resultado}"

def test_solve_bottleneck_funneling():
    # ==========================================
    # 1. Preparação (Arrange): As Camadas
    # ==========================================
    layer_0 = {
        "name": "Car Dumpers",
        "machines": ["CD1", "CD2", "CD3"]
    }
    
    layer_1 = {
        "name": "Conveyors - Stage 1",
        "machines": ["CB1", "CB2"]
    }

    # ==========================================
    # 2. Execução (Act)
    # ==========================================
    resultado = solve_bottleneck(current_layer=layer_0, next_layer=layer_1)

    # ==========================================
    # 3. Verificação (Assert)
    # ==========================================
    # Esperado: 
    # CD1 (0) -> CB1 (0)
    # CD2 (1) -> CB2 (1)
    # CD3 (2) -> CB2 (1) - O afunilamento acontece aqui!
    expected = {
        "CD1": ["CB1"],
        "CD2": ["CB2"],
        "CD3": ["CB2"]
    }
    
    assert resultado == expected, f"Falha no afunilamento. Retornou: {resultado}"
