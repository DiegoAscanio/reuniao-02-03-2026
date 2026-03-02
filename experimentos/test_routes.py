import pdb
import pytest
import numpy as np
import json

from scipy.stats import qmc

from routes import backtrack, get_car_dumpers, generate_routes

# Importando dos seus respectivos estágios
from topology import generate_topology
from pipeline import build_pipeline_graph
from routes import (
    generate_routes,
    sobol_prune_routes
)

def test_sobol_prune_routes_reproducibility():
    mock_large_routes = {i: [f"Node_{i}"] for i in range(111)}

    # O controle de reprodutibilidade fica estritamente no ambiente de teste
    test_sampler = qmc.Sobol(d=1, scramble=True, seed=42)

    pruned_dict = sobol_prune_routes(mock_large_routes, stacks=['Node_0'], target_n=30, sampler=test_sampler)

    assert len(pruned_dict) == 30
    # Como a seed está fixa, os índices escolhidos serão SEMPRE os mesmos
    # assert list(pruned_dict.keys()) == [...]

def test_sobol_prune_routes():
    # Simulando as 111 rotas do seu pipeline_01
    mock_large_routes_dict = {i: [f"MockNode_{i}"] for i in range(111)}

    target_size = 30 # Um número dentro da janela de 7 a 54 da Tabela 5

    pruned_dict = sobol_prune_routes(mock_large_routes_dict, stacks = ['MockNode_1'], target_n=target_size)

    # Validação 1: O tamanho foi reduzido perfeitamente?
    assert len(pruned_dict) == target_size

    # Validação 2: A reindexação está correta? (Chaves de 0 a 29)
    assert list(pruned_dict.keys()) == list(range(target_size))

    # Validação 3: Resiliência contra target maior que o espaço amostral
    oversized_prune = sobol_prune_routes(mock_large_routes_dict, stacks = ['MockNode_1'], target_n=200)
    assert len(oversized_prune) == 111

# ==========================================
# CONSTANTES DE VALIDAÇÃO (GOLDEN MASTER)
# ==========================================

with open('./.references/topology.json', 'r') as f:
    EXPECTED_TOPO = json.load(f)

with open('./.references/pipeline.json', 'r') as f:
    EXPECTED_PIPELINE_GRAPH = json.load(f)

with open('./.references/routes.json', 'r') as f:
    EXPECTED_ROUTES_DICT = json.load(f)
EXPECTED_ROUTES_DICT = {int(k): v for k, v in EXPECTED_ROUTES_DICT.items()}

# ==========================================
# TESTE INTEGRADO (GOLDEN MASTER)
# ==========================================
def test_full_pipeline_golden_master():
    """
    Garante que os estágios 1, 2 e 3 gerem rigorosamente as mesmas estruturas
    de dados estabelecidas pelo Golden Master com a Seed 42.
    """
    # 1. Arrange
    rng_topo = np.random.default_rng(seed=42)
    rng_pipe = np.random.default_rng(seed=42)

    # 2. Act
    topo = generate_topology(rng=rng_topo)
    pipeline_graph = build_pipeline_graph(topo, rng=rng_pipe)
    routes_dict = generate_routes(pipeline_graph)

    # 3. Assert Nível 1: Topologia
    assert topo == EXPECTED_TOPO, "A topologia gerada divergiu do Golden Master."

    # 4. Assert Nível 2: Pipeline
    assert pipeline_graph == EXPECTED_PIPELINE_GRAPH, "O pipeline gerado divergiu do Golden Master."

    # 5. Assert Nível 3: Rotas
    assert routes_dict == EXPECTED_ROUTES_DICT, "As rotas extraídas divergiram do Golden Master."

def test_full_pipeline_integration_seed_42():
    # ==========================================
    # 1. Arrange: Fixando a Semente
    # ==========================================
    rng = np.random.default_rng(seed=42)
    
    # ==========================================
    # 2. Act: O Fluxo de Dados (Estágios 1 a 3)
    # ==========================================
    
    # Estágio 1: Nasce a Topologia
    topo = generate_topology(rng=rng)
    
    # Estágio 2: Nasce o Grafo (passamos o rng para as probabilidades de conexões laterais)
    pipeline_graph = build_pipeline_graph(topo, rng=rng)
    
    # Estágio 3: Nascem as Rotas
    routes_dict = generate_routes(pipeline_graph)
    
    # ==========================================
    # 3. Assert: Validação das Invariantes
    # ==========================================
    
    assert isinstance(routes_dict, dict)
    assert len(routes_dict) > 0, "O pipeline não conseguiu gerar nenhuma rota válida!"
    
    # Vamos validar a física de todas as rotas geradas
    all_routes = list(routes_dict.values())
    
    for route in all_routes:
        # Regra 1: Toda rota DEVE começar em um Car Dumper
        assert route[0].startswith("CD"), f"Rota inválida detectada (não começa com CD): {route}"
        
        # Regra 2: Toda rota DEVE terminar em uma Pilha (S)
        assert route[-1].startswith("S"), f"Rota incompleta detectada (não chega na pilha): {route}"
        
        # Regra 3: Comprimento mínimo (No mínimo: CD -> R -> S, embora seu modelo tenha CBs no meio)
        assert len(route) >= 3, f"Rota curta demais: {route}"
        
        # Regra 4: A Prova de Fogo contra Loops (Nenhum nó pode aparecer duas vezes na mesma rota)
        assert len(route) == len(set(route)), f"Loop infinito detectado na rota: {route}"

    # Print para você ver o tamanho do monstro que a seed 42 criou
    print(f"\n[SUCESSO] Integração perfeita! A Seed 42 gerou um total de {len(all_routes)} rotas únicas.")


# ==========================================
# Fixtures (Massa de Dados)
# ==========================================
@pytest.fixture
def simple_graph():
    # Grafo do seu docstring
    return {
        'A': ['B', 'C'],
        'B': ['D'],
        'C': ['D'],
        'D': []
    }

@pytest.fixture
def mock_pipeline():
    # Pipeline mínimo para testar extração e rotas
    return {
        'CD1': ['CB1'],
        'CD2': ['CB1'],
        'CB1': ['S1', 'S2'],
        'S1': [],
        'S2': []
    }

# ==========================================
# Teste 1: Backtrack
# ==========================================
def test_backtrack(simple_graph):
    routes = []
    backtrack(simple_graph, 'A', [], routes)
    
    assert len(routes) == 2
    assert ['A', 'B', 'D'] in routes
    assert ['A', 'C', 'D'] in routes

# ==========================================
# Teste 2: Filtro de Car Dumpers
# ==========================================
def test_get_car_dumpers(mock_pipeline):
    cds = get_car_dumpers(mock_pipeline)
    
    assert len(cds) == 2
    assert 'CD1' in cds
    assert 'CD2' in cds
    assert 'CB1' not in cds

# ==========================================
# Teste 3: Geração de Rotas (Com Estratégias)
# ==========================================
def test_generate_routes_default_strategy(mock_pipeline):
    # Executa com a estratégia padrão (Backtrack)
    routes_dict = generate_routes(mock_pipeline)
    
    # CD1 e CD2 podem ir para S1 ou S2 (4 rotas no total)
    assert len(routes_dict) == 4
    
    # Verifica a estrutura de dicionário indexado por inteiros
    assert list(routes_dict.keys()) == [0, 1, 2, 3]
    
    # Verifica se as rotas corretas estão nos valores
    routes_list = list(routes_dict.values())
    assert ['CD1', 'CB1', 'S1'] in routes_list
    assert ['CD2', 'CB1', 'S2'] in routes_list

def test_generate_routes_custom_strategy(mock_pipeline):
    # Função "Mock" fingindo ser a sua futura amostragem Sobol
    def fake_sobol_strategy(graph, start_node, limit=1, prefix=""):
        # Retorna apenas uma rota fictícia baseada nos argumentos
        return [[start_node, f"{prefix}_mock_end"]] * limit

    args = {"limit": 2, "prefix": "sobol"}
    
    # Passando o callable e os args
    routes_dict = generate_routes(
        mock_pipeline, 
        strategy=fake_sobol_strategy, 
        strategy_args=args
    )
    
    # Como temos 2 CDs e o limit é 2, devemos ter 4 rotas fictícias
    assert len(routes_dict) == 4
    assert routes_dict[0] == ['CD1', 'sobol_mock_end']
    assert routes_dict[3] == ['CD2', 'sobol_mock_end']
