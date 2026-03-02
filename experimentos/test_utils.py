import pytest
import numpy as np
from utils import (
    calculate_p_bar, 
    calculate_lt, 
    compute_lambda, 
    sample_lambda_from_poisson,
    extract_product_catalog,
    filter_eligible_stockpiles,
    calculate_route_processing_rate
)

# ---------------------------------------------------------
# TESTE 7: Cálculo da Taxa de Processamento da Rota (Gargalo)
# ---------------------------------------------------------

def test_calculate_route_processing_rate_success():
    # Arrange: Dicionário de máquinas com capacidades diferentes
    mock_machines = {
        'CD1': {'processing_rate': 1/45}, # ~0.0222
        'CB1': {'processing_rate': 1/50}, # 0.0200 (GARGALO)
        'CB2': {'processing_rate': 1/40}, # 0.0250
    }
    
    # Rota válida terminando em uma pilha (S01)
    mock_route = ['CD1', 'CB1', 'CB2', 'S01']
    
    # Act
    rate = calculate_route_processing_rate(mock_route, mock_machines)
    
    # Assert: O gargalo deve ser a CB1 (1/50)
    assert rate == 1/50

def test_calculate_route_processing_rate_empty():
    # Arrange & Act & Assert
    # Uma rota vazia deve retornar 0.0 de forma segura
    assert calculate_route_processing_rate([], {}) == 0.0

def test_calculate_route_processing_rate_only_stockpile():
    # Arrange: Uma rota anômala que só tem o destino final
    mock_route = ['S01']
    
    # Act
    rate = calculate_route_processing_rate(mock_route, {})
    
    # Assert: Como a pilha é ignorada pelo [:-1], a lista de máquinas fica vazia
    assert rate == 0.0

# ---------------------------------------------------------
# TESTE 6: Filtragem de Pilhas Elegíveis para um Produto 
#          (Matching Espacial)
# ---------------------------------------------------------

def test_filter_eligible_stockpiles_success():
    # Arrange: Mock de pilhas com produtos misturados
    mock_stacks = {
        'S01': {'product': 'P1', 'capacity': 100},
        'S02': {'product': 'P2', 'capacity': 200},
        'S03': {'product': 'P1', 'capacity': 150},
        'S04': {'product': 'P3', 'capacity': 50},
    }

    # Act & Assert
    # 1. Deve encontrar as duas pilhas que armazenam P1
    eligible_p1 = filter_eligible_stockpiles(mock_stacks, 'P1')
    assert eligible_p1 == ['S01', 'S03']

    # 2. Deve encontrar apenas a pilha que armazena P2
    eligible_p2 = filter_eligible_stockpiles(mock_stacks, 'P2')
    assert eligible_p2 == ['S02']

def test_filter_eligible_stockpiles_no_match():
    # Arrange
    mock_stacks = {
        'S01': {'product': 'P1'},
        'S02': {'product': 'P2'},
    }

    # Act: Buscar um produto inexistente
    eligible = filter_eligible_stockpiles(mock_stacks, 'P99')

    # Assert
    assert eligible == []

def test_filter_eligible_stockpiles_empty_dict():
    # Deve lidar corretamente com um dicionário vazio
    assert filter_eligible_stockpiles({}, 'P1') == []

def test_filter_eligible_stockpiles_missing_keys():
    # Arrange: Dicionário onde uma pilha esqueceu de declarar o produto
    mock_stacks = {
        'S01': {'capacity': 100}, # Falta a chave 'product'
        'S02': {'product': 'P1'},
    }

    # Act
    eligible = filter_eligible_stockpiles(mock_stacks, 'P1')

    # Assert: O uso do .get() deve evitar o KeyError e pular a S01
    assert eligible == ['S02']


# ---------------------------------------------------------
# TESTE 5: Extração do Catálogo de Produtos
# ---------------------------------------------------------
def test_extract_product_catalog():
    # Arrange: Um mock do dicionário de pilhas gerado pelo stacks.py
    mock_stacks = {
        'S01': {'capacity': 100, 'product': 'P2', 'eligible_routes': ['10']},
        'S02': {'capacity': 150, 'product': 'P1', 'eligible_routes': ['20']},
        'S03': {'capacity': 120, 'product': 'P2', 'eligible_routes': ['30']},
        'S04': {'capacity': 80,  'product': 'P3', 'eligible_routes': ['40']},
    }

    # Act
    catalog = extract_product_catalog(mock_stacks)

    # Assert
    # 1. Garante que os produtos são únicos (P2 aparece duas vezes nas pilhas, mas 1 no catálogo)
    assert len(catalog) == 3

    # 2. Garante que estão presentes e ordenados (boa prática para reprodutibilidade)
    assert catalog == ['P1', 'P2', 'P3']

def test_extract_product_catalog_empty():
    assert extract_product_catalog({}) == []

# ---------------------------------------------------------
# TESTE 4: Sorteio do Poisson 100% Integrado com a Física
# ---------------------------------------------------------
def test_sample_lambda_from_poisson_integrated():
    # ==========================================
    # 1. Preparação (Arrange): O Mundo Físico
    # ==========================================
    capacidade_segundo = (4/3) / 60.0
    mock_machines = {
        'A': {'processing_rate': capacidade_segundo},
        'B': {'processing_rate': capacidade_segundo},
        'C': {'processing_rate': capacidade_segundo}
    }
    mock_routes = {
        0: ['A', 'B', 'S1'],
        1: ['B', 'C', 'S2']
    }
    mock_lot_freqs = {100: 90, 50: 19, 25: 2}
    mock_train_compositions = {1: 90, 2: 9, 3: 1}

    # ==========================================
    # 2. Execução (Act): Cadeia de Funções (Pipeline)
    # ==========================================
    p_bar_calculado = calculate_p_bar(mock_routes, mock_machines, mock_lot_freqs)
    lt_calculado = calculate_lt(mock_train_compositions)
    
    # Geramos o lambda real baseado na configuração do porto (u=0.8, cds=2)
    lambda_integrado = compute_lambda(u=0.8, cds=2, p_bar=p_bar_calculado, lt=lt_calculado)

    # ==========================================
    # 3. Execução e Verificação (Assert): Sorteio de Poisson
    # ==========================================
    seed_fixa = 42
    hz_padrao = 43200  # 12 horas
    
    # Cenário A: Teste no horizonte padrão de 12h
    rng_test_12h = np.random.default_rng(seed_fixa)
    resultado_12h = sample_lambda_from_poisson(lambda_rate=lambda_integrado, rng=rng_test_12h, hz=hz_padrao)
    
    rng_expected_12h = np.random.default_rng(seed_fixa)
    expected_12h = rng_expected_12h.poisson(lam=(lambda_integrado * hz_padrao))
    
    assert resultado_12h == expected_12h, \
        f"Falha na integração de 12h. Esperado {expected_12h}, Retornou {resultado_12h}"

    # Cenário B: Teste escalando para 24h
    hz_24h = 86400
    rng_test_24h = np.random.default_rng(seed_fixa)
    resultado_24h = sample_lambda_from_poisson(lambda_rate=lambda_integrado, rng=rng_test_24h, hz=hz_24h)
    
    rng_expected_24h = np.random.default_rng(seed_fixa)
    expected_24h = rng_expected_24h.poisson(lam=(lambda_integrado * hz_24h))
    
    assert resultado_24h == expected_24h, \
        f"Falha na integração de 24h. Esperado {expected_24h}, Retornou {resultado_24h}"

    # Cenário C: Propagação de erro/falta de rota na topologia (Lambda zero)
    lambda_zero = compute_lambda(u=0.8, cds=2, p_bar=0.0, lt=lt_calculado) # Força o erro
    resultado_zero = sample_lambda_from_poisson(lambda_zero, rng=np.random.default_rng(seed_fixa))
    assert resultado_zero == 0, "A função Poisson não protegeu contra a propagação de lambda zero"

# ---------------------------------------------------------
# ATUALIZAÇÃO DO TESTE 3: Lambda Integrado
# ---------------------------------------------------------
def test_compute_lambda_integrated_pipeline():
    # ==========================================
    # 1. Preparação (Arrange): O Mundo Físico
    # ==========================================
    mock_machines = {
        'A': {'processing_rate': 2.0 / (60)},  # 2 vagões a cada minuto 
        'B': {'processing_rate': 4.0 / (60)},  # 4 vagões a cada minuto
        'C': {'processing_rate': 1.0 / (60)}   # 1 vagão a cada minuto
    }
    mock_routes = {
        0: ['A', 'B', 'S1'],
        1: ['B', 'C', 'S2']
    }
    mock_lot_freqs = {100: 90, 50: 19, 25: 2}
    mock_train_compositions = {1: 90, 2: 9, 3: 1}

    # ==========================================
    # 2. Execução (Act): Etapas Anteriores
    # ==========================================
    # p_bar esperado dessa configuração é ~ 20270.27027
    p_bar_calculado = calculate_p_bar(mock_routes, mock_machines, mock_lot_freqs)

    # lt esperado dessa configuração é 1.11
    lt_calculado = calculate_lt(mock_train_compositions)

    # ==========================================
    # 3. Execução e Verificação: Cálculo do Lambda
    # ==========================================

    # Cenário A: Operação normal (80% de saturação, u=0.8, 2 viradores)
    # lambda = (0.8 * 2) / 22500.0 = 1.6 / 22500.0 = 0.000071111111...
    resultado_lambda_normal = compute_lambda(u=0.8, cds=2, p_bar=p_bar_calculado, lt=lt_calculado)
    expected_lambda_normal = 0.0003555555555555555
    assert resultado_lambda_normal == pytest.approx(expected_lambda_normal, rel=1e-5), \
        f"Falha no Cenário A. Retornou {resultado_lambda_normal}"

    # Cenário B: Operação sob estresse máximo (100% de saturação, u=1.0, 2 viradores)
    # lambda = (1.0 * 2) / 22500.0 = 2.0 / 22500.0 = 0.000088888888...
    resultado_lambda_estresse = compute_lambda(u=1.0, cds=2, p_bar=p_bar_calculado, lt=lt_calculado)
    expected_lambda_estresse = 0.00044444444444444436
    assert resultado_lambda_estresse == pytest.approx(expected_lambda_estresse, rel=1e-5), \
        f"Falha no Cenário B. Retornou {resultado_lambda_estresse}"

    # Cenário C: Proteção contra Divisão por Zero
    resultado_zero = compute_lambda(u=0.8, cds=2, p_bar=0.0, lt=lt_calculado)
    assert resultado_zero == 0.0, "Falha na prevenção de divisão por zero"

def test_calculate_lt():
    # 1. Preparação (Arrange)
    # Dicionário onde a Chave = Qtd de lotes no trem, Valor = Frequência (Quantidade de trens)
    mock_train_compositions = {
        1: 90, # 90 trens têm 1 lote
        2: 9,  # 9 trens têm 2 lotes
        3: 1   # 1 trem tem 3 lotes
    }

    # 2. Ação (Act)
    resultado_lt = calculate_lt(mock_train_compositions)

    # 3. Verificação (Assert)
    # Total de lotes: (1*90) + (2*9) + (3*1) = 90 + 18 + 3 = 111
    # Total de trens: 90 + 9 + 1 = 100
    # Média (lt) = 111 / 100 = 1.11
    expected_lt = 1.11

    assert resultado_lt == pytest.approx(expected_lt, rel=1e-5), \
        f"Esperado {expected_lt}, mas retornou {resultado_lt}"

# ---------------------------------------------------------
# ATUALIZAÇÃO DO TESTE 1: p_bar com as novas capacidades
# ---------------------------------------------------------
def test_calculate_p_bar_weighted_average_generalized():
    mock_machines = {
        'A': {'processing_rate': 2.0 / (5 * 60)},
        'B': {'processing_rate': 4.0 / (5 * 60)},
        'C': {'processing_rate': 1.0 / (5 * 60)}
    }
    mock_routes = {0: ['A', 'B', 'S1'], 1: ['B', 'C', 'S2']}
    mock_lot_freqs = {100: 90, 50: 19, 25: 2}

    resultado_p_bar = calculate_p_bar(mock_routes, mock_machines, mock_lot_freqs)

    # 2250000 / 111 = 20270.27027027027
    expected_p_bar = 20270.27027027027
    assert resultado_p_bar == pytest.approx(expected_p_bar, rel=1e-5), \
        f"Esperado {expected_p_bar}, mas retornou {resultado_p_bar}"

