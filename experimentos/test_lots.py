import pytest
import numpy as np
from lots import generate_train_arrivals, generate_lots

def test_generate_train_arrivals_uniformity_and_order():
    # Arrange
    rng = np.random.default_rng(seed=42)
    hz = 43200  # 12 horas em segundos
    total_trains = 5  # Valor hipotético sorteado previamente pelo utils

    # Act
    arrivals = generate_train_arrivals(total_trains, hz, rng=rng)

    # Assert
    # 1. Garante a quantidade exata
    assert len(arrivals) == 5

    # 2. Garante que todos os trens chegam DENTRO do horizonte de planejamento
    assert all(0.0 <= t <= hz for t in arrivals)

    # 3. Garante que a lista está ordenada cronologicamente
    for i in range(1, len(arrivals)):
        assert arrivals[i] >= arrivals[i-1]

def test_generate_train_arrivals_zero_trains():
    # Deve lidar graciosamente com dias em que o Poisson não sorteia nenhum trem
    assert generate_train_arrivals(0, 43200) == []


@pytest.fixture
def mock_infrastructure_for_lots():
    routes = {
        10: ['CD1', 'CB1', 'S01'],
        20: ['CD2', 'CB2', 'S02']
    }
    machines = {
        'CD1': {'processing_rate': 1/45},
        'CD2': {'processing_rate': 1/45},
        'CB1': {'processing_rate': 1/50}, # Gargalo Rota 10
        'CB2': {'processing_rate': 1/50}  # Gargalo Rota 20
    }
    stacks = {
        'S01': {'product': 'P1', 'eligible_routes': [10]}, # Ajustado para int se o ID da rota for int
        'S02': {'product': 'P2', 'eligible_routes': [20]}
    }
    return routes, machines, stacks

def test_generate_lots_integration(mock_infrastructure_for_lots):
    # Arrange
    routes, machines, stacks = mock_infrastructure_for_lots
    rng = np.random.default_rng(seed=42)

    props = {
        'hz': 43200,
        'utilization_factor': 0.8,
        'lot_sizes': [100, 50, 25],
        'lot_probs': [0.81, 0.17, 0.02],
        'lot_sizes_and_frequencies': {100: 90, 50: 19, 25: 2},
        'lt': 1.11,
        'initial_maneuver_time': 1800
    }

    # Act
    lots = generate_lots(routes, machines, stacks, rng=rng, props=props)

    # Assert
    assert isinstance(lots, dict)
    assert len(lots) > 0

    # Validações do Lote 0
    lote_zero = lots[0]

    # 1. Chaves esperadas
    for key in ['arrival_time', 'wagons', 'product', 'eligible_stockpiles', 'eligible_routes', 'processing_times']:
        assert key in lote_zero

    # 2. Física Temporal (Tem que ter o offset da manobra inicial)
    assert lote_zero['arrival_time'] >= 1800.0

    # 3. Matemática do processamento (Tempo = Vagões / Vazão)
    # Se a rota 10 for elegível, a vazão gargalo é 1/50. Tempo = Wagons / (1/50) = Wagons * 50
    if 10 in lote_zero['eligible_routes']:
        expected_time = lote_zero['wagons'] / (1/50)
        assert lote_zero['processing_times'][10] == pytest.approx(expected_time)
