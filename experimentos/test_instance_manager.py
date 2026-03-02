import pytest
from instance_manager import generate_instance

def test_generate_instance_integration_keys():
    # Arrange
    # Vamos rodar com uma seed fixa e sem passar props, 
    # forçando o uso dos defaults (fallback) definidos em cada módulo.
    seed_test = 42
    
    # Act
    instance = generate_instance(seed=seed_test)
    
    # Assert
    assert isinstance(instance, dict)
    
    # Valida se as 6 macroestruturas do porto foram geradas e acopladas
    expected_keys = [
        'topology', 
        'pipeline', 
        'routes', 
        'machines', 
        'stacks', 
        'lots'
    ]
    
    for key in expected_keys:
        assert key in instance
        assert isinstance(instance[key], (dict, list))
        # Nenhuma etapa do pipeline pode retornar uma estrutura vazia
        assert len(instance[key]) > 0
