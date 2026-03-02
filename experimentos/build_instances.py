import os
import json
import numpy as np
import pdb

# Importa o orquestrador que já consolida a porra toda
from instance_manager import generate_instance
from topology import generate_topology

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.int64, np.int32, np.int16, np.int8)):
            return int(o)
        if isinstance(o, (np.float64, np.float32)):
            return float(o)
        return super(NumpyEncoder, self).default(o)

def main():
    output_dir = "instances"
    os.makedirs(output_dir, exist_ok=True)

    # Dicionário mestre de propriedades (pode iterar sobre ele no futuro para variar cenários)
    base_props = {
        'hz': 10800, 
        'utilization_factor': 0.75, 
        'range_num_products': (2, 6), 
        'avg_number_of_piles': 6, 
        'lot_sizes': [12, 6, 3], 
        'lot_probs': [0.81, 0.17, 0.02],
        'lot_sizes_and_frequencies': {12: 90, 6: 19, 3: 2},
        'lt': 1.11, 
        'cap_limiter': 0.1,
        'initial_maneuver_time': 900,
        'topology_props': {
            'cds_info': {
                'range': (2, 5),
                'properties': {
                    'same_layer_interconnection_prob': 0.0
                }
            },
            'cbs_info': {
                'total_range': (2, 4),
                'layer_range': (1, 3), # Note: (2, 4) in numpy integers is 2 to 3 inclusive
                'properties': {
                    'same_layer_interconnection_prob': 0.3,
                    'double_cap_prob_base': 0.2,
                    'next_layer_connection_to_next_index_prob': 0.0
                }
            },
            'stackers_info': {
                'range': (1, 4),
                'properties': {
                    'same_layer_interconnection_prob': 0.0
                }
            },
            'stacks_info' : {
                'mean': 4,
                'std_dev': 1,
                'properties': {
                    'coverage_strategy': 'proportional_split',
                    'adjacent_machine_overlap_prob': 0.45
                }
            }
        }
    }

    # Setup de geração
    num_instances = 8
    base_seed = 42

    print(f"Gerando {num_instances} instâncias na pasta '{output_dir}/' via instance_manager...")

    for i in range(num_instances):
        current_seed = base_seed + i
        filename = os.path.join(output_dir, f"instance_{i+1:03d}.json")
        
        try:
            # Generate a base topology
            topology_props = base_props['topology_props']
            cds_info = topology_props['cds_info']
            cbs_info = topology_props['cbs_info']
            stackers_info = topology_props['stackers_info']
            stacks_info = topology_props['stacks_info']
            topology = generate_topology(
                rng = np.random.default_rng(current_seed),
                cds_info = cds_info,
                cbs_info = cbs_info,
                stackers_info = stackers_info,
                stacks_info = stacks_info
            )
            # Chama a função que já orquestra os Estágios 1 a 6
            instance_data = generate_instance(
                seed = current_seed,
                topology = topology,
                props = base_props
            )
            
            # Injeta as variáveis de topo que o docplex/build_lp vai cobrar depois
            instance_data['hz'] = base_props['hz']
            instance_data['maintenance_tasks'] = {} # Fica vazio por enquanto

            # Salva no disco usando o encoder sanitizado
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(instance_data, f, cls=NumpyEncoder, indent=2)
                
            print(f"  [OK] {filename} gerada.")
            
        except Exception as e:
            print(f"  [ERRO] Falha ao gerar {filename}: {e}")
            raise e

if __name__ == "__main__":
    main()
