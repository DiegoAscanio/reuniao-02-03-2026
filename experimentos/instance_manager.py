import numpy as np

# Importações dos estágios previamente construídos e testados
from topology import generate_topology
from pipeline import build_pipeline_graph
from routes import generate_routes, sobol_prune_routes
from machines import generate_machines
from stacks import generate_stacks
from lots import generate_lots

def generate_instance(
    seed: int = None,
    topology: dict = None,
    props: dict = None
) -> dict:
    '''
    Orquestra a execução em cascata de todos os módulos (Estágios 1 a 6)
    para gerar uma instância completa do Wagon Unloading Problem.
    '''
    rng = np.random.default_rng(seed=seed)
    
    # Estágio 1: DNA (Topologia)
    if topology is None:
        topology = generate_topology(rng=rng)
    
    # Estágio 2: Esqueleto (Pipeline Direcional)
    # Assumindo que build_pipeline_graph aceita a topologia inteira ou suas partes
    pipeline = build_pipeline_graph(topology, rng=rng)
    
    # Estágio 3: Roteamento (Caminhos e Poda)
    all_routes = generate_routes(pipeline)
    # Poda de segurança para evitar explosão combinatória (alvo estocástico entre 7 e 54 rotas)
    target_n = rng.integers(7, 55)
    routes = sobol_prune_routes(
        all_routes,
        target_n=target_n,
        stacks = topology['destination']['stacks'],
        rng=rng
    ) if len(all_routes) > target_n else all_routes
    
    # Estágio 4: Hardware (Máquinas e Tempos de Setup)
    machines = generate_machines(topology, pipeline, routes, rng=rng)
    
    # Estágio 5: Armazenamento (Pilhas)
    stacks = generate_stacks(topology, routes, machines, rng=rng, props=props)
    
    # Estágio 6: Fluxo de Minério (Lotes)
    lots = generate_lots(routes, machines, stacks, rng=rng, props=props)
    
    # Estágio 7: Consolidação
    return {
        'topology': topology,
        'pipeline': pipeline,
        'routes': routes,
        'machines': machines,
        'stacks': stacks,
        'lots': lots
    }
