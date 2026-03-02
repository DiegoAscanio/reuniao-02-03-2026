import os
import glob
import json
import csv
import argparse

# Importa a sua função construtora do modelo
# Ajuste o nome do arquivo se a sua build_lp estiver em outro módulo (ex: from solver import build_lp)
from wagons_unloading import build_lp

TIME_LIMIT = 3600

def run_scenario(instance_data: dict, enable_capacity: bool, time_limit: int = TIME_LIMIT):
    """
    Monta e resolve o modelo para um cenário específico, retornando as métricas.
    """
    flags = {
        'limit_wagons': True,
        'limit_stacks_capacity': enable_capacity,
        'verbose': True # Deixamos False se nã quisermos floodar o terminal no meio do loop
    }

    # Monta o modelo enxuto na memória
    lp_model = build_lp(
        instance_data['lots'],
        instance_data['machines'],
        instance_data['routes'],
        instance_data.get('maintenance_tasks', {}),
        instance_data['hz'],
        instance_data['stacks'],
        _flags=flags
    )

    # Limita o tempo do solver (padrão: TIME_LIMIT)
    lp_model.set_time_limit(time_limit)
    
    # Roda o solver (você pode colocar log_output=True se quiser ver a árvore no terminal)
    sol = lp_model.solve(log_output=False)
    
    # Extrai as métricas
    if sol:
        details = lp_model.solve_details
        obj_val = sol.get_objective_value()
        time_sec = details.time
        
        # Pega o gap (docplex pode retornar um valor não numérico se a sol inicial for muito ruim)
        raw_gap = details.mip_relative_gap
        # Se raw_gap for None ou NaN, assumimos 1.0 (100%) por segurança
        gap = raw_gap if isinstance(raw_gap, float) else 1.0 
    else:
        # Se não achou NADA em 10 minutos (difícil, mas acontece em instâncias gigantes)
        obj_val = 0
        time_sec = time_limit
        gap = 1.0 # 100% de gap
        
    # LIBERA A MEMÓRIA DO MOTOR C++ (Crucial para loops de experimentação!)
    lp_model.end()
    del(lp_model)
    
    return obj_val, time_sec, gap

def main():
    parser = argparse.ArgumentParser(description="Runner de experimentos MMTSP Portuário (CPLEX)")
    parser.add_argument("instance_dir", help="Caminho para o diretório contendo os arquivos JSON")
    args = parser.parse_args()

    instance_dir = args.instance_dir
    
    if not os.path.isdir(instance_dir):
        print(f"Erro: O diretório '{instance_dir}' não existe.")
        return

    # Busca todos os arquivos JSON e ordena alfabeticamente
    json_files = sorted(glob.glob(os.path.join(instance_dir, "*.json")))
    
    if not json_files:
        print(f"Nenhum arquivo JSON encontrado em '{instance_dir}'.")
        return

    csv_path = os.path.join(instance_dir, "results.csv")
    
    print(f"Iniciando bateria de testes em {len(json_files)} instâncias...")
    print(f"Tempo limite por cenário: {TIME_LIMIT//60} minutos ({TIME_LIMIT} segundos).")
    print("-" * 50)

    # Abre o CSV para escrita (já escrevendo linha a linha para não perder dados se o script cair)
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Cabeçalho da Tabela
        writer.writerow([
            "Instance", 
            "Unloaded_NoCap", "Time_NoCap(s)", "Gap_NoCap(%)", 
            "Unloaded_Cap", "Time_Cap(s)", "Gap_Cap(%)"
        ])
        
        for file_path in json_files:
            # Extrai o nome do arquivo (ex: instance_001.json -> instance_001)
            instance_name = os.path.basename(file_path).replace(".json", "")
            print(f"Processando {instance_name}...")
            
            # Carrega os dados da instância
            with open(file_path, 'r', encoding='utf-8') as f:
                instance_data = json.load(f)
            
            # --- CENÁRIO 1: SEM Restrição de Capacidade (C17 Desligada) ---
            print("  -> Resolvendo Cenário: SEM Limite de Pilha...")
            obj_nocap, time_nocap, gap_nocap = run_scenario(instance_data, enable_capacity=False)
            
            # --- CENÁRIO 2: COM Restrição de Capacidade (C17 Ligada) ---
            print("  -> Resolvendo Cenário: COM Limite de Pilha...")
            obj_cap, time_cap, gap_cap = run_scenario(instance_data, enable_capacity=True)
            
            # Formata o output visual
            print(f"  [RESULTADO] Sem Cap: {obj_nocap} vagões | Com Cap: {obj_cap} vagões")
            
            # Escreve a linha no CSV (formatando Gap para percentual limpo)
            writer.writerow([
                instance_name,
                int(obj_nocap), round(time_nocap, 2), round(gap_nocap * 100, 2),
                int(obj_cap), round(time_cap, 2), round(gap_cap * 100, 2)
            ])
            
            # Força o sistema operacional a salvar a linha no disco imediatamente
            csv_file.flush() 

    print("-" * 50)
    print(f"Bateria concluída! Resultados salvos em: {csv_path}")

if __name__ == "__main__":
    main()
