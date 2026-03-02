# We'll make a solver for the problem of unloading iron ore wagons into
# stockpiles at a port. The goal is to maximize the total amount of 
# vagons unloaded in a given planning horizon hz while respecting
# the problems constraints as formulated by Ferreira et al. (2023).

from docplex.mp.model import Model
from utils import restore_numeric_keys
import pdb

def _machines_only(route):
    return route[:-1]  # assuming last element is stack

def build_lp(lots, machines, routes, maintenance_tasks, hz, stacks=dict(), _flags=None):

    if _flags is None:
        _flags = {
            'limit_wagons': True,
            'limit_stacks_capacity': True,
            'verbose': True # <-- Nova flag adicionada
        }

    # 0. SANITIZAÇÃO DO JSON: Restaurar chaves numéricas para inteiros
    lots = restore_numeric_keys(lots)
    routes = restore_numeric_keys(routes)
    maintenance_tasks = restore_numeric_keys(maintenance_tasks)
    stacks = restore_numeric_keys(stacks)
    machines = restore_numeric_keys(machines) # Limpa sub-dicionários (ex: setup_times)

    # process flags
    limit_wagons = _flags.get('limit_wagons', True)
    limit_stacks_capacity = _flags.get('limit_stacks_capacity', True)
    verbose = _flags.get('verbose', False)

    if verbose: print(f"--- Iniciando build_lp (Horizonte: {hz}s) ---")

    # Create the docplex Model
    lp_model = Model(name="wagons_unloading")

    # Create decision variables
    if verbose: print(">> Criando variáveis de decisão...")

    # 1. Y_lr: Binary variable indicating if lot l is unloaded via route r
    if verbose: print("   - Gerando variável binária Y_{lr}...")
    Y = lp_model.binary_var_dict(((l, r) for l in lots for r in routes), name='Y')

    # 2. X_{ml_{1}l_{2}}: Binary variable indicating if
    # lots l1 and l2 use machine m and l1 unloads before l2
    if verbose: print("   - Gerando variável binária X_{m l1 l2}...")
    X = lp_model.binary_var_dict(((m, l1, l2) for m in machines for l1 in lots for l2 in lots), name='X')

    # 3. YP^{m}_{l_{1}r_{1}l_{2}r_{2}}: Binary variable indicating if
    # lots l1 and l2 unload using routes r1 and r2, respectively,
    # and sharing machine m if m services both routes and have
    # double handling capability. This variable will be very hard
    # to model here, but we'll try.
    # First, let's define A_{l_{1}r_{1}} set which contains all
    # the (l2, r2) pairs eligible for simultaneous unloading with
    # lot l1 using route r1.
    A = {}
    for l1 in lots:
        for r1 in routes:
            A[(l1, r1)] = []
            for l2 in set(lots) - {l1}:
                for r2 in set(routes) - {r1}:
                    # l2, r2 is eligible if and only if all machines in
                    # the intersection of machines servicing r1 and r2
                    # have double handling capability
                    #
                    # eligible starts as False as we don't know if there
                    # is any machine servicing both routes
                    eligible = False
                    for m in machines:
                        if (m in routes[r1] and
                            m in routes[r2]):
                            # if we found a machine servicing both routes,
                            # set eligible to True and check if it has
                            # double handling capability
                            eligible = True
                            if not machines[m]['double_capacity']:
                                # if any machine servicing both routes
                                # does not have double handling capability,
                                # then (l2, r2) is not eligible
                                # so we set eligible to False and break
                                # the loop
                                eligible = False
                                break
                    if eligible:
                        A[(l1, r1)].append((l2, r2))

    keys = [(m,l1,r1,l2,r2) for m in machines for l1 in lots for r1 in routes for l2 in lots for r2 in routes]
    ub = {k: 1 if (k[3], k[4]) in A[(k[1], k[2])] else 0 for k in keys}
    YP = lp_model.binary_var_dict(keys, lb=0, ub=ub, name='YP')

    # 4. XMB_{lj} : Binary variable indicating if lot l uses machine m_j
    # before maintenance task j
    #    XMA_{lj} : Binary variable indicating if lot l uses machine m_j
    # after maintenance task j
    # First, we need to discover which routes are affected per each machine
    # in maintenance
    affected_routes_by_machines_in_maintenance = {}
    for j in maintenance_tasks:
        m_j = maintenance_tasks[j]['machine']
        affected_routes_by_machines_in_maintenance[m_j] = set()
        for r in routes:
            if m_j in routes[r]:
                affected_routes_by_machines_in_maintenance[m_j] = (
                    affected_routes_by_machines_in_maintenance[m_j].union({r})
                )
        affected_routes_by_machines_in_maintenance[m_j] = list(
            affected_routes_by_machines_in_maintenance[m_j]
        )
    # we won't use this information now, only later to add the constraints
    # that will prevent unloading using machines in maintenance
    # XMB and XMA variables are indexed by lots l and maintenance tasks j
    XMB = lp_model.binary_var_dict(
        ((l, j) for l in lots for j in maintenance_tasks),
        name='XMB'
    )
    XMA = lp_model.binary_var_dict(
        ((l, j) for l in lots for j in maintenance_tasks),
        name='XMA'
    )
    
    # 5. N_{lr}: Number of wagons unloaded from lot l via route r
    if verbose: print("   - Gerando variável inteira N_{lr}...")
    N = lp_model.integer_var_dict(((l, r) for l in lots for r in routes), lb=0, name='N')
    
    # 6. S_{lr}: Start time of unloading lot l via route r
    #    F_{lr}: Finish time of unloading lot l via route r
    #    Z_{lr}: Minimum between F_{lr} and the end of the planning horizon hz
    if verbose: print("   - Gerando variáveis contínuas S_{lr}, F_{lr}, Z_{lr}...")
    S = lp_model.continuous_var_dict(((l, r) for l in lots for r in routes), lb=0, name='S')
    F = lp_model.continuous_var_dict(((l, r) for l in lots for r in routes), lb=0, name='F')
    Z = lp_model.continuous_var_dict(((l, r) for l in lots for r in routes), lb=0, name='Z')

    # Compute H parameter
    _processing_times = []
    for l in lots:
        for p in lots[l]['processing_times']:
            _processing_times.append(lots[l]['processing_times'][p])
    H = hz + (max(_processing_times) if _processing_times else 0)

    if verbose: print(">> Variáveis criadas com sucesso. Adicionando restrições...")

    # C1. Each lot will be assigned to at most one route for unloading
    if verbose: print("   - Adicionando [C1]: Atribuição única de rotas...")
    for l in lots:
        eligible_routes = lots[l]['eligible_routes']
        lp_model.add_constraint(
            lp_model.sum(Y[(l, r)] for r in eligible_routes) <= 1,
            ctname=f"C1:sum_Y_{{{l}r}} <= 1"
        )

    # C2. Finish time from lot l via route r should be greater or equal
    # than start time plus unloading time minus H upper bound times (1 - Y_{lr})
    # to control the constraint activation
    if verbose: print("   - Adicionando [C2]: Relação Start e Finish Time...")
    for l in lots:
        for r in lots[l]['eligible_routes']:
            processing_time = lots[l]['processing_times'][r]
            lp_model.add_constraint(
                F[(l, r)] >= S[(l, r)] + processing_time - H * (1 - Y[(l, r)]),
                ctname=f"C2:F_{{{l}{r}}} >= S_{{{l}{r}}} + PT_{{{l}{r}}} - H(1 - Y_{{{l}{r}}})"
            )

    # C3. Guarantee that lot l1 is planned to use route r1 in a simultaneous
    # operation only if it is assigned to route r1 in variable Yl1r1
    # That is:
    # Sum((YP_l1_r1_l2_r2**m) for (l2, r2) in A_l1_r1) <= Y_l1_r1
    #   For all l1 in L, for all r1 in R_l1, for all m in M_r1 U M_r2
    if verbose: print("   - Adicionando [C3]: Garantia de operação simultânea (YP vs Y)...")
    for l1 in lots:
        for r1 in lots[l1]['eligible_routes']:
            machines_in_R1 = set(_machines_only(routes[r1]))
            for l2, r2 in A[(l1, r1)]:
                machines_in_R2 = set(_machines_only(routes[r2]))
                for m in machines_in_R1.union(machines_in_R2):
                    lp_model.add_constraint(
                        lp_model.sum(YP[(m, l1, r1, l2, r2)] for l2, r2 in A[(l1, r1)]) <= Y[(l1, r1)],
                        ctname=f"C3:sum_{{l2r2}} YP_{{{m}{l1}{r1}{l2}{r2}}} <= Y_{{{l1}{r1}}}"
                    )

    # C4. Provide consistent planning for all machines that works in 
    # simultaneous operations, that is:
    # Y * P_l1_r1_l2_r2**m1 == Y * P_l1_r1_l2_r2**m2
    #
    # For all l1 in L, for all r1 in R_l1, for all (l2, r2) in A_l1_r1,
    # for all m1, m2 in M_r1 U M_r2
    # the last one we'll assume it is the concise way of telling that
    # m1 and m2 are built over the cartesian product of machines
    # servicing r1 or r2
    if verbose: print("   - Adicionando [C4]: Consistência de YP entre máquinas simultâneas...")
    for l1 in lots:
        for r1 in lots[l1]['eligible_routes']:
            machines_in_R1 = set(_machines_only(routes[r1]))
            for l2, r2 in A[(l1, r1)]:
                machines_in_R2 = set(_machines_only(routes[r2]))
                all_machines = machines_in_R1.union(machines_in_R2)
                for m1 in all_machines:
                    for m2 in all_machines:
                        lp_model.add_constraint(
                            YP[(m1, l1, r1, l2, r2)] == YP[(m2, l1, r1, l2, r2)],
                            ctname=f"C4:YP_{{{m1}{l1}{r1}{l2}{r2}}} == YP_{{{m2}{l1}{r1}{l2}{r2}}}"
                        )

    # C5. Ensure consistency between YP**{m}_{l1r1l2r2} and YP**{m}_{l2r2l1r1}
    # That is:
    # YP**{m}_{l1r1l2r2} == YP**{m}_{l2r2l1r1}
    # For all l1 in L, for all r1 in R_l1, for all (l2, r2) in A_l1_r1,
    # for all m in M_r1 U M_r2
    if verbose: print("   - Adicionando [C5]: Simetria da variável YP...")
    for l1 in lots:
        for r1 in lots[l1]['eligible_routes']:
            machines_in_R1 = set(_machines_only(routes[r1]))
            for l2, r2 in A[(l1, r1)]:
                machines_in_R2 = set(_machines_only(routes[r2]))
                all_machines = machines_in_R1.union(machines_in_R2)
                for m in all_machines:
                    lp_model.add_constraint(
                        YP[(m, l1, r1, l2, r2)] == YP[(m, l2, r2, l1, r1)],
                        ctname=f"C5:YP_{{{m}{l1}{r1}{l2}{r2}}} == YP_{{{m}{l2}{r2}{l1}{r1}}}"
                    )

    # For C6 and C7 we need to build before a T set that contains all tuples
    # (l1, r1, l2, r2, m) where r1 is eligible for r1, r2 for l2 and
    # machine m services both routes r1 and r2
    # again, we'll need to consider the cartesian product of lots at least
    if verbose: print("   - Mapeando conjunto T para restrições disjuntivas...")
    T = set()
    for l1 in lots:
        for l2 in set(lots) - {l1}:
            for r1 in lots[l1]['eligible_routes']:
                for r2 in lots[l2]['eligible_routes']:
                    for m in machines:
                        if m in routes[r1] and m in routes[r2]:
                            T.add((l1, r1, l2, r2, m))
                            
    # Now we can build C6 and C7
    # C6 and C7. Ensure that each single cap machine won't process more than
    # one lot at a time. These are the "no-overlap" constraints.
    # If YP**{m}_{l1r1l2r2} == 1 (double cap machine), all double cap machines
    # in M_r1 & M_r2 may overlap while processing l1 and l2.
    if verbose: print("   - Adicionando [C6 e C7]: Restrições disjuntivas (No-overlap) com Setups...")
    for l1, r1, l2, r2, m in T:
        setup_time_m_r1_r2 = machines[m]['setup_times'][int(r1)][int(r2)]
        setup_time_m_r2_r1 = machines[m]['setup_times'][int(r2)][int(r1)]
        
        # C6
        lp_model.add_constraint(
            S[(l2, r2)] >= F[(l1, r1)] + setup_time_m_r1_r2 + H * (
                Y[(l1, r1)] + Y[(l2, r2)] - YP[(m, l1, r1, l2, r2)] + X[(m, l1, l2)] - 3
            ),
            ctname=(f"C6:S_{{{l2}{r2}}} >= F_{{{l1}{r1}}} + st_{{{m}{r1}{r2}}}"
                    f" + H(Y_{{{l1}{r1}}} + Y_{{{l2}{r2}}} - YP_{{{m}{l1}{r1}{l2}{r2}}} + X_{{{m}{l1}{l2}}} - 3)")
        )
        
        # C7
        lp_model.add_constraint(
            S[(l1, r1)] >= F[(l2, r2)] + setup_time_m_r2_r1 + H * (
                Y[(l1, r1)] + Y[(l2, r2)] - YP[(m, l1, r1, l2, r2)] - X[(m, l1, l2)] - 2
            ),
            ctname=(f"C7:S_{{{l1}{r1}}} >= F_{{{l2}{r2}}} + st_{{{m}{r2}{r1}}}"
                    f" + H(Y_{{{l1}{r1}}} + Y_{{{l2}{r2}}} - YP_{{{m}{l1}{r1}{l2}{r2}}} - X_{{{m}{l1}{l2}}} - 2)")
        )

    # Constraints 8, 9, 10 and 11 bounds start and Z (finish or hz) times to
    # process lots l through eligible routes r within the planning horizon hz
    #
    # C10 and C11 ensures that Z is the minimum between
    # Finish time of each lot and the planning horizon hz
    #
    if verbose: print("   - Adicionando [C8, C9, C10, C11]: Limites de Horizonte de Tempo e Chegadas...")
    for l in lots:
        for r in lots[l]['eligible_routes']:
            # C8. S_{lr} >= lot arrival time * Y_{lr}
            at_l = lots[l]['arrival_time']
            lp_model.add_constraint(S[(l, r)] >= at_l * Y[(l, r)], ctname=f"C8:S_{{{l}{r}}} >= AT_{{{l}}} * Y_{{{l}{r}}}")
            # C9. S_{lr} before hz ends
            #     S_{lr} <= hz * Y_{lr}
            lp_model.add_constraint(S[(l, r)] <= hz * Y[(l, r)], ctname=f"C9:S_{{{l}{r}}} <= hz * Y_{{{l}{r}}}")
            # C10. Z_{lr} <= F_{lr}
            lp_model.add_constraint(Z[(l, r)] <= F[(l, r)], ctname=f"C10:Z_{{{l}{r}}} <= F_{{{l}{r}}}")
            # C11. Z_{lr} <= hz * Y_{lr}
            lp_model.add_constraint(Z[(l, r)] <= hz * Y[(l, r)], ctname=f"C11:Z_{{{l}{r}}} <= hz * Y_{{{l}{r}}}")

    # C12 computes the number of unloaded wagons within the operation time hz
    if verbose: print("   - Adicionando [C12]: Contagem de vagões descarregados...")
    for l in lots:
        for r in lots[l]['eligible_routes']:
            omega_l = lots[l]['wagons']
            p_lr = lots[l]['processing_times'][r]
            lp_model.add_constraint(
                N[(l, r)] <= (omega_l / p_lr) * (Z[(l, r)] - S[(l, r)]),
                ctname=f"C12:N_{{{l}{r}}} <= (omega_{{{l}}} / PT_{{{l}{r}}}) * (Z_{{{l}{r}}} - S_{{{l}{r}}})"
            )

    # Constraints to avoid unloading using machines in maintenance - 13, 14, 15
    if verbose: print("   - Adicionando [C13, C14, C15]: Restrições de manutenção de máquinas...")
    _affected_routes = set()
    for m_j in affected_routes_by_machines_in_maintenance:
        _affected_routes.update(affected_routes_by_machines_in_maintenance[m_j])
    _affected_routes = list(_affected_routes)
    
    for l in lots:
        for r in routes:
            if r in _affected_routes:
                for j in maintenance_tasks:
                    sm_j = maintenance_tasks[j]['start_time']
                    fm_j = maintenance_tasks[j]['end_time']
                    # C13. If machine m_j is in maintenance, XMB_{lj} define if lot l
                    # uses machine m_j before maintenance task j and XMA_{lj} after it.
                    lp_model.add_constraint(Y[(l, r)] <= XMB[(l, j)] + XMA[(l, j)], ctname=f"C13:Y_{{{l}{r}}} <= XMB_{{{l}{j}}} + XMA_{{{l}{j}}}")
                    # C14. ensures that lot l will be unloaded before maintenance task j
                    lp_model.add_constraint(sm_j >= F[(l, r)] + H * (Y[(l, r)] + XMB[(l, j)] - 2), ctname=f"C14:sm_{{{j}}} >= F_{{{l}{r}}} + H(Y_{{{l}{r}}} + XMB_{{{l}{j}}} - 2)")
                    # C15. ensures that lot l will be unloaded after maintenance task j
                    lp_model.add_constraint(S[(l, r)] >= fm_j + H * (Y[(l, r)] + XMA[(l, j)] - 2), ctname=f"C15:S_{{{l}{r}}} >= fm_{{{j}}} + H(Y_{{{l}{r}}} + XMA_{{{l}{j}}} - 2)")

    # C16. A forgotten constraint in the original paper
    # that will ensure that N_{lr} is <= total wagons in lot l times Y_{lr}
    if limit_wagons:
        if verbose: print("   - Adicionando [C16]: Limite estrito de vagões por lote...")
        for l in lots:
            omega_l = lots[l]['wagons']
            for r in lots[l]['eligible_routes']:
                lp_model.add_constraint(
                    N[(l, r)] <= omega_l * Y[(l, r)], 
                    ctname=f"C16:N_{{{l}{r}}} <= omega_{{{l}}} * Y_{{{l}{r}}}"
                )

    # C17. A contribution to the original paper to ensure a capacity
    # constraint on the stacks.
    if limit_stacks_capacity:
        if verbose: print("   - Adicionando [C17]: Restrição de capacidade espacial nas pilhas...")
        for s in stacks:
            stack_routes_set = set(stacks[s]['eligible_routes'])
            K_s = stacks[s]['capacity']
            lp_model.add_constraint(
                lp_model.sum(
                    N[(l, r)] 
                    for l in lots 
                    for r in set(lots[l]['eligible_routes']).intersection(stack_routes_set)
                ) <= K_s,
                ctname=f"C17_cap_stack_{s}"
            )

    # Objective Function
    if verbose: print(">> Construindo Função Objetivo: Maximizar vagões descarregados...")
    obj = lp_model.sum(N[(l, r)] for l in lots for r in lots[l]['eligible_routes'])
    lp_model.maximize(obj)

    if verbose: print("--- Construção do modelo finalizada! ---")

    # Retorna EXCLUSIVAMENTE o docplex Model para economizar memória
    return lp_model
