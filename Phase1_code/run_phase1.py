import main_phase1
import socp
import pandas as pd
import random
random.seed(10)

def algorithm_y_multiply(list_of_values):
    result = 1
    for item in list_of_values:
        result = item * result
    return result

def cplex_gamma_multiply(gamma_value, p):
    result = gamma_value ** p
    return result

with open("output_phase1.csv","w", newline = '') as csv:
    csv.write("instance, y_values_algorithm, objective_value_algorithm, solution_time, cut_iterations, cplex_gamma, best_bound, cplex_socp_optimal, cplex_solution_time, optimality_gap_(%), optimality_gap_best_bound, gap_cplex")

for i in range(10):
    data = pd.read_csv('instance{}.csv'.format(i+1))
    A, b, c, m, n, d, p = main_phase1.required_data(data)
    decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = main_phase1.main_function(A, b, c, m, n, d, p)
    I, not_I = main_phase1.get_integer_index(n)
    optimal_value_algorithm = algorithm_y_multiply(y_values)
    variables_value, cplex_solution_time, best_bound, gamma, gap_cplex = socp.main(A,b,d,p,n,c,I)
    try:
        optimal_value_cplex = cplex_gamma_multiply(gamma, p)
    except:
        optimal_value_cplex = "none"
    try:
        optimality_gap = ((optimal_value_cplex - optimal_value_algorithm) * 100 / optimal_value_cplex)
    except:
        optimality_gap = "none"
    try:
        best_bound_multiplied = cplex_gamma_multiply(best_bound, p)
    except:
        best_bound_multiplied = 'none'
    try:
        optimality_gap_best_bound = ((best_bound_multiplied - optimal_value_algorithm) * 100 / best_bound_multiplied)
    except:
        optimality_gap_best_bound = 'none'

    final_list = [i+1, y_values, optimal_value_algorithm, solution_time, cut_iteration, gamma, best_bound, optimal_value_cplex, cplex_solution_time, optimality_gap, optimality_gap_best_bound, gap_cplex]
    with open("output_phase1.csv", "a", newline = '') as csv:
        csv.write("\n")
        csv.write(str(final_list))
    #print(gamma, optimal_value_gurobi, gurobi_solution_time)
    #print(y_values, optimal_value_algorithm, solution_time, cut_iteration, nIT_list, gamma, optimal_value_gurobi, gurobi_solution_time, optimality_gap)

#print(y_values, solution_time, nIT_list, cut_iteration)