from docplex.mp.model import Model
import random
import numpy as np
import time
from numba import jit
import cplex

random.seed(10)
@jit(nopython = True)
# extract data from csv file and store in respective variables
def required_data(data):
    # number of constraints or rows of matrix A
    m = int(data.loc[0]['m'])
    # print("m = ", m)

    # number of decision variables // number of columns in matrix A
    n = int(data.loc[0]['n'])

    p = int(data.loc[0]['p'])
    d = list(data.loc[0:p - 1]['d'])

    # cost vector (row vector)
    c = []
    for i in range(p):
        c.append(list(data.loc[i * n:(i * n) + (n - 1)]['c']))
        # for i in range(m):
        # c.append(0)
    # c = np.array(c)
    # print("Cost Vector is: ",c)

    # b vector or the RHS vector
    b = np.array(data.loc[0:m - 1]['b'])
    # print("Initial Solution vector is: ",b)

    # decision variables coefficient matrix
    A = np.array(data.loc[:]['A']).reshape(m, n)

    # print("Matrix A in standard form in standard form is:\n", A)

    return A, b, c, m, n, d, p

@jit(nopython = True)
def get_integer_index(num_of_dec_var):
    I = sorted(list(set(random.sample(range(num_of_dec_var), int(8 * num_of_dec_var / 10)))))
    # I = [i for i in range(int(0.8 * num_of_dec_var))]
    not_I = []
    for j in range(num_of_dec_var):
        if j not in I:
            not_I.append(j)

    return I, not_I

def value_of_k(p):
    break_loop = False
    k = 0
    while break_loop == False:
        value = 2 ** k
        if value >= p:
            break
        else:
            k += 1
    return k  # the value of k equals the value of counter


def model_socp(A, b, c, d, I, k, n, p):
    m2 = Model()
    m2.context.solver.log_output = True
    m2.set_time_limit(1200)

    m2.context.cplex_parameters.threads = 1
    m2.parameters.simplex.tolerances.feasibility = 10 ** -7
    # m2.Params.IntFeasTol = 10**-3
    vars = []
    for i in range(n):
        if i in I:
            vars.append(m2.binary_var(name='x{}'.format(i)))
        else:
            vars.append(m2.continuous_var(lb=0, name='x{}'.format(i)))

    multiplicative_y = m2.continuous_var_matrix(1, p, lb=0, name='y')

    tau = []
    for j in range(1, (2 ** k) + 1):
        tau.append(m2.continuous_var(lb=0, name='Tau0_{}'.format(j)))
    for l in range(1, k):
        for j in range(1, 2 ** (k - l) + 1):
            tau.append(m2.continuous_var(lb=0, name='Tau{}_{}'.format(l,j)))

    #print("** length of tau is **: ", len(tau))
    G = m2.continuous_var(lb=0, name='GAMMA')
    g = m2.continuous_var(lb=0, name='gamma')

    m2.add_constraints((m2.sum(vars[j] * A[k][j] for j in range(n)) <= b[k] for k in range(len(A))),
                       names="general'x'constraint")
    m2.add_constraints(vars[j] <= 1 for j in I)
    m2.add_constraints(vars[j] >= 0 for j in I)
    for i in range(p):
        lhs = 0
        for j in range(n):
            lhs += c[i][j] * vars[j]
        m2.add_constraint(lhs + d[i] == multiplicative_y[0, i], ctname='multiplicative_y_constraint{}'.format(i))

    for i in range(p):
        m2.add_constraint(tau[i] == multiplicative_y[0, i], ctname='variable_change_constraint{}'.format(i))
    #print("value of p is:", p)
    #print("value of 2**k is:", 2 ** k)
    if 2 ** k > p:
        for q in range(p, 2 ** k):
            m2.add_constraint(tau[q] == G)
    counter1, counter2 = 0, 0
    #print('!!!!!!!!! k is', k)
    for l in range(1, k):
        for j in range(1, 2 ** (k - l) + 1):
            m2.add_constraint(tau[(2 ** k) + counter1] * tau[(2 ** k) + counter1] <= tau[counter2] * tau[counter2 + 1],
                              ctname='SOCPconstraint{}'.format(counter1))
            counter1 += 1
            counter2 += 2
    m2.add_constraint(G * G <= tau[-1] * tau[-2], ctname='GAMMAconstraint')
    m2.add_constraint(G >= 0)
    m2.add_constraint(g <= G)
    m2.add_constraint(g >= 0)
    m2.set_objective('max', g)
    return m2, tau

def solve_model(m2, tau):
    m2.solve()
    variables_value = []
    try:
        for i in m2.iter_variables():
            variables_value.append(i.solution_value)
        objective_value = m2.objective_value
        gap_cplex = m2.solve_details.mip_relative_gap
        m_cplex = m2.get_cplex()
        best_bound_NEW = m_cplex.solution.MIP.get_best_objective()
    except:
        variables_value = "none"
        objective_value = "none"
        m_cplex = m2.get_cplex()
        best_bound_NEW = m_cplex.solution.MIP.get_best_objective()
        gap_cplex = "none"

    return variables_value, objective_value, best_bound_NEW, tau, gap_cplex

def main(A,b,d,p,n,c,I):

#for instance in range(1):
#data = pd.read_csv('multiplicative_instances/p=6/small_instance{}.csv'.format(1)

    #I, not_I = get_integer_index(n)
    k = value_of_k(p)
    m2, tau = model_socp(A, b, c, d, I, k, n, p)
    cplex_start_time = time.time()
    variables_value, gamma, best_bound_NEW, tau, gap_cplex = solve_model(m2, tau)
    cplex_solution_time = time.time() - cplex_start_time
    return variables_value, cplex_solution_time, best_bound_NEW, gamma, gap_cplex
#print("objective function value is: ", gamma**p)
#print("decision variables value are: ", variables_value)
# print(variables_value)
# print(cplex_solution_time)
# print(best_bound)
# print(n)