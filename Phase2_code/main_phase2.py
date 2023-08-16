import random
import numpy as np
import time
from docplex.mp.model import Model
from numba import jit
random.seed(10)

# extract data from csv file and store in the respective variables
def required_data(data):
    m = int(data.loc[0]['m'])               # number of constraints or rows of matrix A

    #print(" m = ", m)

    n = int(data.loc[0]['n'])               # number of decision variables or number of columns in matrix A
    p = int(data.loc[0]['p'])               # number of multiplicative objective functions
    d = list(data.loc[0:p - 1]['d'])
    #for i in range(len(d)):
        #d[i] = 3 * d[i]
    c = []                                  # cost vector (row vector) (D in the theory/paper)
    for i in range(p):
        c.append(list(data.loc[i * n:((i + 1) * n - 1)]['c']))

    #print(c)
    #for items in c:
        #for j in range(len(items)):
            #items[j] = items[j] * 0.3
    #print(c)
        # for i in range(m):
        # c.append(0)
    # c = np.array(c)
    # print("Cost Vector is: ",c)

    b = np.array(data.loc[0:m - 1]['b'])                        # b vector or the RHS vector
    # print("Initial Solution vector is: ",b)
    # A = np.zeros((m,n))
    A = np.array(data.loc[:]['A']).reshape(m, n)                # coefficient(of decision variables) matrix

    # print("Matrix A in standard form in standard form is:\n", A)
    return A, b, c, m, n, d, p

@jit(nopython = True)
def get_integer_index(num_of_dec_var):
    # generally in andrea lodi's paper 90% of variables are integer in average
    # I = sorted(list(set(random.sample(range(num_of_dec_var), int(9 * num_of_dec_var/ 10)))))

    I = [i for i in range(int(0.8 * num_of_dec_var))]               #first 80% of variables are integers
    not_I = []
    for j in range(num_of_dec_var):
        if j not in I:
            not_I.append(j)
    return I, not_I

def FP_parameters(I, m, n):
    time_limit = np.log(m + n) / 4                          # time interval in seconds CPU time = 1800 CPU seconds (parameter from Feasibility pump paper)
    average_number_of_variable_to_be_flipped = 10           # average number of variables to be flipped T (set from feasibility pump paper) 30 was initial value
    perturberation_frequency = 100                          # perturberation frequency parameter (R) i.e perterburation after every 100 iterations....... nIT
    exact_num_of_int_var_to_flip = int(random.randint(int(average_number_of_variable_to_be_flipped / 2),
                                                  int(3 * average_number_of_variable_to_be_flipped / 2)))            # exact number of integer variables to be flipped TT
    if exact_num_of_int_var_to_flip > len(I):
        exact_num_of_int_var_to_flip = average_number_of_variable_to_be_flipped
    return time_limit, average_number_of_variable_to_be_flipped, exact_num_of_int_var_to_flip, perturberation_frequency

def MILMMP_parameters():
    epsilon = 100
    multiplier = 10**4
    return epsilon, multiplier

def first_linear_model(A, b, c, d, n, p, I):
    m1 = Model()
    m1.context.cplex_parameters.threads = 1             # m1.Params.Threads = 1 running on one thread
    m1.parameters.simplex.tolerances.feasibility = 10 ** -4
    m1.parameters.simplex.tolerances.optimality = 10 ** -5
    x = m1.continuous_var_matrix(1, n, lb=0, name="x")             #x decision variables
    multiplicative_y = []                                       #y decision variables in criterion space
    for i in range(p):
        multiplicative_y.append(m1.continuous_var(name='y{}'.format(i)))
    m1.add_constraints(m1.sum(x[0,j] * A[k][j] for j in range(n)) <= b[k] for k in range(len(A)))
    m1.add_constraints(x[0,j] <= 1 for j in I)
    for k in range(p):
        #lhs = 0
        #for j in range(n):
            #lhs += c[k][j] * x[0, j]
        #lhs =
        # m1.add_constraint(lhs + d[k] == multiplicative_y[k])
        m1.add_constraint(m1.sum([m1.scal_prod([x[0,j] for j in range(n)], [c[k][j] for j in range(n)]),d[k]]) == multiplicative_y[k])
    m1.set_objective('max', m1.sum(multiplicative_y[k] for k in range(p)))
    return m1, multiplicative_y

def check_infeasibility(model):
    if model.solve_details.status_code == 3:
        return True   # true means the model is infeasible
    else:
        return False

def solve_first_linear_model(m1, n):
    # m1.setParam('OutputFlag', 0)
    m1.solve()
    infeasible = check_infeasibility(m1)
    if infeasible == True:
        return ('infeasible', 'none', 'none', 'none')
    else:
        x_y_list = []
        for i in m1.iter_variables():
            x_y_list.append(i.solution_value)
            # x_relaxed = x*
        z_lp = m1.objective_value                   # objective values
        x_relaxed, y_values = [], []                # y values is the list of values of Y for p number of objectives
        for v in range(n):
            x_relaxed.append(float(x_y_list[v]))
        for u in range(n, len(x_y_list)):
            y_values.append(x_y_list[u])
        return m1, z_lp, x_relaxed, y_values

# check whether the floating point number (in our case: values of decision variables) is integer?
# I = index position where the list x_relaxed must have integer values

def check_integer(I, x_relaxed):
    true_count = 0
    for items in I:
        if x_relaxed[items].is_integer():
            true_count += 1
        elif not x_relaxed[items].is_integer():
            return False
    if len(I) == true_count:
        return True
    else:
        return False

def rounding(x_relaxed, I):
    x_tilde = []
    for count in range(len(x_relaxed)):
        if count in I:
            x_tilde.append(round(x_relaxed[count]))         # x_tilde dimension is similar to x*
        else:
            x_tilde.append(x_relaxed[count])
    return x_tilde

def second_model_FP(n, A, b, c, d, p, I):
    f = Model()
    f.context.cplex_parameters.threads = 1
    f.parameters.simplex.tolerances.feasibility = 10 ** -4
    f.parameters.simplex.tolerances.optimality = 10 ** -5
    x = f.continuous_var_matrix(1, n, lb=0, name="x")
    multiplicative_y = []
    for i in range(p):
        multiplicative_y.append(f.continuous_var(name='y{}'.format(i)))
    objvar = f.continuous_var(lb=0)
    for k in range(p):
        #lhs = 0
        #for j in range(n):
            #lhs += c[k][j] * x[0, j]
        #f.add_constraint(lhs + d[k] == multiplicative_y[k])
        f.add_constraint(f.sum([f.scal_prod([x[0, j] for j in range(n)], [c[k][j] for j in range(n)]), d[k]]) == multiplicative_y[k])
    f.add_constraints(f.sum(x[0, j] * A[i][j] for j in range(n)) <= b[i] for i in range(len(A)))
    f.add_constraints(x[0, i] <= 1 for i in I)
    # f.addConstrs(sum(z[0,j] for j in I if x_tilde[j] == 0) + sum(1-z[0,j] for j in I if x_tilde[j]==1) == objvar)
    f.set_objective('min', objvar)
    return f, objvar, x, multiplicative_y

# this function solves distance based second model
def solve_second_model_FP(f, n, cut_iteration):
    # f.setParam('OutputFlag', 0)
    # f.setObjective(sum(z[0,j] for j in I if x_tilde[j] == 0) + sum(1-z[0,j] for j in I if x_tilde[j]==1), GRB.MINIMIZE)
    #f.set_time_limit(1200-cumulative_time)
    if cut_iteration <=1:
        f.set_time_limit(20)
    else:
        f.set_time_limit(2)
    second_model_start_time = time.time()
    f.solve()
    second_model_time_taken = time.time() - second_model_start_time
    #print("second_model_solve_time is ", second_model_time_taken)
    # delta = f.getObjective()
    # delta = delta.getValue()
    infeasible = check_infeasibility(f)
    if infeasible == True:
        return ('infeasible', 'none', 'none', 'none')
    else:
        jj = time.time()
        x_y_list = []
        try:
            for i in f.iter_variables():
                x_y_list.append(i.solution_value)
        except:
            return f, 'none', 'none','none'
        x_list, y_values= [], []
        for v in range(n):
            x_list.append(float(x_y_list[v]))
        for u in range(n, len(x_y_list) - 1):
            y_values.append(x_y_list[u])
        delta = f.objective_value
        kk = time.time() - jj
        #print("answer appending time is ", kk)
        return f, delta, x_list, y_values


def feasibility_pump(f, I,c,d,TT,n, objvar, z, p, time_limit, x_tilde,cut_iteration):
    nIT = 0
    FP_start_time = time.time()
    delta = 0
    while ((time.time() - FP_start_time) < time_limit):
        nIT += 1
        #print("nIT = ", nIT)

        # calling second_distance_based_model_FP
        objfun = f.add_constraint(
            objvar == (f.sum(z[0, j] for j in I if x_tilde[j] == 0) + f.sum(1 - z[0, j] for j in I if x_tilde[j] == 1)))

        f, delta, x_list, y_values = solve_second_model_FP(f, n, cut_iteration)

        if x_list == 'none':
             ## this line of code is essential as for larger instances x_list is returned as none if f is not solved within 1200 - cumulative time. see function solve_second_model
            break
        
        if y_values == 'none':
            break
        f.remove_constraint(objfun)
        boolean = check_integer(I, x_list)
        if boolean == True:
            z_ip = sum((np.dot(c[i], x_list) + d[i]) for i in range(len(c)))
            #print("yyyyyyyy")
            return (f, x_list, y_values, z_ip, delta, time.time() - FP_start_time, nIT)

        if nIT <= 100:
            #print("*******")
            count = 0
            for items in I:
                if round(x_list[items]) != x_tilde[items]:
                    count += 1
                    if count >= 1:
                        break
            if count >= 1:
                #print("######")
                for i in I:
                    x_tilde = rounding(x_list, I)
            else:
                #   print("$$$$$")
                distance_list = []
                for items in I:
                    distance_list.append(abs(x_list[items] - x_tilde[items]))
                if len(distance_list) > 0 and TT <= len(I):
                    temp_list = []
                    for i in range(TT):
                        max_index = np.argmax(distance_list)  # maximum distance index in distance_list
                        temp_list.append(I[max_index])
                        distance_list[max_index] = - 100
                    for items in temp_list:
                        if x_tilde[items] == 0:
                            x_tilde[items] = 1
                        elif x_tilde[items] == 1:
                            x_tilde[items] = 0
        elif nIT > 100:  ##################### need to change the code for checking last 3 iterations as well, not only after 100 iterations

            for j in I:
                ro = np.random.uniform(-0.3, 0.7)
                if (abs(x_list[j] - x_tilde[j]) + max(ro, 0)) > 0.5:
                    if x_tilde[j] == 0:
                        x_tilde[j] = 1
                    elif x_tilde[j] == 1:
                        x_tilde[j] = 0

    return (f, 'none', 'none', 'none', delta, time.time() - FP_start_time, nIT)

def check_for_perturbation(prev_three_x):
    if (np.array_equiv(prev_three_x[0], prev_three_x[1]) == True) and (np.array_equiv(prev_three_x[1], prev_three_x[2]) == True):
        return True
    else:
        return False

def cut_parameters(y_values, p, epsilon, multiplier):
    cut_list = []
    cut_rhs = []
    for counter in range(len(y_values)):
        cut_list.append(multiplier * (1 / y_values[counter]))
        # cut_matrix = np.array(cut_matrix)
    cut_rhs.append(epsilon + (multiplier * p)) # epsilon + (multiplier*p)
    # cut_rhs = np.array(cut_rhs)
    return cut_list, cut_rhs


def add_cut_m(model, cut_list, multiplicative_y, cut_rhs, cut_iteration):
    #print(cut_list, multiplicative_y, cut_rhs)
    model.add_constraint(model.sum(cut_list[i] * multiplicative_y[i] for i in range(len(multiplicative_y))) >= cut_rhs[0], ctname = 'cut{}'.format(cut_iteration))
    #model.addMConstr(cut_matrix, multiplicative_y, '>', cut_rhs, name = 'cut_constraint')
    return model

def add_cut_f(model, cut_list, multiplicative_y, cut_rhs, cut_iteration):
    model.add_constraint(model.sum(cut_list[i] * multiplicative_y[i] for i in range(len(multiplicative_y))) >= cut_rhs[0], ctname = 'cut{}'.format(cut_iteration))
    #model.addMConstr(cut_matrix, multiplicative_y, '>', cut_rhs, name = 'cut_constraint')
    return model

def check_infeasibility(model):
    if model.solve_details.status_code == 3:
    #model.getAttr(GRB.Attr.Status) == 3:
        #print("INFEASIBLE !!!!!!!")
        return True   # true means the model is infeasible
    elif model.solve_details.status_code == 5:
        #print("UNBOUNDED!!!!!!!!!!")
        return True
    else:
        return False

def check_for_termination(multiplicative_y):
    for items in multiplicative_y:
        if items == 0 or items <= 0:
            return True         #true will stop the multiplicative programming
        else:
            continue
    return False                # false continues the multiplicative programming

def multiplicative_FP(z, n, p,I,c,d, TT,TL, epsilon, multiplier, previous_y_values, m1, multiplicative_y_m1, f, multiplicative_y_f, objvar):
    nIT_list = []               # number of FP iteration in each cut iteration
    cut_iteration = 0
    time_limit_2 = 1200
    start_time = time.time()
    while True and (time.time() - start_time) < time_limit_2:
        cc =time.time()
        cumulative_time = time.time() - start_time
        # LP relaxation with help of gurobi
        m1, z_lp, x_relaxed, y_values = solve_first_linear_model(m1, n)
        if y_values == 'none':
            delta = 'none'
            decision_variables, algorithm_objective_value, solution_time, nIT = x_relaxed, z_lp, time.time() - start_time, 0
            nIT_list.append(nIT)
            return (
            m1, f, decision_variables, previous_y_values, algorithm_objective_value, delta, solution_time, nIT_list,
            cut_iteration)

        is_it_integer = check_integer(I, x_relaxed)
        if is_it_integer == True:
            delta = 0
            decision_variables, algorithm_objective_value, solution_time, nIT = x_relaxed, z_lp, time.time() - start_time, 0
            nIT_list.append(nIT)

        else:
            #print("got inside FP")

            x_tilde = rounding(x_relaxed, I)
            f, decision_variables, y_values, algorithm_objective_value, delta, solution_time_FP, nIT = feasibility_pump(f,I,c,d,TT,n,
                                                                                                                     objvar,
                                                                                                                     z,
                                                                                                                     p,
                                                                                                                     TL,
                                                                                                                     x_tilde, cut_iteration)

            #print("FP solution time", solution_time_FP)
            solution_time = time.time()- start_time
            nIT_list.append(nIT)
            if decision_variables == 'none':
                return m1, f, decision_variables, previous_y_values, algorithm_objective_value, delta, time.time()-start_time, nIT_list, cut_iteration
            #print("cut_iteration = ", cut_iteration + 1)
            #print(y_values)

        previous_y_values = y_values

        termination = check_for_termination(y_values)
        if termination == True:
            break
        solution_time = time.time() - start_time    

        cut_iteration += 1
        cut_list, cut_rhs = cut_parameters(y_values, p, epsilon, multiplier)

        m1 = add_cut_m(m1, cut_list, multiplicative_y_m1, cut_rhs, cut_iteration)

        f = add_cut_f(f, cut_list, multiplicative_y_f, cut_rhs, cut_iteration)
        #if cut_iteration >= 2:
            #m1.remove_constraint("cut{}".format(cut_iteration-1))
            #f.remove_constraint("cut{}".format(cut_iteration-1))
        #print("time taken for this cut ", time.time() - cc)
        cumulative_time = time.time() - start_time
        #print("cumulative time taken is ", cumulative_time)

        # print("x ",decision_variables)
    return m1, f, decision_variables, previous_y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration

#********

def main_function(A, b, c, m, n, d, p):

#with open("output.csv","w", newline = '') as csv:
    #csv.write("decision_variables, y_values, optimal_value, delta, solution_time, FP_iterations, cut_iterations")

    #read and get data

# parameters
    epsilon, multiplier = MILMMP_parameters()
    I, not_I = get_integer_index(n)
    TL, T, TT, R = FP_parameters(I, m, n)

    y_values = [0 for i in range(p)]    # previous y_values (for starting the algorithm)
    #print("value of n", n)
    aa = time.time()
    m1, multiplicative_y_m1 = first_linear_model(A, b, c, d, n, p, I)
    #print(" first model generated and time taken", time.time() - aa)
    bb = time.time()
    f, objvar, z, multiplicative_y_f = second_model_FP(n, A, b, c, d, p, I)
    #print("second model generated and time taken", time.time() - bb)


    m1, f, decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = multiplicative_FP(z,
        n, p,I,c,d, TT,TL, epsilon, multiplier, y_values, m1, multiplicative_y_m1, f, multiplicative_y_f, objvar)
    #m1.export_as_lp("model1.lp")
    #print("written")
    #f.export_as_lp("model2.lp")

    #final_list = [decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration]

#with open("output.csv", "a", newline='') as csv:
    #csv.write("\n")
    #csv.write(str(final_list))
#print(solution_time)
    return decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration

