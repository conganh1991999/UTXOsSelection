import pandas as pd
import math
import operator as op
from functools import reduce


# Function 1:
def valid_output(outs_value, f_dust):
    for x in outs_value:
        if x <= f_dust:
            return False
    return True


# Function 2:
def check_block_size(data, outs_size, f_beta, f_block_size):
    if(sum(x[1] for x in data) + sum(outs_size) + f_beta + 10) > f_block_size:
        return False
    else:
        return True


# Function 3:
def nck(n, k):
    k = min(k, n-k)
    num_er = reduce(op.mul, range(n, n-k, -1), 1)
    de_nom = reduce(op.mul, range(1, k+1), 1)
    return num_er/de_nom


# Function 4:
def highest_value_first(sort_values, k, outs_value, outs_size, optimal_hvf, f_beta, f_dust, f_fee_rate, f_block_size):
    target = sum(outs_value) + f_fee_rate * (sum(outs_size) + sum(x[1] for x in sort_values[:k]) + 10)

    if abs(sum(x[0] for x in sort_values[:k]) - target) < 0.00001:
        if check_block_size(sort_values[:k], outs_size, 0, f_block_size):
            optimal_hvf += sort_values[:k]
            return True

    target = sum(outs_value) + f_fee_rate * (sum(outs_size) + sum(x[1] for x in sort_values[:k]) + 10 + f_beta) + f_dust

    if sum(x[0] for x in sort_values[:k]) > target:
        if check_block_size(sort_values[:k], outs_size, f_beta, f_block_size):
            optimal_hvf += sort_values[:k]
            return True

    return False


# Function 5:
def combinations(sort_sizes, outs_value, outs_size, data, start, end, index, r, optimal_a, optimal_b, f_beta, f_dust, f_fee_rate, f_block_size):
    if index == r:
        if abs(sum(x[0] for x in data) - sum(outs_value) - f_fee_rate * (sum(outs_size) + sum(x[1] for x in data) + 10)) < 0.00001:
            if check_block_size(data, outs_size, 0, f_block_size):
                optimal_a += data
        elif sum(x[0] for x in data) > sum(outs_value) + f_fee_rate * (sum(outs_size) + sum(x[1] for x in data) + 10 + f_beta) + f_dust:
            if check_block_size(data, outs_size, f_beta, f_block_size):
                optimal_b += data
        return

    a = start
    while a <= end and end - a + 1 >= r - index:
        data[index] = sort_sizes[a]
        combinations(sort_sizes, outs_value, outs_size, data, a + 1, end, index + 1, r, optimal_a, optimal_b, f_beta, f_dust, f_fee_rate, f_block_size)
        a += 1


# Optimal Function:
def constraints(input_num, sort_sizes, sort_values, outs_value, outs_size, f_block_size, f_dust, f_fee_rate, f_beta, optimal_a, optimal_b, optimal_hvf):

    for num in range(input_num):
        target = sum(outs_value) + f_fee_rate * (sum(outs_size) + 10)
        if sum(x[0] for x in sort_values[:num + 1]) <= target:
            continue
        else:
            target = sum(outs_value) + f_fee_rate * (sum(outs_size) + sum(x[1] for x in sort_sizes[:num + 1]) + 10)
            if sum(x[0] for x in sort_values[:num + 1]) < target:
                continue
            else:
                if nck(input_num, num + 1) > 1000000:
                    if highest_value_first(sort_values, num + 1, outs_value, outs_size, optimal_hvf, f_beta, f_dust, f_fee_rate, f_block_size):
                        return num + 1
                    else:
                        continue
                else:
                    # initialize
                    data = [[0, 0]] * (num + 1)
                    start = 0
                    end = input_num - 1
                    index = 0
                    # combine
                    combinations(sort_sizes, outs_value, outs_size, data, start, end, index, num + 1, optimal_a, optimal_b, f_beta, f_dust, f_fee_rate, f_block_size)
                    if (optimal_a == []) & (optimal_b == []):
                        continue
                    else:
                        combination_num = num + 1
                        return combination_num
    return False


# Maximize UTXOs Function:
def maximize_choice(opcode, sort_values, sort_sizes, outs_value, outs_size, f_gamma, f_size, combination_num, input_num, optimal_c, optimal_d, optimal_hvf_new, f_beta, f_dust, f_fee_rate, f_block_size):
    num_plus = math.floor((f_gamma * f_size)/148)

    new_n = num_plus + combination_num

    if new_n > input_num:
        new_n = input_num

    # initialize
    data = [[0, 0]] * new_n
    start = 0
    end = input_num - 1
    index = 0

    if opcode == 0:
        tag = combination_num + 2
    else:
        tag = combination_num + 1

    if new_n >= tag:
        if nck(input_num, new_n) > 1000000:
            if highest_value_first(sort_values, new_n, outs_value, outs_size, optimal_hvf_new, f_beta, f_dust, f_fee_rate, f_block_size):
                return new_n
            else:
                return combination_num
        else:
            combinations(sort_sizes, outs_value, outs_size, data, start, end, index, new_n, optimal_c, optimal_d, f_beta, f_dust, f_fee_rate, f_block_size)
            if (optimal_c != []) | (optimal_d != []):
                return new_n
            else:
                return combination_num
    else:
        return combination_num


# Optimal Result Function:
def best_solution(in_optimal, num):
    sizes = []
    cb = int(len(in_optimal)/num)

    for n in range(cb):
        sizes += [sum(x[1] for x in in_optimal[n*num:n*num+num])]

    return in_optimal[(sizes.index(min(sizes)) * num):(sizes.index(min(sizes)) * num + num)]


old_sizes = []
new_sizes = []
change_output = []
md1_choice = []
md2_choice = []
new_sizes_new = []
# Test Data:
for i in range(133):
    path1 = "E:\Downloads\data_out\set" + str(i) + "\input_set.csv"
    path2 = "E:\Downloads\data_out\set" + str(i) + "\output_set.csv"
    path3 = "E:\Downloads\data_out\set" + str(i) + "\const_set.csv"
    input_set = pd.read_csv(path1)
    output_set = pd.read_csv(path2)
    const_set = pd.read_csv(path3)

    inputs_value = list(input_set["value"])
    inputs_size = list(input_set["size"])
    outputs_value = list(output_set["value"])
    outputs_size = list(output_set["size"])

    inputs_table = []
    for inp in range(len(inputs_value)):
        inputs_table += [[inputs_value[inp], inputs_size[inp]]]

    block_size = int(const_set["M"])
    fee_rate = float(const_set["alpha"])
    dust = float(const_set["epsilon"])
    beta = int(const_set["beta"])

    old_sizes += [int(const_set["txsize"])]

    number_of_input = len(inputs_table)
    sorted_sizes = sorted(inputs_table, key=lambda x: x[1], reverse=False)
    sorted_values = sorted(inputs_table, key=lambda x: x[0], reverse=True)
    optimal_A = []
    optimal_B = []
    optimal_C = []
    optimal_D = []
    optimal_HVF = []
    optimal_HVF_new = []
    Size = 0.0
    gamma = 0.1
    new_num = 0

    number_of_combination = constraints(number_of_input, sorted_sizes, sorted_values, outputs_value, outputs_size, block_size, dust, fee_rate, beta, optimal_A, optimal_B, optimal_HVF)
    if valid_output(outputs_value, dust):
        if optimal_A:
            optimal = best_solution(optimal_A, number_of_combination)
            md1_choice += [number_of_combination]
            # Data out
            Size = sum(outputs_size) + sum(x[1] for x in optimal) + 10
            new_sizes += [Size]
            change_output += [0]
            # print("Fee = ", fee_rate*(sum(outputs_size) + sum(x[1] for x in optimal) + 10))
        elif optimal_B:
            optimal = best_solution(optimal_B, number_of_combination)
            md1_choice += [number_of_combination]
            # Data out
            Size = sum(outputs_size) + sum(x[1] for x in optimal) + 10 + beta
            new_sizes += [Size]
            change_output += [sum(x[0] for x in optimal) - sum(outputs_value) - fee_rate*(sum(outputs_size) + sum(x[1] for x in optimal) + 10 + beta)]
            # print("Fee = ", fee_rate*(sum(outputs_size) + sum(x[1] for x in optimal) + 10 + beta))
        elif optimal_HVF:
            md1_choice += [number_of_combination]
            Size = sum(outputs_size) + sum(x[1] for x in optimal_HVF) + 10 + beta
            new_sizes += [Size]
            change_output += [sum(x[0] for x in optimal_HVF) - sum(outputs_value) - fee_rate * (sum(outputs_size) + sum(x[1] for x in optimal_HVF) + 10 + beta)]
            # print("Fee = ", fee_rate * (sum(outputs_size) + sum(x[1] for x in optimal_HVF) + 10 + beta))
        else:
            print("no solution!", i)
            md1_choice += [0]
            new_sizes += [False]
            change_output += [0]

        if gamma > 0:
            if optimal_A:
                new_num = maximize_choice(0, sorted_values, sorted_sizes, outputs_value, outputs_size, gamma, Size, number_of_combination, number_of_input, optimal_C, optimal_D, optimal_HVF_new, beta, dust, fee_rate, block_size)
                md2_choice += [new_num]
            elif optimal_B:
                new_num = maximize_choice(1, sorted_values, sorted_sizes, outputs_value, outputs_size, gamma, Size, number_of_combination, number_of_input, optimal_C, optimal_D, optimal_HVF_new, beta, dust, fee_rate, block_size)
                md2_choice += [new_num]
            elif optimal_HVF:
                new_num = maximize_choice(2, sorted_values, sorted_sizes, outputs_value, outputs_size, gamma, Size, number_of_combination, number_of_input, optimal_C, optimal_D, optimal_HVF_new, beta, dust, fee_rate, block_size)
                md2_choice += [new_num]
            else:
                md2_choice += [0]

            if optimal_C:
                optimal_new = best_solution(optimal_C, new_num)
                # Data out
                Size_new = sum(outputs_size) + sum(x[1] for x in optimal_new) + 10
                new_sizes_new += [Size_new]
                # print("Fee = ", fee_rate*(sum(outputs_size) + sum(x[1] for x in optimal) + 10))
            elif optimal_D:
                optimal_new = best_solution(optimal_D, new_num)
                # Data out
                Size_new = sum(outputs_size) + sum(x[1] for x in optimal_new) + 10 + beta
                new_sizes_new += [Size_new]
                # print("Fee = ", fee_rate*(sum(outputs_size) + sum(x[1] for x in optimal) + 10 + beta))
            elif optimal_HVF_new:
                Size_new = sum(outputs_size) + sum(x[1] for x in optimal_HVF_new) + 10 + beta
                new_sizes_new += [Size_new]
                # print("Fee = ", fee_rate * (sum(outputs_size) + sum(x[1] for x in optimal_HVF) + 10 + beta))
            else:
                new_sizes_new += [Size]
    else:
        print("invalid output set!", i)
        md1_choice += [0]
        md2_choice += [0]
        new_sizes += [False]
        new_sizes_new += [False]
        change_output += [0]

print(sum(old_sizes))
print(sum(new_sizes))
print(sum(new_sizes_new))
print(sum(md1_choice))
print(sum(md2_choice))
