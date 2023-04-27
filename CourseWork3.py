#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 04:14:15 2023

@author: dylan
"""

import argparse
import time
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

# Reads a Sudoku grid from a file and returns it as a list of lists.
def read_grid_from_file(filename: str) -> List[List[int]]:
    grid = []
    with open(filename, "r") as file:
        for line in file.readlines():
            row = [int(x.replace(",", "")) for x in line.strip().split()]
            grid.append(row)
    return grid

# Writes a Sudoku grid and optional instructions to a file.
def write_grid_to_file(filename: str, grid: List[List[int]], instructions: Optional[List[str]] = None) -> None:
    with open(filename, "w") as file:
        for row in grid:
            file.write(" ".join(str(x) for x in row) + "\n")
        if instructions:
            file.write("\nInstructions:\n")
            for instr in instructions:
                file.write(instr + "\n")

def write_no_solution_to_file(file_path: str):
    with open(file_path, "w") as file:
        file.write("No solution found\n")

# Finds an empty cell with the least number of viable options in the grid and returns its row and column as a tuple.
def find_empty_cell(grid: List[List[int]]) -> Tuple[int, int]:
    min_options = float('inf')
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                options = get_viable_options(grid, i, j)
                if len(options) < min_options:
                    min_options = len(options)
                    empty_cell = (i, j)
    try:
        return empty_cell
    except UnboundLocalError:
        return None

# Returns a list of viable options for a cell at the specified row and column in the grid.
def get_viable_options(grid: List[List[int]], row: int, col: int) -> List[int]:
    options = set(range(1, len(grid)+1))
    for i in range(len(grid)):
        if grid[i][col] in options:
            options.remove(grid[i][col])
        if grid[row][i] in options:
            options.remove(grid[row][i])
    row_start = (row // int(len(grid)**0.5)) * int(len(grid)**0.5)
    col_start = (col // int(len(grid)**0.5)) * int(len(grid)**0.5)
    for i in range(row_start, row_start + int(len(grid)**0.5)):
        for j in range(col_start, col_start + int(len(grid)**0.5)):
            if grid[i][j] in options:
                options.remove(grid[i][j])
    return list(options)

# Solves the Sudoku puzzle using the backtracking algorithm and returns a tuple with a boolean indicating if the puzzle is solved and a list of instructions if the explain flag is set to True.
def solve_sudoku(grid: List[List[int]], explain: bool = False) -> Tuple[bool, List[str]]:
    empty_cell = find_empty_cell(grid)
    if empty_cell is None:
        return True, []

    row, col = empty_cell
    options = get_viable_options(grid, row, col)
    instructions = []

    for num in options:
        grid[row][col] = num
        if explain:
            instructions.append(f"Put {num} in location ({row}, {col})")

        solved, child_instructions = solve_sudoku(grid, explain=explain)
        if solved:
            if explain:
                instructions.extend(child_instructions)
            return True, instructions

        grid[row][col] = 0

    return False, []

# Wave functions
def propagate(grid: List[List[int]], entropy_map:dict[tuple,list], number, row, col) -> Tuple[bool, dict[tuple,list], dict[tuple,list]]:
    # Associated grids
    po_set = set()
    # Entropy to be zeroed
    empty_map = dict()
    empty_map[(row, col)] = number
    for i in range(len(grid)):
        po_set.add((i,col))
        po_set.add((row,i))
    row_start = (row // int(len(grid)**0.5)) * int(len(grid)**0.5)
    col_start = (col // int(len(grid)**0.5)) * int(len(grid)**0.5)
    for i in range(row_start, row_start + int(len(grid)**0.5)):
        for j in range(col_start, col_start + int(len(grid)**0.5)):
            po_set.add((i,j))
    po_set.remove((row, col))
    # Associated grids
    for item in po_set:
        # Determine if the entropy of the associated lattice is 0
        if item in entropy_map:
            # Not 0, remove number from the grid entropy array
            if number in entropy_map[item]:
                entropy_map[item].remove(number)
                if len(entropy_map[item]) == 0:
                    # The entropy is 0 after deletion, indicating that there is a rule conflict for this operation and an error is returned
                    return (False, entropy_map, empty_map)
                # The entropy is 1 after deletion, and this entropy is updated recursively
                if len(entropy_map[item]) == 1:
                    state, child_entropy_map, child_empty_map = propagate(grid,entropy_map,entropy_map[item][0],item[0],item[1])
                    if state:
                        empty_map.update(child_empty_map)
                    else:
                        return (False, entropy_map, empty_map)
    return (True, entropy_map, empty_map)

# Get the entropy table
def get_entropy_map(grid: List[List[int]]):
    entropy_map = dict()
    min_entropy = tuple()
    min_options = float('inf')
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                options = get_viable_options(grid, i, j)
                entropy_map[(i,j)] = options
                if len(options) < min_options:
                    min_options = len(options)
                    min_entropy = (i, j)
    return entropy_map, min_entropy

# Solves the Sudoku puzzle using the wavefront propagation algorithm and returns a tuple with a boolean indicating if the puzzle is solved and a list of instructions if the explain flag is set to True.
def solve_sudoku_wavefront_propagation(grid: List[List[int]], explain: bool = False) -> Tuple[bool, List[str]]:
    # Get the entropy table and get the minimum entropy
    entropy_map, min_entropy = get_entropy_map(grid)
    if len(entropy_map.items()) == 0:
        return True, [] 

    instructions = []
    for num in entropy_map[min_entropy]:
        row, col = min_entropy
        # Perform wave function and update entropy table
        state, child_entropy_map, child_empty_map = propagate(grid,copy.deepcopy(entropy_map), num, row, col)
        # Updating entropy does not conflict with other entropy updategrid
        if state:
            for empty_cell, empty_num in child_empty_map.items():
                empty_row, empty_col = empty_cell
                del child_entropy_map[empty_cell]
                grid[empty_row][empty_col] = empty_num
                if explain:
                    instructions.append(f"Put {empty_num} in location ({empty_row}, {empty_col})")
            if len(child_entropy_map.items()) == 0:
                return True, instructions 
            
            # There is also an unzeroed entropy, recursively solved for
            solved, child_instructions = solve_sudoku_wavefront_propagation(grid,explain)
            if solved:
                if explain:
                    instructions.extend(child_instructions)
                return True, instructions
            # Solving errors, backtracking
            for empty_cell, empty_num in child_empty_map.items():
                empty_row, empty_col = empty_cell
                grid[empty_row][empty_col] = 0

    return False, []

#Solves the Sudoku puzzle and returns a grid with a specific number of hints instead of a full solution, along with optional instructions.
def solve_sudoku_with_hint(grid: List[List[int]], hint_count: int, explain: bool = False, wp: bool = False) -> Tuple[List[List[int]], List[str]]:
    solved, instructions = [], []
    if wp:
        solved, instructions = solve_sudoku_wavefront_propagation(grid, explain=explain)
    else:
        solved, instructions = solve_sudoku(grid, explain=explain)

    if not solved:
        return None, None

    filled_positions = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j] != 0]
    random.shuffle(filled_positions)

    while len(filled_positions) > hint_count:
        i, j = filled_positions.pop()
        grid[i][j] = 0
        if explain:
            instructions.append(f"Remove the value at position ({i}, {j})")

    return grid, instructions


def main():
    parser = argparse.ArgumentParser(description="Solve Sudoku puzzles")
    parser.add_argument("-e", "--explain", action="store_true", help="Provide step-by-step instructions")
    parser.add_argument("-f", "--file", nargs=2, metavar=("INPUT", "OUTPUT"), help="Read a Sudoku grid from INPUT file and save the solution to OUTPUT file")
    parser.add_argument("-n", "--hint", type=int, metavar="N", help="Provide a grid with N filled values instead of a full solution")
    parser.add_argument("-p", "--profile", action="store_true", help="Profile the solver performance")
    parser.add_argument("-w", "--wavefront", action="store_true", help="Using the Wavefront propagation algorithm")
    args = parser.parse_args()

    if args.profile:
        profile_solver()
        return

    if args.file:
        input_file, output_file = args.file
        grid = read_grid_from_file(input_file)

        if args.hint:
            grid, instructions = solve_sudoku_with_hint(grid, args.hint, explain=args.explain, wp=args.wavefront)
            if grid is None:
                write_no_solution_to_file(output_file)
            else:
                write_grid_to_file(output_file, grid, instructions)
        else:
            solved, instructions = [], []
            if args.wavefront:
                solved, instructions = solve_sudoku_wavefront_propagation(grid, explain=args.explain)
            else:
                solved, instructions = solve_sudoku(grid, explain=args.explain)
            if not solved:
                write_no_solution_to_file(output_file)
            else:
                write_grid_to_file(output_file, grid, instructions)
    else:
        print("Please provide a Sudoku grid or use the -f/--file flag to specify input and output files.")

def profile_solver():
    sizes = [(2, 2), (3, 2), (3, 3)]
    difficulties = [0.3, 0.5, 0.75]
    trials = 10
    results = {}
    wp_results = {}

    for size in sizes:
        for difficulty in difficulties:
            total_time = 0
            wp_total_time = 0
            for _ in range(trials):
                grid = generate_random_sudoku(size, difficulty)

                wp_start_time = time.time()
                solve_sudoku_wavefront_propagation(copy.deepcopy(grid))
                wp_end_time = time.time()
                wp_total_time += round((wp_end_time - wp_start_time)*1000, 2)

                start_time = time.time()
                solve_sudoku(copy.deepcopy(grid))
                end_time = time.time()
                total_time += round((end_time - start_time)*1000, 2)


            average_time = (total_time / trials) 
            results[(size, difficulty)] = average_time

            average_time = (wp_total_time / trials)
            wp_results[(size, difficulty)] = average_time

    plot_profile_results(results,wp_results)

def generate_random_sudoku(size: Tuple[int, int], difficulty: float) -> List[List[int]]:
    rows, cols = size
    grid_size = rows * cols
    grid = [[0] * grid_size for _ in range(grid_size)]
    empty = grid_size ** 2 - int(grid_size ** 2 * (1 - difficulty))
    all_num = [i for i in range(1,grid_size+1)]
    for col in range(cols):
        num = np.random.choice(all_num)
        grid[0][col] = num
        all_num.remove(num)

    solve_sudoku_wavefront_propagation(grid)
    for _ in range(empty):
        i, j = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        grid[i][j] = 0

    return grid

def plot_profile_results(results,wp_results):

    sizes = [(2, 2), (3, 2), (3, 3)]
    x_str_list = ["(2, 2)", "(3, 2)", "(3, 3)"]
    difficulties = [0.3, 0.5, 0.75]

    figure,(ax1,ax2) = plt.subplots(1,2,sharey=True)
    x = np.arange(len(sizes))
    for index,dif in enumerate(difficulties):
        plt_y = []
        for size in sizes:
            plt_y.append(results[(size, dif)])
        ax1.bar(x+index*0.3,plt_y,width=0.3,label=("Empty:"+str(dif)+"%"))
        
    
    for index,dif in enumerate(difficulties):
        plt_y = []
        for size in sizes:
            plt_y.append(wp_results[(size, dif)])
        ax2.bar(x+index*0.3,plt_y,width=0.3,label=("Empty:"+str(dif)+"%"))
    ax1.legend()
    ax2.legend()
    ax1.set_xticks([r + 0.3 for r in range(len(x))])
    ax1.set_xticklabels(x_str_list)
    
    ax1.set_title("Sudoku Solver Performance")
    ax1.set_ylabel("Time(ms)")
    ax1.set_xlabel("Size")


    ax2.set_xticks([r + 0.3 for r in range(len(x))])
    ax2.set_xticklabels(x_str_list)
    ax2.set_title("Sudoku Solver (WP) Performance")
    ax2.set_xlabel("Size")

    figure.subplots_adjust(wspace=0.1) 
    plt.ioff()
    plt.show()

main()
