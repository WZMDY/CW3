# CW3
This is a Sudoku solver script that supports two solving algorithms: the backtracking algorithm and the wavefront propagation algorithm. The script can read Sudoku grids from files and save solutions (or partially filled grids with a specific number of hints) to output files. Additionally, it can provide step-by-step instructions if the `--explain` flag is used, and it can profile the solver performance.

To use the script, you can run it with different command-line arguments. Here are some examples:

1. Solve a Sudoku puzzle from an input file and save the solution to an output file:
```
python CourseWork3.py -f input_file.txt output_file.txt
```

2. Provide step-by-step instructions for solving the puzzle:
```
python CourseWork3.py -f input_file.txt output_file.txt -e
```

3. Solve the Sudoku puzzle and return a grid with a specific number of hints:
```
python CourseWork3 -f input_file.txt output_file.txt -n 20
```

4. Profile the solver's performance:
```
python CourseWork3.py -p
```

5. Use the wavefront propagation algorithm:
```
python CourseWork3.py -f input_file.txt output_file.txt -w
```

Replace `input_file.txt`, and `output_file.txt` with the appropriate filenames for your use case.
