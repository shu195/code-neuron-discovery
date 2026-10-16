CXX ?= g++
CXXFLAGS ?= -O3 -mavx2 -march=native -std=c++17 -Wall -Wextra

all: math_solver benchmark_100

math_solver: src/math_solver_cli.cpp
	$(CXX) $(CXXFLAGS) src/math_solver_cli.cpp -o math_solver

benchmark_100: src/benchmark_100.cpp
	$(CXX) $(CXXFLAGS) src/benchmark_100.cpp -o benchmark_100

clean:
	rm -f math_solver math_solver.exe benchmark_100 benchmark_100.exe

.PHONY: all clean
