# Confluent Accumulation Cells - reference simulator
#
# -ffp-contract=off is required: fused multiply-add would change the rounding
# of the workload generators and of the training loop, and with it every
# number in the result files.
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -ffp-contract=off -pthread
INC      := -Iinclude
SRC      := src/main.cpp src/tests.cpp src/experiments.cpp
HDR      := $(wildcard include/cac/*.h)

all: cac

cac: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(INC) $(SRC) -o $@

test: cac
	./cac test

results: cac
	mkdir -p results
	./cac all results
	./cac dump results
	python3 -I scripts/crosscheck.py results
	python3 -I scripts/summarize.py results

clean:
	rm -f cac

.PHONY: all test results clean
