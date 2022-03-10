HEADERS = utils.h

CC = g++

CFLAGS = -Wall -g

CXXFLAGS = -O3 -std=c++11

CXXFLAGS2 = -std=c++11 -Wall -fopenmp -O3

TARGETS: $(basename $(wildcard *.cpp)) $(basename $(wildcard *.c))

all: $(TARGETS)

%:%.cpp *.h
	$(CC) $(CXXFLAGS) $< $(LIBS) -o $@

%:%.c
	$(CC) $(CXXFLAGS2) $< $(LIBS) -o $@


clean:
	rm -f MMult1 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 val_test01_solved val_test02_solved

.PHONY: all, clean
