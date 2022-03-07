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
	rm -f MMult1 val_test01 val_test02 omp_bug2 omp_bug3 omp_bug4 omp_bug5 omp_bug6

.PHONY: all, clean
