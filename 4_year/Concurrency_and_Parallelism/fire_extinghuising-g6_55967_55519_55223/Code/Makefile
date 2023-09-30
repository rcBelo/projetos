#
# Agent-based Simulation of Fire Extinguishing
#
# Parallel computing (Degree in Computer Engineering)
# 2018/2019
#
# EduHPC 2019: Peachy assignment
#
# (c) 2019 Arturo Gonzalez-Escribano, Jorge Fernandez-Fabeiro
# Grupo Trasgo, Universidad de Valladolid (Spain)
#
# This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-sa/4.0/
#
#
# The current Parallel Computing course includes contests using:
# OpenMP
#

# Compilers
CC=gcc
OMPFLAG=-fopenmp

# Flags for optimization and libs
CFLAGS=-O3
LIBS=-lm

# Deal with macOS specificities
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
ifneq ($(CC),gcc11)
	OMPFLAG:=-Xclang $(OMPFLAG)
	LIBS:=$(LIBS) -lomp
endif
endif



# Targets to build
OBJS=extinguishing_seq extinguishing_omp extinguishing_omp_report

# Rules. By default show help
help:
	@echo
	@echo "Agent-based Simulation of Fire Extinguishing"
	@echo
	@echo "Group Trasgo, Universidad de Valladolid (Spain)"
	@echo "EduHPC 2019: Peachy assignment"
	@echo
	@echo "make extinguishing_seq	Build only the sequential version"
	@echo "make extinguishing_omp	Build only the OpenMP version"
	@echo
	@echo "make all	Build all versions (Sequential, OpenMP)"
	@echo "make debug	Build all version with demo output for small surfaces)"
	@echo "make clean	Remove targets"
	@echo

all: $(OBJS)

extinguishing_seq: extinguishing.c
	$(CC) $(CFLAGS) $(DEBUG) $< $(LIBS) -o $@

extinguishing_omp: extinguishing_omp.c
	$(CC) $(CFLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@

extinguishing_omp_report: extinguishing_omp_report.c
	$(CC) $(CFLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@

# Remove the target files
clean:
	rm -rf $(OBJS)

# Compile in debug mode
debug:
	make DEBUG=-DDEBUG all
