
# if using PNG files:
#PNGCONF=/usr/bin/libpng-config
ifdef PNGCONF
CFLAGS=-DPNG $(shell $(PNGCONF) --cflags)
LDFLAGS=$(shell $(PNGCONF) --ldflags)
OBJ=pngwriter.c
else

endif

all:    heat_s heat_p heat_p_old heat_i heat_i_g

heat_s:	heat_seq.c $(OBJ)
	cc -o heat_s $(CFLAGS) $< $(OBJ) $(LDFLAGS)

heat_p:	heat_sendrecv.c $(OBJ)
	mpicc -o heat_p $(CFLAGS) $< $(OBJ) $(LDFLAGS)

heat_i:	heat_i.c $(OBJ)
	mpicc -o heat_i $(CFLAGS) $< $(OBJ) $(LDFLAGS)

heat_p_old: heat_mpi.c $(OBJ)
	mpicc -o heat_p_old $(CFLAGS) $< $(OBJ) $(LDFLAGS)

heat_i_g: heat_i_gather.c
	mpicc -o heat_i_g $(CFLAGS) $< $(OBJ) $(LDFLAGS)

clean:
	rm -f heat_s heat_p heat_i heat_p_old heat_i_g *.o
