
# if using PNG files:
#PNGCONF=/usr/bin/libpng-config
ifdef PNGCONF
CFLAGS=-DPNG $(shell $(PNGCONF) --cflags)
LDFLAGS=$(shell $(PNGCONF) --ldflags)
OBJ=pngwriter.c
else

endif

all:    main cuda streams

main:	main.c $(OBJ)
	cc -o main $(CFLAGS) $< $(OBJ) $(LDFLAGS)

cuda:	main_par.cu $(OBJ)
	nvcc -o cuda $(CFLAGS) $< $(OBJ) $(LDFLAGS)

streams:	streams.cu $(OBJ)
	nvcc -o streams $(CFLAGS) $< $(OBJ) $(LDFLAGS)

clean:
	rm -f main cuda streams *.o
