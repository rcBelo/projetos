/*
 * Based on materials from:
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 * https://enccs.github.io/OpenACC-CUDA-beginners/2.02_cuda-heat-equation/
 * changed 23 nov 2022 - vad@fct.unl.pt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mpi.h"

#ifdef PNG
#include "pngwriter.h"
#endif

#define BEGIN 1        // usar para enviar os dados no inicio
#define LEFTREQUEST 2  // usar para comunicar com o worker a esquerda/top
#define RIGHTREQUEST 3 // usar para comunicar com o worker a direita/bottom
#define DONE 4         // usar para enviar os dados no fim


double timedif(struct timespec *t, struct timespec *t0) {
    return (t->tv_sec-t0->tv_sec)+1.0e-9*(double)(t->tv_nsec-t0->tv_nsec);
}

int getIndex(const int i, const int j, const int width)
{
    return i * width + j;
}

void initTemp(float *T, int h, int w)
{
    // Initializing the data with heat from top side
    // all other points at zero
    for (int i = 0; i < w; i++)
    {
        T[i] = 100.0;
    }
}

/* write_pgm - write a PGM image ascii file
 */
void write_pgm(FILE *f, float *img, int width, int height, int maxcolors)
{
    // header
    fprintf(f, "P2\n%d %d %d\n", width, height, maxcolors);
    // data
    for (int l = 0; l < height; l++)
    {
        for (int c = 0; c < width; c++)
        {
            int p = (l * width + c);
            fprintf(f, "%d ", (int)(img[p]));
        }
        putc('\n', f);
    }
}

/* write heat map image
 */
void writeTemp(float *T, int h, int w, int n)
{
    char filename[64];
    //printf("print matriz calor\n");
#ifdef PNG
    sprintf(filename, "heatP_%06d.png", n);
    save_png(T, h, w, filename, 'c');
#else
    sprintf(filename, "heatP_%06d.pgm", n);
    FILE *f = fopen(filename, "w");
    write_pgm(f, T, w, h, 100);
    fclose(f);
#endif
}

int main(int argc, char *argv[])
{

    int taskid, numworkers, numtasks, rows_per_worker, rows, row, offset;
    int error, msgtype;
    int top, bottom;
    int top_send, top_recv, bottom_send, bottom_recv;
    int start, end; // trasverse the matrix in the workers

    MPI_Status status;

    const int nx = atoi(argv[1]); // 200;   // Width of the area
    const int ny = atoi(argv[2]); // 200;   // Height of the area

    const float a = 0.5; // Diffusion constant

    const float h = 0.01; // 0.005;   // h=dx=dy  grid spacing

    const float h2 = h * h;

    const float dt = h2 / (4.0 * a); // Largest stable time step
    const int numSteps = atoi(argv[3]);      // Number of time steps to simulate (time=numSteps*dt)
    const int outputEvery = atoi(argv[4]);   // How frequently to write output image

    int numElements = nx * ny;

    float *Tn = (float *)calloc(numElements, sizeof(float));
    float *Tnp1 = (float *)calloc(numElements, sizeof(float));

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    numworkers = numtasks - 1;

    if ((numworkers > 32) || (numworkers < 0))
    {
        printf("ERROR: the number of tasks must be between %d and %d.\n",
               32, 2);
        printf("Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, error);
        exit(1);
    }

    initTemp(Tn, nx, ny);
    initTemp(Tnp1, nx, ny);



    /*********************************************************************************************************
        MASTER
    *********************************************************************************************************/
    if (taskid == 0){
        printf("Simulated time: %g (%d steps of %g)\n", numSteps*dt, numSteps, dt);
        printf("Simulated surface: %gx%g (in %dx%g divisions)\n", nx*h, ny*h, nx, h);
        writeTemp(Tn, nx, ny, 0);
    }

    rows_per_worker = nx / numtasks;
    int workers_with_extra_row = nx % numtasks;
    int workers_with_no_extra_rows = numtasks - workers_with_extra_row;

    rows = (taskid < workers_with_extra_row) ? rows_per_worker + 1 : rows_per_worker;

    start = (taskid < workers_with_extra_row) ? taskid*rows : taskid*rows + workers_with_extra_row;
    end = start+rows-1;
    offset = start*ny;
    if(start == 0){
        start++;
    }
    if (end == (nx-1)){
        end--;
    }

    top = (taskid == 0) ? -1 : taskid-1;
    bottom = (taskid == numworkers) ? -1 : taskid+1;

    top_send = offset;
    top_recv = offset - ny;
    bottom_send = offset+(rows-1)*ny;
    bottom_recv = offset+rows*ny;
    printf("task= %d   start= %d   end= %d\noffset= %d   rows= %d   top= %d   bottom= %d\n", taskid, start, end, offset, rows, top, bottom);
    
    //timing
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int steps = 0; steps < numSteps; steps++)
    {
        //printf("step: %d\n", steps);

        if(top != -1) {
            MPI_Sendrecv(&Tn[top_send], ny, MPI_FLOAT, top, 123, &Tn[top_recv], ny, MPI_FLOAT, top, 123, MPI_COMM_WORLD, &status);
        }

        if (bottom != -1)
        {
            MPI_Sendrecv(&Tn[bottom_send], ny, MPI_FLOAT, bottom, 123, &Tn[bottom_recv], ny, MPI_FLOAT, bottom, 123, MPI_COMM_WORLD, &status);
        }

        for (int i = start; i <= end; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                const int index = getIndex(i, j, ny);
                //printf("index %d\n", index);
                float tij = Tn[index];
                float tim1j = Tn[getIndex(i - 1, j, ny)];
                float tijm1 = Tn[getIndex(i, j - 1, ny)];
                float tip1j = Tn[getIndex(i + 1, j, ny)];
                float tijp1 = Tn[getIndex(i, j + 1, ny)];
                //Tnp1[index] = index;
                Tnp1[index] = tij + a * dt * ((tim1j + tip1j + tijm1 + tijp1 - 4.0 * tij) / h2);
            }
        }
        float *t = Tn;
        Tn = Tnp1;
        Tnp1 = t;

        if ((steps+1) % outputEvery == 0){
            if(taskid == 0){
                /**Gather all data on worker 0**/
                for (int i = 1; i <= numworkers; i++) {
                    MPI_Recv(&offset, 1, MPI_INT, i, DONE, MPI_COMM_WORLD, &status);
                    MPI_Recv(&rows, 1, MPI_INT, i, DONE, MPI_COMM_WORLD, &status);
                    MPI_Recv(&Tn[offset], rows * ny, MPI_FLOAT, i, DONE, MPI_COMM_WORLD, &status);
                }
                writeTemp(Tn, nx, ny, steps+1);
            }
            if(taskid != 0) {
                /**Send data to worker 0**/
                MPI_Send(&offset, 1, MPI_INT, 0, DONE, MPI_COMM_WORLD);
                MPI_Send(&rows, 1, MPI_INT, 0, DONE, MPI_COMM_WORLD);
                MPI_Send(&Tn[offset], rows * ny, MPI_FLOAT, 0, DONE, MPI_COMM_WORLD);
            }
        }
    }

        if(taskid == 0){
            clock_gettime(CLOCK_MONOTONIC, &t);
            printf("time: %f seconds\n", timedif(&t, &t0) );
        }
        MPI_Finalize();


}
