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

int getIndex(const int i, const int j, const int width)
{
    return i * width + j;
}

double timedif(struct timespec *t, struct timespec *t0) {
    return (t->tv_sec-t0->tv_sec)+1.0e-9*(double)(t->tv_nsec-t0->tv_nsec);
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
    //printf("print matriz calor at %d\n", n);
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

    int taskid, numworkers, numtasks, rows_per_worker, rows, row, offset, g_rows, g_offset;
    int error, msgtype;
    int top, bottom;
    int start, end; // trasverse the matrix in the workers

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

    MPI_Status t_status[2];
    MPI_Status b_status[2];
    MPI_Status statussss;
    MPI_Request t_reqs[2];
    MPI_Request b_reqs[2];

    MPI_Status output_status[3];
    MPI_Request output_reqs[3];

    MPI_Status g_status[numworkers-1];
    MPI_Request g_reqs[numworkers-1];



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

    if (taskid == numworkers) {
        printf("Simulated time: %g (%d steps of %g)\n", numSteps * dt, numSteps, dt);
        printf("Simulated surface: %gx%g (in %dx%g divisions)\n", nx * h, ny * h, nx, h);
        writeTemp(Tn, nx, ny, 0);
    }
    /*********************************************************************************************************
        workers
    *********************************************************************************************************/

    if(taskid != numworkers) {
        rows_per_worker = nx / numworkers;
        int workers_with_extra_row = nx % numworkers;
        int workers_with_no_extra_rows = numworkers - workers_with_no_extra_rows;

        rows = (taskid < workers_with_extra_row) ? rows_per_worker + 1 : rows_per_worker;

        start = (taskid < workers_with_extra_row) ? taskid*rows : taskid*rows + workers_with_extra_row;
        end = start + rows - 1;
        offset = start * ny;

        if (start == 0) {
            start++;
        }
        if (end == (nx - 1)) {
            end--;
        }

        top = (taskid == 0) ? -1 : taskid - 1;
        bottom = (taskid == (numworkers-1)) ? -1 : taskid + 1;

        printf("task= %d   start= %d   end= %d\noffset= %d   rows= %d   top= %d   bottom= %d\n", taskid, start, end, offset, rows, top, bottom);



        for (int steps = 0; steps < numSteps; steps++) {
            //printf("task %d step: %d\n",taskid, steps);

            if (top != -1) {
                MPI_Isend(&Tn[offset], ny, MPI_FLOAT, top, LEFTREQUEST, MPI_COMM_WORLD, &t_reqs[0]);
                MPI_Irecv(&Tn[offset - ny], ny, MPI_FLOAT, top, RIGHTREQUEST, MPI_COMM_WORLD, &t_reqs[1]);
            }

            if (bottom != -1) {
                MPI_Isend(&Tn[offset + (rows - 1) * ny], ny, MPI_FLOAT, bottom, RIGHTREQUEST, MPI_COMM_WORLD,
                          &b_reqs[0]);
                MPI_Irecv(&Tn[offset + rows * ny], ny, MPI_FLOAT, bottom, LEFTREQUEST, MPI_COMM_WORLD, &b_reqs[1]);
            }

            if (top != -1) {
                MPI_Waitall(2, t_reqs, b_status);
            }

            if (bottom != -1) {
                MPI_Waitall(2, b_reqs, t_status);
            }


            for (int i = start; i <= end; i++) {
                for (int j = 1; j < ny - 1; j++) {
                    const int index = getIndex(i, j, ny);
                    float tij = Tn[index];
                    float tim1j = Tn[getIndex(i - 1, j, ny)];
                    float tijm1 = Tn[getIndex(i, j - 1, ny)];
                    float tip1j = Tn[getIndex(i + 1, j, ny)];
                    float tijp1 = Tn[getIndex(i, j + 1, ny)];
                    Tnp1[index] = tij + a * dt * ((tim1j + tip1j + tijm1 + tijp1 - 4.0 * tij) / h2);
                }
            }

            float *t = Tn;
            Tn = Tnp1;
            Tnp1 = t;

            if ((steps + 1) % outputEvery == 0) {
                    /**Send data to last node**/
                    MPI_Isend(&offset, 1, MPI_INT, numworkers, DONE, MPI_COMM_WORLD, &output_reqs[0]);
                    MPI_Isend(&rows, 1, MPI_INT, numworkers, DONE, MPI_COMM_WORLD, &output_reqs[1]);
                    MPI_Isend(&Tn[offset], rows * ny, MPI_FLOAT, numworkers, DONE, MPI_COMM_WORLD, &output_reqs[2]);
                    MPI_Waitall(3, output_reqs, output_status);

            }
        }
    }

    if(taskid == numworkers){
        /**Gather all data on last worker**/
        struct timespec t0, t;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int steps = 0; steps < numSteps; steps++) {

            if ((steps + 1) % outputEvery == 0) {
                for (int i = 0; i < numworkers; i++) {
                    MPI_Recv(&g_offset, 1, MPI_INT, i, DONE, MPI_COMM_WORLD, &statussss);
                    MPI_Recv(&g_rows, 1, MPI_INT, i, DONE, MPI_COMM_WORLD, &statussss);
                    MPI_Irecv(&Tn[g_offset], g_rows * ny, MPI_FLOAT, i, DONE, MPI_COMM_WORLD, &g_reqs[i]);
                }
                MPI_Waitall(numworkers-1, g_reqs, g_status);
                writeTemp(Tn, nx, ny, steps + 1);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &t);
        printf("time: %f seconds\n", timedif(&t, &t0) );
    }
    MPI_Finalize();
}
