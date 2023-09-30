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
#include <cuda.h>

#ifdef PNG
#include "pngwriter.h"
#endif

#define THREADS_BLOCK 8

/* Convert 2D index layout to unrolled 1D layout
 * \param[in] i      Row index
 * \param[in] j      Column index
 * \param[in] width  The width of the area
 * \returns An index in the unrolled 1D array.
 */
int __host__ __device__ getIndex(const int i, const int j, const int width)
{
    return i*width + j;
}

double timedif(struct timespec *t, struct timespec *t0) {
    return (t->tv_sec-t0->tv_sec)+1.0e-9*(double)(t->tv_nsec-t0->tv_nsec);
}

void initTemp(float *T, int h, int w) {
    // Initializing the data with heat from top side
    // all other points at zero
    for (int i = 0; i < w; i++)
    {
        T[i] = 100.0;
    }
}

/* write_pgm - write a PGM image ascii file
 */
void write_pgm(FILE *f, float *img, int width, int height, int maxcolors) {
    // header
    fprintf(f, "P2\n%d %d %d\n", width, height, maxcolors);
    // data
    for (int l = 0; l < height; l++) {
        for (int c = 0; c < width; c++) {
            int p = (l * width + c);
            fprintf(f, "%d ", (int)(img[p]));
        }
        putc('\n', f);
    }
}


/* write heat map image
*/
void writeTemp(float *T, int h, int w, int n) {
    char filename[64];
#ifdef PNG
    sprintf(filename, "P_heat_%06d.png", n);
    save_png(T, h, w, filename, 'c');
#else
    sprintf(filename, "P_heat_%06d.pgm", n);
    FILE *f=fopen(filename, "w");
    write_pgm(f, T, w, h, 100);
    fclose(f);
#endif
}

__global__ void computeHeatShared(float *T, float *Tp, int nx, int  ny, float a, float dt, float h2)
{
    __shared__ float s_T[(THREADS_BLOCK + 2)*(THREADS_BLOCK + 2)];
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    int col = threadIdx.y + blockIdx.y*blockDim.y;

    int s_row = threadIdx.x + 1;
    int s_col = threadIdx.y + 1;
    int s_ny = THREADS_BLOCK + 2;

    // Load data into shared memory
    // Central square
    s_T[getIndex(s_row, s_col, s_ny)] = T[getIndex(row, col, ny)];
    // Top border
    if (s_row == 1 && row != 0)
    {
        s_T[getIndex(0, s_col, s_ny)] = T[getIndex(blockIdx.x*blockDim.x - 1, col, ny)];
    }
    // Bottom border
    if (s_row == THREADS_BLOCK && row != nx - 1)
    {
        s_T[getIndex(THREADS_BLOCK + 1, s_col, s_ny)] = T[getIndex((blockIdx.x + 1)*blockDim.x, col, ny)];
    }
    // Left border
    if ( s_col == 1 && col != 0)
    {
        s_T[getIndex(s_row, 0, s_ny)] = T[getIndex(row, blockIdx.y*blockDim.y - 1, ny)];
    }
    // Right border
    if (s_col == THREADS_BLOCK && col != ny - 1)
    {
        s_T[getIndex(s_row, THREADS_BLOCK + 1, s_ny)] = T[getIndex(row, (blockIdx.y + 1)*blockDim.y, ny)];
    }

    // Make sure all the data is loaded before computing
    __syncthreads();

    if(col < nx-1 && row < ny-1 && col > 0 && row > 0)
        {
            float tij = s_T[getIndex(s_row, s_col, s_ny)];
            float tim1j = s_T[getIndex(s_row-1, s_col, s_ny)];
            float tijm1 = s_T[getIndex(s_row, s_col-1, s_ny)];
            float tip1j = s_T[getIndex(s_row+1, s_col, s_ny)];
            float tijp1 = s_T[getIndex(s_row, s_col+1, s_ny)];

            // Explicit scheme
            Tp[getIndex(row,col,ny)] = tij + a * dt * ( (tim1j + tip1j + tijm1 + tijp1 - 4.0*tij)/h2 );

    }
}

__global__ void computeHeat(float *T, float *Tp, int nx, int  ny, float a, float dt, float h2) {
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    int col = threadIdx.y + blockIdx.y*blockDim.y;

    int index = row*ny + col;
    if(col < nx-1 && row < ny-1 && col > 0 && row > 0){
       	float tij = T[index];
	float tim1j = T[(row-1)*ny + col];
	float tijm1 = T[row*ny + (col-1)];
	float tip1j = T[(row + 1)*ny + col];
	float tijp1 = T[row*ny + (col+1)];
	Tp[index] = tij + a * dt * ( (tim1j + tip1j + tijm1 + tijp1 - 4.0*tij)/h2 );
    }


}



int main(int argc, char *argv[])
{

    int shared = atoi(argv[1]);
    const int nx = 200; // 200;   // Width of the area
    const int ny = 200; // 200;   // Height of the area

    const float a = 0.5;     // Diffusion constant

    const float h = 0.005; // 0.005;   // h=dx=dy  grid spacing

    const float h2 = h*h;

    const float dt =  h2 / (4.0 * a); // Largest stable time step
    const int numSteps = 100000;      // Number of time steps to simulate (time=numSteps*dt)
    const int outputEvery = 10000;   // How frequently to write output image

    int numElements = nx*ny;

    // Allocate two sets of data for current and next timesteps
    float* Tn   = (float*)malloc(numElements * sizeof(float));

    // Initializing the data for T0
    initTemp(Tn, nx, ny);

    // Fill in the data on the next step to ensure that the boundaries are identical.

    printf("Simulated time: %g (%d steps of %g)\n", numSteps*dt, numSteps, dt);
    printf("Simulated surface: %gx%g (in %dx%g divisions)\n", nx*h, ny*h, nx, h);
    writeTemp(Tn, nx, ny, 0);

    dim3 dimBlock(THREADS_BLOCK,THREADS_BLOCK,1); // 32*32THREADS_BLOCK);
    dim3 dimGrid(((nx+dimBlock.x-1)/dimBlock.x), ((ny+dimBlock.y-1)/dimBlock.y),1);

    float *cuda_tn;
    cudaMalloc(&cuda_tn, numElements*sizeof(float));
    float *cuda_tnp1;
    cudaMalloc(&cuda_tnp1, numElements*sizeof(float));

    if ( cuda_tn==NULL || cuda_tnp1==NULL ) {
        fprintf(stderr,"No GPU mem!\n");
        return EXIT_FAILURE;
    }

    //timing
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);

  cudaMemcpy(cuda_tnp1, Tn, numElements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_tn, Tn, numElements*sizeof(float),cudaMemcpyHostToDevice);

    // Main loop
    for (int n = 0; n <= numSteps; n++)
    {
        if(shared){
            computeHeatShared<<<dimGrid, dimBlock>>>(cuda_tn, cuda_tnp1, nx,ny,a,dt,h2);
        }else{
            computeHeat<<<dimGrid, dimBlock>>>(cuda_tn, cuda_tnp1, nx,ny,a,dt,h2);
        }
        
        // Write the output if needed
        if ((n+1) % outputEvery == 0) {
        cudaMemcpy(Tn, cuda_tnp1, numElements * sizeof(float), cudaMemcpyDeviceToHost);
            cudaError_t err=cudaGetLastError();
            if (err!=cudaSuccess) {
                fprintf(stderr, "err=%u %s\n%s\n", (unsigned) err, cudaGetErrorString(err),
                        "Problems executing kernel");
                exit(1);
            }
            writeTemp(Tn, nx, ny, n + 1);
        }

        // Swapping the pointers for the next timestep
        float* t = cuda_tn;
        cuda_tn = cuda_tnp1;
        cuda_tnp1 = t;
    }

    // Timing
    clock_gettime(CLOCK_MONOTONIC, &t);
    printf("time: %f seconds\n", timedif(&t, &t0) );

    // Release the memory
    free(Tn);
    cudaFree(cuda_tn);
    cudaFree(cuda_tnp1);

    return 0;
}
