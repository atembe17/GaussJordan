#include "GJCommon.cuh"
#include<stdio.h>
#include <cstdlib> // malloc(), free()
#include <ctime>
#include <cmath>

//Size of vector
const int SIZE = 64;
//No of iters
const int ITERS = 1;

int main()
{
	clock_t start, end;
	float timeCpu, timeGpu;
	int rows,cols;
	rows = SIZE;
	//No of columns = No rows + 1
	cols = SIZE + 1;
	printf("Operating on a %d x %d matrix\n", SIZE, SIZE);
	//Memory allocation for all variables
	float** a = new float* [rows];
	float** c_cpu = new float* [rows];
	float** c_gpu = new float* [rows];
	//Allocate memory for column unit for each variable  
	for (int i = 0; i < rows; i++) {
		a[i] = new float[cols];
		c_cpu[i] = new float[cols];
		c_gpu[i] = new float[cols];
	}
	//Initialize random values between 0 and 1
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			a[i][j] = (float)rand() / (RAND_MAX);
		}
	}
	//Start Clock
	start = clock();  
	for (int i = 0; i < ITERS; i++) { 
		//Invoke the CPU method
		GaussianEliminationCPU(a, rows, cols, c_cpu, false);
	}
	//End clock
	end = clock();  
	timeCpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Host result (direct) took %f ms\n",timeCpu);
	//Trial pass
	bool success = GaussianEliminationGPU(a, rows, cols, c_gpu, false);
	if (!success) { 
		printf("\nDevice Error!!");
		return 1; 
	}
	//Start clock
	start = clock();  for (int i = 0; i < ITERS; i++) { 
		//Invoke the GPU method
		GaussianEliminationGPU(a, rows, cols, c_gpu, false);
	}  
	//End clock
	end = clock();  
	//Compute the time for GPU
	timeGpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Device result (direct) took %f ms\n", timeGpu);

	// Find the L2 norm error for cpu and gpu values 
	float sum = 0, delta = 0;  
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			delta += (c_cpu[i][j] - c_gpu[i][j]) * (c_cpu[i][j] - c_gpu[i][j]);
			sum += (c_cpu[i][j] * c_gpu[i][j]);
		}
	}
	float L2norm = sqrt(delta / sum);
	printf("Error: %f\n", L2norm);
}