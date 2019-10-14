#include <cuda.h>
#include <cuda_runtime_api.h>
#include<stdio.h>
#include<math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void ErrorCheck(cudaError_t);
//Tile width set according to device 
const int TILE_WIDTH = 32;

//Kernel method to scale diagonal elements to unit value
__global__ void ScaleKernel(float* inpMatrix, unsigned int numberOfRows, unsigned int numberOfColumns, float* outMatrix, int row_consider) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	//Check whether row and col does not exceed the provided values
	if (row < numberOfRows && col < numberOfColumns) {
		//Condition to check if the element is on the diagonal 
		if (row == row_consider && col == row_consider) {
			// If diagonal, divide it by itself
			outMatrix[row_consider * numberOfColumns + row_consider] = inpMatrix[row_consider * numberOfColumns + row_consider]/inpMatrix[row_consider * numberOfColumns + row_consider];
		}
		// Condition to check the row to be considered
		else if (row == row_consider && col != row_consider) {
			//Divide each element of considered row by the diagonal element in that row
			outMatrix[row * numberOfColumns + col] = inpMatrix[row * numberOfColumns + col] / inpMatrix[row_consider*numberOfColumns + row_consider];
		}
	}
}

//Kernel Method to subtract each non diagonal value to make it zero
__global__ void SubtractKernel(float* inpMatrix, unsigned int numberOfRows, unsigned int numberOfColumns, float* outMatrix, int row_consider) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	//Check whether row and col does not exceed the provided values
	if (row < numberOfRows && col < numberOfColumns) {
		if (row!=row_consider) {
			outMatrix[row * numberOfColumns + col] = inpMatrix[row * numberOfColumns + col] - (inpMatrix[row*numberOfColumns + row_consider] * inpMatrix[row_consider*numberOfColumns + col]);
		}
	}
}

//Method to allocate memory and compute the gauss jordan elimination
bool GaussianEliminationGPU(float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot) {
	//Variable of type cudaError_t to store error status
	cudaError_t status;
	//Cuda memory allocation to store the input matrix
	float* Md;
	//Cuda memory allocation for computing matrix on device
	float* Rd;
	//storage in bytes
	int bytes = numberOfRows * numberOfColumns * sizeof(float);
	// Memory allocation
	cudaMalloc((void**)&Md, bytes);
	cudaMalloc((void**)&Rd, bytes);
	//Check for error
	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		ErrorCheck(status);
		cudaFree(Md);
		cudaFree(Rd);
		return false;
	}
	//Specifying the cuda block size to tile size
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	//Specifying the no of grids
	int grid1 = (int)ceil((float)numberOfColumns / (float)TILE_WIDTH);
	int grid2 = (int)ceil((float)numberOfRows / (float)TILE_WIDTH);
	dim3 dimGrid(grid1,grid2 );
	//Copy contents of each row at a time from host input matrix
	for (int i = 0; i < numberOfRows; i++)
	{
		cudaMemcpy(&Md[i*numberOfColumns], matrix[i], numberOfColumns*sizeof(float), cudaMemcpyHostToDevice);
	}
	//Memory copy from deviceto device
	cudaMemcpy(Rd, Md, bytes, cudaMemcpyDeviceToDevice);

	//Consider one row at time for each scaling and Subtraction
	for (int row_consider = 0; row_consider < numberOfRows; row_consider++) {
		//Scale kernel to make diagonal elements 1..
		ScaleKernel << <dimGrid, dimBlock >> > (Md, numberOfRows, numberOfColumns, Rd,row_consider);
		cudaThreadSynchronize();
		//Get error status of last performed function
		status = cudaGetLastError();
		if (status != cudaSuccess)
		{
			ErrorCheck(status);
			cudaFree(Md);
			cudaFree(Rd);
			return false;
		}
		//Copy the computed matrix from ScaleKernel method to input matrix
		cudaMemcpy(Md, Rd, bytes, cudaMemcpyDeviceToDevice);
		//Method to make the elements other than diagonal 0
		SubtractKernel << <dimGrid, dimBlock >> > (Md, numberOfRows, numberOfColumns, Rd, row_consider);
		cudaThreadSynchronize();
		//Get error status of last performed function		
		status = cudaGetLastError();
		if (status != cudaSuccess)
		{
			ErrorCheck(status);
			cudaFree(Md);
			cudaFree(Rd);
			return false;
		}
		//Copy the computed matrix from SubtractKernel method to input matrix
		cudaMemcpy(Md, Rd, bytes, cudaMemcpyDeviceToDevice);
	}

	//Copy contents of each row at a time from device output matrix to host
	for (int i = 0; i < numberOfRows; i++)
	{
		cudaMemcpy(outputMatrix[i], &Md[i * numberOfColumns], numberOfColumns * sizeof(float), cudaMemcpyDeviceToHost);
	}
	//Free the memory
	cudaFree(Md);
	cudaFree(Rd);
	return true;
}

//Error detected..Print
void ErrorCheck(cudaError_t status) {	
	printf("\nKernel failed!! %s", cudaGetErrorString(status));
}