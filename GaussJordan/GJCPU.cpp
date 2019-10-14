#include "GJCommon.cuh"
//Method to calculate Gaussian Elimination using CPU
void GaussianEliminationCPU(float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot) {
	//Copy input matrix into output matrix
	for (int i = 0; i < numberOfRows; i++)
	{
		for (int j = 0; j < numberOfColumns; j++)
			outputMatrix[i][j] = matrix[i][j];
	}
	//Take one row at a time
	for (int row_consider = 0; row_consider < numberOfRows; row_consider++) {
		//Copy diagonal element of matrix considered in the row into temp
		float temp = outputMatrix[row_consider][row_consider];
		//Divide all the elements in the considered row by temp, this will make diagonal element of the row as 1
		for (int j = 0; j < numberOfColumns; j++)
			outputMatrix[row_consider][j] = outputMatrix[row_consider][j] / temp;
		//Take all elements which do not belong to the considered row
		for (int i = 0; i < numberOfRows; i++)
		{
			if (i != row_consider) {
				//Copy value of non considered row and row_consider column 
				float temp = outputMatrix[i][row_consider];
				//Subtract each element which does not belong to the considered row with a corresponding element in the row_consider..
				//This will make all the non diagonal elements as zero..
				for (int j = 0; j < numberOfColumns; j++)
					outputMatrix[i][j] = outputMatrix[i][j] - (temp * outputMatrix[row_consider][j]);
			}
		}
	}
}