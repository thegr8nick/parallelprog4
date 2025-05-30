#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

typedef vector<vector<double>> Matrix;

global void matrixMultiplyKernel(double* A, double* B, double* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        double sum = 0;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * p + col];
        C[row * p + col] = sum;
    }
}

Matrix read_matrix(const string& filename, int& rows, int& cols) {
    ifstream file(filename);
    if (!file) {
        cerr << "Failed to open file: " << filename << endl;
        exit(1);
    }

    file >> rows >> cols;
    Matrix matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            file >> matrix[i][j];

    return matrix;
}

void write_matrix(const string& filename, const Matrix& matrix) {
    ofstream fout(filename);
    int rows = matrix.size();
    int cols = matrix[0].size();
    fout << rows << " " << cols << endl;
    for (const auto& row : matrix) {
        for (double val : row)
            fout << val << " ";
        fout << "\n";
    }
}

void gpu_matrix_multiply(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p) {
    size_t sizeA = m * n * sizeof(double);
    size_t sizeB = n * p * sizeof(double);
    size_t sizeC = m * p * sizeof(double);

    double *h_A = new double[m * n];
    double *h_B = new double[n * p];
    double *h_C = new double[m * p];

    // Flatten matrices
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            h_A[i * n + j] = A[i][j];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            h_B[i * p + j] = B[i][j];

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + 15) / 16, (m + 15) / 16);

    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            C[i][j] = h_C[i * p + j];

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    string fileA = "matrixA.txt";
    string fileB = "matrixB.txt";
    string fileC = "result.txt";
    string fileInfo = "info.txt";

    int n1, m1, n2, m2;
    Matrix A = read_matrix(fileA, n1, m1);
    Matrix B = read_matrix(fileB, n2, m2);

    if (m1 != n2) {
        cerr << "It is impossible to multiply matrices: the sizes do not match." << endl;
        return 1;
    }

    Matrix C(n1, vector<double>(m2, 0.0));

    auto start = high_resolution_clock::now();
    gpu_matrix_multiply(A, B, C, n1, m1, m2);
    auto end = high_resolution_clock::now();
    double elapsed = duration_cast<duration<double>>(end - start).count();

    write_matrix(fileC, C);

    ofstream fout(fileInfo);
    fout << "Matrix dimensions A: " << n1 << "x" << m1 << endl;
    fout << "Matrix dimensions B: " << n2 << "x" << m2 << endl;
    fout << "Result dimensions: " << n1 << "x" << m2 << endl;
    fout << "Execution time (sec): " << elapsed << endl;
    fout.close();

    cout << "Multiplication completed. Result in file '" << fileC << "', information in '" << fileInfo << "'." << endl;

    return 0;
}