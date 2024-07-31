#include <stdio.h>
#include <stdlib.h>
#include <chrono>

typedef struct {
    int user_id;
    int item_id;
    float rating;
} Data;

// Function to load CSV
Data* load_csv(const char* path, int* num_rows, char* header) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    printf("File opened\n");

    int capacity = 926055; // Hardcoded number of lines, adjust if necessary
    Data* data = (Data*)malloc(capacity * sizeof(Data));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    *num_rows = 0;

    printf("Memory allocated\n");

    // Remove the first line (header)
    if (fgets(header, 256, file) == NULL) {
        fprintf(stderr, "Error reading header line\n");
        free(data);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    printf("Header read: %s\n", header);

    while (fscanf(file, "%d,%d,%f\n", &data[*num_rows].user_id, &data[*num_rows].item_id, &data[*num_rows].rating) != EOF) {
        (*num_rows)++;
        if (*num_rows >= capacity) {
            fprintf(stderr, "Exceeded pre-allocated capacity\n");
            break;
        }
        if (*num_rows <= 10) { // Print the first few rows for debugging
            printf("Row %d loaded: user_id=%d, item_id=%d, rating=%.2f\n", *num_rows, data[*num_rows-1].user_id, data[*num_rows-1].item_id, data[*num_rows-1].rating);
        }
    }

    printf("Data loaded, total rows: %d\n", *num_rows);
    fclose(file);
    printf("File closed\n");

    return data;
}

// Function to generate labels
int* generate_labels(Data* data, int num_rows, int* num_labels, int use_user_id) {
    int* labels = (int*)malloc(num_rows * sizeof(int));
    int id = 0;

    for (int i = 0; i < num_rows; i++) {
        int value = use_user_id ? data[i].user_id : data[i].item_id;
        int found = 0;
        for (int j = 0; j < i; j++) {
            int compare_value = use_user_id ? data[j].user_id : data[j].item_id;
            if (value == compare_value) {
                labels[i] = labels[j];
                found = 1;
                break;
            }
        }
        if (!found) {
            labels[i] = id++;
        }
        if (i < 10) { // Print the first few labels for debugging
            printf("Row %d: value=%d, label=%d\n", i, value, labels[i]);
        }
    }

    *num_labels = id;
    return labels;
}

// Function to export a matrix to a file
void export_matrix_to_file(float* matrix, int rows, int cols, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%f ", matrix[i * cols + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Matrix exported to %s\n", filename);
}

// CUDA kernel to transpose a matrix
__global__ void transpose_kernel(float* X, float* XT, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        XT[j * rows + i] = X[i * cols + j];
    }
}

// CUDA kernel to multiply two matrices
__global__ void matrix_multiply_kernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float sum = 0.0;
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// CUDA kernel to build the G matrix
__global__ void build_g_matrix_kernel(float* G, Data* data, int* user_labels, int num_rows, int num_user_labels, float reg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        int user_label = user_labels[idx];
        atomicAdd(&G[user_label * num_user_labels + user_label], data[idx].rating * data[idx].rating);
    }
    if (idx < num_user_labels) {
        G[idx * num_user_labels + idx] += reg;
    }
}

// CUDA kernel to invert the G matrix
__global__ void invert_matrix_kernel(float* G, float* P, int num_labels) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_labels && j < num_labels) {
        P[i * num_labels + j] = (i == j) ? 1.0 / G[i * num_labels + i] : -G[i * num_labels + j] / (G[i * num_labels + i] * G[j * num_labels + j]);
    }
}

// CUDA kernel to build the B matrix
__global__ void build_b_matrix_kernel(float* B, float* P, int num_labels) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_labels && j < num_labels) {
        B[i * num_labels + j] = P[i * num_labels + j] / (-P[i * num_labels + i]);
        if (i == j) {
            B[i * num_labels + i] += 1.0;
        }
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now(); // Start time

    printf("Begin\n");
    const char* path = "/content/CAD-2024/goodbooksData/trainBooksSmall.csv";
    int num_rows;

    printf("Start\n");
    // Allocate space for the header
    char header[256];
    Data* data = load_csv(path, &num_rows, header);

    printf("Imported CSV\n");

    int num_user_labels, num_item_labels;
    int* item_labels = generate_labels(data, num_rows, &num_item_labels, 1);  // Switched to 1 for item_id
    int* user_labels = generate_labels(data, num_rows, &num_user_labels, 0);  // Switched to 0 for user_id

    printf("Generated labels\n");
    printf("num_user_labels: %d, num_item_labels: %d\n", num_user_labels, num_item_labels);

    float** G = (float**)malloc(num_user_labels * sizeof(float*));  // Based on number of users (switched)
    float** P = (float**)malloc(num_user_labels * sizeof(float*));  // Based on number of users (switched)
    float** B = (float**)malloc(num_user_labels * sizeof(float*));  // Based on number of users (switched)
    for (int i = 0; i < num_user_labels; i++) {
        G[i] = (float*)malloc(num_user_labels * sizeof(float));    // Based on number of users (switched)
        P[i] = (float*)malloc(num_user_labels * sizeof(float));    // Based on number of users (switched)
        B[i] = (float*)malloc(num_user_labels * sizeof(float));    // Based on number of users (switched)
    }

    printf("Allocated memory for matrices\n");
    printf("Matrix G dimensions: %d x %d\n", num_user_labels, num_user_labels);
    printf("Matrix P dimensions: %d x %d\n", num_user_labels, num_user_labels);
    printf("Matrix B dimensions: %d x %d\n", num_user_labels, num_user_labels);

    // Allocate device memory
    float* d_G, * d_P, * d_B, * d_X, * d_XT, * d_C;
    Data* d_data;
    int* d_user_labels, * d_item_labels;
    cudaMalloc((void**)&d_G, num_user_labels * num_user_labels * sizeof(float));
    cudaMalloc((void**)&d_P, num_user_labels * num_user_labels * sizeof(float));
    cudaMalloc((void**)&d_B, num_user_labels * num_user_labels * sizeof(float));
    cudaMalloc((void**)&d_X, num_user_labels * num_item_labels * sizeof(float));
    cudaMalloc((void**)&d_XT, num_item_labels * num_user_labels * sizeof(float));
    cudaMalloc((void**)&d_C, num_user_labels * num_user_labels * sizeof(float));
    cudaMalloc((void**)&d_data, num_rows * sizeof(Data));
    cudaMalloc((void**)&d_user_labels, num_rows * sizeof(int));
    cudaMalloc((void**)&d_item_labels, num_rows * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, data, num_rows * sizeof(Data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_user_labels, user_labels, num_rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_item_labels, item_labels, num_rows * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernels
    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;
    printf("num de blocos: %d\n", numBlocks);
    build_g_matrix_kernel<<<numBlocks, blockSize>>>(d_G, d_data, d_user_labels, num_rows, num_user_labels, 250.0);

    printf("Matrix X dimensions: %d x %d\n", num_item_labels, num_user_labels);
    printf("Matrix XT dimensions (before transpose): %d x %d\n", num_user_labels, num_item_labels);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((num_user_labels + dimBlock.x - 1) / dimBlock.x, (num_user_labels + dimBlock.y - 1) / dimBlock.y);
    invert_matrix_kernel<<<dimGrid, dimBlock>>>(d_G, d_P, num_user_labels);
    build_b_matrix_kernel<<<dimGrid, dimBlock>>>(d_B, d_P, num_user_labels);

    // Transpose matrix X
    transpose_kernel<<<dimGrid, dimBlock>>>(d_X, d_XT, num_user_labels, num_item_labels);

    // Multiply matrices X and XT
    matrix_multiply_kernel<<<dimGrid, dimBlock>>>(d_X, d_XT, d_C, num_user_labels, num_item_labels, num_user_labels);

    // Copy results back to host
    for (int i = 0; i < num_user_labels; i++) {
        cudaMemcpy(P[i], &d_P[i * num_user_labels], num_user_labels * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(B[i], &d_B[i * num_user_labels], num_user_labels * sizeof(float), cudaMemcpyDeviceToHost);
        if ((i + 1) % 5 == 0) {
            printf("Multiplication progress: Completed %d rows\n", i + 1);
        }
    }

    // Export B matrix to file
    export_matrix_to_file((float*)B, num_user_labels, num_user_labels, "output_matrix.txt");

    // Free device memory
    cudaFree(d_G);
    cudaFree(d_P);
    cudaFree(d_B);
    cudaFree(d_X);
    cudaFree(d_XT);
    cudaFree(d_C);
    cudaFree(d_data);
    cudaFree(d_user_labels);
    cudaFree(d_item_labels);

    // Free host memory
    for (int i = 0; i < num_user_labels; i++) {
        free(G[i]);
        free(P[i]);
        free(B[i]);
    }
    free(G);
    free(P);
    free(B);
    free(data);
    free(user_labels);
    free(item_labels);

    auto end = std::chrono::high_resolution_clock::now(); // End time
    std::chrono::duration<double, std::milli> duration = end - start; // Calculate duration
    printf("Time elapsed: %f ms\n", duration.count()); // Print duration

    return 0;
}
