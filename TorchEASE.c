#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int user_id;
    int item_id;
    float rating;
} Data;

// Func para carrgear CSV
Data* load_csv(const char* path, int* num_rows, char* header) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    printf("File opened\n");

    int capacity = 926055; // Número de linhas hardcoded, mudar se a gente funfar com o CSV
    Data* data = malloc(capacity * sizeof(Data));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    *num_rows = 0;

    printf("Memory allocated\n");

    // Tirar a primeira linha (linha de colunas)
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
        printf("Row %d loaded: %d, %d, %.2f\n", *num_rows, data[*num_rows-1].user_id, data[*num_rows-1].item_id, data[*num_rows-1].rating);
    }

    printf("Data loaded\n");
    fclose(file);
    printf("File closed\n");

    return data;
}

// Gerar labels para o input do modelo
int* generate_labels(Data* data, int num_rows, int* num_labels) {
    int* labels = malloc(num_rows * sizeof(int));
    int id = 0;

    for (int i = 0; i < num_rows; i++) {
        int found = 0;
        for (int j = 0; j < i; j++) {
            if (data[i].user_id == data[j].user_id) {
                labels[i] = labels[j];
                found = 1;
                break;
            }
        }
        if (!found) {
            labels[i] = id++;
        }
    }

    *num_labels = id;
    return labels;
}

// Função para pegar a transposta
void transpose(float** X, float** XT, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            XT[j][i] = X[i][j];
        }
        if ((i + 1) % 5 == 0) {
            printf("Transpose progress: Completed %d rows\n", i + 1);
        }
    }
}

// Função para multiplicar duas matrizes
void matrix_multiply(float** A, float** B, float** C, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < A_cols; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
        if ((i + 1) % 5 == 0) {
            printf("Multiplication progress: Completed %d rows\n", i + 1);
        }
    }
}

// Função para printar a matriz (debugging, não printar se não precisar, mt pesado)
void print_matrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Função para gerar matriz G, referenciar codigo em Python se precisar
void build_g_matrix(float** G, Data* data, int num_rows, int num_labels, float reg) {
    // Allocate memory for temporary matrices
    float** X = malloc(num_labels * sizeof(float*));
    float** XT = malloc(num_labels * sizeof(float*));
    for (int i = 0; i < num_labels; i++) {
        X[i] = malloc(num_labels * sizeof(float));
        XT[i] = malloc(num_labels * sizeof(float));
    }

    // Initialize X matrix to zero
    for (int i = 0; i < num_labels; i++) {
        for (int j = 0; j < num_labels; j++) {
            X[i][j] = 0.0;
        }
    }

    // Build the X matrix
    //for (int i = 0; i < num_rows; i++) {
    //    int user_id = data[i].user_id;
    //    int item_id = data[i].item_id;
    //    X[user_id][item_id] += data[i].rating;
    //}

    // Transpose the X matrix
    transpose(X, XT, num_labels, num_labels);

    printf("Matrix transposed");

    // Multiply XT by X to get G
    matrix_multiply(XT, X, G, num_labels, num_labels, num_labels);

    printf("G matrix after transpose multiplication:\n");
    //print_matrix(G, num_labels, num_labels);

    // Add regularization to the diagonal of G
    for (int i = 0; i < num_labels; i++) {
        G[i][i] += reg;
    }

    printf("G matrix after adding regularization:\n");
    //print_matrix(G, num_labels, num_labels);

    // Free temporary matrices
    for (int i = 0; i < num_labels; i++) {
        free(X[i]);
        free(XT[i]);
    }
    free(X);
    free(XT);
}

// Function to invert the matrix G
void invert_matrix(float** G, float** P, int num_labels) {
    // For simplification, assume the matrix is invertible
    for (int i = 0; i < num_labels; i++) {
        for (int j = 0; j < num_labels; j++) {
            P[i][j] = (i == j) ? 1.0 / G[i][j] : -G[i][j] / (G[i][i] * G[j][j]);
        }
    }
}

// Function to build the B matrix
void build_b_matrix(float** B, float** P, int num_labels) {
    for (int i = 0; i < num_labels; i++) {
        for (int j = 0; j < num_labels; j++) {
            B[i][j] = P[i][j] / (-P[i][i]);
        }
        B[i][i] += 1.0;
    }
}

int main() {
    printf("Begin");
    const char* path = "/home/pressprexx/Code/Cooleg/CAD/CAD-2024/goodbooksData/trainBooks.csv";
    int num_rows;

    printf("Start");
    // Allocate space for the header
    char header[256];
    Data* data = load_csv(path, &num_rows, header);

    printf("Imported CSV");

    int num_labels;
    int* labels = generate_labels(data, num_rows, &num_labels);

    printf("Generated labels\n");

    float** G = malloc(num_labels * sizeof(float*));
    float** P = malloc(num_labels * sizeof(float*));
    float** B = malloc(num_labels * sizeof(float*));
    for (int i = 0; i < num_labels; i++) {
        G[i] = malloc(num_labels * sizeof(float));
        P[i] = malloc(num_labels * sizeof(float));
        B[i] = malloc(num_labels * sizeof(float));
    }

    printf("Allocated memory for matrices\n");

    build_g_matrix(G, data, num_rows, num_labels, 250.0);
    invert_matrix(G, P, num_labels);
    build_b_matrix(B, P, num_labels);

    printf("B matrix built!\n");

    // Só descomentar o código abaixo quando quiser testar a matriz resultante.

    // for (int i = 0; i < num_labels; i++) {
    //     for (int j = 0; j < num_labels; j++) {
    //         printf("%f ", B[i][j]);
    //     }
    //     printf("\n");
    // }

    for (int i = 0; i < num_labels; i++) {
        free(G[i]);
        free(P[i]);
        free(B[i]);
    }
    free(G);
    free(P);
    free(B);
    free(data);
    free(labels);

    return 0;
}
