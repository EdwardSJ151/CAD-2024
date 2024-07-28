#include <stdio.h>
#include <stdlib.h>

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
    Data* data = malloc(capacity * sizeof(Data));
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
    int* labels = malloc(num_rows * sizeof(int));
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
void export_matrix_to_file(float** matrix, int rows, int cols, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%f ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Matrix exported to %s\n", filename);
}

// Function to transpose a matrix
void transpose(float** X, float** XT, int rows, int cols) {
    printf("Transpose started with %d rows and %d cols\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            XT[j][i] = X[i][j];
        }
        if ((i + 1) % 5 == 0) {
            printf("Transpose progress: Completed %d rows\n", i + 1);
        }
    }
    printf("Transpose completed with %d rows and %d cols\n", rows, cols);
}

// Function to multiply two matrices
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

// Function to print a matrix (for debugging, do not use if not needed)
void print_matrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Function to build matrix G
void build_g_matrix(float** G, Data* data, int num_rows, int num_user_labels, int num_item_labels, int* user_labels, int* item_labels, float reg) {
    // Allocate memory for temporary matrices
    float** X = malloc(num_item_labels * sizeof(float*));  // Switched to item labels
    float** XT = malloc(num_user_labels * sizeof(float*)); // Switched to user labels
    for (int i = 0; i < num_item_labels; i++) {
        X[i] = malloc(num_user_labels * sizeof(float));
    }
    for (int i = 0; i < num_user_labels; i++) {
        XT[i] = malloc(num_item_labels * sizeof(float));
    }

    // Initialize X matrix to zero
    for (int i = 0; i < num_item_labels; i++) {
        for (int j = 0; j < num_user_labels; j++) {
            X[i][j] = 0.0;
        }
    }

    // Build the X matrix using labeled user IDs and item IDs
    for (int i = 0; i < num_rows; i++) {
        int user_label = user_labels[i];
        int item_label = item_labels[i];
        X[item_label][user_label] += data[i].rating; // Switched to use item labels for rows and user labels for columns
    }

    // Print dimensions of matrices X and XT before transposition
    printf("Matrix X dimensions: %d x %d\n", num_item_labels, num_user_labels);
    printf("Matrix XT dimensions (before transpose): %d x %d\n", num_user_labels, num_item_labels);

    // Transpose the X matrix
    transpose(X, XT, num_item_labels, num_user_labels); // Switched to item labels for rows and user labels for columns

    printf("Matrix transposed\n");

    // Multiply XT by X to get G
    matrix_multiply(XT, X, G, num_user_labels, num_item_labels, num_user_labels); // Switched to user labels for rows and item labels for columns

    printf("G matrix after transpose multiplication:\n");
    //print_matrix(G, num_user_labels, num_user_labels);

    // Add regularization to the diagonal of G
    for (int i = 0; i < num_user_labels; i++) {
        G[i][i] += reg;
    }

    printf("G matrix after adding regularization:\n");
    //print_matrix(G, num_user_labels, num_user_labels);

    // Free temporary matrices
    for (int i = 0; i < num_item_labels; i++) {
        free(X[i]);
    }
    for (int i = 0; i < num_user_labels; i++) {
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
    printf("Begin\n");
    const char* path = "/home/pressprexx/Code/Cooleg/CAD/CAD-2024/goodbooksData/trainBooksSmall.csv";
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

    float** G = malloc(num_user_labels * sizeof(float*));  // Based on number of users (switched)
    float** P = malloc(num_user_labels * sizeof(float*));  // Based on number of users (switched)
    float** B = malloc(num_user_labels * sizeof(float*));  // Based on number of users (switched)
    for (int i = 0; i < num_user_labels; i++) {
        G[i] = malloc(num_user_labels * sizeof(float));    // Based on number of users (switched)
        P[i] = malloc(num_user_labels * sizeof(float));    // Based on number of users (switched)
        B[i] = malloc(num_user_labels * sizeof(float));    // Based on number of users (switched)
    }

    printf("Allocated memory for matrices\n");
    printf("Matrix G dimensions: %d x %d\n", num_user_labels, num_user_labels);
    printf("Matrix P dimensions: %d x %d\n", num_user_labels, num_user_labels);
    printf("Matrix B dimensions: %d x %d\n", num_user_labels, num_user_labels);

    build_g_matrix(G, data, num_rows, num_user_labels, num_item_labels, user_labels, item_labels, 250.0);
    invert_matrix(G, P, num_user_labels);  // Number of users for inversion (switched)
    build_b_matrix(B, P, num_user_labels); // Number of users for building B (switched)

    printf("B matrix built!\n");

    // export_matrix_to_file(G, num_user_labels, num_user_labels, "output_matrix.txt");

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

    return 0;
}
