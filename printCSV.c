#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Structure to store the data
typedef struct {
    int user_id;
    int item_id;
    float rating;
} Data;

// Function to load data from CSV
// Function to load data from CSV
Data* load_csv(const char* path, int* num_rows, char* header) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    printf("File opened\n");

    // Hardcode the number of lines to load
    int capacity = 926055;
    Data* data = malloc(capacity * sizeof(Data));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file); // Ensure the file is closed if malloc fails
        exit(EXIT_FAILURE);
    }
    *num_rows = 0;

    printf("Memory allocated\n");

    // Read and store the header line
    if (fgets(header, 256, file) == NULL) {
        fprintf(stderr, "Error reading header line\n");
        free(data);
        fclose(file); // Ensure the file is closed if fgets fails
        exit(EXIT_FAILURE);
    }
    printf("Header read: %s\n", header);

    // Load the data
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

int main() {
    printf("Begin");
    const char* path = "/home/pressprexx/Code/Cooleg/CAD/CAD-2024/goodbooksData/trainBooks.csv";
    int num_rows;

    printf("Start");
    // Allocate space for the header
    char header[256];
    Data* data = load_csv(path, &num_rows, header);

    printf("Imported CSV");

    // Print the last line
    if (num_rows > 0) {
        Data last = data[num_rows - 1];
        printf("Last line: %d, %d, %.2f\n", last.user_id, last.item_id, last.rating);
    } else {
        printf("No data loaded\n");
    }

    // Print the value from line 200, column 1 (user_id) and column 2 (item_id)
    if (num_rows >= 200) {
        Data line200 = data[199]; // 0-based index, so line 200 is at index 199
        printf("Line 200, Column 1: %d, Column 2: %d\n", line200.user_id, line200.item_id);
    } else {
        printf("Less than 200 rows loaded\n");
    }

    free(data);

    return 0;
}
