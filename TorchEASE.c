#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int user_id;
    int item_id;
    float score;
} TrainData;

typedef struct {
    int *indices;
    float *values;
    int num_entries;
} SparseTensor;

typedef struct {
    int user_col;
    int item_col;
    int user_id_col;
    int item_id_col;
    float reg;
    SparseTensor sparse;
    float **B;
} TorchEASE;

void generate_labels(TrainData *train, int num_entries, int *labels, int *num_labels) {
    // Simplified function to generate labels
    int current_label = 0;
    for (int i = 0; i < num_entries; i++) {
        int found = 0;
        for (int j = 0; j < current_label; j++) {
            if (labels[j] == train[i].user_id) {
                found = 1;
                break;
            }
        }
        if (!found) {
            labels[current_label++] = train[i].user_id;
        }
    }
    *num_labels = current_label;
}

void build_sparse_tensor(TorchEASE *model, TrainData *train, int num_entries) {
    model->sparse.num_entries = num_entries;
    model->sparse.indices = (int *)malloc(2 * num_entries * sizeof(int));
    model->sparse.values = (float *)malloc(num_entries * sizeof(float));

    for (int i = 0; i < num_entries; i++) {
        model->sparse.indices[2 * i] = train[i].user_id;
        model->sparse.indices[2 * i + 1] = train[i].item_id;
        model->sparse.values[i] = train[i].score;
    }
}

TorchEASE *initialize_model(TrainData *train, int num_entries, float reg) {
    TorchEASE *model = (TorchEASE *)malloc(sizeof(TorchEASE));
    model->reg = reg;

    int user_labels[100], num_users;
    int item_labels[100], num_items;

    generate_labels(train, num_entries, user_labels, &num_users);
    generate_labels(train, num_entries, item_labels, &num_items);

    model->user_col = 0;
    model->item_col = 1;
    model->user_id_col = 2;
    model->item_id_col = 3;

    build_sparse_tensor(model, train, num_entries);

    // Simplified matrix B initialization
    model->B = (float **)malloc(num_users * sizeof(float *));
    for (int i = 0; i < num_users; i++) {
        model->B[i] = (float *)malloc(num_items * sizeof(float));
        for (int j = 0; j < num_items; j++) {
            model->B[i][j] = 0.0;
        }
    }

    return model;
}

void fit(TorchEASE *model) {
    printf("Building G Matrix and B Matrix\n");
    // Simplified G matrix calculation and inversion
    int n = model->sparse.num_entries;
    float G[n][n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            G[i][j] = 0.0;
        }
    }

    for (int i = 0; i < model->sparse.num_entries; i++) {
        int user_id = model->sparse.indices[2 * i];
        int item_id = model->sparse.indices[2 * i + 1];
        G[user_id][item_id] += model->sparse.values[i];
    }

    for (int i = 0; i < n; i++) {
        G[i][i] += model->reg;
    }

    // Simplified matrix inversion (assuming diagonal matrix)
    float P[n][n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            P[i][j] = (i == j) ? 1.0 / G[i][i] : 0.0;
        }
    }

    // Calculate B matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            model->B[i][j] = P[i][j] / (-1 * P[i][i]);
        }
    }

    for (int i = 0; i < n; i++) {
        model->B[i][i] = 1.0;
    }
}

void predict_all(TorchEASE *model, int *users, int num_users, int k) {
    printf("Making predictions\n");
    for (int i = 0; i < num_users; i++) {
        int user_id = users[i];
        printf("Predictions for user %d: ", user_id);
        for (int j = 0; j < k; j++) {
            printf("%f ", model->B[user_id][j]);
        }
        printf("\n");
    }
}

int main() {
    TrainData train_data[] = {
        {0, 1, 5.0}, {0, 2, 3.0}, {1, 1, 4.0}, {1, 3, 2.0}, {2, 2, 1.0}, {2, 3, 5.0}
    };
    int num_entries = sizeof(train_data) / sizeof(TrainData);
    TorchEASE *model = initialize_model(train_data, num_entries, 250.0);

    fit(model);

    int users[] = {0, 1, 2};
    int num_users = sizeof(users) / sizeof(int);

    predict_all(model, users, num_users, 2);

    return 0;
}
