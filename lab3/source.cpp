#include <chrono>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <vector>

using namespace std;

vector<vector<double>> readMatrix(string filename, int &size) {
    ifstream file(filename);
    file >> size;

    vector<vector<double>> matrix(size, vector<double>(size));

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            file >> matrix[i][j];

    return matrix;
}

void writeMatrix(string filename, vector<vector<double>> &matrix) {
    ofstream file(filename);

    int size = matrix.size();
    file << size << endl;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            file << matrix[i][j] << " ";

        file << endl;
    }
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int size;

    vector<vector<double>> A, B;

    if (rank == 0) {
        int size2;

        A = readMatrix("A.txt", size);
        B = readMatrix("B.txt", size2);

        if (size != size2) {
            cout << "Matrices have different sizes" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<double> flatB(size * size);

    if (rank == 0) {
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                flatB[i * size + j] = B[i][j];
    }

    MPI_Bcast(flatB.data(), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rowsPerProcess = size / processes;

    int startRow = rank * rowsPerProcess;
    int endRow = (rank == processes - 1) ? size : startRow + rowsPerProcess;

    vector<double> localResult((endRow - startRow) * size, 0);

    auto start = chrono::high_resolution_clock::now();

    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < size; j++) {

            double sum = 0;

            for (int k = 0; k < size; k++) {
                sum += A[i][k] * flatB[k * size + j];
            }

            localResult[(i - startRow) * size + j] = sum;
        }
    }

    vector<double> finalResult;

    if (rank == 0)
        finalResult.resize(size * size);

    vector<int> recvCounts(processes);
    vector<int> displs(processes);

    for (int i = 0; i < processes; i++) {
        int start = i * rowsPerProcess;
        int end = (i == processes - 1) ? size : start + rowsPerProcess;

        recvCounts[i] = (end - start) * size;
        displs[i] = start * size;
    }

    MPI_Gatherv(localResult.data(),
                localResult.size(),
                MPI_DOUBLE,
                finalResult.data(),
                recvCounts.data(),
                displs.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    auto end = chrono::high_resolution_clock::now();

    if (rank == 0) {

        chrono::duration<double> elapsed = end - start;

        vector<vector<double>> C(size, vector<double>(size));

        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                C[i][j] = finalResult[i * size + j];

        writeMatrix("result.txt", C);

        cout << "Matrix size: " << size << "x" << size << endl;
        cout << "Processes: " << processes << endl;
        cout << "Execution time: " << elapsed.count() << " seconds" << endl;
    }

    MPI_Finalize();

    return 0;
}
