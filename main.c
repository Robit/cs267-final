#include <mpi.h>
#include "mmio.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int *I, *J;
    double *val;

    int num_rows, num_cols, num_nonzero;
    int *rows, *cols;
    double *vals;
    int *sendcounts, *displs;
    int elems_per_proc;

    if(rank == 0) {
        //Root rank reads in the matrix

        if ((f = fopen("494_bus.mtx", "r")) == NULL) 
            exit(1);

        if (mm_read_banner(f, &matcode) != 0)
        {
            printf("Could not process Matrix Market banner.\n");
            exit(1);
        }


        /*  This is how one can screen matrix types if their application */
        /*  only supports a subset of the Matrix Market data types.      */

        if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
                mm_is_sparse(matcode) )
        {
            printf("Sorry, this application does not support ");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
            exit(1);
        }

        /* find out size of sparse matrix .... */

        if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
            exit(1);


        /* reseve memory for matrices */

        I = (int *) malloc(nz * sizeof(int));
        J = (int *) malloc(nz * sizeof(int));
        val = (double *) malloc(nz * sizeof(double));


        /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
        /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
        /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

        for (int i = 0; i < nz; i++)
        {
            fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
            I[i]--;  /* adjust from 1-based to 0-based */
            J[i]--;
        }

        if (f !=stdin) fclose(f);

        sendcounts = (int*) malloc(sizeof(int) * num_procs);
        displs = (int*) malloc(sizeof(int) * num_procs);

        elems_per_proc = nz / num_procs + (nz % num_procs != 0);

        for(int i = 0; i < num_procs; i++) {
            sendcounts[i] = elems_per_proc;
            displs[i] = i * elems_per_proc;
        }
        if(nz % num_procs != 0) {
            sendcounts[num_procs - 1] = nz - elems_per_proc * (num_procs - 1);
        }
        num_rows = M;
        num_cols = N;
        //printf("nz = %d\n", nz);
    }

    //Scatter matrix data to all procs relevant vals - num_rows, num_cols, num_nonzero, rows, cols, vals
    //NOTE THAT M, N, nz, I, J, and val are NOT VALID for all procs, just the root one.
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(sendcounts, 1, MPI_INT, &num_nonzero, 1, MPI_INT, 0, MPI_COMM_WORLD);
    rows = (int*) malloc(sizeof(int) * num_nonzero);
    cols = (int*) malloc(sizeof(int) * num_nonzero);
    vals = (double*) malloc(sizeof(double) * num_nonzero);
    MPI_Scatterv(I, sendcounts, displs, MPI_INT, rows, num_nonzero, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(J, sendcounts, displs, MPI_INT, cols, num_nonzero, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(val, sendcounts, displs, MPI_DOUBLE, vals, num_nonzero, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //printf("Proc %d received %d values from root\nFirst value: %f at %d,%d\nLast value: %f at %d,%d\n", rank, num_nonzero, vals[0], rows[0], cols[0], vals[num_nonzero - 1], rows[num_nonzero - 1], cols[num_nonzero - 1]);

    /************************/
    /* now write out matrix */
    /************************/

    //mm_write_banner(stdout, matcode);
    //mm_write_mtx_crd_size(stdout, M, N, nz);
    //for (i=0; i<nz; i++)
    //    fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);

    double* dense_mat = (double*) calloc(sizeof(double), num_rows * num_cols);
    for (int i = 0; i < num_nonzero; i++) {
        dense_mat[rows[i] * num_cols + cols[i]] = vals[i];
    }

    double max = dense_mat[0];
    for(int i = 1; i < num_rows * num_cols; i++) {
        if(dense_mat[i] > max) {
            max = dense_mat[i];
        }
    }

    double m;
    MPI_Allreduce(&max, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(rank == 1) {
        printf("MAX = %f\n", m);
    }
    MPI_Finalize();
}