#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define LSIZE 10

int main (int argc, char *argv[]) {
 MPI_Status status;
 int i, mpi_rank, mpi_size, vsize;
 int * localVector, * Vector; 

 MPI_Init(&argc, &argv);
 MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
 MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

 vsize= LSIZE*mpi_size; 

 localVector=(int *)malloc(sizeof(int)*LSIZE);

 if (mpi_rank == 0) 
  {
    Vector=(int *)malloc(sizeof(int)*vsize);
    for (i=0; i < vsize; i++)  *(Vector+i)=i;
  }

if (mpi_rank == 0)
   {for (i=0; i < vsize ; i++) printf ("%d ", *(Vector+i));
     printf("\n"); }


MPI_Scatter(Vector, LSIZE, MPI_INT, localVector, LSIZE, MPI_INT,  0, MPI_COMM_WORLD);   

for (i=0; i < LSIZE ; i++) *(localVector+i)+=1 ; // ogni task increneta gli interi ricevuti

MPI_Gather (localVector, LSIZE, MPI_INT, Vector, LSIZE, MPI_INT,  0, MPI_COMM_WORLD);   

 if (mpi_rank == 0)  
   {for (i=0; i < vsize ; i++) printf ("%d ", *(Vector+i));
     printf("\n"); }

 MPI_Finalize();
 return 0;
}
