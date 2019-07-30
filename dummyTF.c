#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#define STEPS 10
#define MAX_LEN 1024
//#define LAYERS  4

#define INPUT -1
#define FC 0
#define CONV 1
#define APOOL 2
#define MPOOL 3


int allreduce(size_t elements, float * buf){

	return MPI_Allreduce(buf,buf,elements,MPI_FLOAT, MPI_SUM,MPI_COMM_WORLD);
}


void communication(int layers, int * backward_neurons, float * buf, int * ready, int rank){
	int l;
    int n = omp_get_thread_num();
	for (l = layers-1; l > 0; l--){ //input w is not updated
		while(ready[l]==0); 
//		printf("Rank %d Thread %d Layer %d Allreduce %d\n",rank,n,l,backward_neurons[l]*sizeof(float));
		allreduce(backward_neurons[l],buf);
		ready[l] = 2; 
	}

}


void computation(int layers, int * forward, int *  backward_cg,
		int * backward_wu, int * ready, int rank){ 

    int n = omp_get_thread_num();
	int l = 1; //input is skipped
	for (l = 1; l < layers; l++){
//		printf("Rank %d Thread %d Layer %d usleep %d FP\n",rank,n,l,forward[l]);
		usleep(forward[l]); // FP
	}
	for (l = layers-1; l > 0; l--){ //input is not updated
//		printf("Rank %d Thread %d Layer %d usleep %d CG\n",rank,n,l,backward_cg[l]);
		usleep(backward_cg[l]); // CG
		ready[l] = 1;
	}
	for (l = layers-1; l > 0; l--){ //input is not updated
		while(ready[l] != 2);
                //printf("Rank %d Thread %d Layer %d usleep %d WU\n",rank,n,l,backward_wu[l]);
		usleep(backward_wu[l]); // WU
		ready[l]=0;
	}

} 


const char* getfield(char* line, int num){
  const char* tok;
  for (tok = strtok(line, ";");
    tok && *tok;
    tok = strtok(NULL, ";\n"))
    if (!--num)
      return tok;
  return NULL;
}

int getfield_int(char* line, int num){
  char *l= strdup(line);
  const char *field= getfield(l, num); 
  if (field != NULL) {
    return atoi(field); 
  }
  free(l);
  return 0;
}

double getfield_double(char* line, int num){
  char *l= strdup(line);
  const char *field= getfield(l, num); 
  if (field != NULL) {
    return atof(field); 
  }
  free(l);
  return 0;
}



int count_layers(FILE *fp){
  int num_layers= 0;
  while(!feof(fp))
  {
    char ch = fgetc(fp);
    if(ch == '\n')
    {
      num_layers++;
    }
  }
  return num_layers;
}


int main(int argc, char * argv []){

    
	if (argc < 2){
      perror("Usage: ./dummyTF model.csv\n");
      exit(-1);
    }

	int nthreads,s;
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    FILE *fp_model, *fp_results;
    int aux, j;
    char auxstr[200], auxstr2[200], *token, *str;
      if(world_rank == 0)
    printf("Model: %s\n", argv[1]);
    fp_model= fopen(argv[1], "r");
	int layers = count_layers(fp_model)-1; //we discard the info line



    //printf("layers = %d\n",layers);
	int l;
	int *type = malloc (layers * sizeof(int));
	int *forward = malloc (layers * sizeof(int));
	int *backward_cg  = malloc (layers * sizeof(int));
	int *backward_wu  = malloc (layers * sizeof(int));
	int *nneurons  = malloc (layers * sizeof(int));



	//int ready[LAYERS] = { 0,0,0,0};
	int * ready = malloc (layers * sizeof(int));

    char line[MAX_LEN]; 
    int i = 0;
    fclose(fp_model);
    fp_model= fopen(argv[1], "r");
    fgets(line, MAX_LEN, fp_model);
    while(fgets(line, MAX_LEN, fp_model)){
      char* tmp = strdup(line);
      const char* typel = getfield(tmp, 2); 
      nneurons[i]  = getfield_int(line, 3);
      forward[i]= getfield_int(line, 4);
      backward_cg[i]  = getfield_int(line, 5);
      backward_wu[i]    = getfield_int(line, 6);
      ready[i] = 0;

      if ( !strcmp(typel, "Input") ){ 
    	type[i] = INPUT; nneurons[i]=0;forward[i]=0;backward_cg[i]=0;backward_wu[i]=0; 
    }
    else if ( !strcmp(typel, "FC") ){ 
    	type[i] = FC;   
	  }
    else if ( !strcmp(typel, "Convolutional") ){ 
    	type[i] = CONV; 
	} 
    else if ( !strcmp(typel, "apool") ){ 
    	type[i] = APOOL; nneurons[i]=0;forward[i]=0;backward_cg[i]=0;backward_wu[i]=0; 
    }
    else if ( !strcmp(typel, "mpool") ){ 
    	type[i] = MPOOL;  nneurons[i]=0;forward[i]=0;backward_cg[i]=0;backward_wu[i]=0;
     }


      if(world_rank == 0)
          printf("type %d, neurons %d, forward %d, backward_cg %d, backward_wu %d\n",type[i],nneurons[i] , forward[i], backward_cg[i],backward_wu[i]);
      i++;
    }
fclose(fp_model);


    int max_neurons = 0;
	for(l=0;l<layers;l++){
		if(nneurons[l] > max_neurons)
			max_neurons = nneurons[l];
	}

	float * buf = malloc(max_neurons *  sizeof(float));


    omp_set_num_threads(2);
    #pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}

	for(s = 0; s < STEPS; s++){
      if(world_rank == 0)
	printf("Starting STEP %d\n",s);
		#pragma omp parallel num_threads(2)
		{
			if(omp_get_thread_num()==0)
			{
				communication(layers,nneurons,buf,ready,world_rank);
			}
			else
			{
				computation(layers,forward,backward_cg,backward_wu,ready,world_rank);
			}
		}

	}
	
	
	MPI_Finalize();

}
