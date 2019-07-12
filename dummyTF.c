#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#define STEPS 5
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


void communication(int layers, int * backward_neurons, float * buf, int * ready){
	int l;
    int n = omp_get_thread_num();
	for (l = layers-1; l > 0; l--){ //input w is not updated
		while(ready[l]==0); 
		printf("Thread %d  Allreduce %d\n",n,backward_neurons[l]);
		allreduce(backward_neurons[l],buf);
		ready[l] = 0; 
	}

}


void computation(int layers, int * forward, int *  backward_cg,
		int * backward_wu, int * ready){ 

    int n = omp_get_thread_num();
	int l = 1; //input is skipped
	for (l = 1; l < layers; l++){
		printf("Thread %d usleep %d FP\n",n,forward[l]);
		usleep(forward[l]); // FP
	}
	for (l = layers-1; l > 0; l--){ //input is not updated
		printf("Thread %d usleep %d CG\n",n,backward_cg[l]);
		usleep(backward_cg[l]); // CG
		ready[l] = 1;
		printf("Thread %d usleep %d WU\n",n,backward_wu[l]);
		usleep(backward_wu[l]); // WU
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

    FILE *fp_model, *fp_results;
    int aux, j;
    char auxstr[200], auxstr2[200], *token, *str;
    printf("Model: %s\n", argv[1]);
    fp_model= fopen(argv[1], "r");
    //printf("layers: %d\n",count_layers(fp_model));
	int layers = count_layers(fp_model)-1; //we discard the info line



//    int l,layers = 4;
//	int forward[LAYERS] = { 2000000,5000000,3500000,2500000};
//	int backward_cg[LAYERS] = { 100000,500000,350000,250000};
//	int backward_wu[LAYERS] = { 10000,50000,35000,25000};
//	int backward_neurons[LAYERS] = { 1000,5000,3500,2500};
    printf("layers = %d\n",layers);
	int l;
	int *type = malloc (layers * sizeof(int));
	int *forward = malloc (layers * sizeof(int));
	int *backward_cg  = malloc (layers * sizeof(int));
	int *backward_wu  = malloc (layers * sizeof(int));
	int *nneurons  = malloc (layers * sizeof(int));


    int max_neurons = 0;
	for(l=0;l<layers;l++){
		if(nneurons[l] > max_neurons)
			max_neurons = nneurons[l];
	}

	float * buf = malloc(max_neurons *  sizeof(float));

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

      if ( !strcmp(typel, "input") ){ 
    	type[i] = INPUT; 
    }
    else if ( !strcmp(typel, "fc") ){ 
    	type[i] = FC;   
	  }
    else if ( !strcmp(typel, "conv") ){ 
    	type[i] = CONV; 
	} 
    else if ( !strcmp(typel, "apool") ){ 
    	type[i] = APOOL;  
    }
    else if ( !strcmp(typel, "mpool") ){ 
    	type[i] = MPOOL;  
     }



      printf("type %d, neurons %d, forward %d, backward_cg %d, backward_wu %d\n",type[i],nneurons[i] , forward[i], backward_cg[i],backward_wu[i]);
      i++;
    }
fclose(fp_model);




    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    omp_set_num_threads(2);
    #pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}

	for(s = 0; s < STEPS; s++){
		#pragma omp parallel num_threads(2)
		{
			if(omp_get_thread_num()==0)
			{
				communication(layers,nneurons,buf,ready);
			}
			else
			{
				computation(layers,forward,backward_cg,backward_wu,ready);
			}
		}

	}
	
	
	MPI_Finalize();

}
