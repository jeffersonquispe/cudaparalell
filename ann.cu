
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string>
#include <stdlib.h>
#include <new>

#define SIZE 300000*4

using namespace std;
__global__ void kMartixByMatrixElementwise(const int nThreads, const float *m1, const float *m2, float *output) {
    /*  Te almacena el el resultados de dos arreglos (elementos acertados)
				Retorna un array  donde los elementos calculados son almacenados aqui
    */
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = m1[i] * m2[i];
	  }
}

__device__ float* dMartixByMatrixElementwise(const float *m1, const float *m2, float *output, const int width, const int height){

	kMartixByMatrixElementwise <<< width, height >>> ( width * height, m1, m2, output );
	//kMartixByMatrixElementwise <<< width/8, height/8 >>> ( width * height, m1, m2, output );
    cudaDeviceSynchronize();
    return output;
}

__global__ void kMartixSubstractMatrix(const int nThreads, const float *m1, const float *m2, float *output) {
    //Computa los elementos diferenciados entre dos arrays
		//Retorna un array  donde los elementos calculados son almacenados aqui
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = m1[i] - m2[i];
	  }
}

__device__ float* dMartixSubstractMatrix(const float *m1, const float *m2, float *output, const int width, const int height){

	//X: size de valores iniciales
	//x_w:Número de entradas
	//x_h:Número de logs
	//y:size de valores espérados
	//y_w:(1)cantidad valores esperados
	//l1:size para capa 1
	//l1_w:numero de neuronas ocultas
	//l_1_d:size derivada de layer 1
	//pred: size de valores de Prediccion
	//pred_d: valores de prediccion derivada
	//WO:sizePesos iniciales
	//w1:size de pesos de capa oculta
	//buffer: size de valores de salidad

//--ACAC ,E QUEDEEEdMartixSubstractMatrix(y, pred, pred_d, X_h, y_w), dSigmoid_d(pred, buffer, X_h, y_w), pred_d, X_h, y_w );


	kMartixSubstractMatrix <<< width, height >>> ( width * height, m1, m2, output );
	//kMartixSubstractMatrix <<< width/8, height/8 >>> ( width * height, m1, m2, output );
    cudaDeviceSynchronize();
    return output;
}

__global__ void kSigmoid(const int nThreads, float const *input, float *output){
    /* caulcula la funcion sigmoidaal f(x) = 1/(1 + e^-x).
    */
		//nThreads: numero de entradas x Numero de neuronas ocultas
		//input: size para capa 1
		//ouput:size para capa 1

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = 1.0 / (1.0 + std::exp(-input[i]));
	  }
}

__device__ void dSigmoid(float const *input, float *output, const int height, const int width){
	//input: funcion dDot(m1*m2)
	//ouput:size para capa 1
	//heigth:numero de entradas
	//width:numero de neuronas ocultas

	kSigmoid <<< height, width >>> (height * width, input, output);
	//kSigmoid <<<  height/8, width/8 >>> (height * width, input, output);
	cudaDeviceSynchronize();
}

__global__ void kSigmoid_d(const int nThreads, float const *input, float *output) {
	/*  calcula la derivada de la funcion sigmoidal f'(x) = f(x)(1 - f(x)),
	    salida: arreglo alamcenado aqui x(1 - x) para cada elemento de la matriz input m1
	*/

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = input[i] * (1 - input[i]);
	  }
}

__device__ float* dSigmoid_d(float const *input, float *output, const int rows, const int columns){
	kSigmoid_d <<< rows, columns >>> (rows*columns, input, output);
	//kSigmoid_d <<< rows/8, columns/8>>> (rows*columns, input, output);
	cudaDeviceSynchronize();
	return output;
}

__global__ void kDot(const int nThreads, const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_columns ){
//Calcula el producto de dos matrices, m1 y m2 arrays inputs,
//salida:m1*m2

//nThreads:multiplicacion por numero de salidas y entradas
//m1:size para capa 1
//m2: size para pesos de capa ocultas
//output: size de valores de prediccións
//m1_rows: Numero de logs
//m1_columns: Numero de neuronas ocultas
//m2_columns; Numero de cantidad de valores esperados

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
		 {
	    int r = (int)i / m2_columns;
	    int c = i % m2_columns;
	    float t_output = 0.f;

	    for( int k = 0; k < m1_columns; ++k ) {
	        t_output += m1[ r * m1_columns + k ] * m2[ k * m2_columns + c ];
	    }
	    output[i] = t_output;
		}
}

__device__ float* dDot(const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_columns ){

	//m1:size para capa 1
	//m2:size de pesos de capa oculta
	//output: size de valores de Prediccion
	//m1_rows:Número de logs
	//m1_columns:numero de neuronas ocultas
	//m2_columns:(1)cantidad valores esperados
	//funcion dDot(l1, W1, pred, X_h, l1_w, y_w)

	kDot <<< m1_rows, m2_columns >>> (m1_rows * m2_columns, m1, m2, output, m1_rows , m1_columns, m2_columns );
	//kDot <<< m1_rows/8, m2_columns/8>>> (m1_rows * m2_columns, m1, m2, output, m1_rows , m1_columns, m2_columns );
	cudaDeviceSynchronize();
	return output;
}

__global__ void kDot_m1_m2T(const int nThreads, const float *m1, const float *m2, float *output, const int m1_columns, const int m2_rows ){
	//actualiza las salidas con el producto de dos dosmatrices trasnpuesta
	//salida: producto de dos arrays

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	{
		int r = (int)i / m2_rows;
		int c = i % m2_rows;
		float t_output = 0.0;
		int id_T;

		for( int k = 0; k < m1_columns; ++k ) {
			id_T = c * m1_columns + k;
			t_output += m1[ r * m1_columns + k ] * m2[ id_T ];
		}

		output[i] = t_output;
	}
}

__device__ float* dDot_m1_m2T(const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_rows )
{
	kDot_m1_m2T <<< m1_rows, m2_rows >>> ( m1_rows * m2_rows, m1, m2, output, m1_columns, m2_rows );
	//kDot_m1_m2T <<< m1_rows/8, m2_rows/8 >>> ( m1_rows * m2_rows, m1, m2, output, m1_columns, m2_rows );
	cudaDeviceSynchronize();
	return output;
}

__global__ void kDot_m1T_m2(const int nThreads, const float *m1, const float *m2, float *output, const int m1_rows,
							const int m1_columns, const int m2_columns ){
								//Incrementa la salida de la matriz con el producto de dos matrices: m1 trasnpuesta con m2

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	{
	    int r = (int)i / m2_columns;
	    int c = i % m2_columns;
	    int id_T;
	    float t_output = 0.0;

	    for( int k = 0; k < m1_rows; ++k ) {
	    	id_T = k * m1_columns + r;
	        t_output += m1[ id_T ] * m2[ k * m2_columns + c ];
	    }

	    output[i] += t_output;
	}
}

__device__ void dDot_m1T_m2(const float *m1, const float *m2, float *output, const int m1_height , const int m1_width, const int m2_width )
{
	kDot_m1T_m2 <<< m1_width, m2_width >>> (m1_width * m2_width, m1, m2, output, m1_height, m1_width, m2_width );
	//kDot_m1T_m2 <<<  m1_width/8, m2_width/8 >>> (m1_width * m2_width, m1, m2, output, m1_height, m1_width, m2_width );
	cudaDeviceSynchronize();
}

__device__ void kPrintMatrix (const float* M, int h, int w) {
    // imprime hxw
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			printf("%f  ", M[i*w+j]);
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void ktrain(	const float* X, const int X_w, const int X_h,
						const float* y, const int y_w,
						float* l1, const int l1_w, float* l_1_d,
						float* pred, float* pred_d,
						float* W0,
						float* W1,
						float* buffer
						)
{
	for (unsigned i = 0; i < 50; ++i) {//numero de epocas
				//X: size de valores iniciales
				//x_w:Número de entradas
				//x_h:Número de logs
				//y:size de valores espérados
				//y_w:(1)cantidad valores esperados
				//l1:size para capa 1
				//l1_w:numero de neuronas ocultas
				//l_1_d:size derivada de layer 1
				//pred: size de valores de Prediccion
				//pred_d: valores de prediccion derivada
				//WO:sizePesos iniciales
				//w1:size de pesos de capa oculta
				//buffer: size de valores de salidad

        dSigmoid(dDot(X, W0, l1, X_h, X_w, l1_w), l1, X_h, l1_w);// capa 1
        dSigmoid(dDot(l1, W1, pred, X_h, l1_w, y_w), pred, X_h, y_w);//cape 2
        dMartixByMatrixElementwise(dMartixSubstractMatrix(y, pred, pred_d, X_h, y_w), dSigmoid_d(pred, buffer, X_h, y_w), pred_d, X_h, y_w );
        dMartixByMatrixElementwise(dDot_m1_m2T(pred_d, W1, l_1_d, X_h, y_w, l1_w), dSigmoid_d(l1, buffer, X_h, l1_w), l_1_d, X_h, l1_w);
        dDot_m1T_m2( l1, pred_d, W1, X_h, l1_w, y_w );
        dDot_m1T_m2( X, l_1_d, W0, X_h, X_w, l1_w );
    }
}

__host__ float *  read(){

  FILE *archivo;

  // float *array = malloc(sizeof(float)*SIZE);
	float *array = new float[SIZE];
	// static float array[SIZE];
  double i;
  float n;
  archivo = fopen("oversample.txt","rt");
  i = fscanf(archivo,"%f, ",&n);
  int k = 0;

  while(i != EOF){

    if(k == SIZE) break;

    array[k] = n;
    fscanf(archivo,"%f, ",&n);
    k++;
  }
  printf("%d\n",k );
  fclose(archivo);

  return array;
}



__host__ float * data_range(int begin, int end , float * array){
   // float * arr = malloc((end-begin)*sizeof(float));
	 //int local_size = end - begin;
	 //static float arr[local_size];
	float * arr=new float[end-begin];
  int i = 0;
  for(int k = begin; k < end; k++ ){
    arr[i] = array[k];
    // printf("%f \n",arr[i] );
    i++;
  }

  return arr;
}

int main(void){

	const int TRAINING_SIZE = 30000;//numero de logs
	const int TRAINING_DIM = 4; //numero de variables
	 int L1_SIZE=4; //numero de neuronas

	clock_t a,b;
	//declaracion de eventos
	cudaEvent_t start;
	cudaEvent_t stop;
	float tiempo;

	float *data = read();
   int inicio = 0*4, fin = 30000*4;
   float *h_X = data_range(inicio, fin, data);
	/*for(int i = 0; i < 100; i++){
		printf("%d \n",data[i]);
	}*/

	/*for(int i=0;i<22;i++){
		printf("%f\n",h_X[i]);
	}*/
	int i=0;
	while(i<=10){
	i++;
	L1_SIZE=i*10;
	//	float h_X;
	tiempo=0;
	//scanf("%i\n",&L1_SIZE );
	a=clock();
	//Creación de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

/*	float h_X[TRAINING_SIZE*TRAINING_DIM] = {
5.1,3.5,1.4,0.2,
4.9,3.0,1.4,0.2,
4.7,3.2,1.3,0.2,
4.6,3.1,1.5,0.2,
5.0,3.6,1.4,0.2,
5.4,3.9,1.7,0.4,
4.6,3.4,1.4,0.3,	//buffer: size de valores de salidad

dMartixSubstractMatrix(y, pred, pred_d, X_h, y_w), dSigmoid_d(pred, buffer, X_h, y_w), pred_d, X_h, y_w );
5.0,3.4,1.5,0.2,
4.4,2.9,1.4,0.2,
4.9,3.1,1.5,0.1,
5.4,3.7,1.5,0.2,
4.8,3.4,1.6,0.2,
4.8,3.0,1.4,0.1,
4.3,3.0,1.1,0.1,
5.8,4.0,1.2,0.2,
5.7,4.4,1.5,0.4,
5.4,3.9,1.3,0.4,
5.1,3.5,1.4,0.3,
5.7,3.8,1.7,0.3,
5.1,3.8,1.5,0.3,
5.4,3.4,1.7,0.2,
5.1,3.7,1.5,0.4,
4.6,3.6,1.0,0.2,
5.1,3.3,1.7,0.5,
4.8,3.4,1.9,0.2,
5.0,3.0,1.6,0.2,
5.0,3.4,1.6,0.4,
5.2,3.5,1.5,0.2,	//buffer: size de valores de salidad

dMartixSubstractMatrix(y, pred, pred_d, X_h, y_w), dSigmoid_d(pred, buffer, X_h, y_w), pred_d, X_h, y_w );
5.2,3.4,1.4,0.2,
4.7,3.2,1.6,0.2,
4.8,3.1,1.6,0.2,
5.4,3.4,1.5,0.4,
5.2,4.1,1.5,0.1,
5.5,4.2,1.4,0.2,
4.9,3.1,1.5,0.1,
5.0,3.2,1.2,0.2,
5.5,3.5,1.3,0.2,
4.9,3.1,1.5,0.1,
4.4,3.0,1.3,0.2,
5.1,3.4,1.5,0.2,
5.0,3.5,1.3,0.3,
4.5,2.3,1.3,0.3,
4.4,3.2,1.3,0.2,
5.0,3.5,1.6,0.6,
5.1,3.8,1.9,0.4,
4.8,3.0,1.4,0.3,
5.1,3.8,1.6,0.2,
4.6,3.2,1.4,0.2,
5.3,3.7,1.5,0.2,
5.0,3.3,1.4,0.2,
//---
		//5.4,3.9,1.3,0.4,
		//5.1,3.5,1.4,0.3,
		//5.7,3.8,1.7,0.3,
		//5.1,3.8,1.5,0.3,
//------------------
	/*	7.4,2.8,6.1,1.9,
		7.9,3.8,6.4,2.0,
		6.4,2.8,5.6,2.2,
		6.3,2.8,5.1,1.5,
		6.1,2.6,5.6,1.4,
		7.7,3.0,6.1,2.3,
		6.3,3.4,5.6,2.4,
		6.4,3.1,5.5,1.8,
		6.0,3.0,4.8,1.8,
		6.9,3.1,5.4,2.1,
		6.7,3.1,5.6,2.4,
		6.9,3.1,5.1,2.3,
		5.8,2.7,5.1,1.9,
		6.8,3.2,5.9,2.3,
		6.7,3.3,5.7,2.5,
		6.7,3.0,5.2,2.3,
		6.3,2.5,5.0,1.9,
		6.2,3.4,5.4,2.3,
		5.9,3.0,5.1,1.8,
		6.5,3.0,5.2,2.0
};*/
	const signed int X_size = sizeof(h_X);
	//printf("tamaño %i\n",X_size );
	float *d_X;
	//marcar inicio
	cudaEventRecord(start,0);
	//asigamos un puntero dx, donce X_size es el tamaño de memoria de los datos entradasXlogs
	cudaMalloc(&d_X,X_size);
	cudaMemcpy(d_X,h_X,X_size,cudaMemcpyHostToDevice);

  //tamaño de los pesos asignados a la memoria device
	const long signed int W0_size = L1_SIZE*TRAINING_DIM*sizeof(float);
	//printf("peso 1 %li\n",W0_size);
	//tamaño de los pesos
	float *h_W0 = (float*)malloc(W0_size);
	for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++){
	//introduce elementos random
			//h_W0[i] = 0.5;
	    h_W0[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	}

	float *d_W0;
	//introduce el tamaño de los pesos
	cudaMalloc(&d_W0, W0_size);
	cudaMemcpy(d_W0, h_W0, W0_size, cudaMemcpyHostToDevice);


	//LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
	const long signed int L1_size = L1_SIZE*TRAINING_SIZE*sizeof(float);

	float* h_layer_1 = (float*)malloc(L1_size);
	float* h_layer_1_delta = (float*)malloc(L1_size);
	float* h_buffer = (float*)malloc(L1_size);

	for (int i = 0; i < L1_SIZE*TRAINING_SIZE; i++){
	    h_layer_1[i] = 0.0;
	    h_buffer[i] = 0.0;
	    h_layer_1_delta[i] = 0.0;
	}

  //Crea y asigna memoria para la capa
	float *d_layer_1;
	cudaMalloc(&d_layer_1, L1_size);
	cudaMemcpy(d_layer_1, h_layer_1, L1_size, cudaMemcpyHostToDevice);

	//Crea y asigna memoria para buffer
	float *d_buffer;
	cudaMalloc(&d_buffer, L1_size);
	cudaMemcpy(d_buffer, h_buffer, L1_size, cudaMemcpyHostToDevice);

	//Crea y asigna memoria para la derivada de
	float *d_layer_1_delta;
	cudaMalloc(&d_layer_1_delta, L1_size);
	cudaMemcpy(d_layer_1_delta, h_layer_1_delta, L1_size, cudaMemcpyHostToDevice);

	//PESOS 1
	const long signed int W1_size = L1_SIZE*sizeof(float);
	float *h_W1 = (float*)malloc(W1_size);
	for (int i = 0; i < L1_SIZE; i++){
	    h_W1[i] = 0.1* (2.0*rand()/RAND_MAX-1.0);
			//h_W1[i]=0.5;
	}

	float *d_W1;
	cudaMalloc(&d_W1, W1_size);
	cudaMemcpy(d_W1, h_W1, W1_size, cudaMemcpyHostToDevice);
	//lectura de datos esperados
	float h_y[30000];
	for(int i=0;i<30000;i++){
			if(i<30000){
				h_y[i]=0;
			}else{
				h_y[i]=1;
			}
	}
	/*for(int i=0;i<22;i++){
		printf("%f\n",h_y[i]);
	}*/
	const signed int y_size = sizeof(h_y);
	float *d_y;
	cudaMalloc(&d_y, y_size);
	cudaMemcpy(d_y, h_y, y_size, cudaMemcpyHostToDevice);

	//prediccion y preddicion delta
	float* h_pred = (float*)malloc(y_size);
	float* h_pred_delta = (float*)malloc(y_size);
	for (int i = 0; i < TRAINING_SIZE; i++){
	    h_pred[i] = 0.0;
	    h_pred_delta[i] = 0.0;
	}
	float *d_pred;
	cudaMalloc(&d_pred, y_size);
	cudaMemcpy(d_pred, h_pred, y_size, cudaMemcpyHostToDevice);

	float *d_pred_delta;
	cudaMalloc(&d_pred_delta, y_size);
	cudaMemcpy(d_pred_delta, h_pred_delta, y_size, cudaMemcpyHostToDevice);

	ktrain<<< 1, 1 >>> (	d_X, TRAINING_DIM, TRAINING_SIZE,
						d_y, 3,
						d_layer_1, L1_SIZE, d_layer_1_delta,
						d_pred,
						d_pred_delta,
						d_W0,
						d_W1,
						d_buffer);

	cudaMemcpy(h_pred, d_pred, y_size, cudaMemcpyDeviceToHost);
	cudaFree(d_pred);
	cudaFree(d_X);
	cudaFree(d_y);
	cudaFree(d_layer_1_delta);
	cudaFree(d_pred_delta);
	cudaFree(d_W0);
	cudaFree(d_W1);
	cudaFree(d_buffer);

	free(h_layer_1_delta);
	free(h_pred_delta);
	free(h_W0);
	free(h_W1);
	free(h_buffer);
	//marcar final
		cudaEventRecord(stop,0);
	//sincronizacion GPU=CPU
		cudaEventSynchronize(stop);
	//calculo del tiempo en milisegundos
		cudaEventElapsedTime(&tiempo,start,stop);
	//Impresion de resultados

	//liberación de recursos
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	b=clock();

	for (int i = 0; i < TRAINING_SIZE; i++){
	/*if(i==9999 || i==9998 || i==9997  || i==19999 || i==19998 || i==19997 || i==29999 || i==29998 || i==29997){
		printf("Prediccion[%i]: %f - valor real[%i]: %f  - Error[%i]: %f\n", i, h_pred[i], i, h_y[i], i, h_pred[i] - h_y[i]);
	}*/
	}
	free(h_pred);

	printf(">Tiempo de ejecución %f ms \n",tiempo);
	//printf(">Tiempo de ejecución cpu %i ms \n",b-a);
	//getchar();
}//end while

}
