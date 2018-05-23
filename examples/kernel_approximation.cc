#include <cstdio>
#include <cstring>
#include <sstream>
#include <fstream>
#include <string>
#include <cassert>
#include <iostream>
#include <cstdio>
#include "random_layer.h"
#include "math_functions.h"
#include "utils.h"
#include<vector>
#include <algorithm>
using namespace std;
void read_data(char* file_name, float **X,  float **y, int &num_data, int &dim_data){
  FILE* fp;
  int buffer_size = 10000;
  char* tok;
  char* buffer = new char [buffer_size];
  int ncount = 0;
  int dcount = 0;
  fp = fopen(file_name,"r");
  if(fp == NULL){
    cout << "Error in opening file: " << file_name << endl;
    exit(0);
  }
  if(fgets(buffer, buffer_size, fp) != NULL){
    tok = strtok(buffer," ");
    num_data = atoi(tok);
    tok = strtok(NULL," ");
    dim_data = atoi(tok);
  }
  *X = new float [num_data * dim_data];
  *y = new float [num_data];
  int counter = 0;
  while(fgets(buffer, buffer_size, fp) != NULL){
    tok = strtok(buffer," ");
    (*y)[ncount] = (float)atof(tok);
    tok = strtok(NULL," ");
    while(tok != NULL){
      (*X)[ncount * dim_data + dcount]  = (float)atof(tok);
      dcount++;
      tok = strtok(NULL," ");
    }
    if(dcount != dim_data){
      cout << "Dimension mismatch while reading data from file: "<< file_name << " at line number: " << ncount+2<<endl;
      exit(1);
    }
    ncount++;
    dcount = 0;
  }
  if(ncount != num_data){
    cout<< "Mismatch in total number of datapoints in file: "<<file_name<<endl;
    exit(2);
  }
  fclose(fp);
  delete [] buffer;
}

float l2dist(const float *x, const float *y, int d) {
  float xx = cpu_dot(d, x, x);
  float yy = cpu_dot(d, y, y);
  float xy = cpu_dot(d, x, y);
  float dist = xx + yy - float(2) * xy;
  return (dist);
}

void compute_rbf_kernel(const float* x_matrix, float* kernel_matrix,
                        float sigma, int num_data, int dim) {
  cpu_scal(num_data * num_data, float(0), kernel_matrix);
  for(int i = 0; i < num_data; i++) {
    const float *x_matrix_i = x_matrix + i * dim;
    float *kernel_matrix_i = kernel_matrix + i  * num_data;
    kernel_matrix_i[i] = float(1);
    for(int j = i + 1; j < num_data; j++) {
      const float *x_matrix_j = x_matrix + j * dim;
      float dist_ij = l2dist(x_matrix_i, x_matrix_j, dim);
      kernel_matrix_i[j] = exp(-dist_ij   / (2.0 * sigma * sigma));
      float *kernel_matrix_j = kernel_matrix + j  * num_data;
      kernel_matrix_j[i] = kernel_matrix_i[j];
    }
  }
}

float compute_sigma(const float* x_matrix, 
                    int num_data, int dim) {
  float sigma = 0;
  for(int i = 0; i < num_data; i++) {
    const float *x_matrix_i = x_matrix + i * dim;
    vector<float> distance;
    for(int j = 0; j < num_data; j++) {
      const float *x_matrix_j = x_matrix + j * dim;
      float dist_ij = l2dist(x_matrix_i, x_matrix_j, dim);
      distance.push_back(dist_ij);
    }
    sort(distance.begin(), distance.end());
    sigma += sqrt(distance[50]);
  }
  sigma /= float(num_data);
  return sigma;
}

void print_matrix(int M, int N, const float *W) {
  for(int i = 0; i < M * N; i++) {
    cout<<W[i]<<" ";
    if((i + 1) % N == 0) {
      cout<<endl;
    }
  }
  cout<<endl;
}

void save_mean_std(int M, int N, const float *W, string filename) {

  float *mean = new float [N];
  float *std = new float [N];
  memset(mean, 0, N * sizeof(float));
  memset(std, 0, N * sizeof(float));
  for(int i = 0; i < M * N; i++) {
    mean[i % N] += W[i] / float(M);
    std[i % N] += pow(W[i], 2) / float(M);
  }
  for(int  i =0; i < N; i++) {
    std[i] -= pow(mean[i], 2);
    std[i] = sqrt(std[i]);
  }
  ofstream fout(filename);
  for(int i = 0; i < N; i++) {
    fout<<i+1<<" "<<mean[i]<<" "<<std[i]<<endl;
  }
  fout.close();  
  delete [] mean;
  delete [] std;
}


int main() {
  Arena::set_device(Arena::GPU);
  float *data = NULL, *labels = NULL;
  int num_data = 0, num_sdata = 0, dim_data = 0;
  read_data("../datasets/usps.train", &data, &labels, num_data, dim_data);
  num_sdata =  100;
  int dim2_data = pow(2, log2(float(dim_data)));
  cout<<dim2_data<<endl;
  float sigma = compute_sigma(data, 1000, dim_data);
  cout<<"Sigma "<<sigma<<endl;
  float* kernel_matrix = new float [num_sdata * num_sdata];
  float* approximate_kernel_matrix = new float [num_sdata * num_sdata];
  compute_rbf_kernel(data, kernel_matrix, sigma, num_sdata, dim_data);
  cout<<num_data<<" "<<dim_data<<endl;
  SyncedMemory* X = new SyncedMemory(num_sdata * dim_data);
  float *x_ptr = X->mutable_cpu_data();
  memcpy(x_ptr, data, num_sdata * dim_data * sizeof(float));
  int max_blocks = 10;
  int num_trials = 10;
  float *rff_error = new float [num_trials * max_blocks];
  float *fastfood_error = new float [num_trials * max_blocks];
  float *orff_error = new float [num_trials * max_blocks];
  float *sorff_error = new float [num_trials * max_blocks];
  int counter = 0;
  for(int t =0; t < num_trials; t++) {
    RandomLayer *rff_layer = new RechtRahimiRandomLayer(dim_data, max_blocks, sigma); 
    cout<<"Init RFF done\n";
    RandomLayer *fastfood_layer = new FastfoodRandomLayer(dim_data, max_blocks, sigma); 
    RandomLayer *orff_layer = new OrthogonalRandomLayer(dim_data, max_blocks, sigma); 
    RandomLayer *sorff_layer = new StructuredOrthogonalRandomLayer(dim_data, max_blocks, sigma); 
    cout<<"Init KRFF done\n";
    Timer timer;
    for(int i = 1; i <= max_blocks; i++) { 
      double et_rff = 0, et_fastfood = 0, et_orff = 0, et_sorff = 0, et_krff = 0;
      rff_layer->set_num_blocks(i);
      timer.startTimer(); 
      rff_layer->forward(X, num_sdata);
      const float *rff_ptr = (rff_layer->output())->cpu_data();
      timer.stopTimer();
      et_rff  += timer.elapsedTimeSeconds();

      fastfood_layer->set_num_blocks(i);
      timer.startTimer(); 
      fastfood_layer->forward(X, num_sdata);
      const float *fastfood_ptr = (fastfood_layer->output())->cpu_data();
      timer.stopTimer();
      et_fastfood  += timer.elapsedTimeSeconds();
      
      orff_layer->set_num_blocks(i);
      timer.startTimer(); 
      orff_layer->forward(X, num_sdata);
      const float *orff_ptr = (orff_layer->output())->cpu_data();
      timer.stopTimer();
      et_orff  += timer.elapsedTimeSeconds();

      sorff_layer->set_num_blocks(i);
      timer.startTimer(); 
      sorff_layer->forward(X, num_sdata);
      const float *sorff_ptr = (sorff_layer->output())->cpu_data();
      timer.stopTimer();
      et_sorff  += timer.elapsedTimeSeconds();

      krff_layer->set_num_blocks(i);
      timer.startTimer(); 
      krff_layer->forward(X, num_sdata);
      const float *krff_ptr = (krff_layer->output())->cpu_data();
      timer.stopTimer();
      et_krff  += timer.elapsedTimeSeconds();
      
      CHECK_EQ((rff_layer->output())->size(), num_sdata *  i * dim_data);
      CHECK_EQ((fastfood_layer->output())->size(), num_sdata *  i * dim_data);
      CHECK_EQ((orff_layer->output())->size(), num_sdata *  i * dim_data);
      CHECK_EQ((sorff_layer->output())->size(), num_sdata *  i * dim_data);
      cpu_gemm(CblasNoTrans, CblasTrans, num_sdata,
               num_sdata, i * dim_data, (float)1.,
               rff_ptr, rff_ptr, (float)0., approximate_kernel_matrix);
      float rff_approx_error = l2dist(kernel_matrix, approximate_kernel_matrix, num_sdata * num_sdata) / float(num_sdata * num_sdata);

      float fastfood_approx_error = 0;
      cpu_gemm(CblasNoTrans, CblasTrans, num_sdata,
               num_sdata,  i * dim_data, (float)1.,
               fastfood_ptr, fastfood_ptr, (float)0., approximate_kernel_matrix);
      fastfood_approx_error = l2dist(kernel_matrix, approximate_kernel_matrix, num_sdata * num_sdata) / float(num_sdata * num_sdata);

      float orff_approx_error = 0;
      cpu_gemm(CblasNoTrans, CblasTrans, num_sdata,
               num_sdata, i * dim_data, (float)1.,
               orff_ptr, orff_ptr, (float)0., approximate_kernel_matrix);
      orff_approx_error = l2dist(kernel_matrix, approximate_kernel_matrix, num_sdata * num_sdata) / float(num_sdata * num_sdata);

      float sorff_approx_error = 0;
      cpu_gemm(CblasNoTrans, CblasTrans, num_sdata,
               num_sdata, i * dim_data, (float)1.,
               sorff_ptr, sorff_ptr, (float)0., approximate_kernel_matrix);
      sorff_approx_error = l2dist(kernel_matrix, approximate_kernel_matrix, num_sdata * num_sdata) / float(num_sdata * num_sdata);

      
      rff_error[counter] = rff_approx_error;
      fastfood_error[counter] = fastfood_approx_error;
      orff_error[counter] = orff_approx_error;
      sorff_error[counter] = sorff_approx_error;
      counter++;
      cout<<"RFF: "<<rff_approx_error<<":"<<et_rff<<" Fastfood: "<<fastfood_approx_error<<":"<<et_fastfood<<" ORFF: "<<orff_approx_error<<":"<<et_orff<<" SORFF: "<<sorff_approx_error<<":"<<et_sorff<<endl;
    }
    delete rff_layer;
    delete fastfood_layer;
    delete orff_layer;
        delete krff_layer;
  }
  print_matrix(num_trials, max_blocks, rff_error);
  print_matrix(num_trials, max_blocks, fastfood_error);
  print_matrix(num_trials, max_blocks, orff_error);
  print_matrix(num_trials, max_blocks, sorff_error);
  save_mean_std(num_trials, max_blocks, rff_error, "rff.dat");
  save_mean_std(num_trials, max_blocks, fastfood_error, "fastfood.dat");
  save_mean_std(num_trials, max_blocks, orff_error, "orff.dat");
  save_mean_std(num_trials, max_blocks, sorff_error, "sorff.dat");
  delete X;
  delete [] kernel_matrix;
  delete [] approximate_kernel_matrix;
  delete [] rff_error;
  delete [] fastfood_error;
  delete [] orff_error;
  delete [] sorff_error;
  return 0;
}
