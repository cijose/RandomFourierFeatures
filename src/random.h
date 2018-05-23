#ifndef _PRAND_H_
#define _PRAND_H_

class Prand {
  static int const N = 624;
  static int const M = 397;
  static unsigned long const MATRIX_A =  0x9908b0dfUL;   /* constant vector a */
  static unsigned long const UPPER_MASK = 0x80000000UL; /* most significant w-r bits */
  static unsigned long const LOWER_MASK =  0x7fffffffUL; /* least significant r bits */
  int mti; /* mti==N+1 means mt[N] is not initialized */
  unsigned long* mt; /* the array for the state vector  */
  bool return_v;
  double v_val;
 public:
  explicit Prand(unsigned long int seed);
  Prand();
  ~Prand() {delete [] mt;}
  //Generate gaussian random variable
  double gauss_rng();
  //Generate a floating point random variable  between a and b
  double uniform_rng(double a, double b);
  //Generate a integer random variable  between a and b
  int randi(int a, int b);
  //Generates a random number on [0,0xffffffff]-interval
  unsigned long int  randi_32();
  //Generates a random number on [0,0x7fffffff]-interval
  long int  randi_31();
  //Generate a random number on [0, 1] real interval
  double rand_1(void);
  //Generate a random number on [0, 1) real interval
  double rand_2(void);
  //Generate a random number on (0, 1) real interval
  double rand_3(void);
  //Generate a gaussian random variable with mean mu and standard deviation std
  double randn(double mu, double stddev);
};

#endif
