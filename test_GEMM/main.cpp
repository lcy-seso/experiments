#include <mkl.h>

int main() {
  void* darray;
  int darray_size = 1000;
  int alignment = 64;
  darray = mkl_malloc(sizeof(double) * darray_size, alignment);
  mkl_free(darray);
  return 0;
}
