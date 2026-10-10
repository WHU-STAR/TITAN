#include "example_api.h"
#include <cstring>

extern "C" {
int add(int a, int b) { return a + b; }

double multiply(double a, double b) { return a * b; }

char *get_message() {
  static char message[] = "Hello from C++!";
  return message;
}
}