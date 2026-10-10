package example

/*
#cgo CFLAGS: -I${SRCDIR}/../../c_api/example
#cgo darwin LDFLAGS: -L${SRCDIR}/../../build -lexample -Wl,-rpath,${SRCDIR}/../../build
#cgo !darwin LDFLAGS: -L${SRCDIR}/../../build -lexample
#include "example_api.h"
*/
import "C"

// Add 调用C API的add函数
func Add(a, b int) int {
	return int(C.add(C.int(a), C.int(b)))
}

// Multiply 调用C API的multiply函数
func Multiply(a, b float64) float64 {
	return float64(C.multiply(C.double(a), C.double(b)))
}

// GetMessage 调用C API的get_message函数
func GetMessage() string {
	cStr := C.get_message()
	return C.GoString(cStr)
}
