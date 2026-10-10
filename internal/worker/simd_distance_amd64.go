//go:build amd64

package worker

import (
	"unsafe"
)

// Assembly entry points for amd64.
//
//go:noescape
func l2DistanceAVX2(a, b unsafe.Pointer, dim int) float32

//go:noescape
func l2DistanceSSE2(a, b unsafe.Pointer, dim int) float32

//go:noescape
func l2DistanceSquaredAVX2(a, b unsafe.Pointer, dim int) float32

//go:noescape
func l2DistanceSquaredSSE2(a, b unsafe.Pointer, dim int) float32

func simdL2(a, b []float32) (float32, bool) {
	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])
	dim := len(a)
	if hasAVX2() {
		return l2DistanceAVX2(aPtr, bPtr, dim), true
	}
	if hasSSE2() {
		return l2DistanceSSE2(aPtr, bPtr, dim), true
	}
	return 0, false
}

func simdL2Squared(a, b []float32) (float32, bool) {
	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])
	dim := len(a)
	if hasAVX2() {
		return l2DistanceSquaredAVX2(aPtr, bPtr, dim), true
	}
	if hasSSE2() {
		return l2DistanceSquaredSSE2(aPtr, bPtr, dim), true
	}
	return 0, false
}
