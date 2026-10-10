//go:build amd64

package worker

import (
	"unsafe"
)

// 汇编函数声明（仅在amd64平台上编译）
//
//go:noescape
func l2DistanceAVX2(a, b unsafe.Pointer, dim int) float32

//go:noescape
func l2DistanceSSE2(a, b unsafe.Pointer, dim int) float32

//go:noescape
func l2DistanceSquaredAVX2(a, b unsafe.Pointer, dim int) float32

//go:noescape
func l2DistanceSquaredSSE2(a, b unsafe.Pointer, dim int) float32
