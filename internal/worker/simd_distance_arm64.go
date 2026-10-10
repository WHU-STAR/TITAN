//go:build arm64

package worker

import (
	"unsafe"
)

// 汇编函数声明（仅在arm64平台上编译）
//
//go:noescape
func l2DistanceNEON(a, b unsafe.Pointer, dim int) float32
