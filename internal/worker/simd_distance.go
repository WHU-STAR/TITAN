package worker

import (
	"math"
	"unsafe"
)

// SIMDDistance 使用SIMD优化的距离计算
func SIMDDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	dim := len(a)
	if dim == 0 {
		return 0
	}

	// 获取向量的底层数据指针
	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])

	// 根据CPU特性选择最优的SIMD实现
	// 优先级：AVX2 > SSE2 > NEON > 标量

	// 尝试AVX2
	if hasAVX2() {
		return l2DistanceAVX2(aPtr, bPtr, dim)
	}

	// 尝试SSE2
	if hasSSE2() {
		return l2DistanceSSE2(aPtr, bPtr, dim)
	}

	// 尝试NEON (ARM) - 仅在arm64平台可用
	if hasNEON() {
		// 这里需要条件编译，暂时注释掉
		// return l2DistanceNEON(aPtr, bPtr, dim)
	}

	// 回退到标量实现
	return scalarL2Distance(a, b)
}

// 标量实现作为回退
func scalarL2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// BatchSIMDDistance 批量计算距离，提高缓存效率
func BatchSIMDDistance(query []float32, candidates [][]float32) []float32 {
	results := make([]float32, len(candidates))

	// 批量处理，提高缓存局部性
	for i, candidate := range candidates {
		results[i] = SIMDDistance(query, candidate)
	}

	return results
}

// SIMDDistanceToCentroids 计算查询向量到所有聚类中心的距离
func SIMDDistanceToCentroids(query []float32, centroids [][]float32) []float32 {
	results := make([]float32, len(centroids))

	for i, centroid := range centroids {
		results[i] = SIMDDistance(query, centroid)
	}

	return results
}

// InnerProductSIMD 计算内积（点积）
func InnerProductSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}

// L2DistanceSquaredSIMD 使用SIMD优化的L2距离平方计算（不计算sqrt，用于PQ距离表）
func L2DistanceSquaredSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	dim := len(a)
	if dim == 0 {
		return 0
	}

	// 获取向量的底层数据指针
	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])

	// 根据CPU特性选择最优的SIMD实现
	// 尝试AVX2
	if hasAVX2() {
		return l2DistanceSquaredAVX2(aPtr, bPtr, dim)
	}

	// 尝试SSE2
	if hasSSE2() {
		return l2DistanceSquaredSSE2(aPtr, bPtr, dim)
	}

	// 回退到标量实现
	return scalarL2DistanceSquared(a, b)
}

// 标量实现作为回退（距离平方，不计算sqrt）
func scalarL2DistanceSquared(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// BatchL2DistanceSquaredSIMD 批量计算L2距离平方（用于PQ距离表）
func BatchL2DistanceSquaredSIMD(query []float32, candidates [][]float32) []float32 {
	results := make([]float32, len(candidates))

	// 批量处理，提高缓存局部性
	for i, candidate := range candidates {
		results[i] = L2DistanceSquaredSIMD(query, candidate)
	}

	return results
}
