package worker

import (
	"math"
)

// SIMDDistance computes a distance with SIMD when available.
func SIMDDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	dim := len(a)
	if dim == 0 {
		return 0
	}

	if dist, ok := simdL2(a, b); ok {
		return dist
	}
	return scalarL2Distance(a, b)
}

// Scalar fallback.
func scalarL2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// BatchSIMDDistance computes many distances.
func BatchSIMDDistance(query []float32, candidates [][]float32) []float32 {
	results := make([]float32, len(candidates))

	// Batch the work for better cache use.
	for i, candidate := range candidates {
		results[i] = SIMDDistance(query, candidate)
	}

	return results
}

// SIMDDistanceToCentroids distances a query against every centroid.
func SIMDDistanceToCentroids(query []float32, centroids [][]float32) []float32 {
	results := make([]float32, len(centroids))

	for i, centroid := range centroids {
		results[i] = SIMDDistance(query, centroid)
	}

	return results
}

// InnerProductSIMD computes an inner product.
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

// L2DistanceSquaredSIMD returns squared L2 without sqrt.
func L2DistanceSquaredSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	dim := len(a)
	if dim == 0 {
		return 0
	}

	if dist, ok := simdL2Squared(a, b); ok {
		return dist
	}
	return scalarL2DistanceSquared(a, b)
}

// Scalar squared-L2 fallback.
func scalarL2DistanceSquared(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// BatchL2DistanceSquaredSIMD computes squared L2 for a PQ table.
func BatchL2DistanceSquaredSIMD(query []float32, candidates [][]float32) []float32 {
	results := make([]float32, len(candidates))

	// Batch the work for better cache use.
	for i, candidate := range candidates {
		results[i] = L2DistanceSquaredSIMD(query, candidate)
	}

	return results
}
