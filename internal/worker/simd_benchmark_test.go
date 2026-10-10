package worker

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// Random test vector.
func generateRandomVectorTest(dim int, rng *rand.Rand) []float32 {
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = rng.Float32()*2 - 1 // [-1, 1]
	}
	return vector
}

// Scalar distance used as the reference.
func scalarDistanceTest(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// TestSIMDDistance checks SIMD against the scalar reference.
func TestSIMDDistanceCorrectness(t *testing.T) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	testCases := []struct {
		dim int
	}{
		{128},  // common dimension
		{256},  // medium dimension
		{512},  // larger dimension
		{1024}, // large dimension
		{7},    // not divisible by 4
		{15},   // not divisible by 8
		{31},   // not divisible by 4 or 8
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("dim_%d", tc.dim), func(t *testing.T) {
			// Build test vectors.
			a := generateRandomVectorTest(tc.dim, rng)
			b := generateRandomVectorTest(tc.dim, rng)

			// Scalar distance.
			scalarResult := scalarDistanceTest(a, b)

			// SIMD distance.
			simdResult := SIMDDistance(a, b)

			// Allow a small floating-point gap.
			diff := float32(math.Abs(float64(scalarResult - simdResult)))
			if diff > 1e-5 {
				t.Errorf("dim %d: SIMD %.6f differs from scalar %.6f by %.6f",
					tc.dim, simdResult, scalarResult, diff)
			}
		})
	}
}

// Throughput benchmark.
func BenchmarkDistanceCalculation(b *testing.B) {
	dimensions := []int{128, 256, 512, 1024}

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))

			// Build test vectors.
			a := generateRandomVectorTest(dim, rng)
			bVec := generateRandomVectorTest(dim, rng)

			b.ResetTimer()

			// Scalar implementation.
			b.Run("scalar", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_ = scalarDistanceTest(a, bVec)
				}
			})

			// SIMD implementation.
			b.Run("simd", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_ = SIMDDistance(a, bVec)
				}
			})
		})
	}
}

// Batch distance benchmark.
func BenchmarkBatchDistanceCalculation(b *testing.B) {
	dim := 256
	numCandidates := 1000

	rng := rand.New(rand.NewSource(42))
	query := generateRandomVectorTest(dim, rng)
	candidates := make([][]float32, numCandidates)

	for i := range candidates {
		candidates[i] = generateRandomVectorTest(dim, rng)
	}

	b.ResetTimer()

	// Scalar batch.
	b.Run("scalar_batch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			results := make([]float32, numCandidates)
			for j, candidate := range candidates {
				results[j] = scalarDistanceTest(query, candidate)
			}
			_ = results
		}
	})

	// SIMD batch.
	b.Run("simd_batch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = BatchSIMDDistance(query, candidates)
		}
	})
}

// Centroid distance benchmark.
func BenchmarkCentroidDistanceCalculation(b *testing.B) {
	dim := 256
	numCentroids := 1000

	rng := rand.New(rand.NewSource(42))
	query := generateRandomVectorTest(dim, rng)
	centroids := make([][]float32, numCentroids)

	for i := range centroids {
		centroids[i] = generateRandomVectorTest(dim, rng)
	}

	b.ResetTimer()

	// Scalar implementation.
	b.Run("scalar_centroids", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			results := make([]float32, numCentroids)
			for j, centroid := range centroids {
				results[j] = scalarDistanceTest(query, centroid)
			}
			_ = results
		}
	})

	// SIMD implementation.
	b.Run("simd_centroids", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = SIMDDistanceToCentroids(query, centroids)
		}
	})
}
