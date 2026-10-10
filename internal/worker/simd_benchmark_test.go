package worker

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// 生成随机向量（测试用）
func generateRandomVectorTest(dim int, rng *rand.Rand) []float32 {
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = rng.Float32()*2 - 1 // [-1, 1]
	}
	return vector
}

// 标量距离计算（测试用）
func scalarDistanceTest(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// 测试SIMD距离计算的正确性
func TestSIMDDistanceCorrectness(t *testing.T) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	testCases := []struct {
		dim int
	}{
		{128},  // 常见维度
		{256},  // 中等维度
		{512},  // 较大维度
		{1024}, // 大维度
		{7},    // 不能被4整除的维度
		{15},   // 不能被8整除的维度
		{31},   // 不能被4和8整除的维度
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("dim_%d", tc.dim), func(t *testing.T) {
			// 生成测试向量
			a := generateRandomVectorTest(tc.dim, rng)
			b := generateRandomVectorTest(tc.dim, rng)

			// 计算标量距离
			scalarResult := scalarDistanceTest(a, b)

			// 计算SIMD距离
			simdResult := SIMDDistance(a, b)

			// 检查结果是否接近（考虑浮点精度误差）
			diff := float32(math.Abs(float64(scalarResult - simdResult)))
			if diff > 1e-5 {
				t.Errorf("维度 %d: SIMD结果 %.6f 与标量结果 %.6f 差异过大: %.6f",
					tc.dim, simdResult, scalarResult, diff)
			}
		})
	}
}

// 性能基准测试
func BenchmarkDistanceCalculation(b *testing.B) {
	dimensions := []int{128, 256, 512, 1024}

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))

			// 生成测试向量
			a := generateRandomVectorTest(dim, rng)
			bVec := generateRandomVectorTest(dim, rng)

			b.ResetTimer()

			// 测试标量实现
			b.Run("scalar", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_ = scalarDistanceTest(a, bVec)
				}
			})

			// 测试SIMD实现
			b.Run("simd", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_ = SIMDDistance(a, bVec)
				}
			})
		})
	}
}

// 批量距离计算性能测试
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

	// 测试标量批量实现
	b.Run("scalar_batch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			results := make([]float32, numCandidates)
			for j, candidate := range candidates {
				results[j] = scalarDistanceTest(query, candidate)
			}
			_ = results
		}
	})

	// 测试SIMD批量实现
	b.Run("simd_batch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = BatchSIMDDistance(query, candidates)
		}
	})
}

// 聚类中心距离计算性能测试
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

	// 测试标量实现
	b.Run("scalar_centroids", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			results := make([]float32, numCentroids)
			for j, centroid := range centroids {
				results[j] = scalarDistanceTest(query, centroid)
			}
			_ = results
		}
	})

	// 测试SIMD实现
	b.Run("simd_centroids", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = SIMDDistanceToCentroids(query, centroids)
		}
	})
}
