//go:build !amd64

package worker

func simdL2(a, b []float32) (float32, bool) {
	return 0, false
}

func simdL2Squared(a, b []float32) (float32, bool) {
	return 0, false
}
