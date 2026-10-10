package definitions

import "testing"

func TestPercentile(t *testing.T) {
	sorted := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if got := Percentile(sorted, 50); got != 5.5 {
		t.Fatalf("p50 = %v, want 5.5", got)
	}
	if got := Percentile(sorted, 95); got < 9.5 || got > 9.6 {
		t.Fatalf("p95 = %v, want ~9.55", got)
	}
	if got := Percentile(sorted, 99); got < 9.9 || got > 10.0 {
		t.Fatalf("p99 = %v, want ~9.91", got)
	}
}
