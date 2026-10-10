package worker

import "testing"

func newTestWorker(strategy PruningStrategy, lambda float64, radii []float32) *Worker {
	return &Worker{
		pruningStrategy: strategy,
		pruningLambda:   lambda,
		bucketRadiuses:  radii,
	}
}

func TestPruneBucketsAngular(t *testing.T) {
	w := newTestWorker(PruningStrategyAngular, 0.8, []float32{1, 1, 1, 1})
	nodeDict := nodeBucketCandidates{
		0: {
			Buckets:   []uint32{0, 1, 2, 3},
			Distances: []float32{1, 2, 5, 10},
		},
	}

	result := w.pruneBucketsAngular(nodeDict)
	kept := result[0]
	if len(kept.Buckets) == 0 {
		t.Fatal("expected to keep at least one bucket")
	}
	if len(kept.Buckets) == len(nodeDict[0].Buckets) {
		t.Fatal("expected angular pruning to remove some far buckets")
	}
}

func TestPruneBucketsCentroidDistance(t *testing.T) {
	w := newTestWorker(PruningStrategyCentroidDistance, 0.5, nil)
	nodeDict := nodeBucketCandidates{
		0: {
			Buckets:   []uint32{0, 1, 2, 3},
			Distances: []float32{1, 3, 5, 9},
		},
	}

	result := w.pruneBucketsCentroidDistance(nodeDict)
	kept := result[0]
	if len(kept.Buckets) != 3 {
		t.Fatalf("expected 3 buckets at lambda=0.5, got %d (%v)", len(kept.Buckets), kept.Buckets)
	}
	if kept.Buckets[0] != 0 || kept.Buckets[1] != 1 || kept.Buckets[2] != 2 {
		t.Fatalf("unexpected kept buckets: %v", kept.Buckets)
	}
}

func TestPruneBucketsDistanceMinusRadius(t *testing.T) {
	w := newTestWorker(PruningStrategyDistanceMinusRadius, 0.5, []float32{0, 0, 4, 8})
	nodeDict := nodeBucketCandidates{
		0: {
			Buckets:   []uint32{0, 1, 2, 3},
			Distances: []float32{1, 3, 5, 9},
		},
	}

	result := w.pruneBucketsDistanceMinusRadius(nodeDict)
	if len(result[0].Buckets) == 0 {
		t.Fatal("expected to keep at least one bucket")
	}
}

func TestPruneBucketsFixedTopK(t *testing.T) {
	w := newTestWorker(PruningStrategyFixedTopK, 0.5, nil)
	nodeDict := nodeBucketCandidates{
		0: {
			Buckets:   []uint32{0, 1, 2, 3},
			Distances: []float32{9, 5, 3, 1},
		},
	}

	result := w.pruneBucketsFixedTopK(nodeDict)
	kept := result[0]
	if len(kept.Buckets) != 2 {
		t.Fatalf("expected 2 buckets for fixed-topk with lambda=0.5, got %d", len(kept.Buckets))
	}
	if kept.Buckets[0] != 3 || kept.Buckets[1] != 2 {
		t.Fatalf("expected closest buckets [3 2], got %v", kept.Buckets)
	}
}

func TestPruneBucketsDisabledWhenLambdaNonPositive(t *testing.T) {
	w := newTestWorker(PruningStrategyFixedTopK, 0, nil)
	nodeDict := nodeBucketCandidates{
		0: {
			Buckets:   []uint32{0, 1},
			Distances: []float32{1, 2},
		},
	}

	result := w.pruneBuckets(nodeDict)
	if len(result[0].Buckets) != 2 {
		t.Fatalf("expected no pruning when lambda <= 0, got %d buckets", len(result[0].Buckets))
	}
}

func TestPruneBucketsAngularV2(t *testing.T) {
	w := &Worker{
		pruningStrategy: PruningStrategyAngularV2,
		pruningLambda:   0.8,
		bucketRadiuses:  []float32{1, 1, 1, 1},
		bucketSizes:     []uint32{100, 100, 500, 100},
		totalNumVectors: 1000,
	}
	nodeDict := nodeBucketCandidates{
		0: {
			Buckets:   []uint32{0, 1, 2, 3},
			Distances: []float32{1, 2, 5, 10},
		},
	}

	result := w.pruneBucketsAngularV2(nodeDict)
	kept := result[0]
	if len(kept.Buckets) == 0 {
		t.Fatal("expected to keep at least one bucket")
	}
	// bucket 2 has largest |B|/N weight; v2 should prune it later than v1 would
	v1 := newTestWorker(PruningStrategyAngular, 0.8, []float32{1, 1, 1, 1})
	v1Result := v1.pruneBucketsAngular(nodeDict)
	if len(kept.Buckets) < len(v1Result[0].Buckets) {
		t.Fatalf("expected v2 to keep >= buckets than v1 for large bucket, v1=%d v2=%d",
			len(v1Result[0].Buckets), len(kept.Buckets))
	}
}

func TestBucketSizeWeight(t *testing.T) {
	w := &Worker{
		bucketSizes:     []uint32{10, 20},
		totalNumVectors: 100,
	}
	if got := w.bucketSizeWeight(1); got != 0.2 {
		t.Fatalf("expected 0.2, got %v", got)
	}
}

func TestParsePruningStrategy(t *testing.T) {
	cases := map[string]PruningStrategy{
		"":                         PruningStrategyAngular,
		"angular":                  PruningStrategyAngular,
		"angular-v2":               PruningStrategyAngularV2,
		"hf-v2":                    PruningStrategyAngularV2,
		"centroid-distance":        PruningStrategyCentroidDistance,
		"distance-minus-radius":    PruningStrategyDistanceMinusRadius,
		"fixed-topk":                 PruningStrategyFixedTopK,
		"unknown-strategy":           PruningStrategyAngular,
	}
	for input, want := range cases {
		if got := ParsePruningStrategy(input); got != want {
			t.Fatalf("ParsePruningStrategy(%q) = %q, want %q", input, got, want)
		}
	}
}
