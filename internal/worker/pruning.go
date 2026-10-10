package worker

import (
	"math"
	"sort"

	"github.com/WHU-STAR/TITAN/pkg/logger"
	"go.uber.org/zap"
)

type nodeBucketCandidates = map[uint32]struct {
	Buckets   []uint32
	Distances []float32
}

type bucketCandidate struct {
	id       uint32
	distance float32
}

// pruneBuckets dispatches to the configured pruning strategy.
// pruningLambda controls aggressiveness for every strategy:
//   - angular: minimum recall guarantee (1 - lambda max volume loss)
//   - centroid-distance / distance-minus-radius: distance-score threshold interpolation
//   - fixed-topk: keep ceil(lambda * n) closest buckets per worker
func (w *Worker) pruneBuckets(nodeDict nodeBucketCandidates) nodeBucketCandidates {
	if w.pruningLambda <= 0 {
		return nodeDict
	}

	switch w.pruningStrategy {
	case PruningStrategyCentroidDistance:
		return w.pruneBucketsCentroidDistance(nodeDict)
	case PruningStrategyDistanceMinusRadius:
		return w.pruneBucketsDistanceMinusRadius(nodeDict)
	case PruningStrategyFixedTopK:
		return w.pruneBucketsFixedTopK(nodeDict)
	case PruningStrategyAngularV2:
		return w.pruneBucketsAngularV2(nodeDict)
	default:
		return w.pruneBucketsAngular(nodeDict)
	}
}

func clampLambda(lambda float64) float64 {
	if lambda > 1 {
		return 1
	}
	if lambda < 0 {
		return 0
	}
	return lambda
}

func (w *Worker) bucketRadius(bucketID uint32) float64 {
	if int(bucketID) < len(w.bucketRadiuses) {
		return float64(w.bucketRadiuses[bucketID])
	}
	return 1
}

func sliceBucketsFrom(candidates []bucketCandidate, start int) ([]uint32, []float32) {
	if start >= len(candidates) {
		return nil, nil
	}
	kept := candidates[start:]
	buckets := make([]uint32, len(kept))
	distances := make([]float32, len(kept))
	for i, c := range kept {
		buckets[i] = c.id
		distances[i] = c.distance
	}
	return buckets, distances
}

func sliceBucketsUpTo(candidates []bucketCandidate, count int) ([]uint32, []float32) {
	if count <= 0 || len(candidates) == 0 {
		return nil, nil
	}
	if count > len(candidates) {
		count = len(candidates)
	}
	kept := candidates[:count]
	buckets := make([]uint32, len(kept))
	distances := make([]float32, len(kept))
	for i, c := range kept {
		buckets[i] = c.id
		distances[i] = c.distance
	}
	return buckets, distances
}

func sortCandidatesByDistance(candidates []bucketCandidate, descending bool) {
	sort.Slice(candidates, func(i, j int) bool {
		if descending {
			return candidates[i].distance > candidates[j].distance
		}
		return candidates[i].distance < candidates[j].distance
	})
}

func buildNodeBucketResult(nodeIndex uint32, buckets []uint32, distances []float32, result nodeBucketCandidates) {
	if len(buckets) == 0 {
		return
	}
	result[nodeIndex] = struct {
		Buckets   []uint32
		Distances []float32
	}{
		Buckets:   buckets,
		Distances: distances,
	}
}

func (w *Worker) bucketSizeWeight(bucketID uint32) float64 {
	if w.totalNumVectors > 0 && int(bucketID) < len(w.bucketSizes) {
		return float64(w.bucketSizes[bucketID]) / float64(w.totalNumVectors)
	}
	if len(w.bucketRadiuses) > 0 {
		return 1.0 / float64(len(w.bucketRadiuses))
	}
	return 1.0
}

func (w *Worker) calculateAngularVolumeRatio(bucketID uint32, distance float64, totalBuckets int) float64 {
	_ = totalBuckets
	radius := w.bucketRadius(bucketID)

	var angle float64
	if distance > 0 {
		angle = math.Atan(radius / distance)
	} else {
		angle = math.Pi / 2
	}

	logger.Debug("calculateAngularVolumeRatio",
		zap.Float64("radius", radius),
		zap.Float64("distance", distance),
		zap.Float64("angle", angle),
	)

	return angle * angle
}

// calculateAngularV2VolumeRatio extends angular pruning with |B_b|/N bucket cardinality weight.
func (w *Worker) calculateAngularV2VolumeRatio(bucketID uint32, distance float64) float64 {
	return w.calculateAngularVolumeRatio(bucketID, distance, 0) * w.bucketSizeWeight(bucketID)
}

// pruneBucketsAngular is TITAN's solid-angle heuristic: prune far buckets while
// the cumulative estimated volume loss stays below (1 - lambda).
func (w *Worker) pruneBucketsAngular(nodeDict nodeBucketCandidates) nodeBucketCandidates {
	result := make(nodeBucketCandidates, len(nodeDict))
	lambda := clampLambda(w.pruningLambda)
	maxAllowedLoss := 1 - lambda

	totalOriginalBuckets := 0
	totalPrunedBuckets := 0

	for nodeIndex, buckets := range nodeDict {
		candidates := make([]bucketCandidate, len(buckets.Buckets))
		for i := range buckets.Buckets {
			candidates[i] = bucketCandidate{
				id:       buckets.Buckets[i],
				distance: buckets.Distances[i],
			}
		}
		sortCandidatesByDistance(candidates, true)

		cumulativeLossRatio := 0.0
		keepStartIdx := 0
		for i, candidate := range candidates {
			currentLossRatio := w.calculateAngularVolumeRatio(
				candidate.id,
				float64(candidate.distance),
				len(candidates),
			)
			if cumulativeLossRatio+currentLossRatio <= maxAllowedLoss {
				cumulativeLossRatio += currentLossRatio
				continue
			}
			keepStartIdx = i
			break
		}

		totalOriginalBuckets += len(candidates)
		totalPrunedBuckets += keepStartIdx
		prunedBuckets, prunedDistances := sliceBucketsFrom(candidates, keepStartIdx)
		buildNodeBucketResult(nodeIndex, prunedBuckets, prunedDistances, result)
	}

	logPruningSummary(w.pruningStrategy, totalOriginalBuckets, totalPrunedBuckets, w.pruningLambda)
	return result
}

// pruneBucketsAngularV2 applies solid-angle pruning weighted by bucket cardinality |B_b|/N.
func (w *Worker) pruneBucketsAngularV2(nodeDict nodeBucketCandidates) nodeBucketCandidates {
	result := make(nodeBucketCandidates, len(nodeDict))
	lambda := clampLambda(w.pruningLambda)
	maxAllowedLoss := 1 - lambda

	totalOriginalBuckets := 0
	totalPrunedBuckets := 0

	for nodeIndex, buckets := range nodeDict {
		candidates := make([]bucketCandidate, len(buckets.Buckets))
		for i := range buckets.Buckets {
			candidates[i] = bucketCandidate{
				id:       buckets.Buckets[i],
				distance: buckets.Distances[i],
			}
		}
		sortCandidatesByDistance(candidates, true)

		cumulativeLossRatio := 0.0
		keepStartIdx := 0
		for i, candidate := range candidates {
			currentLossRatio := w.calculateAngularV2VolumeRatio(
				candidate.id,
				float64(candidate.distance),
			)
			if cumulativeLossRatio+currentLossRatio <= maxAllowedLoss {
				cumulativeLossRatio += currentLossRatio
				continue
			}
			keepStartIdx = i
			break
		}

		totalOriginalBuckets += len(candidates)
		totalPrunedBuckets += keepStartIdx
		prunedBuckets, prunedDistances := sliceBucketsFrom(candidates, keepStartIdx)
		buildNodeBucketResult(nodeIndex, prunedBuckets, prunedDistances, result)
	}

	logPruningSummary(w.pruningStrategy, totalOriginalBuckets, totalPrunedBuckets, w.pruningLambda)
	return result
}

// pruneBucketsCentroidDistance keeps buckets whose centroid distance is within
// a lambda-interpolated threshold between the nearest and farthest candidate.
func (w *Worker) pruneBucketsCentroidDistance(nodeDict nodeBucketCandidates) nodeBucketCandidates {
	result := make(nodeBucketCandidates, len(nodeDict))
	lambda := clampLambda(w.pruningLambda)

	totalOriginalBuckets := 0
	totalPrunedBuckets := 0

	for nodeIndex, buckets := range nodeDict {
		candidates := make([]bucketCandidate, len(buckets.Buckets))
		for i := range buckets.Buckets {
			candidates[i] = bucketCandidate{
				id:       buckets.Buckets[i],
				distance: buckets.Distances[i],
			}
		}
		sortCandidatesByDistance(candidates, false)

		keepCount := len(candidates)
		if len(candidates) > 0 {
			dMin := float64(candidates[0].distance)
			dMax := float64(candidates[len(candidates)-1].distance)
			threshold := dMin + (1-lambda)*(dMax-dMin)
			for i, candidate := range candidates {
				if float64(candidate.distance) > threshold {
					keepCount = i
					break
				}
			}
		}

		totalOriginalBuckets += len(candidates)
		totalPrunedBuckets += len(candidates) - keepCount
		keptBuckets, keptDistances := sliceBucketsUpTo(candidates, keepCount)
		buildNodeBucketResult(nodeIndex, keptBuckets, keptDistances, result)
	}

	logPruningSummary(w.pruningStrategy, totalOriginalBuckets, totalPrunedBuckets, w.pruningLambda)
	return result
}

// pruneBucketsDistanceMinusRadius ranks buckets by (centroid distance - radius)
// and keeps those within a lambda-interpolated score threshold.
func (w *Worker) pruneBucketsDistanceMinusRadius(nodeDict nodeBucketCandidates) nodeBucketCandidates {
	result := make(nodeBucketCandidates, len(nodeDict))
	lambda := clampLambda(w.pruningLambda)

	totalOriginalBuckets := 0
	totalPrunedBuckets := 0

	for nodeIndex, buckets := range nodeDict {
		candidates := make([]bucketCandidate, len(buckets.Buckets))
		for i := range buckets.Buckets {
			candidates[i] = bucketCandidate{
				id:       buckets.Buckets[i],
				distance: buckets.Distances[i],
			}
		}

		sort.Slice(candidates, func(i, j int) bool {
			scoreI := float64(candidates[i].distance) - w.bucketRadius(candidates[i].id)
			scoreJ := float64(candidates[j].distance) - w.bucketRadius(candidates[j].id)
			return scoreI < scoreJ
		})

		keepCount := len(candidates)
		if len(candidates) > 0 {
			scoreMin := float64(candidates[0].distance) - w.bucketRadius(candidates[0].id)
			scoreMax := float64(candidates[len(candidates)-1].distance) - w.bucketRadius(candidates[len(candidates)-1].id)
			threshold := scoreMin + (1-lambda)*(scoreMax-scoreMin)
			for i, candidate := range candidates {
				score := float64(candidate.distance) - w.bucketRadius(candidate.id)
				if score > threshold {
					keepCount = i
					break
				}
			}
		}

		totalOriginalBuckets += len(candidates)
		totalPrunedBuckets += len(candidates) - keepCount
		keptBuckets, keptDistances := sliceBucketsUpTo(candidates, keepCount)
		buildNodeBucketResult(nodeIndex, keptBuckets, keptDistances, result)
	}

	logPruningSummary(w.pruningStrategy, totalOriginalBuckets, totalPrunedBuckets, w.pruningLambda)
	return result
}

// pruneBucketsFixedTopK keeps the closest ceil(lambda * n) buckets per worker node.
func (w *Worker) pruneBucketsFixedTopK(nodeDict nodeBucketCandidates) nodeBucketCandidates {
	result := make(nodeBucketCandidates, len(nodeDict))
	lambda := clampLambda(w.pruningLambda)

	totalOriginalBuckets := 0
	totalPrunedBuckets := 0

	for nodeIndex, buckets := range nodeDict {
		candidates := make([]bucketCandidate, len(buckets.Buckets))
		for i := range buckets.Buckets {
			candidates[i] = bucketCandidate{
				id:       buckets.Buckets[i],
				distance: buckets.Distances[i],
			}
		}
		sortCandidatesByDistance(candidates, false)

		keepCount := int(math.Ceil(lambda * float64(len(candidates))))
		if keepCount < 1 {
			keepCount = 1
		}
		if keepCount > len(candidates) {
			keepCount = len(candidates)
		}

		totalOriginalBuckets += len(candidates)
		totalPrunedBuckets += len(candidates) - keepCount
		keptBuckets, keptDistances := sliceBucketsUpTo(candidates, keepCount)
		buildNodeBucketResult(nodeIndex, keptBuckets, keptDistances, result)
	}

	logPruningSummary(w.pruningStrategy, totalOriginalBuckets, totalPrunedBuckets, w.pruningLambda)
	return result
}

func logPruningSummary(strategy PruningStrategy, totalOriginalBuckets, totalPrunedBuckets int, lambda float64) {
	if totalPrunedBuckets <= 0 || totalOriginalBuckets <= 0 {
		return
	}
	logger.Debug("Pruning summary",
		zap.String("strategy", strategy.String()),
		zap.Int("totalOriginalBuckets", totalOriginalBuckets),
		zap.Int("totalPrunedBuckets", totalPrunedBuckets),
		zap.Float64("pruningRate", float64(totalPrunedBuckets)/float64(totalOriginalBuckets)),
		zap.Float64("pruningLambda", lambda))
}

func (w *Worker) sortBuckets(nodeDict nodeBucketCandidates) nodeBucketCandidates {
	result := make(nodeBucketCandidates, len(nodeDict))

	for nodeID, data := range nodeDict {
		candidates := make([]bucketCandidate, len(data.Buckets))
		for i := range data.Buckets {
			candidates[i] = bucketCandidate{
				id:       data.Buckets[i],
				distance: data.Distances[i],
			}
		}
		sortCandidatesByDistance(candidates, true)
		buckets, distances := sliceBucketsFrom(candidates, 0)
		buildNodeBucketResult(nodeID, buckets, distances, result)
	}
	return result
}
