package worker

import "strings"

// PruningStrategy selects which bucket pruning heuristic to apply at query time.
type PruningStrategy string

const (
	PruningStrategyAngular             PruningStrategy = "angular"
	PruningStrategyCentroidDistance    PruningStrategy = "centroid-distance"
	PruningStrategyDistanceMinusRadius PruningStrategy = "distance-minus-radius"
	PruningStrategyFixedTopK           PruningStrategy = "fixed-topk"
	PruningStrategyAngularV2           PruningStrategy = "angular-v2"
)

// ParsePruningStrategy normalizes user/config input into a supported strategy.
func ParsePruningStrategy(raw string) PruningStrategy {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "angular", "hf", "solid-angle", "solid_angle":
		return PruningStrategyAngular
	case "centroid-distance", "centroid_distance", "centroid":
		return PruningStrategyCentroidDistance
	case "distance-minus-radius", "distance_minus_radius", "dmr", "radius":
		return PruningStrategyDistanceMinusRadius
	case "fixed-topk", "fixed_topk", "topk", "fixed-top-k":
		return PruningStrategyFixedTopK
	case "angular-v2", "angular_v2", "hf-v2", "hf_v2":
		return PruningStrategyAngularV2
	default:
		return PruningStrategyAngular
	}
}

func (s PruningStrategy) String() string {
	return string(s)
}
