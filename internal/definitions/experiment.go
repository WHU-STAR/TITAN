package definitions

import "fmt"

// ExperimentParams 单次实验参数
type ExperimentParams struct {
	FinalTopK     int
	WorkerTopK    int
	EfSearch      int
	MetaK         int
	Threshold     int
	NProbe        int
	PruningLambda float64
}

func (ep *ExperimentParams) ToString() string {
	return fmt.Sprintf("FinalTopK: %d, WorkerTopK: %d, EfSearch: %d, MetaK: %d, Threshold: %d, NProbe: %d, PruningLambda: %.2f",
		ep.FinalTopK, ep.WorkerTopK, ep.EfSearch, ep.MetaK, ep.Threshold, ep.NProbe, ep.PruningLambda)
}
