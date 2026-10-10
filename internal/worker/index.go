package worker

import (
	"errors"
)

// Index defines the interface for graph indices
type Index interface {
	Load(path string, dim int, metric string) error
	Search(vector []float32, k int) ([]uint32, []float32, error)
	SearchWithEntry(vector []float32, bucketId uint32, k int) ([]uint32, []float32, error)
	Unload()
	GetAvgHops() float32
	GetAvgDistComputations() float32
	GetDataSize() uint32
	SetEFSearch(efSearch int) error
	SetNProbe(nProbe int) error
	ShowAllLabels() string
}

// NewIndex creates a new index based on the type
func NewIndex(indexType string) (Index, error) {
	switch indexType {
	case "hnsw", "hnsw_impr":
		return &HNSWIndex{}, nil
	case "vamana":
		return &VamanaIndex{}, nil
	case "bf":
		return &BFIndex{}, nil
	case "ivf":
		return &IVFIndex{}, nil
	default:
		return nil, errors.New("unsupported index type")
	}
}
