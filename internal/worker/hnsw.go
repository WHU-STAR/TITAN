package worker

import (
	"errors"

	"github.com/WHU-STAR/TITAN/internal/engine/go_api/hnsw"
)

// HNSWIndex implements VectorIndex using HNSW
type HNSWIndex struct {
	index *hnsw.Index
}

func (h *HNSWIndex) SetNProbe(nProbe int) error {
	return nil
}

func (h *HNSWIndex) ShowAllLabels() string {
	if h.index == nil {
		return "Index not loaded"
	}
	return h.index.ShowAllLabels()
}

func (h *HNSWIndex) SearchWithEntry(vector []float32, bucketId uint32, k int) ([]uint32, []float32, error) {
	if h.index == nil {
		return nil, nil, errors.New("index not loaded")
	}
	return h.index.SearchKnnWithEntryPoint(vector, k, bucketId)
}

func (h *HNSWIndex) SetEFSearch(efSearch int) error {
	if h.index == nil {
		return errors.New("index not loaded")
	}
	h.index.SetEf(efSearch)
	return nil
}

func (h *HNSWIndex) Load(path string, dim int, metric string) error {
	index, err := hnsw.LoadIndex(path, dim, metric)
	if err != nil {
		return err
	}
	h.index = index
	return nil
}

func (h *HNSWIndex) Search(vector []float32, k int) ([]uint32, []float32, error) {
	return h.index.SearchKNN(vector, k)
}

func (h *HNSWIndex) Unload() {
	h.index.Unload()
}

func (h *HNSWIndex) GetAvgHops() float32 {
	return h.index.GetAvgHops()
}

func (h *HNSWIndex) GetAvgDistComputations() float32 {
	return h.index.GetAvgDistComputations()
}

func (h *HNSWIndex) GetDataSize() uint32 {
	// TODO: implement
	return 0
}
