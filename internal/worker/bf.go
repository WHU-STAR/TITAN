package worker

import (
	"errors"

	"github.com/WHU-STAR/TITAN/internal/engine/go_api/bf"
)

type BFIndex struct {
	index *bf.BruteForceIndex
}

func (b *BFIndex) SetEFSearch(efSearch int) error {
	return nil
}

func (b *BFIndex) ShowAllLabels() string {
	// Brute force does not have labels to show
	return "Brute force index does not support labels"
}

func (b *BFIndex) SetNProbe(nProbe int) error {
	return nil
}

func (b *BFIndex) Load(path string, dim int, metric string) error {
	index, err := bf.LoadIndex(path, dim, metric)
	if err != nil {
		return errors.New("failed to load brute force index: " + err.Error())
	}
	b.index = index
	return nil
}

func (b *BFIndex) SearchWithEntry(vector []float32, bucketId uint32, k int) ([]uint32, []float32, error) {
	return b.Search(vector, k)

}

func (b *BFIndex) Search(vector []float32, k int) ([]uint32, []float32, error) {
	if b.index == nil {
		return nil, nil, errors.New("index not loaded")
	}
	return b.index.SearchKNN(vector, k)
}

func (b *BFIndex) Unload() {
	if b.index != nil {
		b.index.Unload()
	}
}

func (b *BFIndex) GetAvgHops() float32 {
	return 0 // Brute force does not have hops
}

func (b *BFIndex) GetAvgDistComputations() float32 {
	return 0 // Brute force does not have distance computations
}

func (b *BFIndex) GetDataSize() uint32 {
	// TODO: implement
	return 0
}
