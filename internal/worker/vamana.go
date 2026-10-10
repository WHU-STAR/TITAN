package worker

import "github.com/WHU-STAR/TITAN/internal/engine/go_api/vamana"

type VamanaIndex struct {
	index *vamana.VamanaIndex
	dim   int
}

func (v *VamanaIndex) ShowAllLabels() string {
	return "Vamana index does not support labels"
}

func (v *VamanaIndex) SetNProbe(nProbe int) error {
	return nil
}

func (v *VamanaIndex) SearchWithEntry(vector []float32, bucketId uint32, k int) ([]uint32, []float32, error) {
	return v.SearchWithBucketId(vector, bucketId, k)
}

func (v *VamanaIndex) GetPoint(id uint32) ([]float32, error) {
	vector := make([]float32, v.dim)
	err := v.index.GetPoint(id, vector)
	if err != nil {
		return nil, err
	}
	return vector, nil
}

func (v *VamanaIndex) Load(path string, dim int, metric string) error {
	index, err := vamana.LoadIndex(path)
	if err != nil {
		return err
	}
	v.index = index
	v.dim = dim
	return nil
}

func (v *VamanaIndex) Search(vector []float32, k int) ([]uint32, []float32, error) {
	return v.index.Search(vector, uint32(k))
}

func (v *VamanaIndex) SearchWithBucketId(vector []float32, bucketId uint32, k int) ([]uint32, []float32, error) {
	return v.index.SearchWithStartPoint(vector, bucketId, uint32(k))
}

func (v *VamanaIndex) Unload() {
	v.index.Free()
}

func (v *VamanaIndex) GetAvgHops() float32 {
	return v.index.GetAvgHops()
}

func (v *VamanaIndex) GetAvgDistComputations() float32 {
	return v.index.GetAvgDistComputations()
}

func (v *VamanaIndex) GetDataSize() uint32 {
	return v.index.GetDataSize()
}

func (v *VamanaIndex) SetEFSearch(efSearch int) error {
	// vamana does not need to set ef
	return nil
}
