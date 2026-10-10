package ivf

/*
#cgo CFLAGS: -I${SRCDIR}/../../c_api/ivf
#cgo LDFLAGS: -L${SRCDIR}/../../build -livf -Wl,-rpath,${SRCDIR}/../../build
#include "ivf_c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"math"
	"unsafe"

	"github.com/WHU-STAR/TITAN/internal/definitions"
)

type Index struct {
	index *C.IVFIndex
	dim   int
}

// NewIndex creates a new IVF index with the given dimension and bucket IDs
func NewIndex(dim int, bucketIdList []uint32) *Index {
	// Create a new slice with an extra element for the terminator
	cBucketIds := make([]C.int, len(bucketIdList)+1)
	for i, id := range bucketIdList {
		cBucketIds[i] = C.int(id)
	}
	// Add terminator
	cBucketIds[len(bucketIdList)] = C.int(-1)

	index := C.ivf_new(C.size_t(dim), (*C.int)(&cBucketIds[0]), C.size_t(len(cBucketIds)))
	if index == nil {
		return nil
	}
	return &Index{
		index: index,
		dim:   dim,
	}
}

// Free releases the memory allocated for the index
func (idx *Index) Free() {
	if idx.index != nil {
		C.ivf_free(idx.index)
		idx.index = nil
	}
}

// AddPoint adds a point to the index
func (idx *Index) AddPoint(point []float32, bucketId uint32, pointId uint64) error {
	if len(point) != idx.dim {
		return fmt.Errorf("point dimension mismatch: expected %d, got %d", idx.dim, len(point))
	}

	if idx.index == nil {
		return fmt.Errorf("index is not initialized")
	}

	ret := C.ivf_add_point(idx.index, C.int(bucketId), (*C.float)(&point[0]), C.size_t(pointId))
	if ret != 0 {
		return fmt.Errorf("failed to add point")
	}
	return nil
}

// SearchKNN searches for k nearest neighbors of the query vector in the index
func (idx *Index) SearchKNN(query []float32, bucketIds []uint32, topk int) ([]uint32, []float32, error) {
	if len(query) == 0 {
		return nil, nil, fmt.Errorf("empty query")
	}

	if len(bucketIds) == 0 {
		return nil, nil, fmt.Errorf("empty bucket ids")
	}

	if topk <= 0 {
		return nil, nil, fmt.Errorf("invalid topk")
	}

	// Convert bucket ids to a C array terminated by -1.
	cBucketIds := make([]C.int, len(bucketIds)+1)
	for i, id := range bucketIds {
		cBucketIds[i] = C.int(id)
	}
	cBucketIds[len(bucketIds)] = C.int(-1) // terminator

	cQuery := (*C.float)(unsafe.Pointer(&query[0]))

	labels := make([]C.size_t, topk)
	distances := make([]C.float, topk)

	// Call the C search.
	C.ivf_search_knn(
		idx.index,
		cQuery,
		(*C.int)(&cBucketIds[0]),
		C.size_t(topk),
		(*C.size_t)(&labels[0]),
		(*C.float)(&distances[0]),
	)

	// Drop invalid labels.
	validLabels := make([]uint32, 0, topk)
	validDistances := make([]float32, 0, topk)
	for i := len(labels) - 1; i >= 0; i-- {
		label := uint32(labels[i])
		if label == math.MaxUint32 {
			// Skip an empty slot.
			continue
		}
		validLabels = append(validLabels, label)
		validDistances = append(validDistances, float32(distances[i]))
	}

	// Labels are already global ids.
	return validLabels, validDistances, nil
}

// build the index
func (idx *Index) AddItems(buckets []definitions.Bucket) error {
	totalPoints := 0
	for _, bucket := range buckets {
		for _, point := range bucket.Points {
			err := idx.AddPoint(point.Data, uint32(bucket.BucketId), uint64(point.PointId))
			if err != nil {
				return err
			}
			totalPoints++
		}
	}
	return nil
}

func (idx *Index) ContainsBucket(bucketId uint32) bool {
	return C.ivf_contains_bucket(idx.index, C.int(bucketId)) != 0
}
