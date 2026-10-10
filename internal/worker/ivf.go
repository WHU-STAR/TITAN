package worker

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/WHU-STAR/TITAN/pkg/logger"
	"go.uber.org/zap"
)

// IVFIndex is a cluster-based inverted-file index.
type IVFIndex struct {
	NList             int
	NProbe            int
	Dim               int
	Metric            string
	BucketIdList      []uint32
	BucketIdToPointId map[uint32][]uint32
	PointIdToData     map[uint32][]float32 // raw vectors in flat mode; nil in PQ mode
	PointIdToCode     map[uint32][]uint8   // PQ codes
	Centroids         [][]float32          // cluster centroids
	IsTrained         bool
	// PQ fields
	UsePQ       bool          // whether PQ is enabled
	PQM         uint16        // number of PQ subquantizers
	PQKsub      uint16        // PQ codebook size
	PQCodebooks [][][]float32 // PQ codebooks: [M][ksub][dsub]float32
	PQCodeSize  int           // PQ code size in bytes
}

// IVFHeader is the on-disk index header.
type IVFHeader struct {
	Magic     uint32   // magic number
	Version   uint32   // format version
	Dim       int32    // vector dimension
	NList     int32    // number of clusters
	NProbe    int32    // clusters probed at search time
	Metric    [16]byte // distance metric name
	IsTrained uint8    // 1 when the index is trained
	PQFlag    uint8    // 1 = PQ, 0 = flat
	PQM       uint16   // number of PQ subquantizers
	PQKsub    uint16   // PQ codebook size
}

const (
	IVFMagic    = 0x49564F46 // "IVOF"
	IVFVersion  = 1          // version 1: flat index
	IVFVersion2 = 2          // version 2: PQ index
)

// ShowAllLabels reports that this index has no label dump.
func (i *IVFIndex) ShowAllLabels() string {
	return "IVF index does not support labels"
}

// SearchWithEntry searches from the standard IVF path.
func (i *IVFIndex) SearchWithEntry(vector []float32, bucketId uint32, k int) ([]uint32, []float32, error) {
	return i.Search(vector, k)
}

// Search runs an IVF search.
func (i *IVFIndex) Search(vector []float32, k int) ([]uint32, []float32, error) {
	if !i.IsTrained {
		return nil, nil, fmt.Errorf("index is not trained")
	}

	if len(vector) != i.Dim {
		return nil, nil, fmt.Errorf("vector dimension mismatch: expected %d, got %d", i.Dim, len(vector))
	}

	// Probe the nearest nprobe clusters.
	candidateClusters, err := i.findNearestClusters(vector, i.NProbe)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to find nearest clusters: %v", err)
	}

	// Scan each probed cluster.
	var allResults []SearchResult
	for _, clusterID := range candidateClusters {
		clusterResults, err := i.searchInCluster(vector, clusterID, k)
		if err != nil {
			continue
		}
		allResults = append(allResults, clusterResults...)
	}

	// Sort and keep top-k, matching the graph indexes.
	sort.Sort(SearchResults(allResults))

	// Keep the closest k results.
	if len(allResults) > k {
		allResults = allResults[len(allResults)-k:]
	}

	// Extract ids and distances.
	results := make([]uint32, len(allResults))
	distances := make([]float32, len(allResults))
	for i, result := range allResults {
		results[i] = result.PointID
		distances[i] = result.Distance
	}

	return results, distances, nil
}

// SearchResult is one neighbor.
type SearchResult struct {
	PointID  uint32
	Distance float32
}

// SearchResults sorts neighbors.
type SearchResults []SearchResult

func (r SearchResults) Len() int           { return len(r) }
func (r SearchResults) Less(i, j int) bool { return r[i].Distance > r[j].Distance }
func (r SearchResults) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }

// findNearestClusters returns the nearest nprobe centroids.
func (i *IVFIndex) findNearestClusters(vector []float32, nprobe int) ([]uint32, error) {
	if len(i.Centroids) == 0 {
		return nil, fmt.Errorf("centroids are not initialized")
	}

	// Distances from the query to every centroid.
	distances := SIMDDistanceToCentroids(vector, i.Centroids)

	var clusterDistances []ClusterDistance
	for bucketID, distance := range distances {
		clusterDistances = append(clusterDistances, ClusterDistance{
			BucketID: uint32(bucketID),
			Distance: distance,
		})
	}

	// Sort.
	sort.Sort(ClusterDistances(clusterDistances))

	if len(clusterDistances) > nprobe {
		clusterDistances = clusterDistances[:nprobe]
	}

	result := make([]uint32, len(clusterDistances))
	for j, dist := range clusterDistances {
		result[j] = dist.BucketID
	}

	return result, nil
}

// ClusterDistance is a centroid and its distance.
type ClusterDistance struct {
	BucketID uint32
	Distance float32
}

// ClusterDistances sorts centroid distances.
type ClusterDistances []ClusterDistance

func (d ClusterDistances) Len() int           { return len(d) }
func (d ClusterDistances) Less(i, j int) bool { return d[i].Distance < d[j].Distance }
func (d ClusterDistances) Swap(i, j int)      { d[i], d[j] = d[j], d[i] }

// decodePQ reconstructs a vector from PQ codes.
func (i *IVFIndex) decodePQ(code []uint8) []float32 {
	if len(code) != i.PQCodeSize {
		return nil
	}
	dsub := i.Dim / int(i.PQM)
	decoded := make([]float32, i.Dim)

	for m := uint16(0); m < i.PQM; m++ {
		codeIdx := code[m]
		if int(codeIdx) >= int(i.PQKsub) {
			continue
		}
		start := int(m) * dsub
		copy(decoded[start:start+dsub], i.PQCodebooks[m][codeIdx])
	}

	return decoded
}

// computePQDistanceTable builds per-subquantizer L2 tables.
// Returns squared L2 distances as [M][ksub]float32.
// SIMD batch path.
func (i *IVFIndex) computePQDistanceTable(query []float32) [][]float32 {
	dsub := i.Dim / int(i.PQM)
	table := make([][]float32, i.PQM)

	for m := uint16(0); m < i.PQM; m++ {
		start := int(m) * dsub
		qSub := query[start : start+dsub]
		table[m] = make([]float32, i.PQKsub)

		// Pack codewords for a SIMD batch.
		candidates := make([][]float32, i.PQKsub)
		for k := uint16(0); k < i.PQKsub; k++ {
			candidates[k] = i.PQCodebooks[m][k]
		}

		// Squared L2, without sqrt.
		distances := BatchL2DistanceSquaredSIMD(qSub, candidates)
		for k := uint16(0); k < i.PQKsub; k++ {
			table[m][k] = distances[k]
		}
	}

	return table
}

// computePQDistance looks up an asymmetric PQ distance.
func (i *IVFIndex) computePQDistance(query []float32, code []uint8, distTable [][]float32) float32 {
	if len(code) != i.PQCodeSize {
		return math.MaxFloat32
	}

	var distance float32

	if i.Metric == "L2" {
		// L2 uses the precomputed table.
		for m := uint16(0); m < i.PQM; m++ {
			codeIdx := code[m]
			if int(codeIdx) >= int(i.PQKsub) {
				continue
			}
			distance += distTable[m][codeIdx]
		}
	} else {
		// Inner product decodes the full vector.
		decoded := i.decodePQ(code)
		if decoded == nil {
			return math.MaxFloat32
		}
		distance = -InnerProductSIMD(query, decoded) // Negate so smaller scores rank first.
	}

	return distance
}

// BatchPQDistance computes PQ distances for many codes.
func (i *IVFIndex) BatchPQDistance(query []float32, codes [][]uint8) []float32 {
	if len(codes) == 0 {
		return nil
	}

	results := make([]float32, len(codes))

	if i.Metric == "L2" {
		// L2: build the table, then scan codes.
		distTable := i.computePQDistanceTable(query)
		for idx, code := range codes {
			results[idx] = i.computePQDistance(query, code, distTable)
		}
	} else {
		// Inner product: decode each code.
		for idx, code := range codes {
			results[idx] = i.computePQDistance(query, code, nil)
		}
	}

	return results
}

// searchInCluster scans one cluster in flat or PQ mode.
func (i *IVFIndex) searchInCluster(vector []float32, clusterID uint32, k int) ([]SearchResult, error) {
	pointIDs, exists := i.BucketIdToPointId[clusterID]
	if !exists || len(pointIDs) == 0 {
		return nil, fmt.Errorf("cluster %d has no vectors", clusterID)
	}

	if i.UsePQ {
		// PQ mode.
		codes := make([][]uint8, 0, len(pointIDs))
		validPointIDs := make([]uint32, 0, len(pointIDs))

		for _, pointID := range pointIDs {
			code, exists := i.PointIdToCode[pointID]
			if exists {
				codes = append(codes, code)
				validPointIDs = append(validPointIDs, pointID)
			}
		}

		if len(codes) == 0 {
			return nil, fmt.Errorf("cluster %d has no valid vectors", clusterID)
		}

		// Batch distances.
		distances := i.BatchPQDistance(vector, codes)

		// Build the result list.
		results := make([]SearchResult, len(validPointIDs))
		for j, pointID := range validPointIDs {
			results[j] = SearchResult{
				PointID:  pointID,
				Distance: distances[j],
			}
		}

		// Sort.
		sort.Sort(SearchResults(results))

		// Keep the closest k.
		if len(results) > k {
			results = results[len(results)-k:]
		}

		return results, nil
	} else {
		// Flat mode uses raw vectors.
		candidates := make([][]float32, 0, len(pointIDs))
		validPointIDs := make([]uint32, 0, len(pointIDs))

		for _, pointID := range pointIDs {
			data, exists := i.PointIdToData[pointID]
			if exists {
				candidates = append(candidates, data)
				validPointIDs = append(validPointIDs, pointID)
			}
		}

		if len(candidates) == 0 {
			return nil, fmt.Errorf("cluster %d has no valid vectors", clusterID)
		}

		// SIMD batch distances.
		distances := BatchSIMDDistance(vector, candidates)

		// Build the result list.
		results := make([]SearchResult, len(validPointIDs))
		for j, pointID := range validPointIDs {
			results[j] = SearchResult{
				PointID:  pointID,
				Distance: distances[j],
			}
		}

		// Sort.
		sort.Sort(SearchResults(results))

		// Keep the closest k.
		if len(results) > k {
			results = results[len(results)-k:]
		}

		return results, nil
	}
}

// Unload releases the index.
func (i *IVFIndex) Unload() {
	i.BucketIdToPointId = nil
	i.PointIdToData = nil
	i.PointIdToCode = nil
	i.Centroids = nil
	i.PQCodebooks = nil
	i.IsTrained = false
}

// GetAvgHops reports nprobe as the hop proxy.
func (i *IVFIndex) GetAvgHops() float32 {
	return float32(i.NProbe) // IVF probes nprobe lists
}

// GetAvgDistComputations estimates distance computations.
func (i *IVFIndex) GetAvgDistComputations() float32 {
	totalVectors := 0
	for _, pointIDs := range i.BucketIdToPointId {
		totalVectors += len(pointIDs)
	}

	if len(i.BucketIdToPointId) == 0 {
		return 0
	}

	// Distance count is about nprobe times the average list length.
	avgVectorsPerCluster := float32(totalVectors) / float32(len(i.BucketIdToPointId))
	return float32(i.NProbe) * avgVectorsPerCluster
}

// GetDataSize returns the stored vector bytes.
func (i *IVFIndex) GetDataSize() uint32 {

	totalVectors := 0
	for _, pointIDs := range i.BucketIdToPointId {
		totalVectors += len(pointIDs)
	}
	return uint32(totalVectors)
}

func (i *IVFIndex) SetEFSearch(efSearch int) error {
	return nil
}

func (i *IVFIndex) SetNProbe(nProbe int) error {
	if nProbe <= 0 {
		return fmt.Errorf("nProbe must be greater than 0")
	}
	i.NProbe = nProbe
	logger.Debug("NProbe updated", zap.Int("nProbe", nProbe))
	return nil
}

// Load reads an index from disk.
func (i *IVFIndex) Load(path string, dim int, metric string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open index file: %v", err)
	}
	defer file.Close()

	// Read the header.
	var header IVFHeader
	if err := binary.Read(file, binary.LittleEndian, &header); err != nil {
		return fmt.Errorf("failed to read index header: %v", err)
	}

	// Check magic and version.
	if header.Magic != IVFMagic {
		return fmt.Errorf("invalid index file")
	}
	if header.Version != IVFVersion && header.Version != IVFVersion2 {
		return fmt.Errorf("unsupported index version: %d", header.Version)
	}

	// Check dimension.
	if header.Dim != int32(dim) {
		return fmt.Errorf("dimension mismatch: file has %d, expected %d", header.Dim, dim)
	}

	// Read the metric name.
	metricBytes := make([]byte, 0, 16)
	for _, b := range header.Metric {
		if b == 0 {
			break
		}
		metricBytes = append(metricBytes, b)
	}
	fileMetric := string(metricBytes)
	if !strings.EqualFold(fileMetric, metric) {
		return fmt.Errorf("metric mismatch: file has %s, expected %s", fileMetric, metric)
	}

	// Initialize the in-memory index.
	i.NList = int(header.NList)
	i.NProbe = int(header.NProbe)
	i.Dim = int(header.Dim)
	i.Metric = metric
	i.BucketIdList = make([]uint32, header.NList)
	i.BucketIdToPointId = make(map[uint32][]uint32)
	i.Centroids = make([][]float32, header.NList)
	i.IsTrained = header.IsTrained != 0

	// Initialize PQ fields.
	i.UsePQ = header.PQFlag != 0
	if i.UsePQ {
		i.PQM = header.PQM
		i.PQKsub = header.PQKsub
		i.PQCodeSize = int(i.PQM) // one byte per subquantizer
		i.PointIdToCode = make(map[uint32][]uint8)
		i.PointIdToData = nil // PQ mode does not store raw vectors
	} else {
		i.PointIdToData = make(map[uint32][]float32)
		i.PointIdToCode = nil
	}

	// Read centroids.
	for j := 0; j < int(header.NList); j++ {
		centroid := make([]float32, header.Dim)
		if err := binary.Read(file, binary.LittleEndian, &centroid); err != nil {
			return fmt.Errorf("failed to read centroid %d: %v", j, err)
		}
		i.Centroids[j] = centroid
		i.BucketIdList[j] = uint32(j)
	}

	// Read PQ codebooks when PQ is enabled.
	if i.UsePQ {
		if int(header.Dim)%int(i.PQM) != 0 {
			return fmt.Errorf("dimension %d must be divisible by PQ subquantizers %d", header.Dim, i.PQM)
		}
		dsub := int(header.Dim) / int(i.PQM)
		zap.L().Info("reading PQ codebooks",
			zap.Uint16("M", i.PQM),
			zap.Uint16("ksub", i.PQKsub),
			zap.Int("dsub", dsub))

		i.PQCodebooks = make([][][]float32, i.PQM)
		for m := uint16(0); m < i.PQM; m++ {
			i.PQCodebooks[m] = make([][]float32, i.PQKsub)
			for k := uint16(0); k < i.PQKsub; k++ {
				code := make([]float32, dsub)
				vectorBytes := make([]byte, dsub*4)
				if _, err := io.ReadFull(file, vectorBytes); err != nil {
					return fmt.Errorf("failed to read PQ codebook [%d][%d]: %v", m, k, err)
				}
				// Decode float32 values.
				for d := 0; d < dsub; d++ {
					bits := uint32(vectorBytes[d*4]) |
						uint32(vectorBytes[d*4+1])<<8 |
						uint32(vectorBytes[d*4+2])<<16 |
						uint32(vectorBytes[d*4+3])<<24
					code[d] = math.Float32frombits(bits)
				}
				i.PQCodebooks[m][k] = code
			}
		}
		zap.L().Info("PQ codebooks loaded")
	}

	// Read inverted lists.
	totalVectors := 0
	for j := 0; j < int(header.NList); j++ {
		if j%100 == 0 {
			zap.L().Info("reading cluster", zap.Int("cluster_id", j), zap.Int("total_clusters", int(header.NList)))
		}

		var vectorCount uint32
		if err := binary.Read(file, binary.LittleEndian, &vectorCount); err != nil {
			return fmt.Errorf("failed to read vector count of cluster %d: %v", j, err)
		}

		pointIDs := make([]uint32, vectorCount)
		if vectorCount > 0 {
			if err := binary.Read(file, binary.LittleEndian, &pointIDs); err != nil {
				return fmt.Errorf("failed to read ids of cluster %d: %v", j, err)
			}
		}

		i.BucketIdToPointId[uint32(j)] = pointIDs

		// Read raw vectors or PQ codes.
		if i.UsePQ {
			// PQ mode: read codes.
			for _, pointID := range pointIDs {
				code := make([]uint8, i.PQCodeSize)
				if _, err := io.ReadFull(file, code); err != nil {
					return fmt.Errorf("failed to read PQ code of vector %d: %v", pointID, err)
				}
				i.PointIdToCode[pointID] = code
			}
		} else {
			// Flat mode: read raw vectors.
			for _, pointID := range pointIDs {
				vector := make([]float32, header.Dim)
				// Read the full vector payload.
				vectorBytes := make([]byte, int(header.Dim)*4)
				if _, err := io.ReadFull(file, vectorBytes); err != nil {
					return fmt.Errorf("failed to read bytes of vector %d: %v", pointID, err)
				}
				// Decode float32 values.
				for k := 0; k < int(header.Dim); k++ {
					bits := uint32(vectorBytes[k*4]) |
						uint32(vectorBytes[k*4+1])<<8 |
						uint32(vectorBytes[k*4+2])<<16 |
						uint32(vectorBytes[k*4+3])<<24
					vector[k] = math.Float32frombits(bits)
				}
				i.PointIdToData[pointID] = vector
			}
		}

		totalVectors += int(vectorCount)
	}

	logFields := []zap.Field{
		zap.Int("nlist", i.NList),
		zap.Int("nprobe", i.NProbe),
		zap.Int("dim", i.Dim),
		zap.String("metric", i.Metric),
		zap.Bool("is_trained", i.IsTrained),
		zap.Uint32("total_vectors", i.GetDataSize()),
		zap.Bool("use_pq", i.UsePQ),
	}
	if i.UsePQ {
		logFields = append(logFields,
			zap.Uint16("pq_m", i.PQM),
			zap.Uint16("pq_ksub", i.PQKsub),
			zap.Int("pq_code_size", i.PQCodeSize))
	}
	zap.L().Info("IVF index loaded", logFields...)

	return nil
}

// InitIndex allocates empty posting lists.
func (i *IVFIndex) InitIndex(dim int, bucketIdList []uint32) error {
	i.Dim = dim
	i.NList = len(bucketIdList)
	i.BucketIdList = bucketIdList
	i.BucketIdToPointId = make(map[uint32][]uint32)
	i.PointIdToData = make(map[uint32][]float32)
	i.PointIdToCode = make(map[uint32][]uint8)
	i.Centroids = make([][]float32, i.NList)
	i.IsTrained = false
	i.UsePQ = false

	return nil
}

// Save writes the index to disk.
func (i *IVFIndex) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create index file: %v", err)
	}
	defer file.Close()

	// Write the header.
	header := IVFHeader{
		Magic:     IVFMagic,
		Version:   IVFVersion,
		Dim:       int32(i.Dim),
		NList:     int32(i.NList),
		NProbe:    int32(i.NProbe),
		IsTrained: 1,
	}

	// Write the metric name.
	copy(header.Metric[:], []byte(i.Metric))

	if err := binary.Write(file, binary.LittleEndian, &header); err != nil {
		return fmt.Errorf("failed to write index header: %v", err)
	}

	// Write centroids.
	for j := 0; j < i.NList; j++ {
		if err := binary.Write(file, binary.LittleEndian, &i.Centroids[j]); err != nil {
			return fmt.Errorf("failed to write centroid %d: %v", j, err)
		}
	}

	// Write inverted lists.
	for j := 0; j < i.NList; j++ {
		pointIDs := i.BucketIdToPointId[uint32(j)]
		vectorCount := uint32(len(pointIDs))

		// Write the list length.
		if err := binary.Write(file, binary.LittleEndian, &vectorCount); err != nil {
			return fmt.Errorf("failed to write vector count of cluster %d: %v", j, err)
		}

		// Write vector ids.
		if err := binary.Write(file, binary.LittleEndian, &pointIDs); err != nil {
			return fmt.Errorf("failed to write ids of cluster %d: %v", j, err)
		}

		// Write vector data.
		for _, pointID := range pointIDs {
			vector := i.PointIdToData[pointID]
			if err := binary.Write(file, binary.LittleEndian, &vector); err != nil {
				return fmt.Errorf("failed to write vector %d: %v", pointID, err)
			}
		}
	}

	zap.L().Info("IVF index saved",
		zap.String("path", path),
		zap.Int("nlist", i.NList),
		zap.Int("nprobe", i.NProbe),
		zap.Int("dim", i.Dim),
		zap.String("metric", i.Metric),
		zap.Uint32("total_vectors", i.GetDataSize()),
	)

	return nil
}
