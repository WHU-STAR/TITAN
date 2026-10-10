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

// IVFIndex 实现基于聚类的倒排索引
type IVFIndex struct {
	NList             int
	NProbe            int
	Dim               int
	Metric            string
	BucketIdList      []uint32
	BucketIdToPointId map[uint32][]uint32
	PointIdToData     map[uint32][]float32 // Flat模式：原始向量；PQ模式：nil
	PointIdToCode     map[uint32][]uint8   // PQ模式：编码后的向量
	Centroids         [][]float32          // 聚类中心
	IsTrained         bool
	// PQ相关字段
	UsePQ       bool          // 是否使用PQ
	PQM         uint16        // PQ子向量数量
	PQKsub      uint16        // PQ码本大小（码字数量）
	PQCodebooks [][][]float32 // PQ码本: [M][ksub][dsub]float32
	PQCodeSize  int           // PQ编码大小（字节）
}

// IVFHeader 索引文件头结构
type IVFHeader struct {
	Magic     uint32   // 魔数，用于验证文件格式
	Version   uint32   // 版本号
	Dim       int32    // 向量维度
	NList     int32    // 聚类数量
	NProbe    int32    // 搜索时探测的聚类数量
	Metric    [16]byte // 距离度量类型
	IsTrained uint8    // 是否已训练
	PQFlag    uint8    // PQ标志：1=使用PQ，0=使用Flat
	PQM       uint16   // PQ子向量数量
	PQKsub    uint16   // PQ码本大小（码字数量）
}

const (
	IVFMagic    = 0x49564F46 // "IVOF"
	IVFVersion  = 1          // 版本1：Flat索引
	IVFVersion2 = 2          // 版本2：支持PQ索引
)

// ShowAllLabels 显示所有标签
func (i *IVFIndex) ShowAllLabels() string {
	return "IVF index does not support labels"
}

// SearchWithEntry 使用指定的聚类ID进行搜索
func (i *IVFIndex) SearchWithEntry(vector []float32, bucketId uint32, k int) ([]uint32, []float32, error) {
	return i.Search(vector, k)
}

// Search 执行IVF搜索
func (i *IVFIndex) Search(vector []float32, k int) ([]uint32, []float32, error) {
	if !i.IsTrained {
		return nil, nil, fmt.Errorf("索引未训练")
	}

	if len(vector) != i.Dim {
		return nil, nil, fmt.Errorf("向量维度不匹配: 期望 %d, 实际 %d", i.Dim, len(vector))
	}

	// 1. 找到最近的nprobe个聚类
	candidateClusters, err := i.findNearestClusters(vector, i.NProbe)
	if err != nil {
		return nil, nil, fmt.Errorf("查找最近聚类失败: %v", err)
	}

	// 2. 在每个候选聚类中搜索
	var allResults []SearchResult
	for _, clusterID := range candidateClusters {
		clusterResults, err := i.searchInCluster(vector, clusterID, k) // 搜索更多候选
		if err != nil {
			continue
		}
		allResults = append(allResults, clusterResults...)
	}

	// 3. 排序并返回top-k，降序排序，和其他graph 算法保持一致
	sort.Sort(SearchResults(allResults))

	// 取后k个结果
	if len(allResults) > k {
		allResults = allResults[len(allResults)-k:]
	}

	// 提取结果
	results := make([]uint32, len(allResults))
	distances := make([]float32, len(allResults))
	for i, result := range allResults {
		results[i] = result.PointID
		distances[i] = result.Distance
	}

	return results, distances, nil
}

// SearchResult 搜索结果结构
type SearchResult struct {
	PointID  uint32
	Distance float32
}

// SearchResults 搜索结果排序接口
type SearchResults []SearchResult

func (r SearchResults) Len() int           { return len(r) }
func (r SearchResults) Less(i, j int) bool { return r[i].Distance > r[j].Distance }
func (r SearchResults) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }

// findNearestClusters 找到最近的nprobe个聚类 - 使用SIMD批量优化
func (i *IVFIndex) findNearestClusters(vector []float32, nprobe int) ([]uint32, error) {
	if len(i.Centroids) == 0 {
		return nil, fmt.Errorf("聚类中心未初始化")
	}

	// 使用SIMD批量计算到所有聚类中心的距离
	distances := SIMDDistanceToCentroids(vector, i.Centroids)

	var clusterDistances []ClusterDistance
	for bucketID, distance := range distances {
		clusterDistances = append(clusterDistances, ClusterDistance{
			BucketID: uint32(bucketID),
			Distance: distance,
		})
	}

	// 排序
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

// ClusterDistance 聚类距离结构
type ClusterDistance struct {
	BucketID uint32
	Distance float32
}

// ClusterDistances 聚类距离排序接口
type ClusterDistances []ClusterDistance

func (d ClusterDistances) Len() int           { return len(d) }
func (d ClusterDistances) Less(i, j int) bool { return d[i].Distance < d[j].Distance }
func (d ClusterDistances) Swap(i, j int)      { d[i], d[j] = d[j], d[i] }

// decodePQ 解码PQ编码的向量
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

// computePQDistanceTable 预计算查询向量到所有码字的距离表（L2距离）
// 返回: [M][ksub]float32，表示每个子向量到每个码字的距离平方
// 使用SIMD批量优化加速
func (i *IVFIndex) computePQDistanceTable(query []float32) [][]float32 {
	dsub := i.Dim / int(i.PQM)
	table := make([][]float32, i.PQM)

	for m := uint16(0); m < i.PQM; m++ {
		start := int(m) * dsub
		qSub := query[start : start+dsub]
		table[m] = make([]float32, i.PQKsub)

		// 将码字组织成候选向量数组，使用SIMD批量计算
		candidates := make([][]float32, i.PQKsub)
		for k := uint16(0); k < i.PQKsub; k++ {
			candidates[k] = i.PQCodebooks[m][k]
		}

		// 使用SIMD批量计算距离平方（不计算sqrt）
		distances := BatchL2DistanceSquaredSIMD(qSub, candidates)
		for k := uint16(0); k < i.PQKsub; k++ {
			table[m][k] = distances[k]
		}
	}

	return table
}

// computePQDistance 计算查询向量与PQ编码向量的距离（使用距离表加速）
func (i *IVFIndex) computePQDistance(query []float32, code []uint8, distTable [][]float32) float32 {
	if len(code) != i.PQCodeSize {
		return math.MaxFloat32
	}

	var distance float32

	if i.Metric == "L2" {
		// L2距离：使用预计算的距离表，只需查表求和
		for m := uint16(0); m < i.PQM; m++ {
			codeIdx := code[m]
			if int(codeIdx) >= int(i.PQKsub) {
				continue
			}
			distance += distTable[m][codeIdx]
		}
	} else {
		// IP距离：需要解码完整向量
		decoded := i.decodePQ(code)
		if decoded == nil {
			return math.MaxFloat32
		}
		distance = -InnerProductSIMD(query, decoded) // 注意：IP通常返回负值用于排序
	}

	return distance
}

// BatchPQDistance 批量计算PQ距离（优化版本）
func (i *IVFIndex) BatchPQDistance(query []float32, codes [][]uint8) []float32 {
	if len(codes) == 0 {
		return nil
	}

	results := make([]float32, len(codes))

	if i.Metric == "L2" {
		// L2距离：预计算距离表，然后批量查表
		distTable := i.computePQDistanceTable(query)
		for idx, code := range codes {
			results[idx] = i.computePQDistance(query, code, distTable)
		}
	} else {
		// IP距离：需要逐个解码计算
		for idx, code := range codes {
			results[idx] = i.computePQDistance(query, code, nil)
		}
	}

	return results
}

// searchInCluster 在指定聚类中搜索 - 使用SIMD批量优化（支持Flat和PQ）
func (i *IVFIndex) searchInCluster(vector []float32, clusterID uint32, k int) ([]SearchResult, error) {
	pointIDs, exists := i.BucketIdToPointId[clusterID]
	if !exists || len(pointIDs) == 0 {
		return nil, fmt.Errorf("聚类 %d 中没有向量", clusterID)
	}

	if i.UsePQ {
		// PQ模式：批量计算PQ距离（优化版本）
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
			return nil, fmt.Errorf("聚类 %d 中没有有效向量", clusterID)
		}

		// 批量计算距离
		distances := i.BatchPQDistance(vector, codes)

		// 构建结果
		results := make([]SearchResult, len(validPointIDs))
		for j, pointID := range validPointIDs {
			results[j] = SearchResult{
				PointID:  pointID,
				Distance: distances[j],
			}
		}

		// 排序
		sort.Sort(SearchResults(results))

		// 取后k个
		if len(results) > k {
			results = results[len(results)-k:]
		}

		return results, nil
	} else {
		// Flat模式：使用原始向量
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
			return nil, fmt.Errorf("聚类 %d 中没有有效向量", clusterID)
		}

		// 使用SIMD批量计算距离
		distances := BatchSIMDDistance(vector, candidates)

		// 构建结果
		results := make([]SearchResult, len(validPointIDs))
		for j, pointID := range validPointIDs {
			results[j] = SearchResult{
				PointID:  pointID,
				Distance: distances[j],
			}
		}

		// 排序
		sort.Sort(SearchResults(results))

		// 取后k个
		if len(results) > k {
			results = results[len(results)-k:]
		}

		return results, nil
	}
}

// Unload 卸载索引
func (i *IVFIndex) Unload() {
	i.BucketIdToPointId = nil
	i.PointIdToData = nil
	i.PointIdToCode = nil
	i.Centroids = nil
	i.PQCodebooks = nil
	i.IsTrained = false
}

// GetAvgHops 获取平均跳数
func (i *IVFIndex) GetAvgHops() float32 {
	return float32(i.NProbe) // IVF的平均跳数等于nprobe
}

// GetAvgDistComputations 获取平均距离计算次数
func (i *IVFIndex) GetAvgDistComputations() float32 {
	totalVectors := 0
	for _, pointIDs := range i.BucketIdToPointId {
		totalVectors += len(pointIDs)
	}

	if len(i.BucketIdToPointId) == 0 {
		return 0
	}

	// 平均每个查询需要计算的距离数 = nprobe个聚类的平均向量数
	avgVectorsPerCluster := float32(totalVectors) / float32(len(i.BucketIdToPointId))
	return float32(i.NProbe) * avgVectorsPerCluster
}

// GetDataSize 获取数据大小
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
		return fmt.Errorf("nProbe必须大于0")
	}
	i.NProbe = nProbe
	logger.Debug("NProbe updated", zap.Int("nProbe", nProbe))
	return nil
}

// Load 从文件加载索引
func (i *IVFIndex) Load(path string, dim int, metric string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("无法打开索引文件: %v", err)
	}
	defer file.Close()

	// 读取索引头
	var header IVFHeader
	if err := binary.Read(file, binary.LittleEndian, &header); err != nil {
		return fmt.Errorf("读取索引头失败: %v", err)
	}

	// 验证魔数和版本
	if header.Magic != IVFMagic {
		return fmt.Errorf("无效的索引文件格式")
	}
	if header.Version != IVFVersion && header.Version != IVFVersion2 {
		return fmt.Errorf("不支持的索引版本: %d", header.Version)
	}

	// 验证参数
	if header.Dim != int32(dim) {
		return fmt.Errorf("维度不匹配: 文件中的维度 %d, 期望 %d", header.Dim, dim)
	}

	// 提取距离度量类型
	metricBytes := make([]byte, 0, 16)
	for _, b := range header.Metric {
		if b == 0 {
			break
		}
		metricBytes = append(metricBytes, b)
	}
	fileMetric := string(metricBytes)
	if !strings.EqualFold(fileMetric, metric) {
		return fmt.Errorf("距离度量不匹配: 文件中的度量 %s, 期望 %s", fileMetric, metric)
	}

	// 初始化结构
	i.NList = int(header.NList)
	i.NProbe = int(header.NProbe)
	i.Dim = int(header.Dim)
	i.Metric = metric
	i.BucketIdList = make([]uint32, header.NList)
	i.BucketIdToPointId = make(map[uint32][]uint32)
	i.Centroids = make([][]float32, header.NList)
	i.IsTrained = header.IsTrained != 0

	// 初始化PQ相关字段
	i.UsePQ = header.PQFlag != 0
	if i.UsePQ {
		i.PQM = header.PQM
		i.PQKsub = header.PQKsub
		i.PQCodeSize = int(i.PQM) // 每个子向量用1字节表示，所以code_size = M
		i.PointIdToCode = make(map[uint32][]uint8)
		i.PointIdToData = nil // PQ模式不使用原始向量
	} else {
		i.PointIdToData = make(map[uint32][]float32)
		i.PointIdToCode = nil
	}

	// 读取聚类中心
	for j := 0; j < int(header.NList); j++ {
		centroid := make([]float32, header.Dim)
		if err := binary.Read(file, binary.LittleEndian, &centroid); err != nil {
			return fmt.Errorf("读取聚类中心 %d 失败: %v", j, err)
		}
		i.Centroids[j] = centroid
		i.BucketIdList[j] = uint32(j)
	}

	// 读取PQ码本（如果使用PQ）
	if i.UsePQ {
		if int(header.Dim)%int(i.PQM) != 0 {
			return fmt.Errorf("向量维度 %d 必须能被PQ子向量数 %d 整除", header.Dim, i.PQM)
		}
		dsub := int(header.Dim) / int(i.PQM)
		zap.L().Info("正在读取PQ码本",
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
					return fmt.Errorf("读取PQ码本 [%d][%d] 失败: %v", m, k, err)
				}
				// 手动解析float32数据
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
		zap.L().Info("PQ码本读取完成")
	}

	// 读取倒排列表
	totalVectors := 0
	for j := 0; j < int(header.NList); j++ {
		if j%100 == 0 {
			zap.L().Info("正在读取聚类", zap.Int("cluster_id", j), zap.Int("total_clusters", int(header.NList)))
		}

		var vectorCount uint32
		if err := binary.Read(file, binary.LittleEndian, &vectorCount); err != nil {
			return fmt.Errorf("读取聚类 %d 的向量数量失败: %v", j, err)
		}

		pointIDs := make([]uint32, vectorCount)
		if vectorCount > 0 {
			if err := binary.Read(file, binary.LittleEndian, &pointIDs); err != nil {
				return fmt.Errorf("读取聚类 %d 的向量ID失败: %v", j, err)
			}
		}

		i.BucketIdToPointId[uint32(j)] = pointIDs

		// 读取向量数据（Flat模式）或编码（PQ模式）
		if i.UsePQ {
			// PQ模式：读取编码
			for _, pointID := range pointIDs {
				code := make([]uint8, i.PQCodeSize)
				if _, err := io.ReadFull(file, code); err != nil {
					return fmt.Errorf("读取向量 %d 的PQ编码失败: %v", pointID, err)
				}
				i.PointIdToCode[pointID] = code
			}
		} else {
			// Flat模式：读取原始向量
			for _, pointID := range pointIDs {
				vector := make([]float32, header.Dim)
				// 使用io.ReadFull读取完整的向量数据
				vectorBytes := make([]byte, int(header.Dim)*4)
				if _, err := io.ReadFull(file, vectorBytes); err != nil {
					return fmt.Errorf("读取向量 %d 的字节数据失败: %v", pointID, err)
				}
				// 手动解析float32数据
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
	zap.L().Info("IVF索引加载完成", logFields...)

	return nil
}

// InitIndex 初始化索引
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

// Save 保存索引到文件
func (i *IVFIndex) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("无法创建索引文件: %v", err)
	}
	defer file.Close()

	// 写入索引头
	header := IVFHeader{
		Magic:     IVFMagic,
		Version:   IVFVersion,
		Dim:       int32(i.Dim),
		NList:     int32(i.NList),
		NProbe:    int32(i.NProbe),
		IsTrained: 1,
	}

	// 写入距离度量类型
	copy(header.Metric[:], []byte(i.Metric))

	if err := binary.Write(file, binary.LittleEndian, &header); err != nil {
		return fmt.Errorf("写入索引头失败: %v", err)
	}

	// 写入聚类中心
	for j := 0; j < i.NList; j++ {
		if err := binary.Write(file, binary.LittleEndian, &i.Centroids[j]); err != nil {
			return fmt.Errorf("写入聚类中心 %d 失败: %v", j, err)
		}
	}

	// 写入倒排列表
	for j := 0; j < i.NList; j++ {
		pointIDs := i.BucketIdToPointId[uint32(j)]
		vectorCount := uint32(len(pointIDs))

		// 写入向量数量
		if err := binary.Write(file, binary.LittleEndian, &vectorCount); err != nil {
			return fmt.Errorf("写入聚类 %d 的向量数量失败: %v", j, err)
		}

		// 写入向量ID列表
		if err := binary.Write(file, binary.LittleEndian, &pointIDs); err != nil {
			return fmt.Errorf("写入聚类 %d 的向量ID失败: %v", j, err)
		}

		// 写入向量数据
		for _, pointID := range pointIDs {
			vector := i.PointIdToData[pointID]
			if err := binary.Write(file, binary.LittleEndian, &vector); err != nil {
				return fmt.Errorf("写入向量 %d 失败: %v", pointID, err)
			}
		}
	}

	zap.L().Info("IVF索引保存完成",
		zap.String("path", path),
		zap.Int("nlist", i.NList),
		zap.Int("nprobe", i.NProbe),
		zap.Int("dim", i.Dim),
		zap.String("metric", i.Metric),
		zap.Uint32("total_vectors", i.GetDataSize()),
	)

	return nil
}
