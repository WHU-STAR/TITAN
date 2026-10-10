package definitions

type Query struct {
	ID         string    `json:"id"`
	Query      []float32 `json:"query"`
	Topk       int       `json:"topk"`
	Buckets    []uint32  `json:"buckets"`
	Goal       float32   `json:"goal"`
	SearchType string    `json:"search_type"`
}

const (
	ANN        = "ann"
	BruteForce = "bf"
)

const (
	SET_EF_SEARCH = iota
	SET_THRESHOLD
	SET_META_K
	SET_NPROBE
	SET_PRUNING_LAMBDA
)

type IntegerQueryType int32

const (
	GET_TOTAL_BUCKET_COUNT IntegerQueryType = iota
	GET_TOTAL_FILTERED_BUCKET_COUNT
)
