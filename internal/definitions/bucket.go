package definitions

type Bucket struct {
	BucketId uint32
	Count    int32
	Points   []Point
}

type Point struct {
	PointId uint32
	Data    []float32
}
