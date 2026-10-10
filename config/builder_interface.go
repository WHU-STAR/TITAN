package config

type Builder interface {
	Build() *ClusterConfig
}
