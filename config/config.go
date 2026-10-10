package config

import (
	"fmt"
	"os"
)

type ClusterConfig struct {
	ConfigName     string
	SchedulerIP    string
	SchedulerPort  string
	ClientIP       string
	ClientPort     string
	LogLevel       string
	MetaIndexType  string
	NodeIndexType  string
	SearchMethod    string
	PruningStrategy string
	TopK           map[int][]int
	MetaK          []int
	// MetaKAndEfSearch 允许显式指定 MetaK 与 EfSearch 的组合对：
	// 形如 [[metaK, efSearch], ...]。如果该字段非空，则在实验参数生成时
	// 优先使用该字段，而忽略独立的 MetaK 与 EfSearch 列表。
	MetaKAndEfSearch [][]int
	Thresh         []int
	PruningLambdas []float64
	EfSearch       []int
	NProbe         []int
	QueryCount     int
	Epochs         int
	RecallGoal     float64
	RecallMinimum  float64
}

var Config *ClusterConfig
var builder Builder

func init() {
	// 如果没有指定builder，使用默认builder，目前只测试，默认builder
	env := os.Getenv("VCLUSTER_CONFIG_BUILDER")
	if env == "" {
		fmt.Print("VCLUSTER_CONFIG_BUILDER environment variable is not set, setting it to 'default'\n")
	}
	builder = &ConfigBuilder{
		Env: env,
	}
	Config = builder.Build()
}

func (c *ClusterConfig) GetSchedulerAddress() string {
	return c.SchedulerIP + ":" + c.SchedulerPort
}

func (c *ClusterConfig) GetClientAddress() string {
	return c.ClientIP + ":" + c.ClientPort
}

func (c *ClusterConfig) GetLogLevel() string {
	return c.LogLevel
}

func (c *ClusterConfig) GetInfo() string {
	res := "ConfigName: " + c.ConfigName + "\n"
	res += "SchedulerIP: " + c.SchedulerIP + "\n"
	res += "SchedulerPort: " + c.SchedulerPort + "\n"
	res += "ClientIP: " + c.ClientIP + "\n"
	res += "ClientPort: " + c.ClientPort + "\n"
	res += "LogLevel: " + c.LogLevel + "\n"
	res += "MetaIndexType: " + c.MetaIndexType + "\n"
	res += "NodeIndexType: " + c.NodeIndexType + "\n"
	res += "SearchMethod: " + c.SearchMethod + "\n"
	res += "PruningStrategy: " + c.PruningStrategy + "\n"
	res += fmt.Sprintf("RecallGoal: %.2f\n", c.RecallGoal)
	res += fmt.Sprintf("RecallMinimum: %.2f\n", c.RecallMinimum)
	res += fmt.Sprintf("PruningLambda: %v\n", c.PruningLambdas)
	res += fmt.Sprintf("MetaKAndEfSearch: %v\n", c.MetaKAndEfSearch)
	res += fmt.Sprintf("Epochs: %d\n", c.Epochs)
	res += fmt.Sprintf("EfSearch: %v\n", c.EfSearch)
	res += fmt.Sprintf("TopK: %v\n", c.TopK)
	res += fmt.Sprintf("MetaK: %v\n", c.MetaK)
	res += fmt.Sprintf("Thresh: %v\n", c.Thresh)
	res += fmt.Sprintf("NProbe: %v\n", c.NProbe)
	res += fmt.Sprintf("QueryCount: %d\n", c.QueryCount)
	return res
}
