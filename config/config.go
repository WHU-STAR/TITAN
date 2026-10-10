package config

import (
	"fmt"
	"os"
)

type ClusterConfig struct {
	ConfigName      string
	SchedulerIP     string
	SchedulerPort   string
	ClientIP        string
	ClientPort      string
	LogLevel        string
	MetaIndexType   string
	NodeIndexType   string
	SearchMethod    string
	PruningStrategy string
	TopK            map[int][]int
	MetaK           []int
	// MetaKAndEfSearch lists explicit [metaK, efSearch] pairs.
	// When non-empty, experiment generation uses these pairs
	// and ignores the separate MetaK and EfSearch lists.
	MetaKAndEfSearch [][]int
	Thresh           []int
	PruningLambdas   []float64
	EfSearch         []int
	NProbe           []int
	QueryCount       int
	Epochs           int
	RecallGoal       float64
	RecallMinimum    float64
}

var Config *ClusterConfig
var builder Builder

func init() {
	// Fall back to the default builder when the env var is unset.
	env := os.Getenv("TITAN_CONFIG_BUILDER")
	if env == "" {
		env = "default"
		fmt.Print("TITAN_CONFIG_BUILDER is not set, using 'default'\n")
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
