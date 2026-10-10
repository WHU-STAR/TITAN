package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type ConfigBuilder struct {
	Builder
	Env string
}

type EnvConfig struct {
	SchedulerIP   string    `yaml:"schedulerIP"`
	SchedulerPort int       `yaml:"schedulerPort"`
	ClientIP      string    `yaml:"clientIP"`
	ClientPort    int       `yaml:"clientPort"`
	LogLevel      string    `yaml:"logLevel"`
	PruningLambda []float64 `yaml:"pruningLambda"`
	MetaK         []int     `yaml:"metaK"`
	// MetaKAndEfSearch lists explicit [metaK, efSearch] pairs.
	// When set in YAML, experiment generation prefers these pairs.
	MetaKAndEfSearch [][]int       `yaml:"metaKAndEfSearch"`
	Thresh           []int         `yaml:"threshold"`
	TopK             map[int][]int `yaml:"topK"`
	QueryCount       int           `yaml:"queryCount"`
	Epochs           int           `yaml:"epochs"`
	MetaIndexType    string        `yaml:"metaIndexType"`
	NodeIndexType    string        `yaml:"nodeIndexType"`
	SearchMethod     string        `yaml:"searchMethod"`
	PruningStrategy  string        `yaml:"pruningStrategy"`
	EfSearch         []int         `yaml:"efSearch"`
	NProbe           []int         `yaml:"nProbe"`
	RecallGoal       float64       `yaml:"recallGoal"`
	RecallMinimum    float64       `yaml:"recallMinimum"`
}

// recursively build config
func (cb *ConfigBuilder) Build() *ClusterConfig {
	// Read the YAML config.
	configFilePath := "../runconfig.yaml"
	// print pwd
	yamlFile, err := os.ReadFile(configFilePath)
	if err != nil {
		// print err
		fmt.Printf("Error reading config file: %v\n", err)
	}

	// Decode into a generic map.
	allConfigs := make(map[string]EnvConfig)
	err = yaml.Unmarshal(yamlFile, &allConfigs)
	if err != nil {
		// print err
		fmt.Printf("Error parsing config file: %v\n", err)
	}
	envConfig := allConfigs[cb.Env]
	if envConfig.RecallMinimum == 0 {
		envConfig.RecallMinimum = 0.88
	}
	// Build the cluster config.
	return &ClusterConfig{
		ConfigName:       cb.Env,
		SchedulerIP:      envConfig.SchedulerIP,
		SchedulerPort:    fmt.Sprintf("%d", envConfig.SchedulerPort),
		ClientIP:         envConfig.ClientIP,
		ClientPort:       fmt.Sprintf("%d", envConfig.ClientPort),
		LogLevel:         envConfig.LogLevel,
		PruningLambdas:   envConfig.PruningLambda,
		TopK:             envConfig.TopK,
		MetaKAndEfSearch: envConfig.MetaKAndEfSearch,
		MetaK:            envConfig.MetaK,
		Thresh:           envConfig.Thresh,
		QueryCount:       envConfig.QueryCount,
		Epochs:           envConfig.Epochs,
		MetaIndexType:    envConfig.MetaIndexType,
		NodeIndexType:    envConfig.NodeIndexType,
		EfSearch:         envConfig.EfSearch,
		SearchMethod:     envConfig.SearchMethod,
		PruningStrategy:  envConfig.PruningStrategy,
		NProbe:           envConfig.NProbe,
		RecallGoal:       envConfig.RecallGoal,
		RecallMinimum:    envConfig.RecallMinimum,
	}
}
