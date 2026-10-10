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
	SchedulerIP   string        `yaml:"schedulerIP"`
	SchedulerPort int           `yaml:"schedulerPort"`
	ClientIP      string        `yaml:"clientIP"`
	ClientPort    int           `yaml:"clientPort"`
	LogLevel      string        `yaml:"logLevel"`
	PruningLambda []float64     `yaml:"pruningLambda"`
	MetaK         []int         `yaml:"metaK"`
	// MetaKAndEfSearch 是新的组合配置键：
	// 形如 [[metaK, efSearch], ...]。如果该字段在 YAML 中存在且非空，
	// 则在上层逻辑中会优先使用该字段，而忽略独立的 MetaK 与 EfSearch。
	MetaKAndEfSearch [][]int     `yaml:"metaKAndEfSearch"`
	Thresh        []int         `yaml:"threshold"`
	TopK          map[int][]int `yaml:"topK"`
	QueryCount    int           `yaml:"queryCount"`
	Epochs        int           `yaml:"epochs"`
	MetaIndexType string        `yaml:"metaIndexType"`
	NodeIndexType string        `yaml:"nodeIndexType"`
	SearchMethod    string `yaml:"searchMethod"`
	PruningStrategy string `yaml:"pruningStrategy"`
	EfSearch      []int         `yaml:"efSearch"`
	NProbe        []int         `yaml:"nProbe"`
	RecallGoal    float64       `yaml:"recallGoal"`
	RecallMinimum float64       `yaml:"recallMinimum"`
}

// recursively build config
func (cb *ConfigBuilder) Build() *ClusterConfig {
	// 读取 YAML 配置文件
	configFilePath := "../runconfig.yaml"
	// print pwd
	yamlFile, err := os.ReadFile(configFilePath)
	if err != nil {
		// print err
		fmt.Printf("Error reading config file: %v\n", err)
	}

	// 解析为 map[string]interface{}
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
	// 构建最终配置对象
	return &ClusterConfig{
		ConfigName:     cb.Env,
		SchedulerIP:    envConfig.SchedulerIP,
		SchedulerPort:  fmt.Sprintf("%d", envConfig.SchedulerPort),
		ClientIP:       envConfig.ClientIP,
		ClientPort:     fmt.Sprintf("%d", envConfig.ClientPort),
		LogLevel:       envConfig.LogLevel,
		PruningLambdas: envConfig.PruningLambda,
		TopK:           envConfig.TopK,
		MetaKAndEfSearch: envConfig.MetaKAndEfSearch,
		MetaK:          envConfig.MetaK,
		Thresh:         envConfig.Thresh,
		QueryCount:     envConfig.QueryCount,
		Epochs:         envConfig.Epochs,
		MetaIndexType:  envConfig.MetaIndexType,
		NodeIndexType:  envConfig.NodeIndexType,
		EfSearch:       envConfig.EfSearch,
		SearchMethod:    envConfig.SearchMethod,
		PruningStrategy: envConfig.PruningStrategy,
		NProbe:         envConfig.NProbe,
		RecallGoal:     envConfig.RecallGoal,
		RecallMinimum:  envConfig.RecallMinimum,
	}
}
