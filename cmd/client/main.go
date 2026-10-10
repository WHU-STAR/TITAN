package main

import (
	"context"
	"fmt"
	"log"
	"sort"
	"sync"
	"time"

	"github.com/WHU-STAR/TITAN/config"
	"github.com/WHU-STAR/TITAN/internal/client"
	"github.com/WHU-STAR/TITAN/internal/definitions"
	"github.com/WHU-STAR/TITAN/pkg/logger"
	"github.com/spf13/cobra"
)

var (
	ClientQueryCount int
	ClientAddress    string
	QuerySetPath     string
	GroundTruthPath  string
	RunEpochs        int    // 重复运行的次数
	RunName          string // 运行名称，用于保存结果
)

const (
	DEFAULT_QUERY_SET_PATH    = "./data/sift/sift_query.fvecs"
	DEFAULT_GROUND_TRUTH_PATH = "./data/sift/sift_groundtruth.ivecs"
	DEFAULT_RUN_NAME          = "default_run"
	MAX_RETRY_COUNT           = 7
)

// ExperimentConfig 实验配置结构体
type ExperimentConfig struct {
	TopKs           map[int][]int
	EfSearchs       []int
	MetaKs          []int
	// MetaKAndEfSearch 允许显式指定 MetaK 与 EfSearch 的组合：
	// 形如 [[metaK, efSearch], ...]。如果该字段非空，则生成实验参数时
	// 仅使用该字段给出的组合，而忽略独立的 MetaKs 与 EfSearchs。
	MetaKAndEfSearch [][]int
	Thresholds       []int
	NProbes          []int
	PruningLambdas   []float64
	QueryCount       int
	Epochs           int
}

var (
	SkippedParams      = []definitions.ExperimentParams{}
	ActuallySkipped    = []definitions.ExperimentParams{}
	actualSkippedCount = 0
)

func skippable(param definitions.ExperimentParams) bool {
	for _, skippableParam := range SkippedParams {
		paramMatch := param.FinalTopK == skippableParam.FinalTopK &&
			param.Threshold == skippableParam.Threshold &&
			param.NProbe == skippableParam.NProbe
		if paramMatch {
			if skippableParam.WorkerTopK <= param.WorkerTopK && skippableParam.EfSearch <= param.EfSearch && skippableParam.MetaK <= param.MetaK {
				// skippable
				return true
			}
		}
	}
	return false
}

// isInSkippedParams 检查参数是否已经在SkippedParams中
func isInSkippedParams(param definitions.ExperimentParams) bool {
	for _, skippedParam := range SkippedParams {
		if param.FinalTopK == skippedParam.FinalTopK &&
			param.WorkerTopK == skippedParam.WorkerTopK &&
			param.EfSearch == skippedParam.EfSearch &&
			param.MetaK == skippedParam.MetaK &&
			param.NProbe == skippedParam.NProbe &&
			param.Threshold == skippedParam.Threshold &&
			param.PruningLambda == skippedParam.PruningLambda {
			return true
		}
	}
	return false
}

// initializeClient 初始化客户端
func initializeClient() (*client.Client, error) {
	ClientAddress = config.Config.GetClientAddress()
	client := client.NewClient(ClientAddress)
	client.RunName = RunName
	client.RunEpochs = config.Config.Epochs

	if err := client.LoadQuerySet(QuerySetPath); err != nil {
		return nil, fmt.Errorf("failed to load query set: %w", err)
	}

	if err := client.LoadGroundTruth(GroundTruthPath); err != nil {
		return nil, fmt.Errorf("failed to load ground truth: %w", err)
	}

	return client, nil
}

// waitForServerInitialization 等待服务器初始化完成
func waitForServerInitialization(client *client.Client, ctx context.Context) error {
	ticker := time.NewTicker(4 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if client.IsServerInitialized() {
				return nil
			}
			log.Println("Waiting for server to be initialized...")
		}
	}
}

// getExperimentConfig 获取实验配置
func getExperimentConfig() ExperimentConfig {
	return ExperimentConfig{
		TopKs:           config.Config.TopK,
		EfSearchs:       config.Config.EfSearch,
		MetaKs:          config.Config.MetaK,
		MetaKAndEfSearch: config.Config.MetaKAndEfSearch,
		Thresholds:      config.Config.Thresh,
		NProbes:         config.Config.NProbe,
		QueryCount:      config.Config.QueryCount,
		Epochs:          config.Config.Epochs,
		PruningLambdas:  config.Config.PruningLambdas,
	}
}

// generateExperimentParams 生成所有实验参数组合
func generateExperimentParams(config ExperimentConfig) []definitions.ExperimentParams {
	var params []definitions.ExperimentParams

	sort.Ints(config.EfSearchs)
	sort.Ints(config.MetaKs)
	sort.Ints(config.Thresholds)

	useMetaKEfPairs := len(config.MetaKAndEfSearch) > 0
	// LOL, this is a bad code, but maybe couldn't be better
	for finalTopK, workerTopKs := range config.TopKs {
		sort.Ints(workerTopKs)
		for _, workerTopK := range workerTopKs {
			if useMetaKEfPairs {
				for _, pair := range config.MetaKAndEfSearch {
					if len(pair) < 2 {
						continue
					}
					metaK := pair[0]
					efSearch := pair[1]
					for _, nProbe := range config.NProbes {
						for _, pruningLambda := range config.PruningLambdas {
							for _, threshold := range config.Thresholds {
								if efSearch >= workerTopK {
									params = append(params, definitions.ExperimentParams{
										FinalTopK:     finalTopK,
										WorkerTopK:    workerTopK,
										EfSearch:      efSearch,
										MetaK:         metaK,
										NProbe:        nProbe,
										Threshold:     threshold,
										PruningLambda: pruningLambda,
									})
								}
							}
						}
					}
				}
			} else {
				for _, efSearch := range config.EfSearchs {
					for _, metaK := range config.MetaKs {
						for _, nProbe := range config.NProbes {
							for _, pruningLambda := range config.PruningLambdas {
								for _, threshold := range config.Thresholds {
									if efSearch >= workerTopK {
										params = append(params, definitions.ExperimentParams{
											FinalTopK:     finalTopK,
											WorkerTopK:    workerTopK,
											EfSearch:      efSearch,
											MetaK:         metaK,
											NProbe:        nProbe,
											Threshold:     threshold,
											PruningLambda: pruningLambda,
										})
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return params
}

func configureClient(client *client.Client, params definitions.ExperimentParams) error {
	configureFunc := func() error {
		if err := client.SetEFSearch(params.EfSearch); err != nil {
			return fmt.Errorf("failed to set efSearch: %w", err)
		}
		if err := client.SetMetaK(params.MetaK); err != nil {
			return fmt.Errorf("failed to set metaK: %w", err)
		}
		if err := client.SetNProbe(params.NProbe); err != nil {
			return fmt.Errorf("failed to set nProbe: %w", err)
		}
		if err := client.SetThreshold(params.Threshold); err != nil {
			return fmt.Errorf("failed to set threshold: %w", err)
		}
		if err := client.SetLambda(float32(params.PruningLambda)); err != nil {
			return fmt.Errorf("failed to set lambda: %w", err)
		}
		return nil
	}

	var lastErr error
	for range MAX_RETRY_COUNT {
		if err := configureFunc(); err != nil {
			lastErr = err
			continue
		}
		return nil
	}
	return lastErr
}

// runSingleExperiment 运行单次实验
func runSingleExperiment(client *client.Client, params definitions.ExperimentParams, queryCount int) (definitions.RunResult, error) {
	var wg sync.WaitGroup
	wg.Add(2)

	skipFunc := func(recall float64) {
		if recall >= config.Config.RecallGoal {
			// 检查是否已经在SkippedParams中，避免重复添加
			if !isInSkippedParams(params) {
				logger.Info(fmt.Sprintf("Recall reached goal. Adding current parameters to skip list: %s",
					params.ToString()))
				SkippedParams = append(SkippedParams, params)
			}
		}

	}

	logger.Info("Starting experiment with parameters: " + params.ToString())

	// 启动查询
	go func() {
		defer wg.Done()
		client.StartQuery(queryCount, params.FinalTopK, params.WorkerTopK)
	}()

	res := definitions.RunResult{}

	// 处理结果
	go func() {
		defer wg.Done()
		res = client.HandleResult(queryCount, skipFunc)
	}()

	wg.Wait()
	return res, nil
}

// runExperiments 运行所有实验
func runExperiments(client *client.Client, experimentConf ExperimentConfig, params []definitions.ExperimentParams) error {
	totalIterations := len(params)
	startTime := time.Now()

	for i, param := range params {
		if skippable(param) {
			logger.Info(fmt.Sprintf("Skipping parameters: %s", param.ToString()))
			ActuallySkipped = append(ActuallySkipped, param)
			actualSkippedCount++
			continue
		}

		// 配置客户端
		if err := configureClient(client, param); err != nil {
			return fmt.Errorf("failed to configure client: %w", err)
		}
		logger.Info(fmt.Sprintf("EfSearch: %d", param.EfSearch))
		logger.Info(fmt.Sprintf("MetaK: %d", param.MetaK))
		logger.Info(fmt.Sprintf("NProbe: %d", param.NProbe))
		logger.Info(fmt.Sprintf("Threshold: %d", param.Threshold))
		logger.Info(fmt.Sprintf("ClientQueryCount: %d", experimentConf.QueryCount))

		experimentConf.QueryCount = min(experimentConf.QueryCount, int(client.QueryData.Num))
		// 计算进度
		for epoch := range experimentConf.Epochs {
			progress := float64(i+1) / float64(totalIterations) * 100
			elapsed := time.Since(startTime)
			var eta time.Duration
			if i > 0 {
				eta = time.Duration(float64(elapsed) / float64(i+1) * float64(totalIterations-i-1))
			}

			// 记录参数
			log.Printf("Epoch: %d, Progress: %.2f%%, Elapsed: %s, ETA: %s\n",
				epoch+1, progress, elapsed.Truncate(time.Second), eta.Truncate(time.Second))
			log.Printf("TotalIterations: %d, CurrentIteration: %d\n", totalIterations, i+1)

			// 运行实验
			res, err := runSingleExperiment(client, param, experimentConf.QueryCount)
			if err != nil {
				return fmt.Errorf("failed to run experiment: %w", err)
			}

			// check if res.Recall is NaN or lower than RecallMinimum
			if res.Recall != res.Recall {
				logger.Info(fmt.Sprintf("Recall is NaN for parameters: %s. Skipping further epochs.", param.ToString()))
				break
			}

			if res.Recall < config.Config.RecallMinimum {
				logger.Info(fmt.Sprintf("Recall %.4f is below minimum %.4f for parameters: %s. Skipping further epochs.",
					res.Recall, config.Config.RecallMinimum, param.ToString()))
				break
			}

			log.Printf("Epoch %d finished\n", epoch+1)

		}
	}

	// 打印统计信息
	log.Println("\n========== 实验统计信息 ==========")
	log.Printf("总参数组合数: %d\n", totalIterations)
	log.Printf("实际跳过的参数组合数: %d\n", actualSkippedCount)
	log.Printf("实际运行的参数组合数: %d\n", totalIterations-actualSkippedCount)
	log.Printf("达到目标recall的参数组合数: %d\n", len(SkippedParams))

	if len(SkippedParams) > 0 {
		log.Println("\n达到目标recall的参数组合:")
		for i, param := range SkippedParams {
			log.Printf("%d. %s\n", i+1, param.ToString())
		}
	}

	if actualSkippedCount > 0 {
		log.Println("\n实际被跳过的参数组合:")
		for i, param := range ActuallySkipped {
			log.Printf("%d. %s\n", i+1, param.ToString())
		}
	}
	log.Println("================================")

	return nil
}

func NewClientCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "client",
		Short: "Start client",
		Run: func(cmd *cobra.Command, args []string) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			// 初始化客户端
			client, err := initializeClient()
			if err != nil {
				log.Fatalf("Failed to initialize client: %v", err)
			}
			defer client.Stop()

			// 启动gRPC服务器
			go client.StartGRPCServer()

			// 获取实验配置
			expConfig := getExperimentConfig()
			ClientQueryCount = expConfig.QueryCount

			expParams := generateExperimentParams(expConfig)
			// 记录配置信息
			log.Println("MetaK:", expConfig.MetaKs)
			if len(expConfig.MetaKAndEfSearch) > 0 {
				log.Println("MetaKAndEfSearch (pairs [metaK, efSearch]):", expConfig.MetaKAndEfSearch)
			}
			log.Println("TopK:", expConfig.TopKs)
			log.Println("EfSearch:", expConfig.EfSearchs)
			log.Println("NProbe:", expConfig.NProbes)
			log.Println("Thresholds:", expConfig.Thresholds)
			log.Println("Lambdas:", expConfig.PruningLambdas)
			log.Println("Runepochs:", expConfig.Epochs)
			log.Println("QuerySetPath:", QuerySetPath)
			log.Println("GroundTruthPath:", GroundTruthPath)
			log.Println("Total number of experiment params generated: ", len(expParams))

			// 等待服务器初始化
			if err := waitForServerInitialization(client, ctx); err != nil {
				log.Fatalf("Server initialization failed: %v", err)
			}

			// 运行实验
			if err := runExperiments(client, expConfig, expParams); err != nil {
				log.Fatalf("Experiment failed: %v", err)
			}
		},
	}
	cmd.Flags().StringVar(&QuerySetPath, "query_set_path", DEFAULT_QUERY_SET_PATH, "The path of query set")
	cmd.Flags().StringVar(&GroundTruthPath, "ground_truth_path", DEFAULT_GROUND_TRUTH_PATH, "The path of ground truth")
	cmd.Flags().StringVar(&RunName, "run_name", DEFAULT_RUN_NAME, "The name of the run")
	return cmd
}

func main() {
	cmd := NewClientCmd()
	if err := cmd.Execute(); err != nil {
		log.Fatal(err)
	}
}
