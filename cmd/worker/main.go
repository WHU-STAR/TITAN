package main

import (
	"fmt"
	"log"
	"time"

	"github.com/WHU-STAR/TITAN/config"
	"github.com/WHU-STAR/TITAN/internal/worker"
	"github.com/WHU-STAR/TITAN/pkg/logger"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

var (
	WorkerAddress string
	WorkerID      string
	Mode          string
	Strategy      string
)

const (
	DEFAULT_ADDRESS  = "localhost:50051"
	DEFAULT_ID       = "worker-0"
	DEFAULT_MODE     = "singleNode"
	DEFAULT_STRATEGY = "similarity"
	RETRY_TIMES      = 300
	RETRY_WAIT_SEC   = 10
)

func NewWorkerCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "worker",
		Short: "Start worker",
		Run: func(cmd *cobra.Command, args []string) {
			worker := worker.NewWorker(WorkerID, WorkerAddress, Mode, Strategy)
			logger.Info("log level", zap.String("log level", config.Config.LogLevel))
			logger.Info("mode", zap.String("mode", Mode), zap.String("strategy", Strategy))

			err := worker.InitIndex()
			if err != nil {
				logger.Error("Failed to init index, exiting", zap.Error(err))
				log.Fatalf("worker %s failed to init index: %v", WorkerID, err)
			}
			go func() {
				for i := range RETRY_TIMES {
					err = worker.RegisterToScheduler(cmd.Context())
					if err == nil {
						break
					} else {
						logger.Error(fmt.Sprintf("Failed to register to scheduler, try %d of total %d", i+1, RETRY_TIMES))
						if i < RETRY_TIMES-1 {
							time.Sleep(RETRY_WAIT_SEC * time.Second)
						}
					}
				}
				if err != nil {
					logger.Error("Failed to register to scheduler", zap.Error(err))
				}
			}()
			go worker.StartHealthCheck(cmd.Context())
			go worker.AssignTasks(cmd.Context())
			worker.StartGRPCServer()
		},
	}
	cmd.Flags().StringVar(&WorkerID, "id", DEFAULT_ID, "The ID of the worker")
	cmd.Flags().StringVar(&WorkerAddress, "address", DEFAULT_ADDRESS, "The ip address of the worker")
	cmd.Flags().StringVar(&Mode, "mode", DEFAULT_MODE, "The mode of the worker")
	cmd.Flags().StringVar(&Strategy, "strategy", DEFAULT_STRATEGY, "The strategy of the worker")
	return cmd
}

func main() {
	cmd := NewWorkerCmd()
	if err := cmd.Execute(); err != nil {
		log.Fatal(err)
	}
}
