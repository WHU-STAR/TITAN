package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/WHU-STAR/TITAN/config"
	"github.com/WHU-STAR/TITAN/internal/scheduler"
	"github.com/WHU-STAR/TITAN/pkg/logger"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

var ServerAddress string
var Strategy string
var NumWorkers int

const DEFAULT_STRATEGY = "similarity"
const DEFAULT_WORKER_NUM = 16

func NewSchedulerCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "scheduler",
		Short: "Start scheduler",
		Run: func(cmd *cobra.Command, args []string) {
			ServerAddress = config.Config.SchedulerIP + ":" + config.Config.SchedulerPort
			scheduler := scheduler.NewScheduler(ServerAddress, Strategy, NumWorkers)

			logger.Info("Starting scheduler",
				zap.String("address", ServerAddress),
				zap.String("strategy", Strategy),
				zap.String("logLevel", config.Config.LogLevel))

			logger.Info("Config details", zap.String("Config Info", config.Config.GetInfo()))
			sigChan := make(chan os.Signal, 1)
			signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

			go scheduler.StartGRPCServer()

			// Wait for signal
			sig := <-sigChan
			log.Println("Received signal:", sig)

			scheduler.Stop()
		},
	}
	cmd.Flags().StringVarP(&Strategy, "strategy", "s", DEFAULT_STRATEGY, "The strategy of the scheduler")
	cmd.Flags().IntVarP(&NumWorkers, "workers", "w", DEFAULT_WORKER_NUM, "The number of workers")
	return cmd
}

func main() {
	cmd := NewSchedulerCmd()
	if err := cmd.Execute(); err != nil {
		log.Fatal(err)
	}
}
