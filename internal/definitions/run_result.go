package definitions

import (
	"encoding/csv"
	"fmt"
	"os"
)

type RunResult struct {
	MetaK      int     `json:"meta_k"`
	TopK       int     `json:"top_k"`
	WorkerTopK int     `json:"worker_top_k"`
	EFSearch   int     `json:"ef_search"`
	Thresh     int     `json:"thresh"`
	NProbe     int     `json:"n_probe"`
	Lambda     float64 `json:"lambda"`
	Recall     float64 `json:"recall"`
	QPS        float64 `json:"qps"`
	AvgTime    float64 `json:"avg_time"`
	P50Time    float64 `json:"p50_time"`
	P95Time    float64 `json:"p95_time"`
	P99Time    float64 `json:"p99_time"`
}

func (rr *RunResult) ToCsvRow() []string {
	return []string{
		fmt.Sprintf("%d", rr.MetaK),
		fmt.Sprintf("%d", rr.TopK),
		fmt.Sprintf("%d", rr.WorkerTopK),
		fmt.Sprintf("%d", rr.EFSearch),
		fmt.Sprintf("%d", rr.Thresh),
		fmt.Sprintf("%d", rr.NProbe),
		fmt.Sprintf("%.4f", rr.Lambda),
		fmt.Sprintf("%.4f", rr.Recall),
		fmt.Sprintf("%.4f", rr.QPS),
		fmt.Sprintf("%.4f", rr.AvgTime),
		fmt.Sprintf("%.4f", rr.P50Time),
		fmt.Sprintf("%.4f", rr.P95Time),
		fmt.Sprintf("%.4f", rr.P99Time),
	}
}

func (rr *RunResult) WriteToCsv(csvPath string) error {
	fileExists := false

	if _, err := os.Stat(csvPath); err == nil {
		fileExists = true
	}
	file, err := os.OpenFile(csvPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()
	if !fileExists {
		header := []string{"meta_k", "top_k", "worker_top_k", "ef_search", "thresh", "n_probe", "lambda", "recall", "qps", "avg_time", "p50_time", "p95_time", "p99_time"}
		if err := writer.Write(header); err != nil {
			return fmt.Errorf("failed to write header: %v", err)
		}
	}
	row := rr.ToCsvRow()
	if err := writer.Write(row); err != nil {
		return fmt.Errorf("failed to write row: %v", err)
	}

	return nil
}
