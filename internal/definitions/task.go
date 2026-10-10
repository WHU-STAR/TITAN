package definitions

import "time"

// Task 表示一个向量检索任务
type Task struct {
	ID         string    `json:"id"`
	QueryID    uint32    `json:"query_id"`
	Query      []float32 `json:"query"`
	Topk       int64     `json:"topk"`
	WorkerTopK int64     `json:"worker_topk"`
	Type       string    `json:"type"`
	RecallGoal float32   `json:"recall_goal"`
	Status     string    `json:"status"`
	CreatedAt  time.Time `json:"created_at"`
	NodeID     string    `json:"node_id,omitempty"`
	Result     []int64   `json:"result,omitempty"`    // ID表示的检索结果
	Distances  []float32 `json:"distances,omitempty"` // 检索结果的距离
}

// TaskStatus 定义任务状态
const (
	TaskStatusPending  = "pending"
	TaskStatusAssigned = "assigned"
	TaskStatusComplete = "complete"
	TaskStatusFailed   = "failed"
)

// TaskType 定义任务类型
const (
	TaskMetaQuery = "meta-query"
	TaskNodeQuery = "node-query"
)
