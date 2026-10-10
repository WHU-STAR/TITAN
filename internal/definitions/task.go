package definitions

import "time"

// Task is one vector-search request.
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
	Result     []int64   `json:"result,omitempty"`    // neighbor ids
	Distances  []float32 `json:"distances,omitempty"` // distances of the neighbors
}

// TaskStatus is the lifecycle of a task.
const (
	TaskStatusPending  = "pending"
	TaskStatusAssigned = "assigned"
	TaskStatusComplete = "complete"
	TaskStatusFailed   = "failed"
)

// TaskType selects meta-index routing or a broadcast node query.
const (
	TaskMetaQuery = "meta-query"
	TaskNodeQuery = "node-query"
)
