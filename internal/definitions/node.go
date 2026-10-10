package definitions

import (
	"time"
)

type Node struct {
	ID       string
	Address  string
	Score    int
	LastPing time.Time
}
