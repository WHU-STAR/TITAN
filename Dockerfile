# Build stage
FROM golang:1.23-alpine AS builder

# Install build dependencies
RUN apk add --no-cache gcc musl-dev make cmake g++

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Clean and build
RUN make clean && \
    make build

# Final stage
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache libstdc++ libgcc bash

WORKDIR /app

# Copy binaries, config and data from builder
COPY --from=builder /app/bin/ ./bin/
COPY --from=builder /app/config/ ./config/
COPY --from=builder /app/data/ ./data/
COPY --from=builder /app/internal/engine/build/lib* /usr/lib/

# Create necessary directories
RUN mkdir -p log pids

# Create entrypoint script
COPY <<-"EOF" /entrypoint.sh
#!/bin/sh
case "$1" in
    "scheduler")
        ./bin/scheduler
        ;;
    "worker")
        ./bin/worker
        ;;
    "client")
        ./bin/client
        ;;
    *)
        echo "Usage: $0 {scheduler|worker|client}"
        exit 1
        ;;
esac
EOF

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]