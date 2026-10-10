
# Generate a hostname for a worker node based on its index (0-based).

echo "worker-$(printf "%03d" $(($1)))"

