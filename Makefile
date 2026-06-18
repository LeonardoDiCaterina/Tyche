.PHONY: all run-h100-experiments clean

all: run-h100-experiments

run-h100-experiments:
	@echo "Starting H100 Experiment Tracking Suite..."
	@mkdir -p results
	@echo "1. Running Throughput Sweep..."
	python experiments/h100_throughput_sweep.py --output results/throughput_results.csv
	@echo "2. Running Embedding Convergence Sweep..."
	python experiments/embedding_convergence_sweep.py --output results/convergence_results.json
	@echo "Experiments completed successfully! Artifacts are in the results/ directory."

clean:
	rm -rf results/
