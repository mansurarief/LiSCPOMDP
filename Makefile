NUM_THREADS=1


instantiate: 
	@echo "Instantiating environment"
	@julia --project=. -e 'using Pkg; Pkg.instantiate()'
	
experiment:
	@JULIA_NUM_THREADS=$(NUM_THREADS) julia --threads $(NUM_THREADS) --project=. experiment.jl

main:
	@echo "Running main"
	julia --project=. experiment.jl

viz:
	@echo "Running viz"
	julia --project=. test_viz.jl	

run_all: main experiment