.PHONY: data

data: data/lin_reg_data/data.csv
	python -B src/regression.py

height: data/lin_reg_data/Heights.csv
	python -B src/regression.py


