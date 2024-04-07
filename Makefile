.PHONY: data

lin_reg: 
	python3 -B src/lin_reg.py

log_reg: 
	python3 -B src/log_reg.py

mlp:
	echo "Please be patient..."
	python3 -B src/mlp.py
	
