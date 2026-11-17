phenome:
	python3 run_LR.py -r data/phoneme_train.csv -e data/phoneme_test.csv -a 0.02 

sklearn:
	python3 run_LR.py -r data/phoneme_train.csv -e data/phoneme_test.csv -a 0.02 --compare

test:
	python3 -m unittest -v test_hw07 

challenger:
	python3 run_LR.py -r data/challenger_train.csv -e data/challenger_test.csv -a 0.001 --tmax 10000 --eps 0 --compare --plot challenger.pdf
	