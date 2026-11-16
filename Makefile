phenome:
	python3 run_LR.py -r data/phoneme_train.csv -e data/phoneme_test.csv -a 0.02 > phenome.txt

test:
	python3 -m unittest -v test_hw07 
	