python3 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots --log ./logs_t1/ --ngpu 2 --learning_rate 0.05 -b 128 --method tanh
python3 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots --log ./logs_t2/ --ngpu 2 --learning_rate 0.05 -b 128 --method tanh
python3 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots --log ./logs_c1/ --ngpu 2 --learning_rate 0.05 -b 128 --method cos
python3 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots --log ./logs_c2/ --ngpu 2 --learning_rate 0.05 -b 128 --method cos
python3 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots --log ./logs_s1/ --ngpu 2 --learning_rate 0.05 -b 128
python3 train.py ~/DATASETS/cifar.python cifar10 -s ./snapshots --log ./logs_s2/ --ngpu 2 --learning_rate 0.05 -b 128
