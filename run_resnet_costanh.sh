# methods=(step_decay cos_epoch tanh_epoch cos_iteration tanh_iteration cos_restart tanh_restart cos_restart_reducebytanh tanh_restart_reducebytanh)
cosine_weights=(0.1 0.3 0.5 0.7 0.9)
method=cos_tanh
for j in {0..2};
do
	for i in {1..5}; 
	do 
	dir=ResNet32_100_SGD_${method}_CosWeight_${cosine_weights[$j]}_$i
	if [ -d "$dir" ]; then
	  echo [ERROR] folder $dir already exists
	else
	  mkdir $dir
	  cp run_resnet.sh train.py models $dir -r
	  CUDA_VISIBLE_DEVICES=0 python train.py -b 128 -e 200 -d cifar100 -o sgd -lr_m $method -sc ${cosine_weights[$j]} -net resnet -log $dir | tee $dir/log.log
	fi
	done
done
