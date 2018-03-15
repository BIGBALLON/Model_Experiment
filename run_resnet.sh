methods=(step_decay cos_epoch tanh_epoch cos_iteration tanh_iteration cos_restart tanh_restart cos_restart_reducebytanh tanh_restart_reducebytanh)
for j in {0..3};
do
	for i in {1..5}; 
	do 
	dir=ResNet32_10_SGD_${methods[$j]}_$i
	if [ -d "$dir" ]; then
	  echo [ERROR] folder $dir already exists
	else
	  mkdir $dir
	  cp run_resnet.sh train.py models $dir -r
	  CUDA_VISIBLE_DEVICES=0 python train.py -b 128 -e 200 -d cifar10 -o sgd -lr_m ${methods[$j]} -net resnet -log $dir | tee $dir/log.log
	fi
	done
done
