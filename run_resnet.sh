# methods used in ResNet110
methods=(step_decay cos_epoch tanh_epoch cos_iteration tanh_iteration cos_restart tanh_restart cos_restart_tanhdecay tanh_restart_tanhdecay cos_shift_restart_tanhdecay)
# methods used in Wide ResNet
# methods=(step_decay cos_epoch tanh_epoch cos_restart_tanhdecay tanh_restart_tanhdecay)
index=(7)
datasets=(10)
# datasets=(100)
Net=resnet
for k in {0..0}; do
	dataset=${datasets[$k]}
	for j in ${index[*]};
	do
		outer_dir=ResNet32_${dataset}_SGD_${methods[$j]}
		for i in {1..5}; 
		do 
		dir=${outer_dir}_$i
		if [ -d "$dir" ]; then
		  echo [ERROR] folder $dir already exists
		else
		  mkdir $dir
		  cp run_resnet.sh train.py models $dir -r
		  CUDA_VISIBLE_DEVICES=0 python train.py -b 128 -e 200 -d cifar${dataset} -o sgd -lr_m ${methods[$j]} -net ${Net} -log $dir | tee $dir/log.log
		fi
		done
		
		if [ -d "$outer_dir" ]; then
		  echo [ERROR] folder $dir already exists
		else
		  mkdir $outer_dir
		  mv ${outer_dir}_* $outer_dir
		fi	
	done
done
