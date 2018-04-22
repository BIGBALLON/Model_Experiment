#!/usr/bin/env bash
Datasets=(cifar10 cifar100 fashion_mnist)
LRSchedulerList=(step_decay step_decay_two step_decay_three 
exponential linear constant
tanh_restart cos_restart cos_restart_tanhdecay tanh_restart_tanhdecay
tanh_epoch cos_epoch cos_iteration tanh_iteration cos_tanh abs_sin)
# scheduler:
#  0 step_decay
#  1 step_decay_two
#  2 step_decay_three
#  3 exponential
#  4 linear
#  5 constant
#  6 tanh_restart
#  7 cos_restart
#  8 cos_restart_tanhdecay
#  9 tanh_restart_tanhdecay
# 10 tanh_epoch
# 11 cos_epoch
# 12 cos_iteration
# 13 tanh_iteration
# 14 cos_tanh
# 15 abs_sin
SchedulerListIndex=(0)
Depth=5
Width=1
BatchSize=128
Epochs=200
Opt=sgd
Net=resnet
# Net = lenet or resnet or wresnet
CosineConstant=0
TanhBegin=-2.0
TanhEnd=2.0
# FirstLrEpochNum=(40 60 80 100 120 140 160 )
FirstLrEpochNum=(40 160 )
# SecondLrEpochNum=(40 60 80 100 120 140 160 )
SecondLrEpochNum=(160 )
let depth=3*2*Depth+2
for first_eopch in ${FirstLrEpochNum[*]}; do
	for second_epoch in ${SecondLrEpochNum[*]}; do
		for i in {1..4}; do 
			for j in ${SchedulerListIndex[*]}; do
				for k in {0..0}; do
					dataset="${Datasets[$k]^^}"
					opt="${Opt^^}"
					outer_dir=${Net}
					if [ "${Net}" != "lenet" ]; then
						outer_dir+=_${depth}
					fi
					outer_dir+=_${dataset}_$opt_${LRSchedulerList[$j]}

					
					argument=" -b $BatchSize -d ${Datasets[$k]} -o $Opt -lr_m 
								${LRSchedulerList[$j]} -net ${Net} -depth $Depth 
								-width $Width -E1 $first_eopch -E2 $second_epoch "
					outer_dir+=_${first_eopch}_${second_epoch}
					
					# if [ "${LRSchedulerList[$j]}" == "cos_tanh" ]; then
					# 	argument+=" -sc ${CosineConstant}"
					# fi
					# if [ "${LRSchedulerList[$j]}" == "tanh_epoch" ]; then
					# 	argument+=" -tanh_begin ${TanhBegin} -tanh_end ${TanhEnd} "
					# 	outer_dir+=_${TanhBegin}_${TanhEnd}
					# fi
					
					dir=${outer_dir}_$i
					argument+="-log $dir "
					

					if [ -d "$dir" ]; then
					  echo [ERROR] folder $dir already exists
					else
					  mkdir $dir
					  cp run_net.sh ../our_callbacks.py *.py ../models ../Pytorch_ResNeXt $dir -r
		              echo $argument
					  CUDA_VISIBLE_DEVICES=0 python train.py $argument | tee $dir/log.log
					fi
					
					# if [ -d "$outer_dir" ]; then
					#   echo [ERROR] folder $dir already exists
					# else
					#   mkdir $outer_dir
					#   mv ${outer_dir}_* $outer_dir
					# fi	
				done
			done
		done
	done
done
