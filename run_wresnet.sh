# dir=OthersWideResNet28x10_SGD_step_decay_10_BN_momentum0.9
dir=tmp
if [ -d "$dir" ]; then
  echo [ERROR] folder $dir already exists
else
  mkdir $dir
  cp run_wresnet.sh train.py models $dir -r
  CUDA_VISIBLE_DEVICES=0 python train.py -b 128 -e 200 -o sgd -lr_m cos_restart -net resnet -log $dir | tee $dir/log.log
fi