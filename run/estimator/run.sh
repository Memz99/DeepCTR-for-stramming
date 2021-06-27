set -x
root="/home/hemingzhi/.jupyter/ctr"
do="train"

epoch=1
train_batch_size=320

task="multitask_epoch${epoch}_bs${train_batch_size}_sparse_p_MSE"
table="xtr_v2"


train_date="20210608"
train_path="${root}/data/${table}/${train_date}_train_splits"
data_info_path="${root}/data/configs/${table}"
train_info_path="${root}/run/estimator/configs/xtr_v2.json"

eval_date="20210609"
eval_path="${root}/data/${table}/${eval_date}_eval_splits"
checkpoint_load_path="${root}/result/${table}/${train_date}/${task}_train/checkpoint"

if [ "$do" = "train" ]
then
  save_path="${root}/result/${table}/${train_date}/${task}_${do}"
elif [ "$do" = "eval" ]
then
  save_path="${root}/result/${table}/${eval_date}/${task}_${do}"
fi


python ./main_test.py \
  --root $root \
  --do $do \
  --train_path $train_path --eval_path $eval_path \
  --save_path $save_path --data_info_path $data_info_path \
  --train_info_path $train_info_path \
  --checkpoint_load_path $checkpoint_load_path  \
  --epoch $epoch --train_batch_size $train_batch_size

