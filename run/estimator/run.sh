set -x
root="/home/hemingzhi/.jupyter/ctr"
do="train"

epoch=1
train_batch_size=320

task="multitask_epoch${epoch}_bs${train_batch_size}_qp_MSE_7d"
table="xtr_v3"


train_date="20210721"
train_path="${root}/data/${table}/${train_date}_train_splits"
data_info_path="${root}/data/configs/${table}"
model_info_path="${root}/run/estimator/configs/xtr_v3.json"

eval_date="20210721"
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
  --model_info_path $model_info_path \
  --checkpoint_load_path $checkpoint_load_path  \
  --epoch $epoch --train_batch_size $train_batch_size

if [ "$do" = "eval" ]:
then
  python ./post_analyse.py --result_dir $save_path
fi
