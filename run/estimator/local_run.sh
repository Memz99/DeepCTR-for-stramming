set -x
root="/Users/hemingzhi/Documents/Projects/ctr"
do="eval"
task="multitask_epoch1_MSE"
table="xtr_v3"

train_date="20210721_high_photo"
train_path="${root}/data/${table}/${train_date}_train_splits"
data_info_path="${root}/data/configs/${table}"
model_info_path="${root}/run/estimator/configs/xtr_v3.json"

eval_date="20210721_high_photo"
eval_path="${root}/data/${table}/${eval_date}_eval_splits"
checkpoint_load_path="${root}/result/${table}/${train_date}/${task}_train/checkpoint"

if [ $do == "train" ]
then
  save_path="${root}/result/${table}/${train_date}/${task}_${do}"
elif [ $do == "eval" ]
then
  save_path="${root}/result/${table}/${eval_date}/${task}_${do}"
fi

python ./main_test.py \
  --root $root --is_local \
  --do $do \
  --train_path $train_path --eval_path $eval_path \
  --data_info_path $data_info_path \
  --model_info_path $model_info_path \
  --checkpoint_load_path $checkpoint_load_path  \
  --save_path $save_path
