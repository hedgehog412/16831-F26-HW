prob 1.3

python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1 \
--eval_batch_size 10000

python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Hopper.pkl \
--env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
--video_log_freq -1 \
--eval_batch_size 10000

prob 1.4

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 \
    --exp_name bc_ant \
    --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 10000 \
    --video_log_freq -1 \
    --size 16

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 \
    --exp_name bc_ant \
    --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 10000 \
    --video_log_freq -1 \
    --size 32

python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 \
  --exp_name bc_ant \
  --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --eval_batch_size 10000 \
  --video_log_freq -1 \
  --size 64

python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 \
  --exp_name bc_ant \
  --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --eval_batch_size 10000 \
  --video_log_freq -1 \
  --size 128

python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 \
  --exp_name bc_ant \
  --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --eval_batch_size 10000 \
  --video_log_freq -1 \
  --size 256

prob 2.2

python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
--do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1 \
--eval_batch_size 10000

python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Hopper.pkl \
--env_name Hopper-v2 --exp_name dagger_hopper --n_iter 10 \
--do_dagger --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
--video_log_freq -1 \
--eval_batch_size 10000