022-05-12 09:01:22 Starting - Starting the training job...
2022-05-12 09:01:48 Starting - Preparing the instances for trainingProfilerReport-1652346082: InProgress
......
2022-05-12 09:02:53 Downloading - Downloading input data...
2022-05-12 09:03:13 Training - Downloading the training image...........................
2022-05-12 09:07:55 Training - Training image download completed. Training in progress...bash: cannot set terminal process group (-1): Inappropriate ioctl for device
bash: no job control in this shell
2022-05-12 09:07:57,937 sagemaker-training-toolkit INFO     Imported framework sagemaker_pytorch_container.training
2022-05-12 09:07:57,955 sagemaker_pytorch_container.training INFO     Block until all host DNS lookups succeed.
2022-05-12 09:07:57,961 sagemaker_pytorch_container.training INFO     Invoking user training script.
2022-05-12 09:07:58,612 sagemaker-training-toolkit INFO     Invoking user script
Training Env:
{
    "additional_framework_parameters": {},
    "channel_input_dirs": {
        "test": "/opt/ml/input/data/test",
        "train": "/opt/ml/input/data/train",
        "val": "/opt/ml/input/data/val"
    },
    "current_host": "algo-1",
    "framework_module": "sagemaker_pytorch_container.training:main",
    "hosts": [
        "algo-1"
    ],
    "hyperparameters": {
        "do_eval": true,
        "do_train": true,
        "learning_rate": 3e-05,
        "max_seq_length": 128,
        "model_name_or_path": "uer/chinese_roberta_L-12_H-768",
        "num_train_epochs": 1,
        "output_dir": "/opt/ml/model/chinese_roberta",
        "per_device_train_batch_size": 64,
        "train_file": "/opt/ml/input/data/train/train.csv",
        "validation_file": "/opt/ml/input/data/val/val.csv"
    },
    "input_config_dir": "/opt/ml/input/config",
    "input_data_config": {
        "test": {
            "TrainingInputMode": "File",
            "S3DistributionType": "FullyReplicated",
            "RecordWrapperType": "None"
        },
        "train": {
            "TrainingInputMode": "File",
            "S3DistributionType": "FullyReplicated",
            "RecordWrapperType": "None"
        },
        "val": {
            "TrainingInputMode": "File",
            "S3DistributionType": "FullyReplicated",
            "RecordWrapperType": "None"
        }
    },
    "input_dir": "/opt/ml/input",
    "is_master": true,
    "job_name": "huggingface-pytorch-training-2022-05-12-09-01-22-376",
    "log_level": 20,
    "master_hostname": "algo-1",
    "model_dir": "/opt/ml/model",
    "module_dir": "s3://sagemaker-us-east-1-635837196364/huggingface-pytorch-training-2022-05-12-09-01-22-376/source/sourcedir.tar.gz",
    "module_name": "run_glue",
    "network_interface_name": "eth0",
    "num_cpus": 4,
    "num_gpus": 1,
    "output_data_dir": "/opt/ml/output/data",
    "output_dir": "/opt/ml/output",
    "output_intermediate_dir": "/opt/ml/output/intermediate",
    "resource_config": {
        "current_host": "algo-1",
        "current_instance_type": "ml.g5.xlarge",
        "current_group_name": "homogeneousCluster",
        "hosts": [
            "algo-1"
        ],
        "instance_groups": [
            {
                "instance_group_name": "homogeneousCluster",
                "instance_type": "ml.g5.xlarge",
                "hosts": [
                    "algo-1"
                ]
            }
        ],
        "network_interface_name": "eth0"
    },
    "user_entry_point": "run_glue.py"
}
Environment variables:
SM_HOSTS=["algo-1"]
SM_NETWORK_INTERFACE_NAME=eth0
SM_HPS={"do_eval":true,"do_train":true,"learning_rate":3e-05,"max_seq_length":128,"model_name_or_path":"uer/chinese_roberta_L-12_H-768","num_train_epochs":1,"output_dir":"/opt/ml/model/chinese_roberta","per_device_train_batch_size":64,"train_file":"/opt/ml/input/data/train/train.csv","validation_file":"/opt/ml/input/data/val/val.csv"}
SM_USER_ENTRY_POINT=run_glue.py
SM_FRAMEWORK_PARAMS={}
SM_RESOURCE_CONFIG={"current_group_name":"homogeneousCluster","current_host":"algo-1","current_instance_type":"ml.g5.xlarge","hosts":["algo-1"],"instance_groups":[{"hosts":["algo-1"],"instance_group_name":"homogeneousCluster","instance_type":"ml.g5.xlarge"}],"network_interface_name":"eth0"}
SM_INPUT_DATA_CONFIG={"test":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"},"train":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"},"val":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}
SM_OUTPUT_DATA_DIR=/opt/ml/output/data
SM_CHANNELS=["test","train","val"]
SM_CURRENT_HOST=algo-1
SM_MODULE_NAME=run_glue
SM_LOG_LEVEL=20
SM_FRAMEWORK_MODULE=sagemaker_pytorch_container.training:main
SM_INPUT_DIR=/opt/ml/input
SM_INPUT_CONFIG_DIR=/opt/ml/input/config
SM_OUTPUT_DIR=/opt/ml/output
SM_NUM_CPUS=4
SM_NUM_GPUS=1
SM_MODEL_DIR=/opt/ml/model
SM_MODULE_DIR=s3://sagemaker-us-east-1-635837196364/huggingface-pytorch-training-2022-05-12-09-01-22-376/source/sourcedir.tar.gz
SM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"test":"/opt/ml/input/data/test","train":"/opt/ml/input/data/train","val":"/opt/ml/input/data/val"},"current_host":"algo-1","framework_module":"sagemaker_pytorch_container.training:main","hosts":["algo-1"],"hyperparameters":{"do_eval":true,"do_train":true,"learning_rate":3e-05,"max_seq_length":128,"model_name_or_path":"uer/chinese_roberta_L-12_H-768","num_train_epochs":1,"output_dir":"/opt/ml/model/chinese_roberta","per_device_train_batch_size":64,"train_file":"/opt/ml/input/data/train/train.csv","validation_file":"/opt/ml/input/data/val/val.csv"},"input_config_dir":"/opt/ml/input/config","input_data_config":{"test":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"},"train":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"},"val":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","is_master":true,"job_name":"huggingface-pytorch-training-2022-05-12-09-01-22-376","log_level":20,"master_hostname":"algo-1","model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-us-east-1-635837196364/huggingface-pytorch-training-2022-05-12-09-01-22-376/source/sourcedir.tar.gz","module_name":"run_glue","network_interface_name":"eth0","num_cpus":4,"num_gpus":1,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_group_name":"homogeneousCluster","current_host":"algo-1","current_instance_type":"ml.g5.xlarge","hosts":["algo-1"],"instance_groups":[{"hosts":["algo-1"],"instance_group_name":"homogeneousCluster","instance_type":"ml.g5.xlarge"}],"network_interface_name":"eth0"},"user_entry_point":"run_glue.py"}
SM_USER_ARGS=["--do_eval","True","--do_train","True","--learning_rate","3e-05","--max_seq_length","128","--model_name_or_path","uer/chinese_roberta_L-12_H-768","--num_train_epochs","1","--output_dir","/opt/ml/model/chinese_roberta","--per_device_train_batch_size","64","--train_file","/opt/ml/input/data/train/train.csv","--validation_file","/opt/ml/input/data/val/val.csv"]
SM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate
SM_CHANNEL_TEST=/opt/ml/input/data/test
SM_CHANNEL_TRAIN=/opt/ml/input/data/train
SM_CHANNEL_VAL=/opt/ml/input/data/val
SM_HP_DO_EVAL=true
SM_HP_DO_TRAIN=true
SM_HP_LEARNING_RATE=3e-05
SM_HP_MAX_SEQ_LENGTH=128
SM_HP_MODEL_NAME_OR_PATH=uer/chinese_roberta_L-12_H-768
SM_HP_NUM_TRAIN_EPOCHS=1
SM_HP_OUTPUT_DIR=/opt/ml/model/chinese_roberta
SM_HP_PER_DEVICE_TRAIN_BATCH_SIZE=64
SM_HP_TRAIN_FILE=/opt/ml/input/data/train/train.csv
SM_HP_VALIDATION_FILE=/opt/ml/input/data/val/val.csv
PYTHONPATH=/opt/ml/code:/opt/conda/bin:/opt/conda/lib/python38.zip:/opt/conda/lib/python3.8:/opt/conda/lib/python3.8/lib-dynload:/opt/conda/lib/python3.8/site-packages:/opt/conda/lib/python3.8/site-packages/smdebug-1.0.13b20220304-py3.8.egg:/opt/conda/lib/python3.8/site-packages/pyinstrument-3.4.2-py3.8.egg:/opt/conda/lib/python3.8/site-packages/pyinstrument_cext-0.2.4-py3.8-linux-x86_64.egg:/opt/conda/lib/python3.8/site-packages/urllib3-1.26.8-py3.8.egg
Invoking script with the following command:
/opt/conda/bin/python3.8 run_glue.py --do_eval True --do_train True --learning_rate 3e-05 --max_seq_length 128 --model_name_or_path uer/chinese_roberta_L-12_H-768 --num_train_epochs 1 --output_dir /opt/ml/model/chinese_roberta --per_device_train_batch_size 64 --train_file /opt/ml/input/data/train/train.csv --validation_file /opt/ml/input/data/val/val.csv
05/12/2022 09:08:04 - WARNING - __main__ - Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: False
05/12/2022 09:08:04 - INFO - __main__ - Training/evaluation parameters TrainingArguments(
_n_gpu=1,
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
bf16=False,
bf16_full_eval=False,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_pin_memory=True,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
debug=[],
deepspeed=None,
disable_tqdm=False,
do_eval=True,
do_predict=False,
do_train=True,
eval_accumulation_steps=None,
eval_steps=None,
evaluation_strategy=IntervalStrategy.NO,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_model_id=None,
hub_strategy=HubStrategy.EVERY_SAVE,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=3e-05,
length_column_name=length,
load_best_model_at_end=False,
local_rank=-1,
log_level=-1,
log_level_replica=-1,
log_on_each_node=True,
logging_dir=/opt/ml/model/chinese_roberta/runs/May12_09-08-04_algo-1,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=500,
logging_strategy=IntervalStrategy.STEPS,
lr_scheduler_type=SchedulerType.LINEAR,
max_grad_norm=1.0,
max_steps=-1,
metric_for_best_model=None,
mp_parameters=,
no_cuda=False,
num_train_epochs=1.0,
optim=OptimizerNames.ADAMW_HF,
output_dir=/opt/ml/model/chinese_roberta,
overwrite_output_dir=False,
past_index=-1,
per_device_eval_batch_size=8,
per_device_train_batch_size=64,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
remove_unused_columns=True,
report_to=[],
resume_from_checkpoint=None,
run_name=/opt/ml/model/chinese_roberta,
save_on_each_node=False,
save_steps=500,
save_strategy=IntervalStrategy.STEPS,
save_total_limit=None,
seed=42,
sharded_ddp=[],
skip_memory_metrics=True,
tf32=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_legacy_prediction_loop=False,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.0,
xpu_backend=None,
)
05/12/2022 09:08:04 - INFO - __main__ - load a local file for train: /opt/ml/input/data/train/train.csv
05/12/2022 09:08:04 - INFO - __main__ - load a local file for validation: /opt/ml/input/data/val/val.csv
05/12/2022 09:08:04 - WARNING - datasets.builder - Using custom data configuration default-96e977e0854d5d36
05/12/2022 09:08:04 - INFO - datasets.builder - Generating dataset csv (/root/.cache/huggingface/datasets/csv/default-96e977e0854d5d36/0.0.0/433e0ccc46f9880962cc2b12065189766fbb2bee57a221866138fb9203c83519)
Downloading and preparing dataset csv/default to /root/.cache/huggingface/datasets/csv/default-96e977e0854d5d36/0.0.0/433e0ccc46f9880962cc2b12065189766fbb2bee57a221866138fb9203c83519...
0%|          | 0/2 [00:00<?, ?it/s]
100%|██████████| 2/2 [00:00<00:00, 8621.39it/s]
05/12/2022 09:08:04 - INFO - datasets.utils.download_manager - Downloading took 0.0 min
05/12/2022 09:08:04 - INFO - datasets.utils.download_manager - Checksum Computation took 0.0 min
0%|          | 0/2 [00:00<?, ?it/s]
100%|██████████| 2/2 [00:00<00:00, 2105.05it/s]
05/12/2022 09:08:04 - INFO - datasets.utils.info_utils - Unable to verify checksums.
05/12/2022 09:08:04 - INFO - datasets.builder - Generating split train
05/12/2022 09:08:04 - INFO - datasets.builder - Generating split validation
05/12/2022 09:08:04 - INFO - datasets.utils.info_utils - Unable to verify splits sizes.
Dataset csv downloaded and prepared to /root/.cache/huggingface/datasets/csv/default-96e977e0854d5d36/0.0.0/433e0ccc46f9880962cc2b12065189766fbb2bee57a221866138fb9203c83519. Subsequent calls will reuse this data.
0%|          | 0/2 [00:00<?, ?it/s]
100%|██████████| 2/2 [00:00<00:00, 416.70it/s]
[INFO|file_utils.py:2215] 2022-05-12 09:08:04,700 >> https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmph1qsm3sb
[INFO|file_utils.py:2215] 2022-05-12 09:08:04,700 >> https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmph1qsm3sb
Downloading:   0%|          | 0.00/468 [00:00<?, ?B/s]
Downloading: 100%|██████████| 468/468 [00:00<00:00, 512kB/s]
[INFO|file_utils.py:2219] 2022-05-12 09:08:04,731 >> storing https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json in cache at /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|file_utils.py:2219] 2022-05-12 09:08:04,731 >> storing https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json in cache at /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|file_utils.py:2227] 2022-05-12 09:08:04,731 >> creating metadata file for /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|file_utils.py:2227] 2022-05-12 09:08:04,731 >> creating metadata file for /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|configuration_utils.py:648] 2022-05-12 09:08:04,731 >> loading configuration file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|configuration_utils.py:648] 2022-05-12 09:08:04,731 >> loading configuration file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|configuration_utils.py:684] 2022-05-12 09:08:04,732 >> Model config BertConfig {
  "_name_or_path": "uer/chinese_roberta_L-12_H-768",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2",
    "3": "LABEL_3",
    "4": "LABEL_4",
    "5": "LABEL_5"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2,
    "LABEL_3": 3,
    "LABEL_4": 4,
    "LABEL_5": 5
  },
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
[INFO|configuration_utils.py:684] 2022-05-12 09:08:04,732 >> Model config BertConfig {
  "_name_or_path": "uer/chinese_roberta_L-12_H-768",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2",
    "3": "LABEL_3",
    "4": "LABEL_4",
    "5": "LABEL_5"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2,
    "LABEL_3": 3,
    "LABEL_4": 4,
    "LABEL_5": 5
  },
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
[INFO|file_utils.py:2215] 2022-05-12 09:08:04,763 >> https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/tokenizer_config.json not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmphosvqw46
[INFO|file_utils.py:2215] 2022-05-12 09:08:04,763 >> https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/tokenizer_config.json not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmphosvqw46
Downloading:   0%|          | 0.00/264 [00:00<?, ?B/s]
Downloading: 100%|██████████| 264/264 [00:00<00:00, 329kB/s]
[INFO|file_utils.py:2219] 2022-05-12 09:08:04,792 >> storing https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/tokenizer_config.json in cache at /root/.cache/huggingface/transformers/a925345dc1119384750fc4469cae2142ea2531a29053865661d50b08e751ffd4.687752b9857d35e5d69095e1ce9e005030a5996c0fd67687830bb3827270c17e
[INFO|file_utils.py:2219] 2022-05-12 09:08:04,792 >> storing https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/tokenizer_config.json in cache at /root/.cache/huggingface/transformers/a925345dc1119384750fc4469cae2142ea2531a29053865661d50b08e751ffd4.687752b9857d35e5d69095e1ce9e005030a5996c0fd67687830bb3827270c17e
[INFO|file_utils.py:2227] 2022-05-12 09:08:04,792 >> creating metadata file for /root/.cache/huggingface/transformers/a925345dc1119384750fc4469cae2142ea2531a29053865661d50b08e751ffd4.687752b9857d35e5d69095e1ce9e005030a5996c0fd67687830bb3827270c17e
[INFO|file_utils.py:2227] 2022-05-12 09:08:04,792 >> creating metadata file for /root/.cache/huggingface/transformers/a925345dc1119384750fc4469cae2142ea2531a29053865661d50b08e751ffd4.687752b9857d35e5d69095e1ce9e005030a5996c0fd67687830bb3827270c17e
[INFO|configuration_utils.py:648] 2022-05-12 09:08:04,822 >> loading configuration file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|configuration_utils.py:648] 2022-05-12 09:08:04,822 >> loading configuration file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|configuration_utils.py:684] 2022-05-12 09:08:04,823 >> Model config BertConfig {
  "_name_or_path": "uer/chinese_roberta_L-12_H-768",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
[INFO|configuration_utils.py:684] 2022-05-12 09:08:04,823 >> Model config BertConfig {
  "_name_or_path": "uer/chinese_roberta_L-12_H-768",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
[INFO|file_utils.py:2215] 2022-05-12 09:08:04,911 >> https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/vocab.txt not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmpq7xb0l46
[INFO|file_utils.py:2215] 2022-05-12 09:08:04,911 >> https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/vocab.txt not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmpq7xb0l46
Downloading:   0%|          | 0.00/107k [00:00<?, ?B/s]
Downloading: 100%|██████████| 107k/107k [00:00<00:00, 47.7MB/s]
[INFO|file_utils.py:2219] 2022-05-12 09:08:04,945 >> storing https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/vocab.txt in cache at /root/.cache/huggingface/transformers/0be013bc55c3a2fd457b91b27bddb810fc250cdd82faf26e3e1a1ccee896a7e3.accd894ff58c6ff7bd4f3072890776c14f4ea34fcc08e79cd88c2d157756dceb
[INFO|file_utils.py:2219] 2022-05-12 09:08:04,945 >> storing https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/vocab.txt in cache at /root/.cache/huggingface/transformers/0be013bc55c3a2fd457b91b27bddb810fc250cdd82faf26e3e1a1ccee896a7e3.accd894ff58c6ff7bd4f3072890776c14f4ea34fcc08e79cd88c2d157756dceb
[INFO|file_utils.py:2227] 2022-05-12 09:08:04,945 >> creating metadata file for /root/.cache/huggingface/transformers/0be013bc55c3a2fd457b91b27bddb810fc250cdd82faf26e3e1a1ccee896a7e3.accd894ff58c6ff7bd4f3072890776c14f4ea34fcc08e79cd88c2d157756dceb
[INFO|file_utils.py:2227] 2022-05-12 09:08:04,945 >> creating metadata file for /root/.cache/huggingface/transformers/0be013bc55c3a2fd457b91b27bddb810fc250cdd82faf26e3e1a1ccee896a7e3.accd894ff58c6ff7bd4f3072890776c14f4ea34fcc08e79cd88c2d157756dceb
[INFO|file_utils.py:2215] 2022-05-12 09:08:05,035 >> https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/special_tokens_map.json not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmpdibv51za
[INFO|file_utils.py:2215] 2022-05-12 09:08:05,035 >> https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/special_tokens_map.json not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmpdibv51za
Downloading:   0%|          | 0.00/112 [00:00<?, ?B/s]
Downloading: 100%|██████████| 112/112 [00:00<00:00, 124kB/s]
[INFO|file_utils.py:2219] 2022-05-12 09:08:05,064 >> storing https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/special_tokens_map.json in cache at /root/.cache/huggingface/transformers/79f65e354d46fd5886217dd7fe8c70329d047fd2b61e8262f8c191647c1095fd.dd8bd9bfd3664b530ea4e645105f557769387b3da9f79bdb55ed556bdd80611d
[INFO|file_utils.py:2219] 2022-05-12 09:08:05,064 >> storing https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/special_tokens_map.json in cache at /root/.cache/huggingface/transformers/79f65e354d46fd5886217dd7fe8c70329d047fd2b61e8262f8c191647c1095fd.dd8bd9bfd3664b530ea4e645105f557769387b3da9f79bdb55ed556bdd80611d
[INFO|file_utils.py:2227] 2022-05-12 09:08:05,064 >> creating metadata file for /root/.cache/huggingface/transformers/79f65e354d46fd5886217dd7fe8c70329d047fd2b61e8262f8c191647c1095fd.dd8bd9bfd3664b530ea4e645105f557769387b3da9f79bdb55ed556bdd80611d
[INFO|file_utils.py:2227] 2022-05-12 09:08:05,064 >> creating metadata file for /root/.cache/huggingface/transformers/79f65e354d46fd5886217dd7fe8c70329d047fd2b61e8262f8c191647c1095fd.dd8bd9bfd3664b530ea4e645105f557769387b3da9f79bdb55ed556bdd80611d
[INFO|tokenization_utils_base.py:1786] 2022-05-12 09:08:05,095 >> loading file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/vocab.txt from cache at /root/.cache/huggingface/transformers/0be013bc55c3a2fd457b91b27bddb810fc250cdd82faf26e3e1a1ccee896a7e3.accd894ff58c6ff7bd4f3072890776c14f4ea34fcc08e79cd88c2d157756dceb
[INFO|tokenization_utils_base.py:1786] 2022-05-12 09:08:05,096 >> loading file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/tokenizer.json from cache at None
[INFO|tokenization_utils_base.py:1786] 2022-05-12 09:08:05,096 >> loading file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:1786] 2022-05-12 09:08:05,096 >> loading file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/special_tokens_map.json from cache at /root/.cache/huggingface/transformers/79f65e354d46fd5886217dd7fe8c70329d047fd2b61e8262f8c191647c1095fd.dd8bd9bfd3664b530ea4e645105f557769387b3da9f79bdb55ed556bdd80611d
[INFO|tokenization_utils_base.py:1786] 2022-05-12 09:08:05,095 >> loading file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/vocab.txt from cache at /root/.cache/huggingface/transformers/0be013bc55c3a2fd457b91b27bddb810fc250cdd82faf26e3e1a1ccee896a7e3.accd894ff58c6ff7bd4f3072890776c14f4ea34fcc08e79cd88c2d157756dceb
[INFO|tokenization_utils_base.py:1786] 2022-05-12 09:08:05,096 >> loading file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/tokenizer.json from cache at None
[INFO|tokenization_utils_base.py:1786] 2022-05-12 09:08:05,096 >> loading file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:1786] 2022-05-12 09:08:05,096 >> loading file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/special_tokens_map.json from cache at /root/.cache/huggingface/transformers/79f65e354d46fd5886217dd7fe8c70329d047fd2b61e8262f8c191647c1095fd.dd8bd9bfd3664b530ea4e645105f557769387b3da9f79bdb55ed556bdd80611d
[INFO|tokenization_utils_base.py:1786] 2022-05-12 09:08:05,096 >> loading file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/tokenizer_config.json from cache at /root/.cache/huggingface/transformers/a925345dc1119384750fc4469cae2142ea2531a29053865661d50b08e751ffd4.687752b9857d35e5d69095e1ce9e005030a5996c0fd67687830bb3827270c17e
[INFO|tokenization_utils_base.py:1786] 2022-05-12 09:08:05,096 >> loading file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/tokenizer_config.json from cache at /root/.cache/huggingface/transformers/a925345dc1119384750fc4469cae2142ea2531a29053865661d50b08e751ffd4.687752b9857d35e5d69095e1ce9e005030a5996c0fd67687830bb3827270c17e
[INFO|configuration_utils.py:648] 2022-05-12 09:08:05,127 >> loading configuration file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|configuration_utils.py:648] 2022-05-12 09:08:05,127 >> loading configuration file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|configuration_utils.py:684] 2022-05-12 09:08:05,128 >> Model config BertConfig {
  "_name_or_path": "uer/chinese_roberta_L-12_H-768",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
[INFO|configuration_utils.py:684] 2022-05-12 09:08:05,128 >> Model config BertConfig {
  "_name_or_path": "uer/chinese_roberta_L-12_H-768",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
[INFO|configuration_utils.py:648] 2022-05-12 09:08:05,169 >> loading configuration file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|configuration_utils.py:648] 2022-05-12 09:08:05,169 >> loading configuration file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/0581d850743cad42501488567cceb8dc9ce50d9f05ad632d273b4389b2f52f68.042085124aedc502028136283b7bf9a169a238009bd6c309f049b249216061a2
[INFO|configuration_utils.py:684] 2022-05-12 09:08:05,170 >> Model config BertConfig {
  "_name_or_path": "uer/chinese_roberta_L-12_H-768",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
[INFO|configuration_utils.py:684] 2022-05-12 09:08:05,170 >> Model config BertConfig {
  "_name_or_path": "uer/chinese_roberta_L-12_H-768",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.17.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
[INFO|file_utils.py:2215] 2022-05-12 09:08:05,265 >> https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/pytorch_model.bin not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmpadm68cnf
[INFO|file_utils.py:2215] 2022-05-12 09:08:05,265 >> https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/pytorch_model.bin not found in cache or force_download set to True, downloading to /root/.cache/huggingface/transformers/tmpadm68cnf
Downloading:   0%|          | 0.00/390M [00:00<?, ?B/s]
Downloading:   2%|▏         | 7.22M/390M [00:00<00:05, 75.7MB/s]
Downloading:   4%|▎         | 14.4M/390M [00:00<00:05, 71.3MB/s]
Downloading:   5%|▌         | 21.4M/390M [00:00<00:05, 72.1MB/s]
Downloading:   7%|▋         | 28.3M/390M [00:00<00:05, 72.0MB/s]
Downloading:   9%|▉         | 35.2M/390M [00:00<00:05, 72.2MB/s]
Downloading:  11%|█         | 42.2M/390M [00:00<00:05, 72.6MB/s]
Downloading:  13%|█▎        | 49.1M/390M [00:00<00:05, 70.9MB/s]
Downloading:  14%|█▍        | 56.6M/390M [00:00<00:04, 73.1MB/s]
Downloading:  16%|█▋        | 63.5M/390M [00:01<00:06, 51.7MB/s]
Downloading:  18%|█▊        | 71.5M/390M [00:01<00:05, 59.3MB/s]
Downloading:  20%|██        | 79.6M/390M [00:01<00:04, 65.8MB/s]
Downloading:  22%|██▏       | 87.7M/390M [00:01<00:04, 70.7MB/s]
Downloading:  25%|██▍       | 96.2M/390M [00:01<00:04, 75.8MB/s]
Downloading:  27%|██▋       | 104M/390M [00:01<00:03, 78.5MB/s]
Downloading:  29%|██▊       | 112M/390M [00:01<00:03, 75.4MB/s]
Downloading:  31%|███       | 120M/390M [00:01<00:03, 76.3MB/s]
Downloading:  33%|███▎      | 128M/390M [00:01<00:03, 78.6MB/s]
Downloading:  35%|███▍      | 135M/390M [00:01<00:03, 79.6MB/s]
Downloading:  37%|███▋      | 143M/390M [00:02<00:03, 78.9MB/s]
Downloading:  39%|███▊      | 151M/390M [00:02<00:03, 79.7MB/s]
Downloading:  41%|████      | 159M/390M [00:02<00:03, 78.6MB/s]
Downloading:  43%|████▎     | 166M/390M [00:02<00:03, 78.2MB/s]
Downloading:  44%|████▍     | 174M/390M [00:02<00:02, 78.1MB/s]
Downloading:  47%|████▋     | 182M/390M [00:02<00:02, 80.0MB/s]
Downloading:  49%|████▊     | 189M/390M [00:02<00:02, 80.1MB/s]
Downloading:  50%|█████     | 197M/390M [00:02<00:02, 79.2MB/s]
Downloading:  53%|█████▎    | 206M/390M [00:02<00:02, 84.1MB/s]
Downloading:  55%|█████▍    | 214M/390M [00:02<00:02, 85.0MB/s]
Downloading:  57%|█████▋    | 223M/390M [00:03<00:02, 86.5MB/s]
Downloading:  59%|█████▉    | 231M/390M [00:03<00:01, 87.1MB/s]
Downloading:  61%|██████▏   | 240M/390M [00:03<00:01, 84.6MB/s]
Downloading:  64%|██████▎   | 248M/390M [00:03<00:01, 85.3MB/s]
Downloading:  66%|██████▌   | 256M/390M [00:03<00:01, 85.6MB/s]
Downloading:  68%|██████▊   | 265M/390M [00:03<00:01, 87.8MB/s]
Downloading:  70%|███████   | 274M/390M [00:03<00:01, 87.2MB/s]
Downloading:  72%|███████▏  | 282M/390M [00:03<00:01, 85.1MB/s]
Downloading:  74%|███████▍  | 290M/390M [00:03<00:01, 85.5MB/s]
Downloading:  76%|███████▋  | 298M/390M [00:04<00:01, 84.8MB/s]
Downloading:  79%|███████▊  | 307M/390M [00:04<00:01, 85.6MB/s]
Downloading:  81%|████████  | 315M/390M [00:04<00:00, 86.6MB/s]
Downloading:  83%|████████▎ | 324M/390M [00:04<00:00, 89.0MB/s]
Downloading:  85%|████████▌ | 333M/390M [00:04<00:00, 89.4MB/s]
Downloading:  87%|████████▋ | 341M/390M [00:04<00:00, 88.2MB/s]
Downloading:  90%|████████▉ | 350M/390M [00:04<00:00, 88.6MB/s]
Downloading:  92%|█████████▏| 358M/390M [00:04<00:00, 87.3MB/s]
Downloading:  94%|█████████▍| 367M/390M [00:04<00:00, 85.7MB/s]
Downloading:  96%|█████████▌| 375M/390M [00:04<00:00, 86.6MB/s]
Downloading:  98%|█████████▊| 383M/390M [00:05<00:00, 86.4MB/s]
Downloading: 100%|██████████| 390M/390M [00:05<00:00, 80.0MB/s]
[INFO|file_utils.py:2219] 2022-05-12 09:08:10,398 >> storing https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/pytorch_model.bin in cache at /root/.cache/huggingface/transformers/8150d6a23b0f518134caba5b6141414848a8adbe069f59c18ddbdd9b7498045c.a61653c00cec4d0c4b7281005a15c05b7277d845fb82d1967138f8296cc7622b
[INFO|file_utils.py:2219] 2022-05-12 09:08:10,398 >> storing https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/pytorch_model.bin in cache at /root/.cache/huggingface/transformers/8150d6a23b0f518134caba5b6141414848a8adbe069f59c18ddbdd9b7498045c.a61653c00cec4d0c4b7281005a15c05b7277d845fb82d1967138f8296cc7622b
[INFO|file_utils.py:2227] 2022-05-12 09:08:10,398 >> creating metadata file for /root/.cache/huggingface/transformers/8150d6a23b0f518134caba5b6141414848a8adbe069f59c18ddbdd9b7498045c.a61653c00cec4d0c4b7281005a15c05b7277d845fb82d1967138f8296cc7622b
[INFO|file_utils.py:2227] 2022-05-12 09:08:10,398 >> creating metadata file for /root/.cache/huggingface/transformers/8150d6a23b0f518134caba5b6141414848a8adbe069f59c18ddbdd9b7498045c.a61653c00cec4d0c4b7281005a15c05b7277d845fb82d1967138f8296cc7622b
[INFO|modeling_utils.py:1431] 2022-05-12 09:08:10,399 >> loading weights file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/pytorch_model.bin from cache at /root/.cache/huggingface/transformers/8150d6a23b0f518134caba5b6141414848a8adbe069f59c18ddbdd9b7498045c.a61653c00cec4d0c4b7281005a15c05b7277d845fb82d1967138f8296cc7622b
[INFO|modeling_utils.py:1431] 2022-05-12 09:08:10,399 >> loading weights file https://huggingface.co/uer/chinese_roberta_L-12_H-768/resolve/main/pytorch_model.bin from cache at /root/.cache/huggingface/transformers/8150d6a23b0f518134caba5b6141414848a8adbe069f59c18ddbdd9b7498045c.a61653c00cec4d0c4b7281005a15c05b7277d845fb82d1967138f8296cc7622b
[WARNING|modeling_utils.py:1693] 2022-05-12 09:08:11,424 >> Some weights of the model checkpoint at uer/chinese_roberta_L-12_H-768 were not used when initializing BertForSequenceClassification: ['cls.predictions.transform.dense.bias', 'cls.predictions.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
[WARNING|modeling_utils.py:1693] 2022-05-12 09:08:11,424 >> Some weights of the model checkpoint at uer/chinese_roberta_L-12_H-768 were not used when initializing BertForSequenceClassification: ['cls.predictions.transform.dense.bias', 'cls.predictions.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
[WARNING|modeling_utils.py:1704] 2022-05-12 09:08:11,425 >> Some weights of BertForSequenceClassification were not initialized from the model checkpoint at uer/chinese_roberta_L-12_H-768 and are newly initialized: ['classifier.weight', 'bert.pooler.dense.weight', 'classifier.bias', 'bert.pooler.dense.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[WARNING|modeling_utils.py:1704] 2022-05-12 09:08:11,425 >> Some weights of BertForSequenceClassification were not initialized from the model checkpoint at uer/chinese_roberta_L-12_H-768 and are newly initialized: ['classifier.weight', 'bert.pooler.dense.weight', 'classifier.bias', 'bert.pooler.dense.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Running tokenizer on dataset:   0%|          | 0/108 [00:00<?, ?ba/s]
05/12/2022 09:08:11 - INFO - datasets.arrow_dataset - Caching processed dataset at /root/.cache/huggingface/datasets/csv/default-96e977e0854d5d36/0.0.0/433e0ccc46f9880962cc2b12065189766fbb2bee57a221866138fb9203c83519/cache-cff9ff03d74a7beb.arrow
Running tokenizer on dataset:   1%|          | 1/108 [00:00<00:49,  2.16ba/s]
Running tokenizer on dataset:   3%|▎         | 3/108 [00:00<00:16,  6.29ba/s]
Running tokenizer on dataset:   5%|▍         | 5/108 [00:00<00:13,  7.79ba/s]
Running tokenizer on dataset:   6%|▋         | 7/108 [00:00<00:09, 10.40ba/s]
Running tokenizer on dataset:   8%|▊         | 9/108 [00:00<00:08, 12.10ba/s]
Running tokenizer on dataset:  10%|█         | 11/108 [00:01<00:07, 13.73ba/s]
Running tokenizer on dataset:  12%|█▏        | 13/108 [00:01<00:06, 14.59ba/s]
Running tokenizer on dataset:  14%|█▍        | 15/108 [00:01<00:05, 15.73ba/s]
Running tokenizer on dataset:  16%|█▌        | 17/108 [00:01<00:06, 14.18ba/s]
Running tokenizer on dataset:  18%|█▊        | 19/108 [00:01<00:06, 14.74ba/s]
Running tokenizer on dataset:  19%|█▉        | 21/108 [00:01<00:05, 15.30ba/s]
Running tokenizer on dataset:  21%|██▏       | 23/108 [00:01<00:05, 16.00ba/s]
Running tokenizer on dataset:  23%|██▎       | 25/108 [00:01<00:04, 16.96ba/s]
Running tokenizer on dataset:  25%|██▌       | 27/108 [00:02<00:04, 17.51ba/s]
Running tokenizer on dataset:  27%|██▋       | 29/108 [00:02<00:05, 15.23ba/s]
Running tokenizer on dataset:  29%|██▊       | 31/108 [00:02<00:04, 15.95ba/s]
Running tokenizer on dataset:  31%|███       | 33/108 [00:02<00:04, 16.79ba/s]
Running tokenizer on dataset:  32%|███▏      | 35/108 [00:02<00:04, 17.49ba/s]
Running tokenizer on dataset:  34%|███▍      | 37/108 [00:02<00:03, 17.82ba/s]
Running tokenizer on dataset:  36%|███▌      | 39/108 [00:02<00:03, 18.40ba/s]
Running tokenizer on dataset:  38%|███▊      | 41/108 [00:02<00:04, 15.69ba/s]
Running tokenizer on dataset:  40%|███▉      | 43/108 [00:03<00:03, 16.62ba/s]
Running tokenizer on dataset:  42%|████▏     | 45/108 [00:03<00:03, 17.12ba/s]
Running tokenizer on dataset:  44%|████▎     | 47/108 [00:03<00:03, 17.39ba/s]
Running tokenizer on dataset:  45%|████▌     | 49/108 [00:03<00:03, 17.77ba/s]
Running tokenizer on dataset:  47%|████▋     | 51/108 [00:03<00:03, 18.11ba/s]
Running tokenizer on dataset:  49%|████▉     | 53/108 [00:03<00:03, 15.17ba/s]
Running tokenizer on dataset:  51%|█████     | 55/108 [00:03<00:03, 15.77ba/s]
Running tokenizer on dataset:  53%|█████▎    | 57/108 [00:03<00:03, 16.43ba/s]
Running tokenizer on dataset:  55%|█████▍    | 59/108 [00:03<00:02, 16.79ba/s]
Running tokenizer on dataset:  56%|█████▋    | 61/108 [00:04<00:02, 16.70ba/s]
Running tokenizer on dataset:  58%|█████▊    | 63/108 [00:04<00:02, 17.23ba/s]
Running tokenizer on dataset:  60%|██████    | 65/108 [00:04<00:02, 15.14ba/s]
Running tokenizer on dataset:  62%|██████▏   | 67/108 [00:04<00:02, 16.03ba/s]
Running tokenizer on dataset:  64%|██████▍   | 69/108 [00:04<00:02, 16.83ba/s]
Running tokenizer on dataset:  66%|██████▌   | 71/108 [00:04<00:02, 17.31ba/s]
Running tokenizer on dataset:  68%|██████▊   | 73/108 [00:04<00:02, 17.25ba/s]
Running tokenizer on dataset:  69%|██████▉   | 75/108 [00:04<00:01, 17.67ba/s]
Running tokenizer on dataset:  71%|███████▏  | 77/108 [00:05<00:02, 15.15ba/s]
Running tokenizer on dataset:  73%|███████▎  | 79/108 [00:05<00:01, 16.11ba/s]
Running tokenizer on dataset:  75%|███████▌  | 81/108 [00:05<00:01, 16.70ba/s]
Running tokenizer on dataset:  77%|███████▋  | 83/108 [00:05<00:01, 16.93ba/s]
Running tokenizer on dataset:  79%|███████▊  | 85/108 [00:05<00:01, 17.21ba/s]
Running tokenizer on dataset:  81%|████████  | 87/108 [00:05<00:01, 17.64ba/s]
Running tokenizer on dataset:  82%|████████▏ | 89/108 [00:05<00:01, 14.89ba/s]
Running tokenizer on dataset:  84%|████████▍ | 91/108 [00:05<00:01, 15.18ba/s]
Running tokenizer on dataset:  86%|████████▌ | 93/108 [00:06<00:00, 15.54ba/s]
Running tokenizer on dataset:  88%|████████▊ | 95/108 [00:06<00:00, 16.04ba/s]
Running tokenizer on dataset:  90%|████████▉ | 97/108 [00:06<00:00, 16.62ba/s]
Running tokenizer on dataset:  92%|█████████▏| 99/108 [00:06<00:00, 17.13ba/s]
Running tokenizer on dataset:  94%|█████████▎| 101/108 [00:06<00:00, 15.09ba/s]
Running tokenizer on dataset:  95%|█████████▌| 103/108 [00:06<00:00, 15.99ba/s]
Running tokenizer on dataset:  97%|█████████▋| 105/108 [00:06<00:00, 16.51ba/s]
Running tokenizer on dataset:  99%|█████████▉| 107/108 [00:06<00:00, 17.08ba/s]
Running tokenizer on dataset: 100%|██████████| 108/108 [00:06<00:00, 15.48ba/s]
Running tokenizer on dataset:   0%|          | 0/12 [00:00<?, ?ba/s]
05/12/2022 09:08:18 - INFO - datasets.arrow_dataset - Caching processed dataset at /root/.cache/huggingface/datasets/csv/default-96e977e0854d5d36/0.0.0/433e0ccc46f9880962cc2b12065189766fbb2bee57a221866138fb9203c83519/cache-2d176539571eefff.arrow
Running tokenizer on dataset:  17%|█▋        | 2/12 [00:00<00:00, 18.38ba/s]
Running tokenizer on dataset:  33%|███▎      | 4/12 [00:00<00:00, 14.06ba/s]
Running tokenizer on dataset:  50%|█████     | 6/12 [00:00<00:00, 15.71ba/s]
Running tokenizer on dataset:  67%|██████▋   | 8/12 [00:00<00:00, 16.73ba/s]
Running tokenizer on dataset:  83%|████████▎ | 10/12 [00:00<00:00, 17.24ba/s]
Running tokenizer on dataset: 100%|██████████| 12/12 [00:00<00:00, 17.53ba/s]
Running tokenizer on dataset: 100%|██████████| 12/12 [00:00<00:00, 16.85ba/s]
05/12/2022 09:08:19 - INFO - __main__ - Sample 83810 of the training set: {'text': '好像油条啊', 'label': 4, 'input_ids': [101, 1962, 1008, 3779, 3340, 1557, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}.
05/12/2022 09:08:19 - INFO - __main__ - Sample 14592 of the training set: {'text': '电三轮', 'label': 3, 'input_ids': [101, 4510, 676, 6762, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}.
05/12/2022 09:08:19 - INFO - __main__ - Sample 3278 of the training set: {'text': '他们在说什么', 'label': 3, 'input_ids': [101, 800, 812, 1762, 6432, 784, 720, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}.
05/12/2022 09:08:19 - INFO - datasets.utils.file_utils - https://raw.githubusercontent.com/huggingface/datasets/1.18.4/metrics/accuracy/accuracy.py not found in cache or force_download set to True, downloading to /root/.cache/huggingface/datasets/downloads/tmp0_66666k
Downloading:   0%|          | 0.00/1.41k [00:00<?, ?B/s]
Downloading: 3.19kB [00:00, 3.05MB/s]
05/12/2022 09:08:19 - INFO - datasets.utils.file_utils - storing https://raw.githubusercontent.com/huggingface/datasets/1.18.4/metrics/accuracy/accuracy.py in cache at /root/.cache/huggingface/datasets/downloads/18ec2a1ed9dbcfd6ecff70a4f0d0d33fd5cc40c51c3c816376dc3d0b3e30219f.6913c0dc30de3cef9d6bc88cc182661800cb937f0fe5b01ffa731617105a32ac.py
05/12/2022 09:08:19 - INFO - datasets.utils.file_utils - creating metadata file for /root/.cache/huggingface/datasets/downloads/18ec2a1ed9dbcfd6ecff70a4f0d0d33fd5cc40c51c3c816376dc3d0b3e30219f.6913c0dc30de3cef9d6bc88cc182661800cb937f0fe5b01ffa731617105a32ac.py
[INFO|trainer.py:570] 2022-05-12 09:08:22,496 >> The following columns in the training set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
[INFO|trainer.py:570] 2022-05-12 09:08:22,496 >> The following columns in the training set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
/opt/conda/lib/python3.8/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
[INFO|trainer.py:1279] 2022-05-12 09:08:22,511 >> ***** Running training *****
[INFO|trainer.py:1280] 2022-05-12 09:08:22,511 >>   Num examples = 107985
[INFO|trainer.py:1281] 2022-05-12 09:08:22,511 >>   Num Epochs = 1
[INFO|trainer.py:1282] 2022-05-12 09:08:22,511 >>   Instantaneous batch size per device = 64
[INFO|trainer.py:1279] 2022-05-12 09:08:22,511 >> ***** Running training *****
[INFO|trainer.py:1280] 2022-05-12 09:08:22,511 >>   Num examples = 107985
[INFO|trainer.py:1281] 2022-05-12 09:08:22,511 >>   Num Epochs = 1
[INFO|trainer.py:1282] 2022-05-12 09:08:22,511 >>   Instantaneous batch size per device = 64
[INFO|trainer.py:1283] 2022-05-12 09:08:22,511 >>   Total train batch size (w. parallel, distributed & accumulation) = 64
[INFO|trainer.py:1284] 2022-05-12 09:08:22,511 >>   Gradient Accumulation steps = 1
[INFO|trainer.py:1285] 2022-05-12 09:08:22,511 >>   Total optimization steps = 1688
[INFO|trainer.py:1283] 2022-05-12 09:08:22,511 >>   Total train batch size (w. parallel, distributed & accumulation) = 64
[INFO|trainer.py:1284] 2022-05-12 09:08:22,511 >>   Gradient Accumulation steps = 1
[INFO|trainer.py:1285] 2022-05-12 09:08:22,511 >>   Total optimization steps = 1688
0%|          | 0/1688 [00:00<?, ?it/s]
[2022-05-12 09:08:22.824 algo-1:27 INFO utils.py:27] RULE_JOB_STOP_SIGNAL_FILENAME: None
/opt/conda/lib/python3.8/site-packages/smdebug-1.0.13b20220304-py3.8.egg/smdebug/profiler/system_metrics_reader.py:63: SyntaxWarning: "is not" with a literal. Did you mean "!="?
/opt/conda/lib/python3.8/site-packages/smdebug-1.0.13b20220304-py3.8.egg/smdebug/profiler/system_metrics_reader.py:63: SyntaxWarning: "is not" with a literal. Did you mean "!="?
[2022-05-12 09:08:23.073 algo-1:27 INFO profiler_config_parser.py:111] User has disabled profiler.
[2022-05-12 09:08:23.075 algo-1:27 INFO json_config.py:91] Creating hook from json_config at /opt/ml/input/config/debughookconfig.json.
[2022-05-12 09:08:23.075 algo-1:27 INFO hook.py:201] tensorboard_dir has not been set for the hook. SMDebug will not be exporting tensorboard summaries.
[2022-05-12 09:08:23.076 algo-1:27 INFO hook.py:254] Saving to /opt/ml/output/tensors
[2022-05-12 09:08:23.076 algo-1:27 INFO state_store.py:77] The checkpoint config file /opt/ml/input/config/checkpointconfig.json does not exist.
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.embeddings.word_embeddings.weight count_params:16226304
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.embeddings.position_embeddings.weight count_params:393216
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.embeddings.token_type_embeddings.weight count_params:1536
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.embeddings.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.embeddings.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.847 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.output.dense.bias count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.0.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.848 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.output.dense.bias count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.1.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.output.dense.bias count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.2.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.849 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.output.dense.bias count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.3.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.850 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.output.dense.bias count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.4.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.851 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.output.dense.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.5.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.output.dense.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.6.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.852 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.output.dense.bias count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.7.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.853 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.output.dense.bias count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.8.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.output.dense.bias count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.9.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.854 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.output.dense.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.10.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.attention.self.query.weight count_params:589824
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.attention.self.query.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.attention.self.key.weight count_params:589824
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.attention.self.key.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.attention.self.value.weight count_params:589824
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.attention.self.value.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.attention.output.dense.weight count_params:589824
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.attention.output.dense.bias count_params:768
[2022-05-12 09:08:23.855 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.attention.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.attention.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.intermediate.dense.weight count_params:2359296
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.intermediate.dense.bias count_params:3072
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.output.dense.weight count_params:2359296
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.output.dense.bias count_params:768
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.output.LayerNorm.weight count_params:768
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:bert.encoder.layer.11.output.LayerNorm.bias count_params:768
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:bert.pooler.dense.weight count_params:589824
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:bert.pooler.dense.bias count_params:768
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:classifier.weight count_params:4608
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:560] name:classifier.bias count_params:6
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:562] Total Trainable Params: 102272262
[2022-05-12 09:08:23.856 algo-1:27 INFO hook.py:421] Monitoring the collections: losses
[2022-05-12 09:08:23.857 algo-1:27 INFO hook.py:485] Hook is writing from the hook with pid: 27
0%|          | 1/1688 [00:03<1:44:11,  3.71s/it]
0%|          | 2/1688 [00:04<48:01,  1.71s/it]
0%|          | 3/1688 [00:04<30:01,  1.07s/it]
0%|          | 4/1688 [00:04<21:33,  1.30it/s]
0%|          | 5/1688 [00:04<16:53,  1.66it/s]
0%|          | 6/1688 [00:05<14:04,  1.99it/s]
0%|          | 7/1688 [00:05<12:16,  2.28it/s]
0%|          | 8/1688 [00:05<11:07,  2.52it/s]
1%|          | 9/1688 [00:06<10:24,  2.69it/s]
1%|          | 10/1688 [00:06<09:55,  2.82it/s]
1%|          | 11/1688 [00:06<09:35,  2.91it/s]
1%|          | 12/1688 [00:07<09:22,  2.98it/s]
1%|          | 13/1688 [00:07<09:12,  3.03it/s]
1%|          | 14/1688 [00:07<09:04,  3.08it/s]
1%|          | 15/1688 [00:08<08:59,  3.10it/s]
1%|          | 16/1688 [00:08<08:50,  3.15it/s]
1%|          | 17/1688 [00:08<08:45,  3.18it/s]
1%|          | 18/1688 [00:09<08:43,  3.19it/s]
1%|          | 19/1688 [00:09<08:39,  3.21it/s]
1%|          | 20/1688 [00:09<08:39,  3.21it/s]
1%|          | 21/1688 [00:09<08:41,  3.20it/s]
1%|▏         | 22/1688 [00:10<08:39,  3.21it/s]
1%|▏         | 23/1688 [00:10<08:37,  3.22it/s]
1%|▏         | 24/1688 [00:10<08:35,  3.23it/s]
1%|▏         | 25/1688 [00:11<08:33,  3.24it/s]
2%|▏         | 26/1688 [00:11<08:33,  3.24it/s]
2%|▏         | 27/1688 [00:11<08:30,  3.25it/s]
2%|▏         | 28/1688 [00:12<08:29,  3.26it/s]
2%|▏         | 29/1688 [00:12<08:30,  3.25it/s]
2%|▏         | 30/1688 [00:12<08:28,  3.26it/s]
2%|▏         | 31/1688 [00:13<08:29,  3.25it/s]
2%|▏         | 32/1688 [00:13<08:28,  3.26it/s]
2%|▏         | 33/1688 [00:13<08:27,  3.26it/s]
2%|▏         | 34/1688 [00:13<08:27,  3.26it/s]
2%|▏         | 35/1688 [00:14<08:27,  3.26it/s]
2%|▏         | 36/1688 [00:14<08:28,  3.25it/s]
2%|▏         | 37/1688 [00:14<08:28,  3.24it/s]
2%|▏         | 38/1688 [00:15<08:30,  3.23it/s]
2%|▏         | 39/1688 [00:15<08:29,  3.24it/s]
2%|▏         | 40/1688 [00:15<08:27,  3.24it/s]
2%|▏         | 41/1688 [00:16<08:26,  3.25it/s]
2%|▏         | 42/1688 [00:16<08:26,  3.25it/s]
3%|▎         | 43/1688 [00:16<08:25,  3.25it/s]
3%|▎         | 44/1688 [00:17<08:24,  3.26it/s]
3%|▎         | 45/1688 [00:17<08:23,  3.27it/s]
3%|▎         | 46/1688 [00:17<08:21,  3.27it/s]
3%|▎         | 47/1688 [00:17<08:20,  3.28it/s]
3%|▎         | 48/1688 [00:18<08:22,  3.26it/s]
3%|▎         | 49/1688 [00:18<08:24,  3.25it/s]
3%|▎         | 50/1688 [00:18<08:23,  3.25it/s]
3%|▎         | 51/1688 [00:19<08:23,  3.25it/s]
3%|▎         | 52/1688 [00:19<08:22,  3.25it/s]
3%|▎         | 53/1688 [00:19<08:20,  3.27it/s]
3%|▎         | 54/1688 [00:20<08:20,  3.27it/s]
3%|▎         | 55/1688 [00:20<08:20,  3.26it/s]
3%|▎         | 56/1688 [00:20<08:23,  3.24it/s]
3%|▎         | 57/1688 [00:21<08:23,  3.24it/s]
3%|▎         | 58/1688 [00:21<08:21,  3.25it/s]
3%|▎         | 59/1688 [00:21<08:20,  3.25it/s]
4%|▎         | 60/1688 [00:21<08:18,  3.26it/s]
4%|▎         | 61/1688 [00:22<08:19,  3.26it/s]
4%|▎         | 62/1688 [00:22<08:19,  3.25it/s]
4%|▎         | 63/1688 [00:22<08:19,  3.26it/s]
4%|▍         | 64/1688 [00:23<08:17,  3.26it/s]
4%|▍         | 65/1688 [00:23<08:16,  3.27it/s]
4%|▍         | 66/1688 [00:23<08:15,  3.27it/s]
4%|▍         | 67/1688 [00:24<08:14,  3.28it/s]
4%|▍         | 68/1688 [00:24<08:15,  3.27it/s]
4%|▍         | 69/1688 [00:24<08:15,  3.27it/s]
4%|▍         | 70/1688 [00:24<08:14,  3.27it/s]
4%|▍         | 71/1688 [00:25<08:14,  3.27it/s]
4%|▍         | 72/1688 [00:25<08:13,  3.27it/s]
4%|▍         | 73/1688 [00:25<08:12,  3.28it/s]
4%|▍         | 74/1688 [00:26<08:13,  3.27it/s]
4%|▍         | 75/1688 [00:26<08:15,  3.25it/s]
5%|▍         | 76/1688 [00:26<08:15,  3.25it/s]
5%|▍         | 77/1688 [00:27<08:16,  3.24it/s]
5%|▍         | 78/1688 [00:27<08:15,  3.25it/s]
5%|▍         | 79/1688 [00:27<08:12,  3.26it/s]
5%|▍         | 80/1688 [00:28<08:12,  3.26it/s]
5%|▍         | 81/1688 [00:28<08:11,  3.27it/s]
5%|▍         | 82/1688 [00:28<08:10,  3.27it/s]
5%|▍         | 83/1688 [00:28<08:09,  3.28it/s]
5%|▍         | 84/1688 [00:29<08:09,  3.28it/s]
5%|▌         | 85/1688 [00:29<08:08,  3.28it/s]
5%|▌         | 86/1688 [00:29<08:08,  3.28it/s]
5%|▌         | 87/1688 [00:30<08:07,  3.28it/s]
5%|▌         | 88/1688 [00:30<08:07,  3.28it/s]
5%|▌         | 89/1688 [00:30<08:09,  3.27it/s]
5%|▌         | 90/1688 [00:31<08:10,  3.26it/s]
5%|▌         | 91/1688 [00:31<08:10,  3.26it/s]
5%|▌         | 92/1688 [00:31<08:10,  3.25it/s]
6%|▌         | 93/1688 [00:32<08:13,  3.24it/s]
6%|▌         | 94/1688 [00:32<08:11,  3.24it/s]
6%|▌         | 95/1688 [00:32<08:09,  3.25it/s]
6%|▌         | 96/1688 [00:32<08:07,  3.27it/s]
6%|▌         | 97/1688 [00:33<08:07,  3.26it/s]
6%|▌         | 98/1688 [00:33<08:08,  3.26it/s]
6%|▌         | 99/1688 [00:33<08:07,  3.26it/s]
6%|▌         | 100/1688 [00:34<08:07,  3.26it/s]
6%|▌         | 101/1688 [00:34<08:08,  3.25it/s]
6%|▌         | 102/1688 [00:34<08:06,  3.26it/s]
6%|▌         | 103/1688 [00:35<08:05,  3.27it/s]
6%|▌         | 104/1688 [00:35<08:03,  3.27it/s]
6%|▌         | 105/1688 [00:35<08:03,  3.28it/s]
6%|▋         | 106/1688 [00:36<08:02,  3.28it/s]
6%|▋         | 107/1688 [00:36<08:01,  3.28it/s]
6%|▋         | 108/1688 [00:36<08:01,  3.28it/s]
6%|▋         | 109/1688 [00:36<08:00,  3.29it/s]
7%|▋         | 110/1688 [00:37<08:00,  3.29it/s]
7%|▋         | 111/1688 [00:37<08:00,  3.28it/s]
7%|▋         | 112/1688 [00:37<08:02,  3.26it/s]
7%|▋         | 113/1688 [00:38<08:03,  3.25it/s]
7%|▋         | 114/1688 [00:38<08:04,  3.25it/s]
7%|▋         | 115/1688 [00:38<08:05,  3.24it/s]
7%|▋         | 116/1688 [00:39<08:07,  3.22it/s]
7%|▋         | 117/1688 [00:39<08:08,  3.21it/s]
7%|▋         | 118/1688 [00:39<08:07,  3.22it/s]
7%|▋         | 119/1688 [00:40<08:04,  3.24it/s]
7%|▋         | 120/1688 [00:40<08:03,  3.25it/s]
7%|▋         | 121/1688 [00:40<08:04,  3.23it/s]
7%|▋         | 122/1688 [00:40<08:04,  3.23it/s]
7%|▋         | 123/1688 [00:41<08:04,  3.23it/s]
7%|▋         | 124/1688 [00:41<08:05,  3.22it/s]
7%|▋         | 125/1688 [00:41<08:05,  3.22it/s]
7%|▋         | 126/1688 [00:42<08:06,  3.21it/s]
8%|▊         | 127/1688 [00:42<08:05,  3.22it/s]
8%|▊         | 128/1688 [00:42<08:02,  3.23it/s]
8%|▊         | 129/1688 [00:43<08:01,  3.24it/s]
8%|▊         | 130/1688 [00:43<07:58,  3.25it/s]
8%|▊         | 131/1688 [00:43<07:57,  3.26it/s]
8%|▊         | 132/1688 [00:44<07:55,  3.27it/s]
8%|▊         | 133/1688 [00:44<07:56,  3.26it/s]
8%|▊         | 134/1688 [00:44<07:58,  3.25it/s]
8%|▊         | 135/1688 [00:44<07:59,  3.24it/s]
8%|▊         | 136/1688 [00:45<07:59,  3.24it/s]
8%|▊         | 137/1688 [00:45<07:56,  3.25it/s]
8%|▊         | 138/1688 [00:45<07:54,  3.26it/s]
8%|▊         | 139/1688 [00:46<07:54,  3.26it/s]
8%|▊         | 140/1688 [00:46<07:53,  3.27it/s]
8%|▊         | 141/1688 [00:46<07:53,  3.27it/s]
8%|▊         | 142/1688 [00:47<07:53,  3.27it/s]
8%|▊         | 143/1688 [00:47<07:52,  3.27it/s]
9%|▊         | 144/1688 [00:47<07:52,  3.27it/s]
9%|▊         | 145/1688 [00:48<07:54,  3.25it/s]
9%|▊         | 146/1688 [00:48<07:52,  3.26it/s]
9%|▊         | 147/1688 [00:48<07:52,  3.26it/s]
9%|▉         | 148/1688 [00:48<07:51,  3.27it/s]
9%|▉         | 149/1688 [00:49<07:50,  3.27it/s]
9%|▉         | 150/1688 [00:49<07:50,  3.27it/s]
9%|▉         | 151/1688 [00:49<07:51,  3.26it/s]
9%|▉         | 152/1688 [00:50<07:52,  3.25it/s]
9%|▉         | 153/1688 [00:50<07:54,  3.24it/s]
9%|▉         | 154/1688 [00:50<07:55,  3.22it/s]
9%|▉         | 155/1688 [00:51<07:53,  3.24it/s]
9%|▉         | 156/1688 [00:51<07:50,  3.25it/s]
9%|▉         | 157/1688 [00:51<07:49,  3.26it/s]
9%|▉         | 158/1688 [00:52<07:51,  3.24it/s]
9%|▉         | 159/1688 [00:52<07:54,  3.23it/s]
9%|▉         | 160/1688 [00:52<07:53,  3.23it/s]
10%|▉         | 161/1688 [00:52<07:50,  3.24it/s]
10%|▉         | 162/1688 [00:53<07:51,  3.24it/s]
10%|▉         | 163/1688 [00:53<07:52,  3.23it/s]
10%|▉         | 164/1688 [00:53<07:51,  3.23it/s]
10%|▉         | 165/1688 [00:54<07:49,  3.24it/s]
10%|▉         | 166/1688 [00:54<07:47,  3.25it/s]
10%|▉         | 167/1688 [00:54<07:45,  3.27it/s]
10%|▉         | 168/1688 [00:55<07:44,  3.27it/s]
10%|█         | 169/1688 [00:55<07:44,  3.27it/s]
10%|█         | 170/1688 [00:55<07:46,  3.26it/s]
10%|█         | 171/1688 [00:56<07:47,  3.25it/s]
10%|█         | 172/1688 [00:56<07:45,  3.26it/s]
10%|█         | 173/1688 [00:56<07:43,  3.27it/s]
10%|█         | 174/1688 [00:56<07:42,  3.28it/s]
10%|█         | 175/1688 [00:57<07:41,  3.28it/s]
10%|█         | 176/1688 [00:57<07:41,  3.28it/s]
10%|█         | 177/1688 [00:57<07:43,  3.26it/s]
11%|█         | 178/1688 [00:58<07:44,  3.25it/s]
11%|█         | 179/1688 [00:58<07:45,  3.24it/s]
11%|█         | 180/1688 [00:58<07:46,  3.23it/s]
11%|█         | 181/1688 [00:59<07:49,  3.21it/s]
11%|█         | 182/1688 [00:59<07:48,  3.22it/s]
11%|█         | 183/1688 [00:59<07:47,  3.22it/s]
11%|█         | 184/1688 [01:00<07:45,  3.23it/s]
11%|█         | 185/1688 [01:00<07:44,  3.24it/s]
11%|█         | 186/1688 [01:00<07:42,  3.24it/s]
11%|█         | 187/1688 [01:00<07:41,  3.26it/s]
11%|█         | 188/1688 [01:01<07:39,  3.26it/s]
11%|█         | 189/1688 [01:01<07:39,  3.26it/s]
11%|█▏        | 190/1688 [01:01<07:40,  3.25it/s]
11%|█▏        | 191/1688 [01:02<07:40,  3.25it/s]
11%|█▏        | 192/1688 [01:02<07:38,  3.26it/s]
11%|█▏        | 193/1688 [01:02<07:39,  3.26it/s]
11%|█▏        | 194/1688 [01:03<07:38,  3.26it/s]
12%|█▏        | 195/1688 [01:03<07:38,  3.26it/s]
12%|█▏        | 196/1688 [01:03<07:39,  3.25it/s]
12%|█▏        | 197/1688 [01:04<07:36,  3.26it/s]
12%|█▏        | 198/1688 [01:04<07:36,  3.27it/s]
12%|█▏        | 199/1688 [01:04<07:35,  3.27it/s]
12%|█▏        | 200/1688 [01:04<07:34,  3.27it/s]
12%|█▏        | 201/1688 [01:05<07:34,  3.28it/s]
12%|█▏        | 202/1688 [01:05<07:37,  3.25it/s]
12%|█▏        | 203/1688 [01:05<07:36,  3.25it/s]
12%|█▏        | 204/1688 [01:06<07:34,  3.26it/s]
12%|█▏        | 205/1688 [01:06<07:32,  3.28it/s]
12%|█▏        | 206/1688 [01:06<07:32,  3.28it/s]
12%|█▏        | 207/1688 [01:07<07:33,  3.27it/s]
12%|█▏        | 208/1688 [01:07<07:32,  3.27it/s]
12%|█▏        | 209/1688 [01:07<07:32,  3.27it/s]
12%|█▏        | 210/1688 [01:07<07:31,  3.27it/s]
12%|█▎        | 211/1688 [01:08<07:34,  3.25it/s]
13%|█▎        | 212/1688 [01:08<07:36,  3.23it/s]
13%|█▎        | 213/1688 [01:08<07:35,  3.24it/s]
13%|█▎        | 214/1688 [01:09<07:34,  3.24it/s]
13%|█▎        | 215/1688 [01:09<07:35,  3.24it/s]
13%|█▎        | 216/1688 [01:09<07:33,  3.25it/s]
13%|█▎        | 217/1688 [01:10<07:31,  3.26it/s]
13%|█▎        | 218/1688 [01:10<07:31,  3.26it/s]
13%|█▎        | 219/1688 [01:10<07:32,  3.25it/s]
13%|█▎        | 220/1688 [01:11<07:31,  3.25it/s]
13%|█▎        | 221/1688 [01:11<07:32,  3.24it/s]
13%|█▎        | 222/1688 [01:11<07:33,  3.23it/s]
13%|█▎        | 223/1688 [01:12<07:34,  3.23it/s]
13%|█▎        | 224/1688 [01:12<07:30,  3.25it/s]
13%|█▎        | 225/1688 [01:12<07:28,  3.26it/s]
13%|█▎        | 226/1688 [01:12<07:27,  3.27it/s]
13%|█▎        | 227/1688 [01:13<07:26,  3.27it/s]
14%|█▎        | 228/1688 [01:13<07:27,  3.26it/s]
14%|█▎        | 229/1688 [01:13<07:27,  3.26it/s]
14%|█▎        | 230/1688 [01:14<07:29,  3.25it/s]
14%|█▎        | 231/1688 [01:14<07:31,  3.23it/s]
14%|█▎        | 232/1688 [01:14<07:31,  3.22it/s]
14%|█▍        | 233/1688 [01:15<07:29,  3.24it/s]
14%|█▍        | 234/1688 [01:15<07:28,  3.24it/s]
14%|█▍        | 235/1688 [01:15<07:27,  3.25it/s]
14%|█▍        | 236/1688 [01:16<07:28,  3.24it/s]
14%|█▍        | 237/1688 [01:16<07:27,  3.24it/s]
14%|█▍        | 238/1688 [01:16<07:25,  3.25it/s]
14%|█▍        | 239/1688 [01:16<07:24,  3.26it/s]
14%|█▍        | 240/1688 [01:17<07:23,  3.27it/s]
14%|█▍        | 241/1688 [01:17<07:22,  3.27it/s]
14%|█▍        | 242/1688 [01:17<07:22,  3.27it/s]
14%|█▍        | 243/1688 [01:18<07:21,  3.27it/s]
14%|█▍        | 244/1688 [01:18<07:22,  3.26it/s]
15%|█▍        | 245/1688 [01:18<07:23,  3.25it/s]
15%|█▍        | 246/1688 [01:19<07:21,  3.26it/s]
15%|█▍        | 247/1688 [01:19<07:20,  3.27it/s]
15%|█▍        | 248/1688 [01:19<07:19,  3.27it/s]
15%|█▍        | 249/1688 [01:19<07:19,  3.28it/s]
15%|█▍        | 250/1688 [01:20<07:18,  3.28it/s]
15%|█▍        | 251/1688 [01:20<07:18,  3.28it/s]
15%|█▍        | 252/1688 [01:20<07:18,  3.28it/s]
15%|█▍        | 253/1688 [01:21<07:19,  3.26it/s]
15%|█▌        | 254/1688 [01:21<07:19,  3.26it/s]
15%|█▌        | 255/1688 [01:21<07:18,  3.27it/s]
15%|█▌        | 256/1688 [01:22<07:17,  3.27it/s]
15%|█▌        | 257/1688 [01:22<07:18,  3.26it/s]
15%|█▌        | 258/1688 [01:22<07:17,  3.27it/s]
15%|█▌        | 259/1688 [01:23<07:16,  3.27it/s]
15%|█▌        | 260/1688 [01:23<07:16,  3.27it/s]
15%|█▌        | 261/1688 [01:23<07:16,  3.27it/s]
16%|█▌        | 262/1688 [01:23<07:17,  3.26it/s]
16%|█▌        | 263/1688 [01:24<07:18,  3.25it/s]
16%|█▌        | 264/1688 [01:24<07:18,  3.24it/s]
16%|█▌        | 265/1688 [01:24<07:17,  3.26it/s]
16%|█▌        | 266/1688 [01:25<07:16,  3.26it/s]
16%|█▌        | 267/1688 [01:25<07:14,  3.27it/s]
16%|█▌        | 268/1688 [01:25<07:13,  3.27it/s]
16%|█▌        | 269/1688 [01:26<07:12,  3.28it/s]
16%|█▌        | 270/1688 [01:26<07:12,  3.28it/s]
16%|█▌        | 271/1688 [01:26<07:11,  3.28it/s]
16%|█▌        | 272/1688 [01:27<07:11,  3.28it/s]
16%|█▌        | 273/1688 [01:27<07:13,  3.26it/s]
16%|█▌        | 274/1688 [01:27<07:11,  3.28it/s]
16%|█▋        | 275/1688 [01:27<07:10,  3.28it/s]
16%|█▋        | 276/1688 [01:28<07:10,  3.28it/s]
16%|█▋        | 277/1688 [01:28<07:11,  3.27it/s]
16%|█▋        | 278/1688 [01:28<07:11,  3.27it/s]
17%|█▋        | 279/1688 [01:29<07:10,  3.27it/s]
17%|█▋        | 280/1688 [01:29<07:09,  3.28it/s]
17%|█▋        | 281/1688 [01:29<07:09,  3.28it/s]
17%|█▋        | 282/1688 [01:30<07:08,  3.28it/s]
17%|█▋        | 283/1688 [01:30<07:07,  3.28it/s]
17%|█▋        | 284/1688 [01:30<07:07,  3.29it/s]
17%|█▋        | 285/1688 [01:30<07:06,  3.29it/s]
17%|█▋        | 286/1688 [01:31<07:09,  3.27it/s]
17%|█▋        | 287/1688 [01:31<07:08,  3.27it/s]
17%|█▋        | 288/1688 [01:31<07:10,  3.25it/s]
17%|█▋        | 289/1688 [01:32<07:08,  3.26it/s]
17%|█▋        | 290/1688 [01:32<07:07,  3.27it/s]
17%|█▋        | 291/1688 [01:32<07:09,  3.26it/s]
17%|█▋        | 292/1688 [01:33<07:11,  3.23it/s]
17%|█▋        | 293/1688 [01:33<07:09,  3.25it/s]
17%|█▋        | 294/1688 [01:33<07:08,  3.25it/s]
17%|█▋        | 295/1688 [01:34<07:10,  3.24it/s]
18%|█▊        | 296/1688 [01:34<07:09,  3.24it/s]
18%|█▊        | 297/1688 [01:34<07:07,  3.25it/s]
18%|█▊        | 298/1688 [01:34<07:06,  3.26it/s]
18%|█▊        | 299/1688 [01:35<07:05,  3.27it/s]
18%|█▊        | 300/1688 [01:35<07:04,  3.27it/s]
18%|█▊        | 301/1688 [01:35<07:03,  3.27it/s]
18%|█▊        | 302/1688 [01:36<07:02,  3.28it/s]
18%|█▊        | 303/1688 [01:36<07:02,  3.28it/s]
18%|█▊        | 304/1688 [01:36<07:01,  3.28it/s]
18%|█▊        | 305/1688 [01:37<07:01,  3.28it/s]
18%|█▊        | 306/1688 [01:37<07:00,  3.28it/s]
18%|█▊        | 307/1688 [01:37<07:00,  3.29it/s]
18%|█▊        | 308/1688 [01:38<07:02,  3.27it/s]
18%|█▊        | 309/1688 [01:38<07:00,  3.28it/s]
18%|█▊        | 310/1688 [01:38<07:00,  3.27it/s]
18%|█▊        | 311/1688 [01:38<07:00,  3.28it/s]
18%|█▊        | 312/1688 [01:39<07:02,  3.26it/s]
19%|█▊        | 313/1688 [01:39<07:03,  3.25it/s]
19%|█▊        | 314/1688 [01:39<07:03,  3.24it/s]
19%|█▊        | 315/1688 [01:40<07:04,  3.23it/s]
19%|█▊        | 316/1688 [01:40<07:05,  3.23it/s]
19%|█▉        | 317/1688 [01:40<07:04,  3.23it/s]
19%|█▉        | 318/1688 [01:41<07:04,  3.23it/s]
19%|█▉        | 319/1688 [01:41<07:03,  3.24it/s]
19%|█▉        | 320/1688 [01:41<07:07,  3.20it/s]
19%|█▉        | 321/1688 [01:42<07:06,  3.20it/s]
19%|█▉        | 322/1688 [01:42<07:06,  3.20it/s]
19%|█▉        | 323/1688 [01:42<07:06,  3.20it/s]
19%|█▉        | 324/1688 [01:42<07:03,  3.22it/s]
19%|█▉        | 325/1688 [01:43<07:02,  3.23it/s]
19%|█▉        | 326/1688 [01:43<06:59,  3.25it/s]
19%|█▉        | 327/1688 [01:43<06:57,  3.26it/s]
19%|█▉        | 328/1688 [01:44<06:59,  3.25it/s]
19%|█▉        | 329/1688 [01:44<06:57,  3.25it/s]
20%|█▉        | 330/1688 [01:44<06:56,  3.26it/s]
20%|█▉        | 331/1688 [01:45<06:57,  3.25it/s]
20%|█▉        | 332/1688 [01:45<06:59,  3.23it/s]
20%|█▉        | 333/1688 [01:45<07:00,  3.22it/s]
20%|█▉        | 334/1688 [01:46<06:59,  3.23it/s]
20%|█▉        | 335/1688 [01:46<06:59,  3.23it/s]
20%|█▉        | 336/1688 [01:46<06:56,  3.25it/s]
20%|█▉        | 337/1688 [01:46<06:54,  3.26it/s]
20%|██        | 338/1688 [01:47<06:52,  3.27it/s]
20%|██        | 339/1688 [01:47<06:53,  3.26it/s]
20%|██        | 340/1688 [01:47<06:52,  3.27it/s]
20%|██        | 341/1688 [01:48<06:51,  3.27it/s]
20%|██        | 342/1688 [01:48<06:51,  3.27it/s]
20%|██        | 343/1688 [01:48<06:50,  3.28it/s]
20%|██        | 344/1688 [01:49<06:52,  3.26it/s]
20%|██        | 345/1688 [01:49<06:55,  3.23it/s]
20%|██        | 346/1688 [01:49<06:54,  3.24it/s]
21%|██        | 347/1688 [01:50<06:52,  3.25it/s]
21%|██        | 348/1688 [01:50<06:52,  3.25it/s]
21%|██        | 349/1688 [01:50<06:52,  3.25it/s]
21%|██        | 350/1688 [01:50<06:51,  3.25it/s]
21%|██        | 351/1688 [01:51<06:50,  3.26it/s]
21%|██        | 352/1688 [01:51<06:48,  3.27it/s]
21%|██        | 353/1688 [01:51<06:48,  3.27it/s]
21%|██        | 354/1688 [01:52<06:47,  3.27it/s]
21%|██        | 355/1688 [01:52<06:47,  3.27it/s]
21%|██        | 356/1688 [01:52<06:46,  3.28it/s]
21%|██        | 357/1688 [01:53<06:46,  3.28it/s]
21%|██        | 358/1688 [01:53<06:47,  3.27it/s]
21%|██▏       | 359/1688 [01:53<06:48,  3.25it/s]
21%|██▏       | 360/1688 [01:54<06:49,  3.24it/s]
21%|██▏       | 361/1688 [01:54<06:49,  3.24it/s]
21%|██▏       | 362/1688 [01:54<06:48,  3.25it/s]
22%|██▏       | 363/1688 [01:54<06:46,  3.26it/s]
22%|██▏       | 364/1688 [01:55<06:47,  3.25it/s]
22%|██▏       | 365/1688 [01:55<06:46,  3.25it/s]
22%|██▏       | 366/1688 [01:55<06:45,  3.26it/s]
22%|██▏       | 367/1688 [01:56<06:46,  3.25it/s]
22%|██▏       | 368/1688 [01:56<06:44,  3.26it/s]
22%|██▏       | 369/1688 [01:56<06:46,  3.25it/s]
22%|██▏       | 370/1688 [01:57<06:47,  3.23it/s]
22%|██▏       | 371/1688 [01:57<06:47,  3.23it/s]
22%|██▏       | 372/1688 [01:57<06:49,  3.22it/s]
22%|██▏       | 373/1688 [01:58<06:49,  3.21it/s]
22%|██▏       | 374/1688 [01:58<06:46,  3.23it/s]
22%|██▏       | 375/1688 [01:58<06:45,  3.24it/s]
22%|██▏       | 376/1688 [01:58<06:45,  3.24it/s]
22%|██▏       | 377/1688 [01:59<06:43,  3.25it/s]
22%|██▏       | 378/1688 [01:59<06:42,  3.25it/s]
22%|██▏       | 379/1688 [01:59<06:40,  3.27it/s]
23%|██▎       | 380/1688 [02:00<06:40,  3.27it/s]
23%|██▎       | 381/1688 [02:00<06:39,  3.27it/s]
23%|██▎       | 382/1688 [02:00<06:38,  3.28it/s]
23%|██▎       | 383/1688 [02:01<06:39,  3.27it/s]
23%|██▎       | 384/1688 [02:01<06:40,  3.25it/s]
23%|██▎       | 385/1688 [02:01<06:42,  3.24it/s]
23%|██▎       | 386/1688 [02:02<06:43,  3.23it/s]
23%|██▎       | 387/1688 [02:02<06:41,  3.24it/s]
23%|██▎       | 388/1688 [02:02<06:39,  3.25it/s]
23%|██▎       | 389/1688 [02:02<06:40,  3.24it/s]
23%|██▎       | 390/1688 [02:03<06:40,  3.24it/s]
23%|██▎       | 391/1688 [02:03<06:38,  3.25it/s]
23%|██▎       | 392/1688 [02:03<06:37,  3.26it/s]
23%|██▎       | 393/1688 [02:04<06:36,  3.27it/s]
23%|██▎       | 394/1688 [02:04<06:38,  3.25it/s]
23%|██▎       | 395/1688 [02:04<06:37,  3.25it/s]
23%|██▎       | 396/1688 [02:05<06:37,  3.25it/s]
24%|██▎       | 397/1688 [02:05<06:36,  3.26it/s]
24%|██▎       | 398/1688 [02:05<06:35,  3.26it/s]
24%|██▎       | 399/1688 [02:06<06:35,  3.26it/s]
24%|██▎       | 400/1688 [02:06<06:36,  3.25it/s]
24%|██▍       | 401/1688 [02:06<06:36,  3.25it/s]
24%|██▍       | 402/1688 [02:06<06:34,  3.26it/s]
24%|██▍       | 403/1688 [02:07<06:34,  3.25it/s]
24%|██▍       | 404/1688 [02:07<06:35,  3.25it/s]
24%|██▍       | 405/1688 [02:07<06:34,  3.25it/s]
24%|██▍       | 406/1688 [02:08<06:33,  3.25it/s]
24%|██▍       | 407/1688 [02:08<06:32,  3.26it/s]
24%|██▍       | 408/1688 [02:08<06:31,  3.27it/s]
24%|██▍       | 409/1688 [02:09<06:30,  3.27it/s]
24%|██▍       | 410/1688 [02:09<06:30,  3.28it/s]
24%|██▍       | 411/1688 [02:09<06:29,  3.28it/s]
24%|██▍       | 412/1688 [02:10<06:28,  3.28it/s]
24%|██▍       | 413/1688 [02:10<06:28,  3.28it/s]
25%|██▍       | 414/1688 [02:10<06:30,  3.26it/s]
25%|██▍       | 415/1688 [02:10<06:30,  3.26it/s]
25%|██▍       | 416/1688 [02:11<06:29,  3.27it/s]
25%|██▍       | 417/1688 [02:11<06:29,  3.26it/s]
25%|██▍       | 418/1688 [02:11<06:28,  3.27it/s]
25%|██▍       | 419/1688 [02:12<06:28,  3.27it/s]
25%|██▍       | 420/1688 [02:12<06:27,  3.27it/s]
25%|██▍       | 421/1688 [02:12<06:28,  3.26it/s]
25%|██▌       | 422/1688 [02:13<06:28,  3.26it/s]
25%|██▌       | 423/1688 [02:13<06:28,  3.26it/s]
25%|██▌       | 424/1688 [02:13<06:26,  3.27it/s]
25%|██▌       | 425/1688 [02:14<06:25,  3.27it/s]
25%|██▌       | 426/1688 [02:14<06:26,  3.27it/s]
25%|██▌       | 427/1688 [02:14<06:26,  3.27it/s]
25%|██▌       | 428/1688 [02:14<06:25,  3.27it/s]
25%|██▌       | 429/1688 [02:15<06:24,  3.27it/s]
25%|██▌       | 430/1688 [02:15<06:25,  3.26it/s]
26%|██▌       | 431/1688 [02:15<06:26,  3.25it/s]
26%|██▌       | 432/1688 [02:16<06:25,  3.26it/s]
26%|██▌       | 433/1688 [02:16<06:25,  3.25it/s]
26%|██▌       | 434/1688 [02:16<06:25,  3.25it/s]
26%|██▌       | 435/1688 [02:17<06:24,  3.26it/s]
26%|██▌       | 436/1688 [02:17<06:23,  3.26it/s]
26%|██▌       | 437/1688 [02:17<06:22,  3.27it/s]
26%|██▌       | 438/1688 [02:17<06:21,  3.27it/s]
26%|██▌       | 439/1688 [02:18<06:20,  3.28it/s]
26%|██▌       | 440/1688 [02:18<06:20,  3.28it/s]
26%|██▌       | 441/1688 [02:18<06:19,  3.28it/s]
26%|██▌       | 442/1688 [02:19<06:21,  3.27it/s]
26%|██▌       | 443/1688 [02:19<06:20,  3.27it/s]
26%|██▋       | 444/1688 [02:19<06:21,  3.26it/s]
26%|██▋       | 445/1688 [02:20<06:23,  3.24it/s]
26%|██▋       | 446/1688 [02:20<06:23,  3.24it/s]
26%|██▋       | 447/1688 [02:20<06:22,  3.24it/s]
27%|██▋       | 448/1688 [02:21<06:20,  3.26it/s]
27%|██▋       | 449/1688 [02:21<06:21,  3.25it/s]
27%|██▋       | 450/1688 [02:21<06:21,  3.25it/s]
27%|██▋       | 451/1688 [02:21<06:19,  3.26it/s]
27%|██▋       | 452/1688 [02:22<06:18,  3.26it/s]
27%|██▋       | 453/1688 [02:22<06:18,  3.26it/s]
27%|██▋       | 454/1688 [02:22<06:19,  3.25it/s]
27%|██▋       | 455/1688 [02:23<06:18,  3.25it/s]
27%|██▋       | 456/1688 [02:23<06:18,  3.26it/s]
27%|██▋       | 457/1688 [02:23<06:19,  3.24it/s]
27%|██▋       | 458/1688 [02:24<06:19,  3.24it/s]
27%|██▋       | 459/1688 [02:24<06:20,  3.23it/s]
27%|██▋       | 460/1688 [02:24<06:18,  3.24it/s]
27%|██▋       | 461/1688 [02:25<06:17,  3.25it/s]
27%|██▋       | 462/1688 [02:25<06:16,  3.26it/s]
27%|██▋       | 463/1688 [02:25<06:15,  3.27it/s]
27%|██▋       | 464/1688 [02:25<06:14,  3.27it/s]
28%|██▊       | 465/1688 [02:26<06:13,  3.27it/s]
28%|██▊       | 466/1688 [02:26<06:13,  3.27it/s]
28%|██▊       | 467/1688 [02:26<06:12,  3.28it/s]
28%|██▊       | 468/1688 [02:27<06:12,  3.28it/s]
28%|██▊       | 469/1688 [02:27<06:14,  3.26it/s]
28%|██▊       | 470/1688 [02:27<06:14,  3.25it/s]
28%|██▊       | 471/1688 [02:28<06:14,  3.25it/s]
28%|██▊       | 472/1688 [02:28<06:12,  3.26it/s]
28%|██▊       | 473/1688 [02:28<06:11,  3.27it/s]
28%|██▊       | 474/1688 [02:29<06:11,  3.27it/s]
28%|██▊       | 475/1688 [02:29<06:11,  3.27it/s]
28%|██▊       | 476/1688 [02:29<06:12,  3.25it/s]
28%|██▊       | 477/1688 [02:29<06:11,  3.26it/s]
28%|██▊       | 478/1688 [02:30<06:11,  3.26it/s]
28%|██▊       | 479/1688 [02:30<06:11,  3.26it/s]
28%|██▊       | 480/1688 [02:30<06:11,  3.25it/s]
28%|██▊       | 481/1688 [02:31<06:11,  3.25it/s]
29%|██▊       | 482/1688 [02:31<06:10,  3.26it/s]
29%|██▊       | 483/1688 [02:31<06:09,  3.27it/s]
29%|██▊       | 484/1688 [02:32<06:07,  3.27it/s]
29%|██▊       | 485/1688 [02:32<06:07,  3.28it/s]
29%|██▉       | 486/1688 [02:32<06:08,  3.26it/s]
29%|██▉       | 487/1688 [02:33<06:09,  3.25it/s]
29%|██▉       | 488/1688 [02:33<06:07,  3.26it/s]
29%|██▉       | 489/1688 [02:33<06:08,  3.26it/s]
29%|██▉       | 490/1688 [02:33<06:09,  3.25it/s]
29%|██▉       | 491/1688 [02:34<06:11,  3.22it/s]
29%|██▉       | 492/1688 [02:34<06:09,  3.24it/s]
29%|██▉       | 493/1688 [02:34<06:09,  3.23it/s]
29%|██▉       | 494/1688 [02:35<06:08,  3.24it/s]
29%|██▉       | 495/1688 [02:35<06:06,  3.25it/s]
29%|██▉       | 496/1688 [02:35<06:05,  3.26it/s]
29%|██▉       | 497/1688 [02:36<06:06,  3.25it/s]
30%|██▉       | 498/1688 [02:36<06:05,  3.26it/s]
30%|██▉       | 499/1688 [02:36<06:05,  3.26it/s]
30%|██▉       | 500/1688 [02:37<06:04,  3.26it/s]
{'loss': 1.4824, 'learning_rate': 2.1113744075829386e-05, 'epoch': 0.3}
30%|██▉       | 500/1688 [02:37<06:04,  3.26it/s]
[INFO|trainer.py:2139] 2022-05-12 09:10:59,541 >> Saving model checkpoint to /opt/ml/model/chinese_roberta/checkpoint-500
[INFO|trainer.py:2139] 2022-05-12 09:10:59,541 >> Saving model checkpoint to /opt/ml/model/chinese_roberta/checkpoint-500
[INFO|configuration_utils.py:439] 2022-05-12 09:10:59,542 >> Configuration saved in /opt/ml/model/chinese_roberta/checkpoint-500/config.json
[INFO|configuration_utils.py:439] 2022-05-12 09:10:59,542 >> Configuration saved in /opt/ml/model/chinese_roberta/checkpoint-500/config.json
[INFO|modeling_utils.py:1084] 2022-05-12 09:11:00,001 >> Model weights saved in /opt/ml/model/chinese_roberta/checkpoint-500/pytorch_model.bin
[INFO|modeling_utils.py:1084] 2022-05-12 09:11:00,001 >> Model weights saved in /opt/ml/model/chinese_roberta/checkpoint-500/pytorch_model.bin
[INFO|tokenization_utils_base.py:2094] 2022-05-12 09:11:00,001 >> tokenizer config file saved in /opt/ml/model/chinese_roberta/checkpoint-500/tokenizer_config.json
[INFO|tokenization_utils_base.py:2094] 2022-05-12 09:11:00,001 >> tokenizer config file saved in /opt/ml/model/chinese_roberta/checkpoint-500/tokenizer_config.json
[INFO|tokenization_utils_base.py:2100] 2022-05-12 09:11:00,001 >> Special tokens file saved in /opt/ml/model/chinese_roberta/checkpoint-500/special_tokens_map.json
[INFO|tokenization_utils_base.py:2100] 2022-05-12 09:11:00,001 >> Special tokens file saved in /opt/ml/model/chinese_roberta/checkpoint-500/special_tokens_map.json
30%|██▉       | 501/1688 [02:38<14:20,  1.38it/s]
30%|██▉       | 502/1688 [02:39<11:52,  1.66it/s]
30%|██▉       | 503/1688 [02:39<10:09,  1.94it/s]
30%|██▉       | 504/1688 [02:39<08:55,  2.21it/s]
30%|██▉       | 505/1688 [02:39<08:06,  2.43it/s]
30%|██▉       | 506/1688 [02:40<07:29,  2.63it/s]
30%|███       | 507/1688 [02:40<07:05,  2.78it/s]
30%|███       | 508/1688 [02:40<06:45,  2.91it/s]
30%|███       | 509/1688 [02:41<06:31,  3.01it/s]
30%|███       | 510/1688 [02:41<06:22,  3.08it/s]
30%|███       | 511/1688 [02:41<06:19,  3.10it/s]
30%|███       | 512/1688 [02:42<06:15,  3.13it/s]
30%|███       | 513/1688 [02:42<06:12,  3.16it/s]
30%|███       | 514/1688 [02:42<06:08,  3.19it/s]
31%|███       | 515/1688 [02:43<06:07,  3.19it/s]
31%|███       | 516/1688 [02:43<06:07,  3.19it/s]
31%|███       | 517/1688 [02:43<06:05,  3.21it/s]
31%|███       | 518/1688 [02:44<06:04,  3.21it/s]
31%|███       | 519/1688 [02:44<06:02,  3.22it/s]
31%|███       | 520/1688 [02:44<06:03,  3.22it/s]
31%|███       | 521/1688 [02:44<06:02,  3.22it/s]
31%|███       | 522/1688 [02:45<06:03,  3.21it/s]
31%|███       | 523/1688 [02:45<06:01,  3.22it/s]
31%|███       | 524/1688 [02:45<06:00,  3.23it/s]
31%|███       | 525/1688 [02:46<05:58,  3.24it/s]
31%|███       | 526/1688 [02:46<05:57,  3.25it/s]
31%|███       | 527/1688 [02:46<05:56,  3.26it/s]
31%|███▏      | 528/1688 [02:47<05:55,  3.26it/s]
31%|███▏      | 529/1688 [02:47<05:54,  3.27it/s]
31%|███▏      | 530/1688 [02:47<05:53,  3.27it/s]
31%|███▏      | 531/1688 [02:47<05:52,  3.28it/s]
32%|███▏      | 532/1688 [02:48<05:52,  3.28it/s]
32%|███▏      | 533/1688 [02:48<05:52,  3.28it/s]
32%|███▏      | 534/1688 [02:48<05:53,  3.26it/s]
32%|███▏      | 535/1688 [02:49<05:52,  3.27it/s]
32%|███▏      | 536/1688 [02:49<05:54,  3.25it/s]
32%|███▏      | 537/1688 [02:49<05:56,  3.23it/s]
32%|███▏      | 538/1688 [02:50<05:54,  3.24it/s]
32%|███▏      | 539/1688 [02:50<05:55,  3.23it/s]
32%|███▏      | 540/1688 [02:50<05:55,  3.23it/s]
32%|███▏      | 541/1688 [02:51<05:53,  3.25it/s]
32%|███▏      | 542/1688 [02:51<05:52,  3.25it/s]
32%|███▏      | 543/1688 [02:51<05:51,  3.26it/s]
32%|███▏      | 544/1688 [02:52<05:52,  3.25it/s]
32%|███▏      | 545/1688 [02:52<05:51,  3.25it/s]
32%|███▏      | 546/1688 [02:52<05:50,  3.26it/s]
32%|███▏      | 547/1688 [02:52<05:50,  3.26it/s]
32%|███▏      | 548/1688 [02:53<05:49,  3.26it/s]
33%|███▎      | 549/1688 [02:53<05:50,  3.25it/s]
33%|███▎      | 550/1688 [02:53<05:50,  3.24it/s]
33%|███▎      | 551/1688 [02:54<05:50,  3.24it/s]
33%|███▎      | 552/1688 [02:54<05:50,  3.24it/s]
33%|███▎      | 553/1688 [02:54<05:49,  3.25it/s]
33%|███▎      | 554/1688 [02:55<05:50,  3.23it/s]
33%|███▎      | 555/1688 [02:55<05:49,  3.24it/s]
33%|███▎      | 556/1688 [02:55<05:47,  3.26it/s]
33%|███▎      | 557/1688 [02:56<05:46,  3.26it/s]
33%|███▎      | 558/1688 [02:56<05:46,  3.26it/s]
33%|███▎      | 559/1688 [02:56<05:45,  3.27it/s]
33%|███▎      | 560/1688 [02:56<05:45,  3.26it/s]
33%|███▎      | 561/1688 [02:57<05:45,  3.26it/s]
33%|███▎      | 562/1688 [02:57<05:44,  3.27it/s]
33%|███▎      | 563/1688 [02:57<05:43,  3.27it/s]
33%|███▎      | 564/1688 [02:58<05:43,  3.27it/s]
33%|███▎      | 565/1688 [02:58<05:43,  3.27it/s]
34%|███▎      | 566/1688 [02:58<05:42,  3.28it/s]
34%|███▎      | 567/1688 [02:59<05:44,  3.26it/s]
34%|███▎      | 568/1688 [02:59<05:44,  3.25it/s]
34%|███▎      | 569/1688 [02:59<05:43,  3.25it/s]
34%|███▍      | 570/1688 [02:59<05:44,  3.24it/s]
34%|███▍      | 571/1688 [03:00<05:44,  3.24it/s]
34%|███▍      | 572/1688 [03:00<05:43,  3.25it/s]
34%|███▍      | 573/1688 [03:00<05:41,  3.26it/s]
34%|███▍      | 574/1688 [03:01<05:41,  3.27it/s]
34%|███▍      | 575/1688 [03:01<05:40,  3.27it/s]
34%|███▍      | 576/1688 [03:01<05:42,  3.25it/s]
34%|███▍      | 577/1688 [03:02<05:40,  3.26it/s]
34%|███▍      | 578/1688 [03:02<05:40,  3.26it/s]
34%|███▍      | 579/1688 [03:02<05:41,  3.24it/s]
34%|███▍      | 580/1688 [03:03<05:40,  3.26it/s]
34%|███▍      | 581/1688 [03:03<05:42,  3.23it/s]
34%|███▍      | 582/1688 [03:03<05:42,  3.23it/s]
35%|███▍      | 583/1688 [03:03<05:41,  3.24it/s]
35%|███▍      | 584/1688 [03:04<05:41,  3.23it/s]
35%|███▍      | 585/1688 [03:04<05:41,  3.23it/s]
35%|███▍      | 586/1688 [03:04<05:40,  3.24it/s]
35%|███▍      | 587/1688 [03:05<05:38,  3.25it/s]
35%|███▍      | 588/1688 [03:05<05:37,  3.26it/s]
35%|███▍      | 589/1688 [03:05<05:36,  3.27it/s]
35%|███▍      | 590/1688 [03:06<05:35,  3.27it/s]
35%|███▌      | 591/1688 [03:06<05:34,  3.28it/s]
35%|███▌      | 592/1688 [03:06<05:34,  3.28it/s]
35%|███▌      | 593/1688 [03:07<05:34,  3.27it/s]
35%|███▌      | 594/1688 [03:07<05:33,  3.28it/s]
35%|███▌      | 595/1688 [03:07<05:33,  3.28it/s]
35%|███▌      | 596/1688 [03:07<05:35,  3.25it/s]
35%|███▌      | 597/1688 [03:08<05:35,  3.25it/s]
35%|███▌      | 598/1688 [03:08<05:34,  3.26it/s]
35%|███▌      | 599/1688 [03:08<05:33,  3.26it/s]
36%|███▌      | 600/1688 [03:09<05:32,  3.27it/s]
36%|███▌      | 601/1688 [03:09<05:31,  3.28it/s]
36%|███▌      | 602/1688 [03:09<05:30,  3.28it/s]
36%|███▌      | 603/1688 [03:10<05:30,  3.28it/s]
36%|███▌      | 604/1688 [03:10<05:30,  3.28it/s]
36%|███▌      | 605/1688 [03:10<05:29,  3.28it/s]
36%|███▌      | 606/1688 [03:11<05:29,  3.28it/s]
36%|███▌      | 607/1688 [03:11<05:30,  3.27it/s]
36%|███▌      | 608/1688 [03:11<05:31,  3.26it/s]
36%|███▌      | 609/1688 [03:11<05:32,  3.25it/s]
36%|███▌      | 610/1688 [03:12<05:32,  3.24it/s]
36%|███▌      | 611/1688 [03:12<05:31,  3.25it/s]
36%|███▋      | 612/1688 [03:12<05:30,  3.26it/s]
36%|███▋      | 613/1688 [03:13<05:32,  3.23it/s]
36%|███▋      | 614/1688 [03:13<05:32,  3.23it/s]
36%|███▋      | 615/1688 [03:13<05:33,  3.22it/s]
36%|███▋      | 616/1688 [03:14<05:31,  3.23it/s]
37%|███▋      | 617/1688 [03:14<05:30,  3.24it/s]
37%|███▋      | 618/1688 [03:14<05:28,  3.26it/s]
37%|███▋      | 619/1688 [03:15<05:27,  3.26it/s]
37%|███▋      | 620/1688 [03:15<05:28,  3.25it/s]
37%|███▋      | 621/1688 [03:15<05:27,  3.26it/s]
37%|███▋      | 622/1688 [03:15<05:28,  3.24it/s]
37%|███▋      | 623/1688 [03:16<05:27,  3.25it/s]
37%|███▋      | 624/1688 [03:16<05:26,  3.26it/s]
37%|███▋      | 625/1688 [03:16<05:25,  3.26it/s]
37%|███▋      | 626/1688 [03:17<05:24,  3.27it/s]
37%|███▋      | 627/1688 [03:17<05:25,  3.26it/s]
37%|███▋      | 628/1688 [03:17<05:27,  3.24it/s]
37%|███▋      | 629/1688 [03:18<05:25,  3.25it/s]
37%|███▋      | 630/1688 [03:18<05:24,  3.26it/s]
37%|███▋      | 631/1688 [03:18<05:24,  3.26it/s]
37%|███▋      | 632/1688 [03:19<05:24,  3.25it/s]
38%|███▊      | 633/1688 [03:19<05:25,  3.24it/s]
38%|███▊      | 634/1688 [03:19<05:23,  3.26it/s]
38%|███▊      | 635/1688 [03:19<05:23,  3.25it/s]
38%|███▊      | 636/1688 [03:20<05:22,  3.26it/s]
38%|███▊      | 637/1688 [03:20<05:21,  3.27it/s]
38%|███▊      | 638/1688 [03:20<05:20,  3.27it/s]
38%|███▊      | 639/1688 [03:21<05:20,  3.28it/s]
38%|███▊      | 640/1688 [03:21<05:20,  3.27it/s]
38%|███▊      | 641/1688 [03:21<05:20,  3.27it/s]
38%|███▊      | 642/1688 [03:22<05:20,  3.26it/s]
38%|███▊      | 643/1688 [03:22<05:20,  3.26it/s]
38%|███▊      | 644/1688 [03:22<05:19,  3.26it/s]
38%|███▊      | 645/1688 [03:23<05:20,  3.25it/s]
38%|███▊      | 646/1688 [03:23<05:21,  3.24it/s]
38%|███▊      | 647/1688 [03:23<05:21,  3.24it/s]
38%|███▊      | 648/1688 [03:23<05:22,  3.22it/s]
38%|███▊      | 649/1688 [03:24<05:21,  3.23it/s]
39%|███▊      | 650/1688 [03:24<05:19,  3.25it/s]
39%|███▊      | 651/1688 [03:24<05:18,  3.25it/s]
39%|███▊      | 652/1688 [03:25<05:19,  3.24it/s]
39%|███▊      | 653/1688 [03:25<05:19,  3.24it/s]
39%|███▊      | 654/1688 [03:25<05:21,  3.22it/s]
39%|███▉      | 655/1688 [03:26<05:20,  3.22it/s]
39%|███▉      | 656/1688 [03:26<05:19,  3.23it/s]
39%|███▉      | 657/1688 [03:26<05:19,  3.23it/s]
39%|███▉      | 658/1688 [03:27<05:18,  3.24it/s]
39%|███▉      | 659/1688 [03:27<05:18,  3.23it/s]
39%|███▉      | 660/1688 [03:27<05:16,  3.25it/s]
39%|███▉      | 661/1688 [03:27<05:15,  3.26it/s]
39%|███▉      | 662/1688 [03:28<05:14,  3.26it/s]
39%|███▉      | 663/1688 [03:28<05:15,  3.25it/s]
39%|███▉      | 664/1688 [03:28<05:16,  3.24it/s]
39%|███▉      | 665/1688 [03:29<05:14,  3.26it/s]
39%|███▉      | 666/1688 [03:29<05:13,  3.26it/s]
40%|███▉      | 667/1688 [03:29<05:12,  3.27it/s]
40%|███▉      | 668/1688 [03:30<05:13,  3.26it/s]
40%|███▉      | 669/1688 [03:30<05:13,  3.25it/s]
40%|███▉      | 670/1688 [03:30<05:12,  3.26it/s]
40%|███▉      | 671/1688 [03:31<05:11,  3.27it/s]
40%|███▉      | 672/1688 [03:31<05:10,  3.27it/s]
40%|███▉      | 673/1688 [03:31<05:09,  3.27it/s]
40%|███▉      | 674/1688 [03:31<05:11,  3.25it/s]
40%|███▉      | 675/1688 [03:32<05:12,  3.24it/s]
40%|████      | 676/1688 [03:32<05:11,  3.25it/s]
40%|████      | 677/1688 [03:32<05:10,  3.26it/s]
40%|████      | 678/1688 [03:33<05:10,  3.26it/s]
40%|████      | 679/1688 [03:33<05:08,  3.27it/s]
40%|████      | 680/1688 [03:33<05:08,  3.26it/s]
40%|████      | 681/1688 [03:34<05:08,  3.26it/s]
40%|████      | 682/1688 [03:34<05:07,  3.27it/s]
40%|████      | 683/1688 [03:34<05:06,  3.27it/s]
41%|████      | 684/1688 [03:35<05:06,  3.28it/s]
41%|████      | 685/1688 [03:35<05:05,  3.28it/s]
41%|████      | 686/1688 [03:35<05:05,  3.28it/s]
41%|████      | 687/1688 [03:35<05:05,  3.28it/s]
41%|████      | 688/1688 [03:36<05:06,  3.26it/s]
41%|████      | 689/1688 [03:36<05:07,  3.25it/s]
41%|████      | 690/1688 [03:36<05:07,  3.25it/s]
41%|████      | 691/1688 [03:37<05:07,  3.24it/s]
41%|████      | 692/1688 [03:37<05:08,  3.23it/s]
41%|████      | 693/1688 [03:37<05:07,  3.24it/s]
41%|████      | 694/1688 [03:38<05:06,  3.24it/s]
41%|████      | 695/1688 [03:38<05:05,  3.25it/s]
41%|████      | 696/1688 [03:38<05:04,  3.26it/s]
41%|████▏     | 697/1688 [03:38<05:03,  3.26it/s]
41%|████▏     | 698/1688 [03:39<05:03,  3.26it/s]
41%|████▏     | 699/1688 [03:39<05:02,  3.27it/s]
41%|████▏     | 700/1688 [03:39<05:01,  3.28it/s]
42%|████▏     | 701/1688 [03:40<05:03,  3.25it/s]
42%|████▏     | 702/1688 [03:40<05:04,  3.24it/s]
42%|████▏     | 703/1688 [03:40<05:03,  3.24it/s]
42%|████▏     | 704/1688 [03:41<05:02,  3.25it/s]
42%|████▏     | 705/1688 [03:41<05:01,  3.26it/s]
42%|████▏     | 706/1688 [03:41<05:02,  3.24it/s]
42%|████▏     | 707/1688 [03:42<05:02,  3.25it/s]
42%|████▏     | 708/1688 [03:42<05:03,  3.23it/s]
42%|████▏     | 709/1688 [03:42<05:02,  3.24it/s]
42%|████▏     | 710/1688 [03:42<05:00,  3.25it/s]
42%|████▏     | 711/1688 [03:43<04:59,  3.26it/s]
42%|████▏     | 712/1688 [03:43<04:58,  3.27it/s]
42%|████▏     | 713/1688 [03:43<04:57,  3.27it/s]
42%|████▏     | 714/1688 [03:44<04:57,  3.28it/s]
42%|████▏     | 715/1688 [03:44<04:56,  3.28it/s]
42%|████▏     | 716/1688 [03:44<04:56,  3.27it/s]
42%|████▏     | 717/1688 [03:45<04:56,  3.28it/s]
43%|████▎     | 718/1688 [03:45<04:58,  3.24it/s]
43%|████▎     | 719/1688 [03:45<04:59,  3.24it/s]
43%|████▎     | 720/1688 [03:46<04:59,  3.23it/s]
43%|████▎     | 721/1688 [03:46<04:58,  3.24it/s]
43%|████▎     | 722/1688 [03:46<04:58,  3.24it/s]
43%|████▎     | 723/1688 [03:46<04:57,  3.25it/s]
43%|████▎     | 724/1688 [03:47<04:56,  3.25it/s]
43%|████▎     | 725/1688 [03:47<04:57,  3.24it/s]
43%|████▎     | 726/1688 [03:47<04:57,  3.24it/s]
43%|████▎     | 727/1688 [03:48<04:57,  3.23it/s]
43%|████▎     | 728/1688 [03:48<04:56,  3.24it/s]
43%|████▎     | 729/1688 [03:48<04:54,  3.25it/s]
43%|████▎     | 730/1688 [03:49<04:53,  3.26it/s]
43%|████▎     | 731/1688 [03:49<04:52,  3.27it/s]
43%|████▎     | 732/1688 [03:49<04:52,  3.27it/s]
43%|████▎     | 733/1688 [03:50<04:53,  3.25it/s]
43%|████▎     | 734/1688 [03:50<04:53,  3.25it/s]
44%|████▎     | 735/1688 [03:50<04:52,  3.26it/s]
44%|████▎     | 736/1688 [03:50<04:51,  3.27it/s]
44%|████▎     | 737/1688 [03:51<04:52,  3.25it/s]
44%|████▎     | 738/1688 [03:51<04:53,  3.23it/s]
44%|████▍     | 739/1688 [03:51<04:55,  3.22it/s]
44%|████▍     | 740/1688 [03:52<04:54,  3.22it/s]
44%|████▍     | 741/1688 [03:52<04:53,  3.23it/s]
44%|████▍     | 742/1688 [03:52<04:52,  3.23it/s]
44%|████▍     | 743/1688 [03:53<04:51,  3.24it/s]
44%|████▍     | 744/1688 [03:53<04:50,  3.25it/s]
44%|████▍     | 745/1688 [03:53<04:49,  3.26it/s]
44%|████▍     | 746/1688 [03:54<04:48,  3.27it/s]
44%|████▍     | 747/1688 [03:54<04:47,  3.28it/s]
44%|████▍     | 748/1688 [03:54<04:46,  3.28it/s]
44%|████▍     | 749/1688 [03:54<04:46,  3.27it/s]
44%|████▍     | 750/1688 [03:55<04:48,  3.25it/s]
44%|████▍     | 751/1688 [03:55<04:47,  3.26it/s]
45%|████▍     | 752/1688 [03:55<04:46,  3.27it/s]
45%|████▍     | 753/1688 [03:56<04:45,  3.27it/s]
45%|████▍     | 754/1688 [03:56<04:44,  3.28it/s]
45%|████▍     | 755/1688 [03:56<04:45,  3.27it/s]
45%|████▍     | 756/1688 [03:57<04:44,  3.27it/s]
45%|████▍     | 757/1688 [03:57<04:44,  3.27it/s]
45%|████▍     | 758/1688 [03:57<04:43,  3.28it/s]
45%|████▍     | 759/1688 [03:58<04:44,  3.27it/s]
45%|████▌     | 760/1688 [03:58<04:43,  3.27it/s]
45%|████▌     | 761/1688 [03:58<04:42,  3.28it/s]
45%|████▌     | 762/1688 [03:58<04:42,  3.28it/s]
45%|████▌     | 763/1688 [03:59<04:42,  3.27it/s]
45%|████▌     | 764/1688 [03:59<04:42,  3.27it/s]
45%|████▌     | 765/1688 [03:59<04:42,  3.27it/s]
45%|████▌     | 766/1688 [04:00<04:41,  3.27it/s]
45%|████▌     | 767/1688 [04:00<04:41,  3.27it/s]
45%|████▌     | 768/1688 [04:00<04:40,  3.28it/s]
46%|████▌     | 769/1688 [04:01<04:40,  3.27it/s]
46%|████▌     | 770/1688 [04:01<04:42,  3.25it/s]
46%|████▌     | 771/1688 [04:01<04:41,  3.26it/s]
46%|████▌     | 772/1688 [04:02<04:40,  3.27it/s]
46%|████▌     | 773/1688 [04:02<04:40,  3.27it/s]
46%|████▌     | 774/1688 [04:02<04:40,  3.26it/s]
46%|████▌     | 775/1688 [04:02<04:41,  3.25it/s]
46%|████▌     | 776/1688 [04:03<04:40,  3.25it/s]
46%|████▌     | 777/1688 [04:03<04:41,  3.24it/s]
46%|████▌     | 778/1688 [04:03<04:40,  3.24it/s]
46%|████▌     | 779/1688 [04:04<04:39,  3.25it/s]
46%|████▌     | 780/1688 [04:04<04:38,  3.26it/s]
46%|████▋     | 781/1688 [04:04<04:37,  3.26it/s]
46%|████▋     | 782/1688 [04:05<04:38,  3.25it/s]
46%|████▋     | 783/1688 [04:05<04:38,  3.25it/s]
46%|████▋     | 784/1688 [04:05<04:37,  3.25it/s]
47%|████▋     | 785/1688 [04:06<04:36,  3.26it/s]
47%|████▋     | 786/1688 [04:06<04:36,  3.26it/s]
47%|████▋     | 787/1688 [04:06<04:36,  3.26it/s]
47%|████▋     | 788/1688 [04:06<04:36,  3.26it/s]
47%|████▋     | 789/1688 [04:07<04:36,  3.25it/s]
47%|████▋     | 790/1688 [04:07<04:35,  3.26it/s]
47%|████▋     | 791/1688 [04:07<04:34,  3.26it/s]
47%|████▋     | 792/1688 [04:08<04:33,  3.27it/s]
47%|████▋     | 793/1688 [04:08<04:33,  3.27it/s]
47%|████▋     | 794/1688 [04:08<04:33,  3.27it/s]
47%|████▋     | 795/1688 [04:09<04:34,  3.26it/s]
47%|████▋     | 796/1688 [04:09<04:34,  3.24it/s]
47%|████▋     | 797/1688 [04:09<04:34,  3.25it/s]
47%|████▋     | 798/1688 [04:10<04:35,  3.23it/s]
47%|████▋     | 799/1688 [04:10<04:35,  3.22it/s]
47%|████▋     | 800/1688 [04:10<04:35,  3.22it/s]
47%|████▋     | 801/1688 [04:10<04:36,  3.21it/s]
48%|████▊     | 802/1688 [04:11<04:36,  3.21it/s]
48%|████▊     | 803/1688 [04:11<04:35,  3.21it/s]
48%|████▊     | 804/1688 [04:11<04:34,  3.22it/s]
48%|████▊     | 805/1688 [04:12<04:34,  3.22it/s]
48%|████▊     | 806/1688 [04:12<04:32,  3.24it/s]
48%|████▊     | 807/1688 [04:12<04:30,  3.25it/s]
48%|████▊     | 808/1688 [04:13<04:29,  3.27it/s]
48%|████▊     | 809/1688 [04:13<04:28,  3.27it/s]
48%|████▊     | 810/1688 [04:13<04:28,  3.27it/s]
48%|████▊     | 811/1688 [04:14<04:28,  3.27it/s]
48%|████▊     | 812/1688 [04:14<04:29,  3.25it/s]
48%|████▊     | 813/1688 [04:14<04:30,  3.24it/s]
48%|████▊     | 814/1688 [04:14<04:30,  3.23it/s]
48%|████▊     | 815/1688 [04:15<04:31,  3.22it/s]
48%|████▊     | 816/1688 [04:15<04:30,  3.22it/s]
48%|████▊     | 817/1688 [04:15<04:29,  3.23it/s]
48%|████▊     | 818/1688 [04:16<04:30,  3.22it/s]
49%|████▊     | 819/1688 [04:16<04:30,  3.22it/s]
49%|████▊     | 820/1688 [04:16<04:28,  3.23it/s]
49%|████▊     | 821/1688 [04:17<04:27,  3.25it/s]
49%|████▊     | 822/1688 [04:17<04:25,  3.26it/s]
49%|████▉     | 823/1688 [04:17<04:24,  3.27it/s]
49%|████▉     | 824/1688 [04:18<04:24,  3.27it/s]
49%|████▉     | 825/1688 [04:18<04:23,  3.28it/s]
49%|████▉     | 826/1688 [04:18<04:23,  3.28it/s]
49%|████▉     | 827/1688 [04:18<04:22,  3.28it/s]
49%|████▉     | 828/1688 [04:19<04:23,  3.26it/s]
49%|████▉     | 829/1688 [04:19<04:22,  3.27it/s]
49%|████▉     | 830/1688 [04:19<04:23,  3.26it/s]
49%|████▉     | 831/1688 [04:20<04:23,  3.25it/s]
49%|████▉     | 832/1688 [04:20<04:23,  3.25it/s]
49%|████▉     | 833/1688 [04:20<04:22,  3.25it/s]
49%|████▉     | 834/1688 [04:21<04:23,  3.25it/s]
49%|████▉     | 835/1688 [04:21<04:22,  3.24it/s]
50%|████▉     | 836/1688 [04:21<04:24,  3.22it/s]
50%|████▉     | 837/1688 [04:22<04:23,  3.23it/s]
50%|████▉     | 838/1688 [04:22<04:21,  3.25it/s]
50%|████▉     | 839/1688 [04:22<04:21,  3.24it/s]
50%|████▉     | 840/1688 [04:22<04:21,  3.24it/s]
50%|████▉     | 841/1688 [04:23<04:22,  3.22it/s]
50%|████▉     | 842/1688 [04:23<04:21,  3.23it/s]
50%|████▉     | 843/1688 [04:23<04:20,  3.24it/s]
50%|█████     | 844/1688 [04:24<04:19,  3.25it/s]
50%|█████     | 845/1688 [04:24<04:20,  3.24it/s]
50%|█████     | 846/1688 [04:24<04:19,  3.24it/s]
50%|█████     | 847/1688 [04:25<04:19,  3.24it/s]
50%|█████     | 848/1688 [04:25<04:19,  3.24it/s]
50%|█████     | 849/1688 [04:25<04:18,  3.25it/s]
50%|█████     | 850/1688 [04:26<04:18,  3.24it/s]
50%|█████     | 851/1688 [04:26<04:18,  3.24it/s]
50%|█████     | 852/1688 [04:26<04:18,  3.24it/s]
51%|█████     | 853/1688 [04:26<04:17,  3.25it/s]
51%|█████     | 854/1688 [04:27<04:16,  3.26it/s]
51%|█████     | 855/1688 [04:27<04:15,  3.26it/s]
51%|█████     | 856/1688 [04:27<04:16,  3.24it/s]
51%|█████     | 857/1688 [04:28<04:16,  3.24it/s]
51%|█████     | 858/1688 [04:28<04:15,  3.25it/s]
51%|█████     | 859/1688 [04:28<04:14,  3.25it/s]
51%|█████     | 860/1688 [04:29<04:15,  3.24it/s]
51%|█████     | 861/1688 [04:29<04:15,  3.24it/s]
51%|█████     | 862/1688 [04:29<04:14,  3.24it/s]
51%|█████     | 863/1688 [04:30<04:15,  3.23it/s]
51%|█████     | 864/1688 [04:30<04:19,  3.17it/s]
51%|█████     | 865/1688 [04:30<04:17,  3.19it/s]
51%|█████▏    | 866/1688 [04:31<04:16,  3.21it/s]
51%|█████▏    | 867/1688 [04:31<04:14,  3.23it/s]
51%|█████▏    | 868/1688 [04:31<04:12,  3.25it/s]
51%|█████▏    | 869/1688 [04:31<04:11,  3.26it/s]
52%|█████▏    | 870/1688 [04:32<04:10,  3.27it/s]
52%|█████▏    | 871/1688 [04:32<04:10,  3.26it/s]
52%|█████▏    | 872/1688 [04:32<04:09,  3.27it/s]
52%|█████▏    | 873/1688 [04:33<04:10,  3.25it/s]
52%|█████▏    | 874/1688 [04:33<04:10,  3.25it/s]
52%|█████▏    | 875/1688 [04:33<04:11,  3.23it/s]
52%|█████▏    | 876/1688 [04:34<04:10,  3.24it/s]
52%|█████▏    | 877/1688 [04:34<04:10,  3.23it/s]
52%|█████▏    | 878/1688 [04:34<04:10,  3.24it/s]
52%|█████▏    | 879/1688 [04:35<04:09,  3.24it/s]
52%|█████▏    | 880/1688 [04:35<04:09,  3.24it/s]
52%|█████▏    | 881/1688 [04:35<04:08,  3.25it/s]
52%|█████▏    | 882/1688 [04:35<04:07,  3.25it/s]
52%|█████▏    | 883/1688 [04:36<04:06,  3.27it/s]
52%|█████▏    | 884/1688 [04:36<04:06,  3.26it/s]
52%|█████▏    | 885/1688 [04:36<04:08,  3.23it/s]
52%|█████▏    | 886/1688 [04:37<04:07,  3.23it/s]
53%|█████▎    | 887/1688 [04:37<04:07,  3.24it/s]
53%|█████▎    | 888/1688 [04:37<04:06,  3.24it/s]
53%|█████▎    | 889/1688 [04:38<04:05,  3.25it/s]
53%|█████▎    | 890/1688 [04:38<04:05,  3.26it/s]
53%|█████▎    | 891/1688 [04:38<04:05,  3.25it/s]
53%|█████▎    | 892/1688 [04:39<04:05,  3.24it/s]
53%|█████▎    | 893/1688 [04:39<04:06,  3.23it/s]
53%|█████▎    | 894/1688 [04:39<04:05,  3.23it/s]
53%|█████▎    | 895/1688 [04:39<04:05,  3.23it/s]
53%|█████▎    | 896/1688 [04:40<04:05,  3.22it/s]
53%|█████▎    | 897/1688 [04:40<04:05,  3.23it/s]
53%|█████▎    | 898/1688 [04:40<04:04,  3.23it/s]
53%|█████▎    | 899/1688 [04:41<04:03,  3.24it/s]
53%|█████▎    | 900/1688 [04:41<04:02,  3.25it/s]
53%|█████▎    | 901/1688 [04:41<04:03,  3.24it/s]
53%|█████▎    | 902/1688 [04:42<04:04,  3.22it/s]
53%|█████▎    | 903/1688 [04:42<04:04,  3.21it/s]
54%|█████▎    | 904/1688 [04:42<04:03,  3.22it/s]
54%|█████▎    | 905/1688 [04:43<04:01,  3.24it/s]
54%|█████▎    | 906/1688 [04:43<04:00,  3.25it/s]
54%|█████▎    | 907/1688 [04:43<04:00,  3.25it/s]
54%|█████▍    | 908/1688 [04:43<03:59,  3.25it/s]
54%|█████▍    | 909/1688 [04:44<03:58,  3.26it/s]
54%|█████▍    | 910/1688 [04:44<03:58,  3.27it/s]
54%|█████▍    | 911/1688 [04:44<03:58,  3.26it/s]
54%|█████▍    | 912/1688 [04:45<03:58,  3.25it/s]
54%|█████▍    | 913/1688 [04:45<03:59,  3.23it/s]
54%|█████▍    | 914/1688 [04:45<03:58,  3.25it/s]
54%|█████▍    | 915/1688 [04:46<03:57,  3.26it/s]
54%|█████▍    | 916/1688 [04:46<03:56,  3.27it/s]
54%|█████▍    | 917/1688 [04:46<03:57,  3.24it/s]
54%|█████▍    | 918/1688 [04:47<03:57,  3.24it/s]
54%|█████▍    | 919/1688 [04:47<03:56,  3.25it/s]
55%|█████▍    | 920/1688 [04:47<03:56,  3.24it/s]
55%|█████▍    | 921/1688 [04:47<03:56,  3.24it/s]
55%|█████▍    | 922/1688 [04:48<03:56,  3.24it/s]
55%|█████▍    | 923/1688 [04:48<03:56,  3.23it/s]
55%|█████▍    | 924/1688 [04:48<03:55,  3.25it/s]
55%|█████▍    | 925/1688 [04:49<03:54,  3.25it/s]
55%|█████▍    | 926/1688 [04:49<03:55,  3.24it/s]
55%|█████▍    | 927/1688 [04:49<03:54,  3.24it/s]
55%|█████▍    | 928/1688 [04:50<03:55,  3.23it/s]
55%|█████▌    | 929/1688 [04:50<03:54,  3.24it/s]
55%|█████▌    | 930/1688 [04:50<03:53,  3.24it/s]
55%|█████▌    | 931/1688 [04:51<03:52,  3.25it/s]
55%|█████▌    | 932/1688 [04:51<03:51,  3.26it/s]
55%|█████▌    | 933/1688 [04:51<03:50,  3.27it/s]
55%|█████▌    | 934/1688 [04:51<03:50,  3.28it/s]
55%|█████▌    | 935/1688 [04:52<03:49,  3.28it/s]
55%|█████▌    | 936/1688 [04:52<03:49,  3.28it/s]
56%|█████▌    | 937/1688 [04:52<03:49,  3.28it/s]
56%|█████▌    | 938/1688 [04:53<03:48,  3.28it/s]
56%|█████▌    | 939/1688 [04:53<03:48,  3.28it/s]
56%|█████▌    | 940/1688 [04:53<03:47,  3.28it/s]
56%|█████▌    | 941/1688 [04:54<03:47,  3.28it/s]
56%|█████▌    | 942/1688 [04:54<03:47,  3.28it/s]
56%|█████▌    | 943/1688 [04:54<03:47,  3.27it/s]
56%|█████▌    | 944/1688 [04:54<03:47,  3.27it/s]
56%|█████▌    | 945/1688 [04:55<03:46,  3.28it/s]
56%|█████▌    | 946/1688 [04:55<03:46,  3.28it/s]
56%|█████▌    | 947/1688 [04:55<03:46,  3.26it/s]
56%|█████▌    | 948/1688 [04:56<03:47,  3.25it/s]
56%|█████▌    | 949/1688 [04:56<03:48,  3.23it/s]
56%|█████▋    | 950/1688 [04:56<03:48,  3.23it/s]
56%|█████▋    | 951/1688 [04:57<03:48,  3.23it/s]
56%|█████▋    | 952/1688 [04:57<03:48,  3.22it/s]
56%|█████▋    | 953/1688 [04:57<03:47,  3.24it/s]
57%|█████▋    | 954/1688 [04:58<03:47,  3.23it/s]
57%|█████▋    | 955/1688 [04:58<03:46,  3.24it/s]
57%|█████▋    | 956/1688 [04:58<03:45,  3.25it/s]
57%|█████▋    | 957/1688 [04:59<03:45,  3.24it/s]
57%|█████▋    | 958/1688 [04:59<03:44,  3.25it/s]
57%|█████▋    | 959/1688 [04:59<03:44,  3.25it/s]
57%|█████▋    | 960/1688 [04:59<03:44,  3.24it/s]
57%|█████▋    | 961/1688 [05:00<03:45,  3.22it/s]
57%|█████▋    | 962/1688 [05:00<03:45,  3.23it/s]
57%|█████▋    | 963/1688 [05:00<03:43,  3.24it/s]
57%|█████▋    | 964/1688 [05:01<03:44,  3.23it/s]
57%|█████▋    | 965/1688 [05:01<03:46,  3.19it/s]
57%|█████▋    | 966/1688 [05:01<03:46,  3.19it/s]
57%|█████▋    | 967/1688 [05:02<03:44,  3.21it/s]
57%|█████▋    | 968/1688 [05:02<03:44,  3.21it/s]
57%|█████▋    | 969/1688 [05:02<03:43,  3.22it/s]
57%|█████▋    | 970/1688 [05:03<03:43,  3.21it/s]
58%|█████▊    | 971/1688 [05:03<03:43,  3.21it/s]
58%|█████▊    | 972/1688 [05:03<03:41,  3.23it/s]
58%|█████▊    | 973/1688 [05:03<03:41,  3.22it/s]
58%|█████▊    | 974/1688 [05:04<03:40,  3.23it/s]
58%|█████▊    | 975/1688 [05:04<03:40,  3.24it/s]
58%|█████▊    | 976/1688 [05:04<03:41,  3.22it/s]
58%|█████▊    | 977/1688 [05:05<03:39,  3.23it/s]
58%|█████▊    | 978/1688 [05:05<03:38,  3.25it/s]
58%|█████▊    | 979/1688 [05:05<03:37,  3.26it/s]
58%|█████▊    | 980/1688 [05:06<03:37,  3.25it/s]
58%|█████▊    | 981/1688 [05:06<03:37,  3.25it/s]
58%|█████▊    | 982/1688 [05:06<03:36,  3.25it/s]
58%|█████▊    | 983/1688 [05:07<03:37,  3.24it/s]
58%|█████▊    | 984/1688 [05:07<03:37,  3.24it/s]
58%|█████▊    | 985/1688 [05:07<03:37,  3.24it/s]
58%|█████▊    | 986/1688 [05:07<03:37,  3.23it/s]
58%|█████▊    | 987/1688 [05:08<03:36,  3.24it/s]
59%|█████▊    | 988/1688 [05:08<03:36,  3.23it/s]
59%|█████▊    | 989/1688 [05:08<03:35,  3.25it/s]
59%|█████▊    | 990/1688 [05:09<03:35,  3.24it/s]
59%|█████▊    | 991/1688 [05:09<03:34,  3.24it/s]
59%|█████▉    | 992/1688 [05:09<03:34,  3.25it/s]
59%|█████▉    | 993/1688 [05:10<03:34,  3.24it/s]
59%|█████▉    | 994/1688 [05:10<03:34,  3.23it/s]
59%|█████▉    | 995/1688 [05:10<03:33,  3.25it/s]
59%|█████▉    | 996/1688 [05:11<03:32,  3.26it/s]
59%|█████▉    | 997/1688 [05:11<03:31,  3.27it/s]
59%|█████▉    | 998/1688 [05:11<03:31,  3.27it/s]
59%|█████▉    | 999/1688 [05:11<03:30,  3.28it/s]
59%|█████▉    | 1000/1688 [05:12<03:30,  3.27it/s]
{'loss': 1.3823, 'learning_rate': 1.2227488151658767e-05, 'epoch': 0.59}
59%|█████▉    | 1000/1688 [05:12<03:30,  3.27it/s]
[INFO|trainer.py:2139] 2022-05-12 09:13:34,801 >> Saving model checkpoint to /opt/ml/model/chinese_roberta/checkpoint-1000
[INFO|trainer.py:2139] 2022-05-12 09:13:34,801 >> Saving model checkpoint to /opt/ml/model/chinese_roberta/checkpoint-1000
[INFO|configuration_utils.py:439] 2022-05-12 09:13:34,802 >> Configuration saved in /opt/ml/model/chinese_roberta/checkpoint-1000/config.json
[INFO|configuration_utils.py:439] 2022-05-12 09:13:34,802 >> Configuration saved in /opt/ml/model/chinese_roberta/checkpoint-1000/config.json
[INFO|modeling_utils.py:1084] 2022-05-12 09:13:35,248 >> Model weights saved in /opt/ml/model/chinese_roberta/checkpoint-1000/pytorch_model.bin
[INFO|modeling_utils.py:1084] 2022-05-12 09:13:35,248 >> Model weights saved in /opt/ml/model/chinese_roberta/checkpoint-1000/pytorch_model.bin
[INFO|tokenization_utils_base.py:2094] 2022-05-12 09:13:35,248 >> tokenizer config file saved in /opt/ml/model/chinese_roberta/checkpoint-1000/tokenizer_config.json
[INFO|tokenization_utils_base.py:2094] 2022-05-12 09:13:35,248 >> tokenizer config file saved in /opt/ml/model/chinese_roberta/checkpoint-1000/tokenizer_config.json
[INFO|tokenization_utils_base.py:2100] 2022-05-12 09:13:35,248 >> Special tokens file saved in /opt/ml/model/chinese_roberta/checkpoint-1000/special_tokens_map.json
[INFO|tokenization_utils_base.py:2100] 2022-05-12 09:13:35,248 >> Special tokens file saved in /opt/ml/model/chinese_roberta/checkpoint-1000/special_tokens_map.json
59%|█████▉    | 1001/1688 [05:13<08:16,  1.38it/s]
59%|█████▉    | 1002/1688 [05:14<06:51,  1.67it/s]
59%|█████▉    | 1003/1688 [05:14<05:51,  1.95it/s]
59%|█████▉    | 1004/1688 [05:14<05:08,  2.21it/s]
60%|█████▉    | 1005/1688 [05:15<04:38,  2.45it/s]
60%|█████▉    | 1006/1688 [05:15<04:16,  2.66it/s]
60%|█████▉    | 1007/1688 [05:15<04:02,  2.81it/s]
60%|█████▉    | 1008/1688 [05:16<03:52,  2.93it/s]
60%|█████▉    | 1009/1688 [05:16<03:44,  3.03it/s]
60%|█████▉    | 1010/1688 [05:16<03:39,  3.08it/s]
60%|█████▉    | 1011/1688 [05:17<03:35,  3.14it/s]
60%|█████▉    | 1012/1688 [05:17<03:32,  3.18it/s]
60%|██████    | 1013/1688 [05:17<03:30,  3.21it/s]
60%|██████    | 1014/1688 [05:17<03:28,  3.23it/s]
60%|██████    | 1015/1688 [05:18<03:28,  3.23it/s]
60%|██████    | 1016/1688 [05:18<03:27,  3.23it/s]
60%|██████    | 1017/1688 [05:18<03:28,  3.22it/s]
60%|██████    | 1018/1688 [05:19<03:27,  3.23it/s]
60%|██████    | 1019/1688 [05:19<03:26,  3.23it/s]
60%|██████    | 1020/1688 [05:19<03:25,  3.24it/s]
60%|██████    | 1021/1688 [05:20<03:25,  3.25it/s]
61%|██████    | 1022/1688 [05:20<03:25,  3.25it/s]
61%|██████    | 1023/1688 [05:20<03:24,  3.25it/s]
61%|██████    | 1024/1688 [05:21<03:23,  3.26it/s]
61%|██████    | 1025/1688 [05:21<03:23,  3.27it/s]
61%|██████    | 1026/1688 [05:21<03:22,  3.27it/s]
61%|██████    | 1027/1688 [05:21<03:21,  3.27it/s]
61%|██████    | 1028/1688 [05:22<03:22,  3.26it/s]
61%|██████    | 1029/1688 [05:22<03:21,  3.27it/s]
61%|██████    | 1030/1688 [05:22<03:21,  3.26it/s]
61%|██████    | 1031/1688 [05:23<03:21,  3.26it/s]
61%|██████    | 1032/1688 [05:23<03:21,  3.26it/s]
61%|██████    | 1033/1688 [05:23<03:20,  3.27it/s]
61%|██████▏   | 1034/1688 [05:24<03:19,  3.28it/s]
61%|██████▏   | 1035/1688 [05:24<03:19,  3.27it/s]
61%|██████▏   | 1036/1688 [05:24<03:20,  3.25it/s]
61%|██████▏   | 1037/1688 [05:25<03:21,  3.23it/s]
61%|██████▏   | 1038/1688 [05:25<03:21,  3.23it/s]
62%|██████▏   | 1039/1688 [05:25<03:20,  3.23it/s]
62%|██████▏   | 1040/1688 [05:25<03:21,  3.22it/s]
62%|██████▏   | 1041/1688 [05:26<03:19,  3.24it/s]
62%|██████▏   | 1042/1688 [05:26<03:19,  3.24it/s]
62%|██████▏   | 1043/1688 [05:26<03:19,  3.23it/s]
62%|██████▏   | 1044/1688 [05:27<03:18,  3.25it/s]
62%|██████▏   | 1045/1688 [05:27<03:17,  3.26it/s]
62%|██████▏   | 1046/1688 [05:27<03:16,  3.27it/s]
62%|██████▏   | 1047/1688 [05:28<03:15,  3.27it/s]
62%|██████▏   | 1048/1688 [05:28<03:15,  3.27it/s]
62%|██████▏   | 1049/1688 [05:28<03:15,  3.27it/s]
62%|██████▏   | 1050/1688 [05:29<03:14,  3.27it/s]
62%|██████▏   | 1051/1688 [05:29<03:14,  3.27it/s]
62%|██████▏   | 1052/1688 [05:29<03:14,  3.27it/s]
62%|██████▏   | 1053/1688 [05:29<03:14,  3.26it/s]
62%|██████▏   | 1054/1688 [05:30<03:14,  3.27it/s]
62%|██████▎   | 1055/1688 [05:30<03:13,  3.27it/s]
63%|██████▎   | 1056/1688 [05:30<03:13,  3.26it/s]
63%|██████▎   | 1057/1688 [05:31<03:13,  3.26it/s]
63%|██████▎   | 1058/1688 [05:31<03:12,  3.27it/s]
63%|██████▎   | 1059/1688 [05:31<03:12,  3.27it/s]
63%|██████▎   | 1060/1688 [05:32<03:11,  3.27it/s]
63%|██████▎   | 1061/1688 [05:32<03:11,  3.28it/s]
63%|██████▎   | 1062/1688 [05:32<03:10,  3.28it/s]
63%|██████▎   | 1063/1688 [05:33<03:11,  3.27it/s]
63%|██████▎   | 1064/1688 [05:33<03:12,  3.25it/s]
63%|██████▎   | 1065/1688 [05:33<03:11,  3.25it/s]
63%|██████▎   | 1066/1688 [05:33<03:10,  3.26it/s]
63%|██████▎   | 1067/1688 [05:34<03:10,  3.27it/s]
63%|██████▎   | 1068/1688 [05:34<03:10,  3.26it/s]
63%|██████▎   | 1069/1688 [05:34<03:09,  3.27it/s]
63%|██████▎   | 1070/1688 [05:35<03:08,  3.27it/s]
63%|██████▎   | 1071/1688 [05:35<03:09,  3.26it/s]
64%|██████▎   | 1072/1688 [05:35<03:09,  3.24it/s]
64%|██████▎   | 1073/1688 [05:36<03:10,  3.22it/s]
64%|██████▎   | 1074/1688 [05:36<03:10,  3.23it/s]
64%|██████▎   | 1075/1688 [05:36<03:09,  3.23it/s]
64%|██████▎   | 1076/1688 [05:37<03:08,  3.24it/s]
64%|██████▍   | 1077/1688 [05:37<03:09,  3.23it/s]
64%|██████▍   | 1078/1688 [05:37<03:07,  3.25it/s]
64%|██████▍   | 1079/1688 [05:37<03:07,  3.25it/s]
64%|██████▍   | 1080/1688 [05:38<03:06,  3.26it/s]
64%|██████▍   | 1081/1688 [05:38<03:05,  3.27it/s]
64%|██████▍   | 1082/1688 [05:38<03:06,  3.25it/s]
64%|██████▍   | 1083/1688 [05:39<03:07,  3.23it/s]
64%|██████▍   | 1084/1688 [05:39<03:05,  3.25it/s]
64%|██████▍   | 1085/1688 [05:39<03:05,  3.25it/s]
64%|██████▍   | 1086/1688 [05:40<03:05,  3.25it/s]
64%|██████▍   | 1087/1688 [05:40<03:05,  3.23it/s]
64%|██████▍   | 1088/1688 [05:40<03:05,  3.23it/s]
65%|██████▍   | 1089/1688 [05:41<03:05,  3.22it/s]
65%|██████▍   | 1090/1688 [05:41<03:05,  3.22it/s]
65%|██████▍   | 1091/1688 [05:41<03:05,  3.21it/s]
65%|██████▍   | 1092/1688 [05:41<03:05,  3.22it/s]
65%|██████▍   | 1093/1688 [05:42<03:05,  3.21it/s]
65%|██████▍   | 1094/1688 [05:42<03:05,  3.20it/s]
65%|██████▍   | 1095/1688 [05:42<03:05,  3.20it/s]
65%|██████▍   | 1096/1688 [05:43<03:04,  3.21it/s]
65%|██████▍   | 1097/1688 [05:43<03:03,  3.23it/s]
65%|██████▌   | 1098/1688 [05:43<03:02,  3.23it/s]
65%|██████▌   | 1099/1688 [05:44<03:01,  3.24it/s]
65%|██████▌   | 1100/1688 [05:44<03:00,  3.25it/s]
65%|██████▌   | 1101/1688 [05:44<03:00,  3.26it/s]
65%|██████▌   | 1102/1688 [05:45<02:59,  3.27it/s]
65%|██████▌   | 1103/1688 [05:45<02:58,  3.27it/s]
65%|██████▌   | 1104/1688 [05:45<02:58,  3.27it/s]
65%|██████▌   | 1105/1688 [05:45<02:59,  3.25it/s]
66%|██████▌   | 1106/1688 [05:46<02:59,  3.24it/s]
66%|██████▌   | 1107/1688 [05:46<02:59,  3.24it/s]
66%|██████▌   | 1108/1688 [05:46<02:58,  3.25it/s]
66%|██████▌   | 1109/1688 [05:47<02:57,  3.26it/s]
66%|██████▌   | 1110/1688 [05:47<02:57,  3.26it/s]
66%|██████▌   | 1111/1688 [05:47<02:57,  3.26it/s]
66%|██████▌   | 1112/1688 [05:48<02:57,  3.25it/s]
66%|██████▌   | 1113/1688 [05:48<02:56,  3.26it/s]
66%|██████▌   | 1114/1688 [05:48<02:55,  3.26it/s]
66%|██████▌   | 1115/1688 [05:49<02:55,  3.26it/s]
66%|██████▌   | 1116/1688 [05:49<02:55,  3.26it/s]
66%|██████▌   | 1117/1688 [05:49<02:54,  3.27it/s]
66%|██████▌   | 1118/1688 [05:49<02:54,  3.28it/s]
66%|██████▋   | 1119/1688 [05:50<02:53,  3.27it/s]
66%|██████▋   | 1120/1688 [05:50<02:53,  3.27it/s]
66%|██████▋   | 1121/1688 [05:50<02:52,  3.28it/s]
66%|██████▋   | 1122/1688 [05:51<02:52,  3.28it/s]
67%|██████▋   | 1123/1688 [05:51<02:52,  3.28it/s]
67%|██████▋   | 1124/1688 [05:51<02:51,  3.29it/s]
67%|██████▋   | 1125/1688 [05:52<02:51,  3.29it/s]
67%|██████▋   | 1126/1688 [05:52<02:51,  3.28it/s]
67%|██████▋   | 1127/1688 [05:52<02:50,  3.28it/s]
67%|██████▋   | 1128/1688 [05:53<02:51,  3.27it/s]
67%|██████▋   | 1129/1688 [05:53<02:51,  3.26it/s]
67%|██████▋   | 1130/1688 [05:53<02:50,  3.27it/s]
67%|██████▋   | 1131/1688 [05:53<02:49,  3.28it/s]
67%|██████▋   | 1132/1688 [05:54<02:49,  3.28it/s]
67%|██████▋   | 1133/1688 [05:54<02:49,  3.28it/s]
67%|██████▋   | 1134/1688 [05:54<02:48,  3.29it/s]
67%|██████▋   | 1135/1688 [05:55<02:48,  3.28it/s]
67%|██████▋   | 1136/1688 [05:55<02:48,  3.28it/s]
67%|██████▋   | 1137/1688 [05:55<02:47,  3.28it/s]
67%|██████▋   | 1138/1688 [05:56<02:48,  3.27it/s]
67%|██████▋   | 1139/1688 [05:56<02:47,  3.27it/s]
68%|██████▊   | 1140/1688 [05:56<02:47,  3.28it/s]
68%|██████▊   | 1141/1688 [05:56<02:46,  3.28it/s]
68%|██████▊   | 1142/1688 [05:57<02:46,  3.28it/s]
68%|██████▊   | 1143/1688 [05:57<02:45,  3.28it/s]
68%|██████▊   | 1144/1688 [05:57<02:45,  3.28it/s]
68%|██████▊   | 1145/1688 [05:58<02:46,  3.27it/s]
68%|██████▊   | 1146/1688 [05:58<02:47,  3.24it/s]
68%|██████▊   | 1147/1688 [05:58<02:47,  3.24it/s]
68%|██████▊   | 1148/1688 [05:59<02:47,  3.23it/s]
68%|██████▊   | 1149/1688 [05:59<02:47,  3.22it/s]
68%|██████▊   | 1150/1688 [05:59<02:46,  3.23it/s]
68%|██████▊   | 1151/1688 [06:00<02:45,  3.24it/s]
68%|██████▊   | 1152/1688 [06:00<02:46,  3.22it/s]
68%|██████▊   | 1153/1688 [06:00<02:45,  3.24it/s]
68%|██████▊   | 1154/1688 [06:00<02:44,  3.25it/s]
68%|██████▊   | 1155/1688 [06:01<02:43,  3.26it/s]
68%|██████▊   | 1156/1688 [06:01<02:42,  3.26it/s]
69%|██████▊   | 1157/1688 [06:01<02:42,  3.27it/s]
69%|██████▊   | 1158/1688 [06:02<02:41,  3.27it/s]
69%|██████▊   | 1159/1688 [06:02<02:41,  3.27it/s]
69%|██████▊   | 1160/1688 [06:02<02:41,  3.27it/s]
69%|██████▉   | 1161/1688 [06:03<02:41,  3.27it/s]
69%|██████▉   | 1162/1688 [06:03<02:41,  3.26it/s]
69%|██████▉   | 1163/1688 [06:03<02:40,  3.27it/s]
69%|██████▉   | 1164/1688 [06:04<02:40,  3.27it/s]
69%|██████▉   | 1165/1688 [06:04<02:39,  3.27it/s]
69%|██████▉   | 1166/1688 [06:04<02:40,  3.26it/s]
69%|██████▉   | 1167/1688 [06:04<02:40,  3.25it/s]
69%|██████▉   | 1168/1688 [06:05<02:39,  3.26it/s]
69%|██████▉   | 1169/1688 [06:05<02:38,  3.27it/s]
69%|██████▉   | 1170/1688 [06:05<02:38,  3.27it/s]
69%|██████▉   | 1171/1688 [06:06<02:37,  3.27it/s]
69%|██████▉   | 1172/1688 [06:06<02:37,  3.28it/s]
69%|██████▉   | 1173/1688 [06:06<02:36,  3.28it/s]
70%|██████▉   | 1174/1688 [06:07<02:36,  3.28it/s]
70%|██████▉   | 1175/1688 [06:07<02:36,  3.28it/s]
70%|██████▉   | 1176/1688 [06:07<02:36,  3.28it/s]
70%|██████▉   | 1177/1688 [06:08<02:35,  3.28it/s]
70%|██████▉   | 1178/1688 [06:08<02:36,  3.27it/s]
70%|██████▉   | 1179/1688 [06:08<02:35,  3.27it/s]
70%|██████▉   | 1180/1688 [06:08<02:35,  3.26it/s]
70%|██████▉   | 1181/1688 [06:09<02:35,  3.27it/s]
70%|███████   | 1182/1688 [06:09<02:34,  3.27it/s]
70%|███████   | 1183/1688 [06:09<02:34,  3.26it/s]
70%|███████   | 1184/1688 [06:10<02:34,  3.25it/s]
70%|███████   | 1185/1688 [06:10<02:35,  3.24it/s]
70%|███████   | 1186/1688 [06:10<02:34,  3.24it/s]
70%|███████   | 1187/1688 [06:11<02:33,  3.25it/s]
70%|███████   | 1188/1688 [06:11<02:33,  3.26it/s]
70%|███████   | 1189/1688 [06:11<02:32,  3.27it/s]
70%|███████   | 1190/1688 [06:11<02:32,  3.27it/s]
71%|███████   | 1191/1688 [06:12<02:31,  3.27it/s]
71%|███████   | 1192/1688 [06:12<02:31,  3.27it/s]
71%|███████   | 1193/1688 [06:12<02:31,  3.28it/s]
71%|███████   | 1194/1688 [06:13<02:30,  3.28it/s]
71%|███████   | 1195/1688 [06:13<02:30,  3.28it/s]
71%|███████   | 1196/1688 [06:13<02:29,  3.28it/s]
71%|███████   | 1197/1688 [06:14<02:29,  3.28it/s]
71%|███████   | 1198/1688 [06:14<02:29,  3.27it/s]
71%|███████   | 1199/1688 [06:14<02:29,  3.27it/s]
71%|███████   | 1200/1688 [06:15<02:30,  3.25it/s]
71%|███████   | 1201/1688 [06:15<02:29,  3.25it/s]
71%|███████   | 1202/1688 [06:15<02:29,  3.26it/s]
71%|███████▏  | 1203/1688 [06:15<02:28,  3.26it/s]
71%|███████▏  | 1204/1688 [06:16<02:28,  3.26it/s]
71%|███████▏  | 1205/1688 [06:16<02:27,  3.27it/s]
71%|███████▏  | 1206/1688 [06:16<02:27,  3.27it/s]
72%|███████▏  | 1207/1688 [06:17<02:26,  3.28it/s]
72%|███████▏  | 1208/1688 [06:17<02:26,  3.28it/s]
72%|███████▏  | 1209/1688 [06:17<02:26,  3.28it/s]
72%|███████▏  | 1210/1688 [06:18<02:26,  3.27it/s]
72%|███████▏  | 1211/1688 [06:18<02:26,  3.26it/s]
72%|███████▏  | 1212/1688 [06:18<02:25,  3.26it/s]
72%|███████▏  | 1213/1688 [06:19<02:25,  3.26it/s]
72%|███████▏  | 1214/1688 [06:19<02:25,  3.26it/s]
72%|███████▏  | 1215/1688 [06:19<02:25,  3.25it/s]
72%|███████▏  | 1216/1688 [06:19<02:25,  3.25it/s]
72%|███████▏  | 1217/1688 [06:20<02:25,  3.24it/s]
72%|███████▏  | 1218/1688 [06:20<02:25,  3.23it/s]
72%|███████▏  | 1219/1688 [06:20<02:25,  3.23it/s]
72%|███████▏  | 1220/1688 [06:21<02:24,  3.24it/s]
72%|███████▏  | 1221/1688 [06:21<02:23,  3.25it/s]
72%|███████▏  | 1222/1688 [06:21<02:24,  3.22it/s]
72%|███████▏  | 1223/1688 [06:22<02:25,  3.20it/s]
73%|███████▎  | 1224/1688 [06:22<02:23,  3.23it/s]
73%|███████▎  | 1225/1688 [06:22<02:22,  3.24it/s]
73%|███████▎  | 1226/1688 [06:23<02:22,  3.23it/s]
73%|███████▎  | 1227/1688 [06:23<02:22,  3.24it/s]
73%|███████▎  | 1228/1688 [06:23<02:21,  3.25it/s]
73%|███████▎  | 1229/1688 [06:23<02:21,  3.25it/s]
73%|███████▎  | 1230/1688 [06:24<02:20,  3.26it/s]
73%|███████▎  | 1231/1688 [06:24<02:19,  3.27it/s]
73%|███████▎  | 1232/1688 [06:24<02:20,  3.25it/s]
73%|███████▎  | 1233/1688 [06:25<02:20,  3.23it/s]
73%|███████▎  | 1234/1688 [06:25<02:20,  3.22it/s]
73%|███████▎  | 1235/1688 [06:25<02:20,  3.23it/s]
73%|███████▎  | 1236/1688 [06:26<02:19,  3.24it/s]
73%|███████▎  | 1237/1688 [06:26<02:19,  3.24it/s]
73%|███████▎  | 1238/1688 [06:26<02:19,  3.22it/s]
73%|███████▎  | 1239/1688 [06:27<02:19,  3.23it/s]
73%|███████▎  | 1240/1688 [06:27<02:18,  3.24it/s]
74%|███████▎  | 1241/1688 [06:27<02:17,  3.26it/s]
74%|███████▎  | 1242/1688 [06:27<02:16,  3.26it/s]
74%|███████▎  | 1243/1688 [06:28<02:16,  3.25it/s]
74%|███████▎  | 1244/1688 [06:28<02:16,  3.25it/s]
74%|███████▍  | 1245/1688 [06:28<02:15,  3.26it/s]
74%|███████▍  | 1246/1688 [06:29<02:15,  3.27it/s]
74%|███████▍  | 1247/1688 [06:29<02:14,  3.28it/s]
74%|███████▍  | 1248/1688 [06:29<02:14,  3.27it/s]
74%|███████▍  | 1249/1688 [06:30<02:14,  3.26it/s]
74%|███████▍  | 1250/1688 [06:30<02:14,  3.27it/s]
74%|███████▍  | 1251/1688 [06:30<02:13,  3.27it/s]
74%|███████▍  | 1252/1688 [06:31<02:13,  3.27it/s]
74%|███████▍  | 1253/1688 [06:31<02:12,  3.28it/s]
74%|███████▍  | 1254/1688 [06:31<02:12,  3.28it/s]
74%|███████▍  | 1255/1688 [06:31<02:12,  3.26it/s]
74%|███████▍  | 1256/1688 [06:32<02:13,  3.24it/s]
74%|███████▍  | 1257/1688 [06:32<02:13,  3.23it/s]
75%|███████▍  | 1258/1688 [06:32<02:12,  3.24it/s]
75%|███████▍  | 1259/1688 [06:33<02:12,  3.24it/s]
75%|███████▍  | 1260/1688 [06:33<02:11,  3.25it/s]
75%|███████▍  | 1261/1688 [06:33<02:10,  3.26it/s]
75%|███████▍  | 1262/1688 [06:34<02:10,  3.27it/s]
75%|███████▍  | 1263/1688 [06:34<02:10,  3.26it/s]
75%|███████▍  | 1264/1688 [06:34<02:10,  3.25it/s]
75%|███████▍  | 1265/1688 [06:35<02:10,  3.25it/s]
75%|███████▌  | 1266/1688 [06:35<02:10,  3.23it/s]
75%|███████▌  | 1267/1688 [06:35<02:10,  3.22it/s]
75%|███████▌  | 1268/1688 [06:35<02:10,  3.21it/s]
75%|███████▌  | 1269/1688 [06:36<02:10,  3.21it/s]
75%|███████▌  | 1270/1688 [06:36<02:09,  3.23it/s]
75%|███████▌  | 1271/1688 [06:36<02:08,  3.25it/s]
75%|███████▌  | 1272/1688 [06:37<02:08,  3.24it/s]
75%|███████▌  | 1273/1688 [06:37<02:07,  3.25it/s]
75%|███████▌  | 1274/1688 [06:37<02:07,  3.25it/s]
76%|███████▌  | 1275/1688 [06:38<02:07,  3.23it/s]
76%|███████▌  | 1276/1688 [06:38<02:07,  3.23it/s]
76%|███████▌  | 1277/1688 [06:38<02:08,  3.21it/s]
76%|███████▌  | 1278/1688 [06:39<02:08,  3.20it/s]
76%|███████▌  | 1279/1688 [06:39<02:07,  3.20it/s]
76%|███████▌  | 1280/1688 [06:39<02:07,  3.20it/s]
76%|███████▌  | 1281/1688 [06:40<02:06,  3.21it/s]
76%|███████▌  | 1282/1688 [06:40<02:06,  3.22it/s]
76%|███████▌  | 1283/1688 [06:40<02:05,  3.23it/s]
76%|███████▌  | 1284/1688 [06:40<02:05,  3.22it/s]
76%|███████▌  | 1285/1688 [06:41<02:05,  3.22it/s]
76%|███████▌  | 1286/1688 [06:41<02:04,  3.23it/s]
76%|███████▌  | 1287/1688 [06:41<02:04,  3.23it/s]
76%|███████▋  | 1288/1688 [06:42<02:04,  3.22it/s]
76%|███████▋  | 1289/1688 [06:42<02:04,  3.21it/s]
76%|███████▋  | 1290/1688 [06:42<02:03,  3.22it/s]
76%|███████▋  | 1291/1688 [06:43<02:02,  3.24it/s]
77%|███████▋  | 1292/1688 [06:43<02:01,  3.25it/s]
77%|███████▋  | 1293/1688 [06:43<02:01,  3.25it/s]
77%|███████▋  | 1294/1688 [06:44<02:01,  3.24it/s]
77%|███████▋  | 1295/1688 [06:44<02:00,  3.25it/s]
77%|███████▋  | 1296/1688 [06:44<02:00,  3.26it/s]
77%|███████▋  | 1297/1688 [06:44<02:00,  3.25it/s]
77%|███████▋  | 1298/1688 [06:45<02:00,  3.25it/s]
77%|███████▋  | 1299/1688 [06:45<02:00,  3.23it/s]
77%|███████▋  | 1300/1688 [06:45<01:59,  3.25it/s]
77%|███████▋  | 1301/1688 [06:46<01:59,  3.25it/s]
77%|███████▋  | 1302/1688 [06:46<01:58,  3.25it/s]
77%|███████▋  | 1303/1688 [06:46<01:58,  3.24it/s]
77%|███████▋  | 1304/1688 [06:47<01:58,  3.25it/s]
77%|███████▋  | 1305/1688 [06:47<01:57,  3.25it/s]
77%|███████▋  | 1306/1688 [06:47<01:57,  3.26it/s]
77%|███████▋  | 1307/1688 [06:48<01:57,  3.26it/s]
77%|███████▋  | 1308/1688 [06:48<01:56,  3.26it/s]
78%|███████▊  | 1309/1688 [06:48<01:56,  3.26it/s]
78%|███████▊  | 1310/1688 [06:48<01:56,  3.24it/s]
78%|███████▊  | 1311/1688 [06:49<01:56,  3.23it/s]
78%|███████▊  | 1312/1688 [06:49<01:56,  3.23it/s]
78%|███████▊  | 1313/1688 [06:49<01:56,  3.23it/s]
78%|███████▊  | 1314/1688 [06:50<01:55,  3.23it/s]
78%|███████▊  | 1315/1688 [06:50<01:54,  3.25it/s]
78%|███████▊  | 1316/1688 [06:50<01:54,  3.26it/s]
78%|███████▊  | 1317/1688 [06:51<01:53,  3.26it/s]
78%|███████▊  | 1318/1688 [06:51<01:53,  3.27it/s]
78%|███████▊  | 1319/1688 [06:51<01:52,  3.27it/s]
78%|███████▊  | 1320/1688 [06:52<01:52,  3.28it/s]
78%|███████▊  | 1321/1688 [06:52<01:52,  3.27it/s]
78%|███████▊  | 1322/1688 [06:52<01:51,  3.27it/s]
78%|███████▊  | 1323/1688 [06:52<01:51,  3.28it/s]
78%|███████▊  | 1324/1688 [06:53<01:50,  3.28it/s]
78%|███████▊  | 1325/1688 [06:53<01:50,  3.28it/s]
79%|███████▊  | 1326/1688 [06:53<01:50,  3.28it/s]
79%|███████▊  | 1327/1688 [06:54<01:50,  3.27it/s]
79%|███████▊  | 1328/1688 [06:54<01:50,  3.24it/s]
79%|███████▊  | 1329/1688 [06:54<01:51,  3.23it/s]
79%|███████▉  | 1330/1688 [06:55<01:50,  3.24it/s]
79%|███████▉  | 1331/1688 [06:55<01:50,  3.24it/s]
79%|███████▉  | 1332/1688 [06:55<01:49,  3.24it/s]
79%|███████▉  | 1333/1688 [06:56<01:49,  3.24it/s]
79%|███████▉  | 1334/1688 [06:56<01:49,  3.24it/s]
79%|███████▉  | 1335/1688 [06:56<01:48,  3.25it/s]
79%|███████▉  | 1336/1688 [06:56<01:47,  3.26it/s]
79%|███████▉  | 1337/1688 [06:57<01:47,  3.27it/s]
79%|███████▉  | 1338/1688 [06:57<01:46,  3.27it/s]
79%|███████▉  | 1339/1688 [06:57<01:46,  3.27it/s]
79%|███████▉  | 1340/1688 [06:58<01:46,  3.27it/s]
79%|███████▉  | 1341/1688 [06:58<01:46,  3.27it/s]
80%|███████▉  | 1342/1688 [06:58<01:46,  3.25it/s]
80%|███████▉  | 1343/1688 [06:59<01:46,  3.24it/s]
80%|███████▉  | 1344/1688 [06:59<01:45,  3.25it/s]
80%|███████▉  | 1345/1688 [06:59<01:45,  3.24it/s]
80%|███████▉  | 1346/1688 [07:00<01:45,  3.24it/s]
80%|███████▉  | 1347/1688 [07:00<01:44,  3.25it/s]
80%|███████▉  | 1348/1688 [07:00<01:44,  3.25it/s]
80%|███████▉  | 1349/1688 [07:00<01:44,  3.24it/s]
80%|███████▉  | 1350/1688 [07:01<01:44,  3.24it/s]
80%|████████  | 1351/1688 [07:01<01:43,  3.25it/s]
80%|████████  | 1352/1688 [07:01<01:43,  3.26it/s]
80%|████████  | 1353/1688 [07:02<01:42,  3.26it/s]
80%|████████  | 1354/1688 [07:02<01:42,  3.27it/s]
80%|████████  | 1355/1688 [07:02<01:41,  3.27it/s]
80%|████████  | 1356/1688 [07:03<01:41,  3.27it/s]
80%|████████  | 1357/1688 [07:03<01:41,  3.26it/s]
80%|████████  | 1358/1688 [07:03<01:41,  3.26it/s]
81%|████████  | 1359/1688 [07:04<01:41,  3.25it/s]
81%|████████  | 1360/1688 [07:04<01:41,  3.24it/s]
81%|████████  | 1361/1688 [07:04<01:41,  3.23it/s]
81%|████████  | 1362/1688 [07:04<01:40,  3.24it/s]
81%|████████  | 1363/1688 [07:05<01:40,  3.22it/s]
81%|████████  | 1364/1688 [07:05<01:40,  3.23it/s]
81%|████████  | 1365/1688 [07:05<01:39,  3.25it/s]
81%|████████  | 1366/1688 [07:06<01:38,  3.25it/s]
81%|████████  | 1367/1688 [07:06<01:38,  3.26it/s]
81%|████████  | 1368/1688 [07:06<01:37,  3.27it/s]
81%|████████  | 1369/1688 [07:07<01:37,  3.27it/s]
81%|████████  | 1370/1688 [07:07<01:37,  3.25it/s]
81%|████████  | 1371/1688 [07:07<01:37,  3.25it/s]
81%|████████▏ | 1372/1688 [07:08<01:37,  3.24it/s]
81%|████████▏ | 1373/1688 [07:08<01:36,  3.25it/s]
81%|████████▏ | 1374/1688 [07:08<01:36,  3.26it/s]
81%|████████▏ | 1375/1688 [07:08<01:35,  3.26it/s]
82%|████████▏ | 1376/1688 [07:09<01:35,  3.27it/s]
82%|████████▏ | 1377/1688 [07:09<01:35,  3.26it/s]
82%|████████▏ | 1378/1688 [07:09<01:35,  3.25it/s]
82%|████████▏ | 1379/1688 [07:10<01:35,  3.25it/s]
82%|████████▏ | 1380/1688 [07:10<01:34,  3.25it/s]
82%|████████▏ | 1381/1688 [07:10<01:34,  3.25it/s]
82%|████████▏ | 1382/1688 [07:11<01:33,  3.26it/s]
82%|████████▏ | 1383/1688 [07:11<01:33,  3.26it/s]
82%|████████▏ | 1384/1688 [07:11<01:33,  3.26it/s]
82%|████████▏ | 1385/1688 [07:11<01:33,  3.25it/s]
82%|████████▏ | 1386/1688 [07:12<01:33,  3.24it/s]
82%|████████▏ | 1387/1688 [07:12<01:33,  3.24it/s]
82%|████████▏ | 1388/1688 [07:12<01:32,  3.24it/s]
82%|████████▏ | 1389/1688 [07:13<01:32,  3.23it/s]
82%|████████▏ | 1390/1688 [07:13<01:32,  3.21it/s]
82%|████████▏ | 1391/1688 [07:13<01:32,  3.22it/s]
82%|████████▏ | 1392/1688 [07:14<01:31,  3.24it/s]
83%|████████▎ | 1393/1688 [07:14<01:30,  3.24it/s]
83%|████████▎ | 1394/1688 [07:14<01:30,  3.25it/s]
83%|████████▎ | 1395/1688 [07:15<01:29,  3.26it/s]
83%|████████▎ | 1396/1688 [07:15<01:29,  3.27it/s]
83%|████████▎ | 1397/1688 [07:15<01:29,  3.27it/s]
83%|████████▎ | 1398/1688 [07:16<01:29,  3.26it/s]
83%|████████▎ | 1399/1688 [07:16<01:29,  3.24it/s]
83%|████████▎ | 1400/1688 [07:16<01:29,  3.23it/s]
83%|████████▎ | 1401/1688 [07:16<01:29,  3.22it/s]
83%|████████▎ | 1402/1688 [07:17<01:28,  3.22it/s]
83%|████████▎ | 1403/1688 [07:17<01:28,  3.22it/s]
83%|████████▎ | 1404/1688 [07:17<01:28,  3.21it/s]
83%|████████▎ | 1405/1688 [07:18<01:27,  3.23it/s]
83%|████████▎ | 1406/1688 [07:18<01:27,  3.24it/s]
83%|████████▎ | 1407/1688 [07:18<01:26,  3.25it/s]
83%|████████▎ | 1408/1688 [07:19<01:26,  3.25it/s]
83%|████████▎ | 1409/1688 [07:19<01:26,  3.24it/s]
84%|████████▎ | 1410/1688 [07:19<01:25,  3.24it/s]
84%|████████▎ | 1411/1688 [07:20<01:25,  3.24it/s]
84%|████████▎ | 1412/1688 [07:20<01:24,  3.25it/s]
84%|████████▎ | 1413/1688 [07:20<01:24,  3.24it/s]
84%|████████▍ | 1414/1688 [07:20<01:24,  3.25it/s]
84%|████████▍ | 1415/1688 [07:21<01:23,  3.26it/s]
84%|████████▍ | 1416/1688 [07:21<01:23,  3.26it/s]
84%|████████▍ | 1417/1688 [07:21<01:23,  3.26it/s]
84%|████████▍ | 1418/1688 [07:22<01:23,  3.25it/s]
84%|████████▍ | 1419/1688 [07:22<01:23,  3.24it/s]
84%|████████▍ | 1420/1688 [07:22<01:23,  3.22it/s]
84%|████████▍ | 1421/1688 [07:23<01:22,  3.22it/s]
84%|████████▍ | 1422/1688 [07:23<01:22,  3.23it/s]
84%|████████▍ | 1423/1688 [07:23<01:22,  3.22it/s]
84%|████████▍ | 1424/1688 [07:24<01:21,  3.24it/s]
84%|████████▍ | 1425/1688 [07:24<01:20,  3.25it/s]
84%|████████▍ | 1426/1688 [07:24<01:20,  3.26it/s]
85%|████████▍ | 1427/1688 [07:24<01:20,  3.26it/s]
85%|████████▍ | 1428/1688 [07:25<01:19,  3.26it/s]
85%|████████▍ | 1429/1688 [07:25<01:19,  3.25it/s]
85%|████████▍ | 1430/1688 [07:25<01:19,  3.25it/s]
85%|████████▍ | 1431/1688 [07:26<01:19,  3.25it/s]
85%|████████▍ | 1432/1688 [07:26<01:18,  3.26it/s]
85%|████████▍ | 1433/1688 [07:26<01:18,  3.27it/s]
85%|████████▍ | 1434/1688 [07:27<01:17,  3.27it/s]
85%|████████▌ | 1435/1688 [07:27<01:17,  3.27it/s]
85%|████████▌ | 1436/1688 [07:27<01:17,  3.25it/s]
85%|████████▌ | 1437/1688 [07:28<01:17,  3.25it/s]
85%|████████▌ | 1438/1688 [07:28<01:16,  3.26it/s]
85%|████████▌ | 1439/1688 [07:28<01:16,  3.26it/s]
85%|████████▌ | 1440/1688 [07:28<01:16,  3.25it/s]
85%|████████▌ | 1441/1688 [07:29<01:15,  3.25it/s]
85%|████████▌ | 1442/1688 [07:29<01:15,  3.26it/s]
85%|████████▌ | 1443/1688 [07:29<01:15,  3.25it/s]
86%|████████▌ | 1444/1688 [07:30<01:14,  3.26it/s]
86%|████████▌ | 1445/1688 [07:30<01:14,  3.26it/s]
86%|████████▌ | 1446/1688 [07:30<01:14,  3.26it/s]
86%|████████▌ | 1447/1688 [07:31<01:13,  3.27it/s]
86%|████████▌ | 1448/1688 [07:31<01:13,  3.27it/s]
86%|████████▌ | 1449/1688 [07:31<01:13,  3.27it/s]
86%|████████▌ | 1450/1688 [07:32<01:12,  3.27it/s]
86%|████████▌ | 1451/1688 [07:32<01:12,  3.28it/s]
86%|████████▌ | 1452/1688 [07:32<01:12,  3.26it/s]
86%|████████▌ | 1453/1688 [07:32<01:12,  3.26it/s]
86%|████████▌ | 1454/1688 [07:33<01:11,  3.25it/s]
86%|████████▌ | 1455/1688 [07:33<01:11,  3.26it/s]
86%|████████▋ | 1456/1688 [07:33<01:11,  3.26it/s]
86%|████████▋ | 1457/1688 [07:34<01:10,  3.25it/s]
86%|████████▋ | 1458/1688 [07:34<01:10,  3.25it/s]
86%|████████▋ | 1459/1688 [07:34<01:10,  3.24it/s]
86%|████████▋ | 1460/1688 [07:35<01:10,  3.25it/s]
87%|████████▋ | 1461/1688 [07:35<01:09,  3.26it/s]
87%|████████▋ | 1462/1688 [07:35<01:09,  3.26it/s]
87%|████████▋ | 1463/1688 [07:36<01:08,  3.26it/s]
87%|████████▋ | 1464/1688 [07:36<01:08,  3.27it/s]
87%|████████▋ | 1465/1688 [07:36<01:08,  3.28it/s]
87%|████████▋ | 1466/1688 [07:36<01:08,  3.26it/s]
87%|████████▋ | 1467/1688 [07:37<01:07,  3.26it/s]
87%|████████▋ | 1468/1688 [07:37<01:07,  3.27it/s]
87%|████████▋ | 1469/1688 [07:37<01:06,  3.27it/s]
87%|████████▋ | 1470/1688 [07:38<01:06,  3.26it/s]
87%|████████▋ | 1471/1688 [07:38<01:06,  3.25it/s]
87%|████████▋ | 1472/1688 [07:38<01:06,  3.24it/s]
87%|████████▋ | 1473/1688 [07:39<01:06,  3.23it/s]
87%|████████▋ | 1474/1688 [07:39<01:05,  3.25it/s]
87%|████████▋ | 1475/1688 [07:39<01:05,  3.25it/s]
87%|████████▋ | 1476/1688 [07:39<01:04,  3.26it/s]
88%|████████▊ | 1477/1688 [07:40<01:04,  3.26it/s]
88%|████████▊ | 1478/1688 [07:40<01:04,  3.25it/s]
88%|████████▊ | 1479/1688 [07:40<01:04,  3.23it/s]
88%|████████▊ | 1480/1688 [07:41<01:04,  3.24it/s]
88%|████████▊ | 1481/1688 [07:41<01:03,  3.24it/s]
88%|████████▊ | 1482/1688 [07:41<01:03,  3.23it/s]
88%|████████▊ | 1483/1688 [07:42<01:03,  3.21it/s]
88%|████████▊ | 1484/1688 [07:42<01:03,  3.22it/s]
88%|████████▊ | 1485/1688 [07:42<01:03,  3.21it/s]
88%|████████▊ | 1486/1688 [07:43<01:02,  3.22it/s]
88%|████████▊ | 1487/1688 [07:43<01:02,  3.22it/s]
88%|████████▊ | 1488/1688 [07:43<01:01,  3.24it/s]
88%|████████▊ | 1489/1688 [07:44<01:01,  3.25it/s]
88%|████████▊ | 1490/1688 [07:44<01:00,  3.26it/s]
88%|████████▊ | 1491/1688 [07:44<01:00,  3.27it/s]
88%|████████▊ | 1492/1688 [07:44<00:59,  3.27it/s]
88%|████████▊ | 1493/1688 [07:45<00:59,  3.28it/s]
89%|████████▊ | 1494/1688 [07:45<00:59,  3.27it/s]
89%|████████▊ | 1495/1688 [07:45<00:58,  3.27it/s]
89%|████████▊ | 1496/1688 [07:46<00:58,  3.27it/s]
89%|████████▊ | 1497/1688 [07:46<00:58,  3.28it/s]
89%|████████▊ | 1498/1688 [07:46<00:58,  3.28it/s]
89%|████████▉ | 1499/1688 [07:47<00:57,  3.28it/s]
89%|████████▉ | 1500/1688 [07:47<00:57,  3.28it/s]
{'loss': 1.3394, 'learning_rate': 3.341232227488152e-06, 'epoch': 0.89}
89%|████████▉ | 1500/1688 [07:47<00:57,  3.28it/s]
[INFO|trainer.py:2139] 2022-05-12 09:16:09,889 >> Saving model checkpoint to /opt/ml/model/chinese_roberta/checkpoint-1500
[INFO|trainer.py:2139] 2022-05-12 09:16:09,889 >> Saving model checkpoint to /opt/ml/model/chinese_roberta/checkpoint-1500
[INFO|configuration_utils.py:439] 2022-05-12 09:16:09,890 >> Configuration saved in /opt/ml/model/chinese_roberta/checkpoint-1500/config.json
[INFO|configuration_utils.py:439] 2022-05-12 09:16:09,890 >> Configuration saved in /opt/ml/model/chinese_roberta/checkpoint-1500/config.json
[INFO|modeling_utils.py:1084] 2022-05-12 09:16:10,327 >> Model weights saved in /opt/ml/model/chinese_roberta/checkpoint-1500/pytorch_model.bin
[INFO|modeling_utils.py:1084] 2022-05-12 09:16:10,327 >> Model weights saved in /opt/ml/model/chinese_roberta/checkpoint-1500/pytorch_model.bin
[INFO|tokenization_utils_base.py:2094] 2022-05-12 09:16:10,327 >> tokenizer config file saved in /opt/ml/model/chinese_roberta/checkpoint-1500/tokenizer_config.json
[INFO|tokenization_utils_base.py:2100] 2022-05-12 09:16:10,327 >> Special tokens file saved in /opt/ml/model/chinese_roberta/checkpoint-1500/special_tokens_map.json
[INFO|tokenization_utils_base.py:2094] 2022-05-12 09:16:10,327 >> tokenizer config file saved in /opt/ml/model/chinese_roberta/checkpoint-1500/tokenizer_config.json
[INFO|tokenization_utils_base.py:2100] 2022-05-12 09:16:10,327 >> Special tokens file saved in /opt/ml/model/chinese_roberta/checkpoint-1500/special_tokens_map.json
89%|████████▉ | 1501/1688 [07:49<02:13,  1.40it/s]
89%|████████▉ | 1502/1688 [07:49<01:50,  1.69it/s]
89%|████████▉ | 1503/1688 [07:49<01:33,  1.98it/s]
89%|████████▉ | 1504/1688 [07:49<01:22,  2.24it/s]
89%|████████▉ | 1505/1688 [07:50<01:13,  2.48it/s]
89%|████████▉ | 1506/1688 [07:50<01:08,  2.67it/s]
89%|████████▉ | 1507/1688 [07:50<01:04,  2.82it/s]
89%|████████▉ | 1508/1688 [07:51<01:01,  2.94it/s]
89%|████████▉ | 1509/1688 [07:51<00:59,  3.03it/s]
89%|████████▉ | 1510/1688 [07:51<00:57,  3.10it/s]
90%|████████▉ | 1511/1688 [07:52<00:56,  3.15it/s]
90%|████████▉ | 1512/1688 [07:52<00:55,  3.18it/s]
90%|████████▉ | 1513/1688 [07:52<00:54,  3.19it/s]
90%|████████▉ | 1514/1688 [07:53<00:54,  3.20it/s]
90%|████████▉ | 1515/1688 [07:53<00:53,  3.22it/s]
90%|████████▉ | 1516/1688 [07:53<00:53,  3.22it/s]
90%|████████▉ | 1517/1688 [07:53<00:53,  3.21it/s]
90%|████████▉ | 1518/1688 [07:54<00:53,  3.20it/s]
90%|████████▉ | 1519/1688 [07:54<00:52,  3.22it/s]
90%|█████████ | 1520/1688 [07:54<00:52,  3.22it/s]
90%|█████████ | 1521/1688 [07:55<00:51,  3.23it/s]
90%|█████████ | 1522/1688 [07:55<00:51,  3.23it/s]
90%|█████████ | 1523/1688 [07:55<00:51,  3.23it/s]
90%|█████████ | 1524/1688 [07:56<00:50,  3.23it/s]
90%|█████████ | 1525/1688 [07:56<00:50,  3.24it/s]
90%|█████████ | 1526/1688 [07:56<00:49,  3.24it/s]
90%|█████████ | 1527/1688 [07:57<00:49,  3.24it/s]
91%|█████████ | 1528/1688 [07:57<00:49,  3.22it/s]
91%|█████████ | 1529/1688 [07:57<00:49,  3.24it/s]
91%|█████████ | 1530/1688 [07:57<00:48,  3.24it/s]
91%|█████████ | 1531/1688 [07:58<00:48,  3.26it/s]
91%|█████████ | 1532/1688 [07:58<00:47,  3.27it/s]
91%|█████████ | 1533/1688 [07:58<00:47,  3.26it/s]
91%|█████████ | 1534/1688 [07:59<00:47,  3.27it/s]
91%|█████████ | 1535/1688 [07:59<00:46,  3.26it/s]
91%|█████████ | 1536/1688 [07:59<00:46,  3.27it/s]
91%|█████████ | 1537/1688 [08:00<00:46,  3.27it/s]
91%|█████████ | 1538/1688 [08:00<00:45,  3.26it/s]
91%|█████████ | 1539/1688 [08:00<00:45,  3.27it/s]
91%|█████████ | 1540/1688 [08:01<00:45,  3.26it/s]
91%|█████████▏| 1541/1688 [08:01<00:45,  3.25it/s]
91%|█████████▏| 1542/1688 [08:01<00:44,  3.25it/s]
91%|█████████▏| 1543/1688 [08:01<00:44,  3.26it/s]
91%|█████████▏| 1544/1688 [08:02<00:44,  3.27it/s]
92%|█████████▏| 1545/1688 [08:02<00:43,  3.27it/s]
92%|█████████▏| 1546/1688 [08:02<00:43,  3.28it/s]
92%|█████████▏| 1547/1688 [08:03<00:42,  3.28it/s]
92%|█████████▏| 1548/1688 [08:03<00:42,  3.28it/s]
92%|█████████▏| 1549/1688 [08:03<00:42,  3.27it/s]
92%|█████████▏| 1550/1688 [08:04<00:42,  3.24it/s]
92%|█████████▏| 1551/1688 [08:04<00:42,  3.25it/s]
92%|█████████▏| 1552/1688 [08:04<00:41,  3.26it/s]
92%|█████████▏| 1553/1688 [08:05<00:41,  3.26it/s]
92%|█████████▏| 1554/1688 [08:05<00:41,  3.23it/s]
92%|█████████▏| 1555/1688 [08:05<00:41,  3.21it/s]
92%|█████████▏| 1556/1688 [08:05<00:41,  3.19it/s]
92%|█████████▏| 1557/1688 [08:06<00:41,  3.18it/s]
92%|█████████▏| 1558/1688 [08:06<00:40,  3.19it/s]
92%|█████████▏| 1559/1688 [08:06<00:40,  3.20it/s]
92%|█████████▏| 1560/1688 [08:07<00:39,  3.20it/s]
92%|█████████▏| 1561/1688 [08:07<00:39,  3.21it/s]
93%|█████████▎| 1562/1688 [08:07<00:39,  3.22it/s]
93%|█████████▎| 1563/1688 [08:08<00:38,  3.23it/s]
93%|█████████▎| 1564/1688 [08:08<00:38,  3.23it/s]
93%|█████████▎| 1565/1688 [08:08<00:37,  3.24it/s]
93%|█████████▎| 1566/1688 [08:09<00:37,  3.23it/s]
93%|█████████▎| 1567/1688 [08:09<00:37,  3.24it/s]
93%|█████████▎| 1568/1688 [08:09<00:37,  3.24it/s]
93%|█████████▎| 1569/1688 [08:09<00:36,  3.25it/s]
93%|█████████▎| 1570/1688 [08:10<00:36,  3.24it/s]
93%|█████████▎| 1571/1688 [08:10<00:36,  3.23it/s]
93%|█████████▎| 1572/1688 [08:10<00:36,  3.22it/s]
93%|█████████▎| 1573/1688 [08:11<00:35,  3.21it/s]
93%|█████████▎| 1574/1688 [08:11<00:35,  3.20it/s]
93%|█████████▎| 1575/1688 [08:11<00:35,  3.22it/s]
93%|█████████▎| 1576/1688 [08:12<00:34,  3.23it/s]
93%|█████████▎| 1577/1688 [08:12<00:34,  3.25it/s]
93%|█████████▎| 1578/1688 [08:12<00:33,  3.26it/s]
94%|█████████▎| 1579/1688 [08:13<00:33,  3.27it/s]
94%|█████████▎| 1580/1688 [08:13<00:32,  3.28it/s]
94%|█████████▎| 1581/1688 [08:13<00:32,  3.28it/s]
94%|█████████▎| 1582/1688 [08:13<00:32,  3.28it/s]
94%|█████████▍| 1583/1688 [08:14<00:32,  3.26it/s]
94%|█████████▍| 1584/1688 [08:14<00:31,  3.26it/s]
94%|█████████▍| 1585/1688 [08:14<00:31,  3.26it/s]
94%|█████████▍| 1586/1688 [08:15<00:31,  3.27it/s]
94%|█████████▍| 1587/1688 [08:15<00:31,  3.26it/s]
94%|█████████▍| 1588/1688 [08:15<00:30,  3.25it/s]
94%|█████████▍| 1589/1688 [08:16<00:30,  3.24it/s]
94%|█████████▍| 1590/1688 [08:16<00:30,  3.22it/s]
94%|█████████▍| 1591/1688 [08:16<00:30,  3.23it/s]
94%|█████████▍| 1592/1688 [08:17<00:29,  3.25it/s]
94%|█████████▍| 1593/1688 [08:17<00:29,  3.25it/s]
94%|█████████▍| 1594/1688 [08:17<00:28,  3.24it/s]
94%|█████████▍| 1595/1688 [08:18<00:28,  3.25it/s]
95%|█████████▍| 1596/1688 [08:18<00:28,  3.26it/s]
95%|█████████▍| 1597/1688 [08:18<00:27,  3.27it/s]
95%|█████████▍| 1598/1688 [08:18<00:27,  3.27it/s]
95%|█████████▍| 1599/1688 [08:19<00:27,  3.26it/s]
95%|█████████▍| 1600/1688 [08:19<00:26,  3.27it/s]
95%|█████████▍| 1601/1688 [08:19<00:26,  3.28it/s]
95%|█████████▍| 1602/1688 [08:20<00:26,  3.28it/s]
95%|█████████▍| 1603/1688 [08:20<00:25,  3.28it/s]
95%|█████████▌| 1604/1688 [08:20<00:25,  3.29it/s]
95%|█████████▌| 1605/1688 [08:21<00:25,  3.28it/s]
95%|█████████▌| 1606/1688 [08:21<00:24,  3.28it/s]
95%|█████████▌| 1607/1688 [08:21<00:24,  3.27it/s]
95%|█████████▌| 1608/1688 [08:21<00:24,  3.26it/s]
95%|█████████▌| 1609/1688 [08:22<00:24,  3.26it/s]
95%|█████████▌| 1610/1688 [08:22<00:23,  3.26it/s]
95%|█████████▌| 1611/1688 [08:22<00:23,  3.24it/s]
95%|█████████▌| 1612/1688 [08:23<00:23,  3.24it/s]
96%|█████████▌| 1613/1688 [08:23<00:23,  3.23it/s]
96%|█████████▌| 1614/1688 [08:23<00:22,  3.24it/s]
96%|█████████▌| 1615/1688 [08:24<00:22,  3.23it/s]
96%|█████████▌| 1616/1688 [08:24<00:22,  3.23it/s]
96%|█████████▌| 1617/1688 [08:24<00:21,  3.25it/s]
96%|█████████▌| 1618/1688 [08:25<00:21,  3.26it/s]
96%|█████████▌| 1619/1688 [08:25<00:21,  3.24it/s]
96%|█████████▌| 1620/1688 [08:25<00:21,  3.24it/s]
96%|█████████▌| 1621/1688 [08:25<00:20,  3.23it/s]
96%|█████████▌| 1622/1688 [08:26<00:20,  3.23it/s]
96%|█████████▌| 1623/1688 [08:26<00:20,  3.25it/s]
96%|█████████▌| 1624/1688 [08:26<00:19,  3.26it/s]
96%|█████████▋| 1625/1688 [08:27<00:19,  3.25it/s]
96%|█████████▋| 1626/1688 [08:27<00:19,  3.24it/s]
96%|█████████▋| 1627/1688 [08:27<00:18,  3.25it/s]
96%|█████████▋| 1628/1688 [08:28<00:18,  3.25it/s]
97%|█████████▋| 1629/1688 [08:28<00:18,  3.24it/s]
97%|█████████▋| 1630/1688 [08:28<00:17,  3.25it/s]
97%|█████████▋| 1631/1688 [08:29<00:17,  3.25it/s]
97%|█████████▋| 1632/1688 [08:29<00:17,  3.26it/s]
97%|█████████▋| 1633/1688 [08:29<00:16,  3.26it/s]
97%|█████████▋| 1634/1688 [08:29<00:16,  3.26it/s]
97%|█████████▋| 1635/1688 [08:30<00:16,  3.24it/s]
97%|█████████▋| 1636/1688 [08:30<00:16,  3.24it/s]
97%|█████████▋| 1637/1688 [08:30<00:15,  3.25it/s]
97%|█████████▋| 1638/1688 [08:31<00:15,  3.25it/s]
97%|█████████▋| 1639/1688 [08:31<00:15,  3.25it/s]
97%|█████████▋| 1640/1688 [08:31<00:14,  3.25it/s]
97%|█████████▋| 1641/1688 [08:32<00:14,  3.26it/s]
97%|█████████▋| 1642/1688 [08:32<00:14,  3.26it/s]
97%|█████████▋| 1643/1688 [08:32<00:13,  3.26it/s]
97%|█████████▋| 1644/1688 [08:33<00:13,  3.26it/s]
97%|█████████▋| 1645/1688 [08:33<00:13,  3.26it/s]
98%|█████████▊| 1646/1688 [08:33<00:12,  3.25it/s]
98%|█████████▊| 1647/1688 [08:33<00:12,  3.25it/s]
98%|█████████▊| 1648/1688 [08:34<00:12,  3.23it/s]
98%|█████████▊| 1649/1688 [08:34<00:12,  3.21it/s]
98%|█████████▊| 1650/1688 [08:34<00:11,  3.21it/s]
98%|█████████▊| 1651/1688 [08:35<00:11,  3.22it/s]
98%|█████████▊| 1652/1688 [08:35<00:11,  3.22it/s]
98%|█████████▊| 1653/1688 [08:35<00:10,  3.24it/s]
98%|█████████▊| 1654/1688 [08:36<00:10,  3.25it/s]
98%|█████████▊| 1655/1688 [08:36<00:10,  3.24it/s]
98%|█████████▊| 1656/1688 [08:36<00:09,  3.22it/s]
98%|█████████▊| 1657/1688 [08:37<00:09,  3.24it/s]
98%|█████████▊| 1658/1688 [08:37<00:09,  3.23it/s]
98%|█████████▊| 1659/1688 [08:37<00:08,  3.24it/s]
98%|█████████▊| 1660/1688 [08:38<00:08,  3.24it/s]
98%|█████████▊| 1661/1688 [08:38<00:08,  3.24it/s]
98%|█████████▊| 1662/1688 [08:38<00:07,  3.25it/s]
99%|█████████▊| 1663/1688 [08:38<00:07,  3.24it/s]
99%|█████████▊| 1664/1688 [08:39<00:07,  3.23it/s]
99%|█████████▊| 1665/1688 [08:39<00:07,  3.23it/s]
99%|█████████▊| 1666/1688 [08:39<00:06,  3.22it/s]
99%|█████████▉| 1667/1688 [08:40<00:06,  3.24it/s]
99%|█████████▉| 1668/1688 [08:40<00:06,  3.25it/s]
99%|█████████▉| 1669/1688 [08:40<00:05,  3.25it/s]
99%|█████████▉| 1670/1688 [08:41<00:05,  3.24it/s]
99%|█████████▉| 1671/1688 [08:41<00:05,  3.25it/s]
99%|█████████▉| 1672/1688 [08:41<00:04,  3.25it/s]
99%|█████████▉| 1673/1688 [08:42<00:04,  3.23it/s]
99%|█████████▉| 1674/1688 [08:42<00:04,  3.21it/s]
99%|█████████▉| 1675/1688 [08:42<00:04,  3.20it/s]
99%|█████████▉| 1676/1688 [08:42<00:03,  3.20it/s]
99%|█████████▉| 1677/1688 [08:43<00:03,  3.21it/s]
99%|█████████▉| 1678/1688 [08:43<00:03,  3.23it/s]
99%|█████████▉| 1679/1688 [08:43<00:02,  3.22it/s]
100%|█████████▉| 1680/1688 [08:44<00:02,  3.23it/s]
100%|█████████▉| 1681/1688 [08:44<00:02,  3.25it/s]
100%|█████████▉| 1682/1688 [08:44<00:01,  3.25it/s]
100%|█████████▉| 1683/1688 [08:45<00:01,  3.25it/s]
100%|█████████▉| 1684/1688 [08:45<00:01,  3.26it/s]
100%|█████████▉| 1685/1688 [08:45<00:00,  3.25it/s]
100%|█████████▉| 1686/1688 [08:46<00:00,  3.25it/s]
100%|█████████▉| 1687/1688 [08:46<00:00,  3.24it/s]
100%|██████████| 1688/1688 [08:46<00:00,  4.00it/s]
[INFO|trainer.py:1508] 2022-05-12 09:17:08,985 >> 
Training completed. Do not forget to share your model on huggingface.co/models =)
[INFO|trainer.py:1508] 2022-05-12 09:17:08,985 >> 
Training completed. Do not forget to share your model on huggingface.co/models =)
{'train_runtime': 526.4766, 'train_samples_per_second': 205.109, 'train_steps_per_second': 3.206, 'train_loss': 1.392846428387538, 'epoch': 1.0}
100%|██████████| 1688/1688 [08:46<00:00,  4.00it/s]
100%|██████████| 1688/1688 [08:46<00:00,  3.21it/s]
[INFO|trainer.py:2139] 2022-05-12 09:17:08,988 >> Saving model checkpoint to /opt/ml/model/chinese_roberta
[INFO|trainer.py:2139] 2022-05-12 09:17:08,988 >> Saving model checkpoint to /opt/ml/model/chinese_roberta
[INFO|configuration_utils.py:439] 2022-05-12 09:17:08,989 >> Configuration saved in /opt/ml/model/chinese_roberta/config.json
[INFO|configuration_utils.py:439] 2022-05-12 09:17:08,989 >> Configuration saved in /opt/ml/model/chinese_roberta/config.json
[INFO|modeling_utils.py:1084] 2022-05-12 09:17:09,440 >> Model weights saved in /opt/ml/model/chinese_roberta/pytorch_model.bin
[INFO|modeling_utils.py:1084] 2022-05-12 09:17:09,440 >> Model weights saved in /opt/ml/model/chinese_roberta/pytorch_model.bin
[INFO|tokenization_utils_base.py:2094] 2022-05-12 09:17:09,440 >> tokenizer config file saved in /opt/ml/model/chinese_roberta/tokenizer_config.json
[INFO|tokenization_utils_base.py:2094] 2022-05-12 09:17:09,440 >> tokenizer config file saved in /opt/ml/model/chinese_roberta/tokenizer_config.json
[INFO|tokenization_utils_base.py:2100] 2022-05-12 09:17:09,440 >> Special tokens file saved in /opt/ml/model/chinese_roberta/special_tokens_map.json
[INFO|tokenization_utils_base.py:2100] 2022-05-12 09:17:09,440 >> Special tokens file saved in /opt/ml/model/chinese_roberta/special_tokens_map.json
***** train metrics *****
  epoch                    =        1.0
  train_loss               =     1.3928
train_runtime            = 0:08:46.47
  train_samples            =     107985
  train_samples_per_second =    205.109
  train_steps_per_second   =      3.206
05/12/2022 09:17:09 - INFO - __main__ - *** Evaluate ***
[INFO|trainer.py:570] 2022-05-12 09:17:09,469 >> The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
[INFO|trainer.py:570] 2022-05-12 09:17:09,469 >> The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
[INFO|trainer.py:2389] 2022-05-12 09:17:09,471 >> ***** Running Evaluation *****
[INFO|trainer.py:2389] 2022-05-12 09:17:09,471 >> ***** Running Evaluation *****
[INFO|trainer.py:2391] 2022-05-12 09:17:09,471 >>   Num examples = 11999
[INFO|trainer.py:2394] 2022-05-12 09:17:09,472 >>   Batch size = 8
[INFO|trainer.py:2391] 2022-05-12 09:17:09,471 >>   Num examples = 11999
[INFO|trainer.py:2394] 2022-05-12 09:17:09,472 >>   Batch size = 8
0%|          | 0/1500 [00:00<?, ?it/s]
0%|          | 6/1500 [00:00<00:28, 52.88it/s]
1%|          | 12/1500 [00:00<00:31, 47.19it/s]
1%|          | 17/1500 [00:00<00:31, 46.85it/s]
1%|▏         | 22/1500 [00:00<00:32, 45.74it/s]
2%|▏         | 27/1500 [00:00<00:32, 45.09it/s]
2%|▏         | 32/1500 [00:00<00:32, 45.17it/s]
2%|▏         | 37/1500 [00:00<00:32, 44.38it/s]
3%|▎         | 42/1500 [00:00<00:32, 44.53it/s]
3%|▎         | 47/1500 [00:01<00:32, 44.42it/s]
3%|▎         | 52/1500 [00:01<00:32, 44.68it/s]
4%|▍         | 57/1500 [00:01<00:32, 44.71it/s]
4%|▍         | 62/1500 [00:01<00:32, 44.55it/s]
4%|▍         | 67/1500 [00:01<00:32, 44.50it/s]
5%|▍         | 72/1500 [00:01<00:31, 44.79it/s]
5%|▌         | 77/1500 [00:01<00:31, 45.05it/s]
5%|▌         | 82/1500 [00:01<00:32, 44.10it/s]
6%|▌         | 87/1500 [00:01<00:32, 44.14it/s]
6%|▌         | 92/1500 [00:02<00:32, 43.66it/s]
6%|▋         | 97/1500 [00:02<00:31, 44.31it/s]
7%|▋         | 102/1500 [00:02<00:31, 44.33it/s]
7%|▋         | 107/1500 [00:02<00:31, 43.72it/s]
7%|▋         | 112/1500 [00:02<00:31, 44.39it/s]
8%|▊         | 117/1500 [00:02<00:31, 44.39it/s]
8%|▊         | 122/1500 [00:02<00:31, 44.04it/s]
8%|▊         | 127/1500 [00:02<00:31, 43.09it/s]
9%|▉         | 132/1500 [00:02<00:31, 43.91it/s]
9%|▉         | 137/1500 [00:03<00:31, 43.75it/s]
9%|▉         | 142/1500 [00:03<00:30, 43.84it/s]
10%|▉         | 147/1500 [00:03<00:30, 44.32it/s]
10%|█         | 152/1500 [00:03<00:30, 44.50it/s]
10%|█         | 157/1500 [00:03<00:30, 44.17it/s]
11%|█         | 162/1500 [00:03<00:30, 44.54it/s]
11%|█         | 167/1500 [00:03<00:30, 44.37it/s]
11%|█▏        | 172/1500 [00:03<00:29, 44.36it/s]
12%|█▏        | 177/1500 [00:03<00:29, 44.72it/s]
12%|█▏        | 182/1500 [00:04<00:29, 44.01it/s]
12%|█▏        | 187/1500 [00:04<00:29, 43.84it/s]
13%|█▎        | 192/1500 [00:04<00:29, 44.25it/s]
13%|█▎        | 197/1500 [00:04<00:29, 43.91it/s]
13%|█▎        | 202/1500 [00:04<00:29, 44.22it/s]
14%|█▍        | 207/1500 [00:04<00:29, 44.36it/s]
14%|█▍        | 212/1500 [00:04<00:28, 44.42it/s]
14%|█▍        | 217/1500 [00:04<00:28, 44.70it/s]
15%|█▍        | 222/1500 [00:04<00:28, 45.01it/s]
15%|█▌        | 227/1500 [00:05<00:28, 45.01it/s]
15%|█▌        | 232/1500 [00:05<00:33, 37.76it/s]
16%|█▌        | 237/1500 [00:05<00:31, 40.26it/s]
16%|█▌        | 242/1500 [00:05<00:30, 41.53it/s]
16%|█▋        | 247/1500 [00:05<00:29, 41.94it/s]
17%|█▋        | 252/1500 [00:05<00:29, 42.21it/s]
17%|█▋        | 257/1500 [00:05<00:29, 42.14it/s]
17%|█▋        | 262/1500 [00:05<00:28, 42.92it/s]
18%|█▊        | 267/1500 [00:06<00:28, 43.54it/s]
18%|█▊        | 272/1500 [00:06<00:27, 44.06it/s]
18%|█▊        | 277/1500 [00:06<00:27, 44.24it/s]
19%|█▉        | 282/1500 [00:06<00:27, 44.39it/s]
19%|█▉        | 287/1500 [00:06<00:27, 44.09it/s]
19%|█▉        | 292/1500 [00:06<00:27, 44.35it/s]
20%|█▉        | 297/1500 [00:06<00:27, 44.35it/s]
20%|██        | 302/1500 [00:06<00:27, 44.09it/s]
20%|██        | 307/1500 [00:06<00:27, 43.49it/s]
21%|██        | 312/1500 [00:07<00:26, 44.23it/s]
21%|██        | 317/1500 [00:07<00:27, 43.70it/s]
21%|██▏       | 322/1500 [00:07<00:27, 43.37it/s]
22%|██▏       | 327/1500 [00:07<00:26, 43.62it/s]
22%|██▏       | 332/1500 [00:07<00:26, 43.65it/s]
22%|██▏       | 337/1500 [00:07<00:26, 43.92it/s]
23%|██▎       | 342/1500 [00:07<00:26, 44.18it/s]
23%|██▎       | 347/1500 [00:07<00:26, 43.09it/s]
23%|██▎       | 352/1500 [00:08<00:26, 43.69it/s]
24%|██▍       | 357/1500 [00:08<00:26, 43.93it/s]
24%|██▍       | 362/1500 [00:08<00:25, 43.85it/s]
24%|██▍       | 367/1500 [00:08<00:25, 44.12it/s]
25%|██▍       | 372/1500 [00:08<00:25, 44.62it/s]
25%|██▌       | 377/1500 [00:08<00:25, 44.76it/s]
25%|██▌       | 382/1500 [00:08<00:25, 44.69it/s]
26%|██▌       | 387/1500 [00:08<00:24, 44.58it/s]
26%|██▌       | 392/1500 [00:08<00:24, 44.78it/s]
26%|██▋       | 397/1500 [00:09<00:24, 44.97it/s]
27%|██▋       | 402/1500 [00:09<00:24, 45.20it/s]
27%|██▋       | 407/1500 [00:09<00:24, 45.07it/s]
27%|██▋       | 412/1500 [00:09<00:24, 44.47it/s]
28%|██▊       | 417/1500 [00:09<00:24, 44.64it/s]
28%|██▊       | 422/1500 [00:09<00:24, 44.51it/s]
28%|██▊       | 427/1500 [00:09<00:24, 44.30it/s]
29%|██▉       | 432/1500 [00:09<00:24, 44.36it/s]
29%|██▉       | 437/1500 [00:09<00:24, 44.06it/s]
29%|██▉       | 442/1500 [00:10<00:23, 44.32it/s]
30%|██▉       | 447/1500 [00:10<00:24, 43.30it/s]
30%|███       | 452/1500 [00:10<00:23, 43.77it/s]
30%|███       | 457/1500 [00:10<00:23, 44.24it/s]
31%|███       | 462/1500 [00:10<00:23, 44.10it/s]
31%|███       | 467/1500 [00:10<00:23, 43.89it/s]
31%|███▏      | 472/1500 [00:10<00:23, 44.09it/s]
32%|███▏      | 477/1500 [00:10<00:23, 44.39it/s]
32%|███▏      | 482/1500 [00:10<00:23, 43.31it/s]
32%|███▏      | 487/1500 [00:11<00:23, 43.40it/s]
33%|███▎      | 492/1500 [00:11<00:22, 44.06it/s]
33%|███▎      | 497/1500 [00:11<00:22, 44.41it/s]
33%|███▎      | 502/1500 [00:11<00:22, 44.66it/s]
34%|███▍      | 507/1500 [00:11<00:22, 44.69it/s]
34%|███▍      | 512/1500 [00:11<00:22, 44.86it/s]
34%|███▍      | 517/1500 [00:11<00:22, 44.54it/s]
35%|███▍      | 522/1500 [00:11<00:22, 43.99it/s]
35%|███▌      | 527/1500 [00:11<00:22, 43.59it/s]
35%|███▌      | 532/1500 [00:12<00:22, 43.43it/s]
36%|███▌      | 537/1500 [00:12<00:22, 42.15it/s]
36%|███▌      | 542/1500 [00:12<00:22, 42.62it/s]
36%|███▋      | 547/1500 [00:12<00:22, 42.90it/s]
37%|███▋      | 552/1500 [00:12<00:21, 43.79it/s]
37%|███▋      | 557/1500 [00:12<00:21, 44.37it/s]
37%|███▋      | 562/1500 [00:12<00:21, 44.38it/s]
38%|███▊      | 567/1500 [00:12<00:21, 43.96it/s]
38%|███▊      | 572/1500 [00:12<00:21, 43.56it/s]
38%|███▊      | 577/1500 [00:13<00:20, 43.95it/s]
39%|███▉      | 582/1500 [00:13<00:20, 43.80it/s]
39%|███▉      | 587/1500 [00:13<00:21, 43.36it/s]
39%|███▉      | 592/1500 [00:13<00:20, 43.81it/s]
40%|███▉      | 597/1500 [00:13<00:20, 44.28it/s]
40%|████      | 602/1500 [00:13<00:20, 44.24it/s]
40%|████      | 607/1500 [00:13<00:20, 44.04it/s]
41%|████      | 612/1500 [00:13<00:20, 44.23it/s]
41%|████      | 617/1500 [00:14<00:19, 44.32it/s]
41%|████▏     | 622/1500 [00:14<00:19, 44.74it/s]
42%|████▏     | 627/1500 [00:14<00:19, 45.08it/s]
42%|████▏     | 632/1500 [00:14<00:19, 44.75it/s]
42%|████▏     | 637/1500 [00:14<00:19, 44.69it/s]
43%|████▎     | 642/1500 [00:14<00:19, 44.47it/s]
43%|████▎     | 647/1500 [00:14<00:19, 44.63it/s]
43%|████▎     | 652/1500 [00:14<00:18, 45.01it/s]
44%|████▍     | 657/1500 [00:14<00:19, 44.15it/s]
44%|████▍     | 662/1500 [00:15<00:18, 44.17it/s]
44%|████▍     | 667/1500 [00:15<00:18, 44.32it/s]
45%|████▍     | 672/1500 [00:15<00:18, 44.05it/s]
45%|████▌     | 677/1500 [00:15<00:18, 44.08it/s]
45%|████▌     | 682/1500 [00:15<00:18, 44.10it/s]
46%|████▌     | 687/1500 [00:15<00:18, 44.06it/s]
46%|████▌     | 692/1500 [00:15<00:18, 44.34it/s]
46%|████▋     | 697/1500 [00:15<00:18, 44.37it/s]
47%|████▋     | 702/1500 [00:15<00:17, 44.35it/s]
47%|████▋     | 707/1500 [00:16<00:17, 44.87it/s]
47%|████▋     | 712/1500 [00:16<00:17, 44.21it/s]
48%|████▊     | 717/1500 [00:16<00:17, 43.96it/s]
48%|████▊     | 722/1500 [00:16<00:17, 43.96it/s]
48%|████▊     | 727/1500 [00:16<00:17, 43.83it/s]
49%|████▉     | 732/1500 [00:16<00:17, 43.80it/s]
49%|████▉     | 737/1500 [00:16<00:17, 43.98it/s]
49%|████▉     | 742/1500 [00:16<00:16, 44.67it/s]
50%|████▉     | 747/1500 [00:16<00:16, 44.79it/s]
50%|█████     | 752/1500 [00:17<00:16, 45.11it/s]
50%|█████     | 757/1500 [00:17<00:16, 44.38it/s]
51%|█████     | 762/1500 [00:17<00:16, 44.63it/s]
51%|█████     | 767/1500 [00:17<00:16, 44.19it/s]
51%|█████▏    | 772/1500 [00:17<00:16, 43.86it/s]
52%|█████▏    | 777/1500 [00:17<00:16, 43.98it/s]
52%|█████▏    | 782/1500 [00:17<00:16, 43.86it/s]
52%|█████▏    | 787/1500 [00:17<00:16, 43.62it/s]
53%|█████▎    | 792/1500 [00:17<00:16, 42.67it/s]
53%|█████▎    | 797/1500 [00:18<00:16, 42.66it/s]
53%|█████▎    | 802/1500 [00:18<00:16, 43.23it/s]
54%|█████▍    | 807/1500 [00:18<00:15, 43.38it/s]
54%|█████▍    | 812/1500 [00:18<00:15, 44.09it/s]
54%|█████▍    | 817/1500 [00:18<00:15, 42.79it/s]
55%|█████▍    | 822/1500 [00:18<00:16, 41.87it/s]
55%|█████▌    | 827/1500 [00:18<00:15, 42.41it/s]
55%|█████▌    | 832/1500 [00:18<00:15, 43.47it/s]
56%|█████▌    | 837/1500 [00:19<00:15, 43.87it/s]
56%|█████▌    | 842/1500 [00:19<00:14, 44.16it/s]
56%|█████▋    | 847/1500 [00:19<00:14, 43.95it/s]
57%|█████▋    | 852/1500 [00:19<00:14, 44.04it/s]
57%|█████▋    | 857/1500 [00:19<00:14, 44.32it/s]
57%|█████▋    | 862/1500 [00:19<00:14, 44.65it/s]
58%|█████▊    | 867/1500 [00:19<00:14, 44.68it/s]
58%|█████▊    | 872/1500 [00:19<00:14, 44.79it/s]
58%|█████▊    | 877/1500 [00:19<00:13, 44.75it/s]
59%|█████▉    | 882/1500 [00:20<00:13, 44.58it/s]
59%|█████▉    | 887/1500 [00:20<00:13, 44.89it/s]
59%|█████▉    | 892/1500 [00:20<00:13, 45.26it/s]
60%|█████▉    | 897/1500 [00:20<00:13, 45.36it/s]
60%|██████    | 902/1500 [00:20<00:13, 45.34it/s]
60%|██████    | 907/1500 [00:20<00:13, 44.74it/s]
61%|██████    | 912/1500 [00:20<00:13, 44.62it/s]
61%|██████    | 917/1500 [00:20<00:13, 44.51it/s]
61%|██████▏   | 922/1500 [00:20<00:12, 44.79it/s]
62%|██████▏   | 927/1500 [00:21<00:13, 43.85it/s]
62%|██████▏   | 932/1500 [00:21<00:13, 43.60it/s]
62%|██████▏   | 937/1500 [00:21<00:12, 44.19it/s]
63%|██████▎   | 942/1500 [00:21<00:12, 44.13it/s]
63%|██████▎   | 947/1500 [00:21<00:12, 43.58it/s]
63%|██████▎   | 952/1500 [00:21<00:12, 43.87it/s]
64%|██████▍   | 957/1500 [00:21<00:12, 44.19it/s]
64%|██████▍   | 962/1500 [00:21<00:12, 44.47it/s]
64%|██████▍   | 967/1500 [00:21<00:12, 44.12it/s]
65%|██████▍   | 972/1500 [00:22<00:12, 43.60it/s]
65%|██████▌   | 977/1500 [00:22<00:11, 44.04it/s]
65%|██████▌   | 982/1500 [00:22<00:11, 44.50it/s]
66%|██████▌   | 987/1500 [00:22<00:11, 43.56it/s]
66%|██████▌   | 992/1500 [00:22<00:11, 42.83it/s]
66%|██████▋   | 997/1500 [00:22<00:11, 43.42it/s]
67%|██████▋   | 1002/1500 [00:22<00:11, 43.37it/s]
67%|██████▋   | 1007/1500 [00:22<00:11, 44.04it/s]
67%|██████▋   | 1012/1500 [00:22<00:11, 43.72it/s]
68%|██████▊   | 1017/1500 [00:23<00:10, 44.00it/s]
68%|██████▊   | 1022/1500 [00:23<00:10, 44.09it/s]
68%|██████▊   | 1027/1500 [00:23<00:10, 43.75it/s]
69%|██████▉   | 1032/1500 [00:23<00:10, 44.32it/s]
69%|██████▉   | 1037/1500 [00:23<00:10, 44.06it/s]
69%|██████▉   | 1042/1500 [00:23<00:10, 44.10it/s]
70%|██████▉   | 1047/1500 [00:23<00:10, 42.62it/s]
70%|███████   | 1052/1500 [00:23<00:10, 41.74it/s]
70%|███████   | 1057/1500 [00:24<00:10, 42.34it/s]
71%|███████   | 1062/1500 [00:24<00:10, 43.02it/s]
71%|███████   | 1067/1500 [00:24<00:10, 43.01it/s]
71%|███████▏  | 1072/1500 [00:24<00:09, 43.71it/s]
72%|███████▏  | 1077/1500 [00:24<00:09, 44.28it/s]
72%|███████▏  | 1082/1500 [00:24<00:09, 43.85it/s]
72%|███████▏  | 1087/1500 [00:24<00:09, 44.24it/s]
73%|███████▎  | 1092/1500 [00:24<00:09, 44.74it/s]
73%|███████▎  | 1097/1500 [00:24<00:08, 44.79it/s]
73%|███████▎  | 1102/1500 [00:25<00:08, 44.32it/s]
74%|███████▍  | 1107/1500 [00:25<00:08, 44.55it/s]
74%|███████▍  | 1112/1500 [00:25<00:08, 44.77it/s]
74%|███████▍  | 1117/1500 [00:25<00:08, 44.88it/s]
75%|███████▍  | 1122/1500 [00:25<00:08, 44.95it/s]
75%|███████▌  | 1127/1500 [00:25<00:08, 45.26it/s]
75%|███████▌  | 1132/1500 [00:25<00:08, 45.30it/s]
76%|███████▌  | 1137/1500 [00:25<00:08, 45.22it/s]
76%|███████▌  | 1142/1500 [00:25<00:07, 45.30it/s]
76%|███████▋  | 1147/1500 [00:26<00:07, 45.22it/s]
77%|███████▋  | 1152/1500 [00:26<00:07, 45.73it/s]
77%|███████▋  | 1157/1500 [00:26<00:07, 45.74it/s]
77%|███████▋  | 1162/1500 [00:26<00:07, 45.59it/s]
78%|███████▊  | 1167/1500 [00:26<00:07, 45.56it/s]
78%|███████▊  | 1172/1500 [00:26<00:07, 44.87it/s]
78%|███████▊  | 1177/1500 [00:26<00:07, 44.75it/s]
79%|███████▉  | 1182/1500 [00:26<00:07, 44.90it/s]
79%|███████▉  | 1187/1500 [00:26<00:06, 44.99it/s]
79%|███████▉  | 1192/1500 [00:27<00:06, 44.63it/s]
80%|███████▉  | 1197/1500 [00:27<00:06, 43.77it/s]
80%|████████  | 1202/1500 [00:27<00:06, 44.07it/s]
80%|████████  | 1207/1500 [00:27<00:06, 43.50it/s]
81%|████████  | 1212/1500 [00:27<00:06, 44.29it/s]
81%|████████  | 1217/1500 [00:27<00:06, 43.64it/s]
81%|████████▏ | 1222/1500 [00:27<00:06, 43.62it/s]
82%|████████▏ | 1227/1500 [00:27<00:06, 44.37it/s]
82%|████████▏ | 1232/1500 [00:27<00:06, 44.44it/s]
82%|████████▏ | 1237/1500 [00:28<00:06, 43.00it/s]
83%|████████▎ | 1242/1500 [00:28<00:05, 43.67it/s]
83%|████████▎ | 1247/1500 [00:28<00:05, 43.80it/s]
83%|████████▎ | 1252/1500 [00:28<00:05, 43.61it/s]
84%|████████▍ | 1257/1500 [00:28<00:05, 44.16it/s]
84%|████████▍ | 1262/1500 [00:28<00:05, 44.05it/s]
84%|████████▍ | 1267/1500 [00:28<00:05, 44.03it/s]
85%|████████▍ | 1272/1500 [00:28<00:05, 44.29it/s]
85%|████████▌ | 1277/1500 [00:28<00:05, 44.42it/s]
85%|████████▌ | 1282/1500 [00:29<00:04, 44.02it/s]
86%|████████▌ | 1287/1500 [00:29<00:04, 44.64it/s]
86%|████████▌ | 1292/1500 [00:29<00:04, 44.40it/s]
86%|████████▋ | 1297/1500 [00:29<00:04, 44.93it/s]
87%|████████▋ | 1302/1500 [00:29<00:04, 44.86it/s]
87%|████████▋ | 1307/1500 [00:29<00:04, 44.03it/s]
87%|████████▋ | 1312/1500 [00:29<00:04, 44.42it/s]
88%|████████▊ | 1317/1500 [00:29<00:04, 43.14it/s]
88%|████████▊ | 1322/1500 [00:30<00:04, 36.11it/s]
88%|████████▊ | 1327/1500 [00:30<00:04, 37.74it/s]
89%|████████▉ | 1332/1500 [00:30<00:04, 39.56it/s]
89%|████████▉ | 1337/1500 [00:30<00:03, 41.27it/s]
89%|████████▉ | 1342/1500 [00:30<00:03, 42.56it/s]
90%|████████▉ | 1347/1500 [00:30<00:03, 43.32it/s]
90%|█████████ | 1352/1500 [00:30<00:03, 43.61it/s]
90%|█████████ | 1357/1500 [00:30<00:03, 44.09it/s]
91%|█████████ | 1362/1500 [00:30<00:03, 44.60it/s]
91%|█████████ | 1367/1500 [00:31<00:02, 44.89it/s]
91%|█████████▏| 1372/1500 [00:31<00:02, 43.73it/s]
92%|█████████▏| 1377/1500 [00:31<00:02, 44.46it/s]
92%|█████████▏| 1382/1500 [00:31<00:02, 44.58it/s]
92%|█████████▏| 1387/1500 [00:31<00:02, 45.04it/s]
93%|█████████▎| 1392/1500 [00:31<00:02, 44.23it/s]
93%|█████████▎| 1397/1500 [00:31<00:02, 44.31it/s]
93%|█████████▎| 1402/1500 [00:31<00:02, 45.05it/s]
94%|█████████▍| 1407/1500 [00:31<00:02, 44.81it/s]
94%|█████████▍| 1412/1500 [00:32<00:01, 45.14it/s]
94%|█████████▍| 1417/1500 [00:32<00:01, 43.10it/s]
95%|█████████▍| 1422/1500 [00:32<00:01, 43.34it/s]
95%|█████████▌| 1427/1500 [00:32<00:01, 43.49it/s]
95%|█████████▌| 1432/1500 [00:32<00:01, 44.21it/s]
96%|█████████▌| 1437/1500 [00:32<00:01, 44.88it/s]
96%|█████████▌| 1442/1500 [00:32<00:01, 44.15it/s]
96%|█████████▋| 1447/1500 [00:32<00:01, 44.70it/s]
97%|█████████▋| 1452/1500 [00:32<00:01, 44.96it/s]
97%|█████████▋| 1457/1500 [00:33<00:00, 44.86it/s]
97%|█████████▋| 1462/1500 [00:33<00:00, 44.59it/s]
98%|█████████▊| 1467/1500 [00:33<00:00, 44.63it/s]
98%|█████████▊| 1472/1500 [00:33<00:00, 43.78it/s]
98%|█████████▊| 1477/1500 [00:33<00:00, 44.07it/s]
99%|█████████▉| 1482/1500 [00:33<00:00, 44.04it/s]
99%|█████████▉| 1487/1500 [00:33<00:00, 44.02it/s]
99%|█████████▉| 1492/1500 [00:33<00:00, 44.01it/s]
100%|█████████▉| 1497/1500 [00:33<00:00, 43.88it/s]
100%|██████████| 1500/1500 [00:34<00:00, 44.03it/s]
***** eval metrics *****
  epoch                   =        1.0
  eval_accuracy           =     0.4899
  eval_loss               =      1.316
  eval_runtime            = 0:00:34.10
  eval_samples            =      11999
  eval_samples_per_second =    351.873
  eval_steps_per_second   =     43.988
[INFO|modelcard.py:460] 2022-05-12 09:17:43,621 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Text Classification', 'type': 'text-classification'}, 'metrics': [{'name': 'Accuracy', 'type': 'accuracy', 'value': 0.4898741543292999}]}
[INFO|modelcard.py:460] 2022-05-12 09:17:43,621 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Text Classification', 'type': 'text-classification'}, 'metrics': [{'name': 'Accuracy', 'type': 'accuracy', 'value': 0.4898741543292999}]}
2022-05-12 09:17:44,511 sagemaker-training-toolkit INFO     Reporting training SUCCESS

2022-05-12 09:18:09 Uploading - Uploading generated training model
2022-05-12 09:26:12 Completed - Training job completed
Training seconds: 1399
Billable seconds: 1399