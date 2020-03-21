export PIPELINE_CONFIG_PATH=/home/lesliefang/Kannada-MNIST/objection_detection/tensorflow_models/models/ssd_mobilenet_v1_coco_2018_01_28/pipeline_jayz.config
MODEL_DIR=/home/lesliefang/Kannada-MNIST/objection_detection/tensorflow_models/mytunedmodel2
#NUM_TRAIN_STEPS=200000
#NUM_TRAIN_STEPS=4433036882696484067
NUM_TRAIN_STEPS=1000
#NUM_EVL_STEPS=500
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python /home/lesliefang/Kannada-MNIST/objection_detection/tensorflow_models/research/object_detection/model_main.py --pipeline_config_path=$PIPELINE_CONFIG_PATH --model_dir=$MODEL_DIR --num_train_steps=${NUM_TRAIN_STEPS} --sample_1_of_n_eval_examples=1 --alsologtostderr 2>&1 |tee train_jayz.log
