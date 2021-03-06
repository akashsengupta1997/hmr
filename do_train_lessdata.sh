# TODO: Replace with where you downloaded your resnet_v2_50.
PRETRAINED=~/Documents/Github/hmr/models/resnet_v2_50/resnet_v2_50.ckpt
# TODO: Replace with where you generated tf_record!
DATA_DIR=~/Documents/Github/hmr/tf_datasets/

CMD="python -m src.main --d_lr 1e-4 --e_lr 1e-5 --log_img_step 1000 --pretrained_model_path=${PRETRAINED} --data_dir ${DATA_DIR} --e_loss_weight 60. --batch_size=64 --e_3d_weight 60. --datasets lsp,lsp_ext,mpii --epoch 75 --log_dir logs"

# To pick up training/training from a previous model, set LP
# LP='logs/<WITH_YOUR_TRAINED_MODEL>'
# CMD="python -m src.main --d_lr 1e-4 --e_lr 1e-5 --log_img_step 1000 --load_path=${LP} --e_loss_weight 60. --batch_size=64 --e_3d_weight 60. --datasets lsp lsp_ext mpii --epoch 75"

echo $CMD
$CMD

