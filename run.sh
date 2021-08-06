python mvpnet/test_2d.py \
    --cfg configs/scannet/unet_resnet34.yaml \
    --ckpt-path outputs/scannet/aml_unet_resnet34/model_080000.pth \
    --split val_mini

python mvpnet/test_2d_chunks.py \
    --cfg configs/scannet/unet_resnet34.yaml \
    --ckpt-path outputs/scannet/aml_unet_resnet34/model_080000.pth \
    --split val_mini \
    --cache-dir data/scannet/mvpnet \
    --image-dir data/scannet/scans_resize_160x120/

python mvpnet/test_mvpnet_3d.py \
    --cfg configs/scannet/mvpnet_3d_unet_resnet34_pn2ssg.yaml \
    --num-views 5 \
    --ckpt-path outputs/scannet/aml_mvpnet_3d_unet_resnet34_pn2ssg/model_040000.pth \
    --split val_mini \
    --cache-dir data/scannet/mvpnet \
    --image-dir data/scannet/scans_resize_160x120/ \
    MODEL_2D.CKPT_PATH outputs/scannet/aml_unet_resnet34/model_080000.pth

