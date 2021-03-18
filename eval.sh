python eval.py --trained_model=weights/yolact_plus_resnet50_95_20000.pth --score_threshold=0.15 --top_k=15 --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/0_image_synthetic_rle_20210113:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/0_image_synthetic_rle_20210113_eval
python eval.py --trained_model=weights/yolact_plus_resnet50_95_20000.pth --score_threshold=0.15 --top_k=15 --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/20210108_1260_crop_single:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/20210108_1260_crop_single_eval_20210115

python eval.py --trained_model=weights/yolact_plus_base_93_100000.pth --score_threshold=0.15 --top_k=15 --image=/home/dell/zhanglimin/data/card/valid/v2_export_142_2020-05-14_2020-05-17/det/ori_image/0/06c260d4-b1a6-4e79-bfe7-fbb3627fe9e6_371379258.jpg --output_coco_json
python eval.py --trained_model=weights/card_weight_yolact++_20200602/yolact_plus_base_93_100000.pth --output_coco_json --dataset=card_dataset

python eval.py --trained_model=weights/yolact_plus_resnet50_158_20000.pth --score_threshold=0.15 --top_k=15 --no_crop --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_test_thickness8:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_test_thickness8_eval_20210129_nocrop
python eval.py --trained_model=weights/yolact_plus_resnet50_119_15000.pth --score_threshold=0.15 --top_k=15 --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_crop_single:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_crop_single_eval_thickness8_newanchor
python eval.py --trained_model=weights/20210129/yolact_plus_resnet50_158_20000.pth --score_threshold=0.15 --top_k=15 --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_crop_single:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_crop_single_eval_thickness8_nms0.3
python eval.py --trained_model=weights/yolact_plus_resnet50_158_20000.pth --score_threshold=0.15 --top_k=15 --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_crop_single:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_crop_single_20210128
python eval.py --trained_model=weights/yolact_plus_resnet50_158_20000.pth --score_threshold=0.15 --top_k=15 --no_crop --display_lincomb=1 --mask_proto_debug --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_crop_single:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_crop_single_eval_4 

python eval.py --trained_model=weights/yolact_plus_resnet50_198_25000.pth --score_threshold=0.15 --top_k=15 --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_crop_single:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_crop_single_eval_thickness8_pos0.7_neg0.6
python eval.py --trained_model=weights/yolact_plus_resnet50_158_20000.pth --score_threshold=0.15 --top_k=15 --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210130_446_crop_single:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210130_446_crop_single_thickness8_origin

python eval.py --trained_model=weights/yolact_plus_resnet50_198_25000.pth --score_threshold=0.15 --top_k=15 --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210205_thickness7_test:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210205_thickness7_test_eval_pos0.6_neg0.5

python eval.py  --display_text=False --trained_model=weights/yolact_plus_resnet50_146_30000.pth --score_threshold=0.15 --top_k=15 --images=/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210205_thickness7_test:/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210205_thickness7_test_eval_pos0.6_neg0.5_notext

# train
python train.py --config=yolact_plus_base_config
python train.py --config=yolact_plus_resnet50_config --num_workers=8  --batch_size=10 ----save_interval=5000 --validation_epoch=5 --no_autoscale
