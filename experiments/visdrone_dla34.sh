cd src
python train.py mot --exp_id visdrone_ft_mix_dla34 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/visdrone.json'
cd ..