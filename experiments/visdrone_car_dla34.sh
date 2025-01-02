cd src
python train.py mot --exp_id visdrone_car_dla34 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/visdrone_car.json'
cd ..