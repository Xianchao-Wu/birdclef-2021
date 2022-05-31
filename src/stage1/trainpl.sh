# root@22c9bba09d41:/workspace/asr/birdclef-2021/logs/stage1# tree resnet34/
#resnet34/
#├── fold0, fold1, fold2, fold3
#│   └── lightning_logs
#│       ├── version_0
#│       │   ├── checkpoints
#│       │   │   ├── best_f1.ckpt
#│       │   │   └── best_loss.ckpt
# copy a100's result to dgx1!


python make_pseudolabel_debug.py --weight_stage1 resnet34


#xianchaow@dgxa100jp:/raid/xianchaow/asr/birdclef-2021/logs/stage1/repvgg_b0$ tree
#.
#├── fold0
#│   └── lightning_logs
#│       └── version_0
#│           ├── checkpoints
#│           │   ├── best_f1.ckpt
#│           │   └── best_loss.ckpt
# copy a100's result to dgx1!

python make_pseudolabel_debug.py --weight_stage1 repvgg_b0
