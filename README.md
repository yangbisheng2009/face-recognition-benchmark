# face-recognition-benchmark
Easy to use.

## Useage
```
# train
python train.py --backbone mobile --batch-size 512 --train-path /home/work/faces/

# val
python val.py --which db --checkpoint checkpoints/lfw/x.pth

# predict
python predict.py -db db/lfw.db --input-images ./input-images
```

## ChangeLog
[2020-08-04] Train/val/predict recognition model.
[2020-08-04] Change train data.  

## Tricks
1. If you don't have enough faces. Fuse open source data to help the model convergence.
2. The convergence of the model is not stabel when you train it. I need to try a few time make it one.  
3. Make consin score 0.5 to be the diving line.