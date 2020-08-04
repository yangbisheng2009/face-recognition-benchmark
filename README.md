# face-recognition-benchmark
Easy to use.

## Useage
```
# train
python train.py -p configs/lfw.yaml --batch-size 16

# val
python val.py -p configs/lfw.yaml --which db --checkpoint checkpoints/lfw/x.pth

# predict
python predict.py -p configs/lfw.yaml -db db/lfw.db --input-images ./input-images
```

## ChangeLog
[2020-08-04] Train/val/predict recognition model.  

## Tricks
1. If you don't have enough faces. Fuse open source data to help the model convergence.
2. The convergence of the model is not stabel when you train it. I need to try a few time make it one.  
3. Make consin score 0.5 to be the diving line.
