# Tacotron2-Pytorch
Support DataParallel

## Training
1. Put LJSpeech dataset in `data`
2. Run `python preprocess.py`
3. modify hyperparameters in `hparams.py`
4. Run `python train.py`

## Inference
1. Put [Nvidia pretrained waveglow model](https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing) in `waveglow/pre_trained_model`
2. Run `python inference.py`