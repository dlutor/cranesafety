# Environment
## FairMOT
* pytorch 1.2
## Transformer
* pytorch 1.8
# Usage
```sh
git clone https://github.com/dlutor/cranesafety.git
```
## FairMOT
Download dataset from [Datasets](https://drive.google.com/file/d/1I1ldqKXFhxF1h9yaVJdMXjvhF2i02O7F/view).

Move and extract to `./FairMOT/datasets`.

Download pretrained model from [pretrained](https://drive.google.com/file/d/1I1ldqKXFhxF1h9yaVJdMXjvhF2i02O7F/view).

Move and extract to `./FairMOT/models`.

Train
```sh
cd FairMOT
python ./src/train.py
```

Test
```sh
python demo.py
```
## Transformer
Train and test
```sh
cd Trajectory-Transformer
python train_individualTF.py
```
or `python train_newTF.py`
