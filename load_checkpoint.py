import torch

# prereq: pip install -r requirements.txt
# i recommend doing this in a conda environment

# first mkdir weights
# put https://huggingface.co/Srijith-rkr/Whispering-LLaMA/tree/main content into weights/

# run the following to load the weights
a = torch.load("weights/a.pth")
b = torch.load("weights/b.pth")
c = torch.load("weights/c.pth")

# merge the weights (only need to do once)
merged = a | b | c
pretrained_path = "weights/alpaca.pth"
torch.save(merged, pretrained_path)

# after this you may run the following shell command to run inference
# python Inference/WL-M.py --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data 'path to your dataset'

# you may have to edit Inference/WL-M.py to read the dataset exactly how you want!

# lastly if you want to train the model, you can run the following shell command (from the README)
# python training/WL-S.py --lr 1e-3 --d 1 --pretrained_path 'weights/alpaca.pth' --tokenizer_path 'weights/tokenizer.model' --data 'path to your dataset'
