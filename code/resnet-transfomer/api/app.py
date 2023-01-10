import sys
sys.path.append("/home/zzengae/WangTianyin/Latex_rc/src_v12")
from http import HTTPStatus

import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from data_preprocess.my_build_vocab import Vocabulary
import data_preprocess.my_build_vocab
import pickle
from models.lit_models import Lit_Resnet_Transformer
from models.utils import get_best_checkpoint
import torch
from torchvision import transforms
from models.utils import *
app = FastAPI(
    title="Image to Latex Convert",
    desription="Convert an image of math equation into LaTex code.",
)

@app.on_event("startup")
async def load_model():
    global lit_model
    global transform
    global vocab
    
    global device
    # Device configuration
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    #加载模型
    checkpoint_path = get_best_checkpoint("/data/zzengae/tywang/save_model/math/from_MLM_pretrain_256")
    checkpoint = torch.load(checkpoint_path,map_location ='cpu')
    args = checkpoint['args']
    args.cuda_index = -1  #cpu
    #加载词典
    # vocab_pkl_file_path = "/home/zzengae/WangTianyin/Latex_rc/scr_v8/full_data/vocab.pkl"
    # with open(vocab_pkl_file_path, 'rb') as f:
    #     vocab = pickle.load(f)
    vocab_txt_file_path = "/home/zzengae/WangTianyin/Latex_rc/scr_v8/full_data/vocab.txt"
    vocab = build_vocab(vocab_txt_file_path)
    vocab.add_word('[MASK]')
    num_classes = len(vocab.word2idx)

    lit_model = Lit_Resnet_Transformer(args,d_model=256, dim_feedforward=256,
                                       nhead=4, dropout=0.2, num_decoder_layers=3,
                                       max_output_len=200, sos_index=vocab('<start>'),
                                       eos_index=vocab('<end>'), pad_index=vocab('<pad>'),
                                       unk_index=vocab('<unk>'), num_classes=num_classes)
    lit_model.models.load_state_dict(checkpoint['model_state_dict'])


@app.get("/", tags=["General"])
def read_root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict/", tags=["Prediction"])
def predict(file: UploadFile = File(...)):
    lit_model.models.eval()
    with torch.no_grad():
        image = Image.open(file.file)
        image = resize_image(image,size=256)
        # image = image.to(device)
        # image_tensor = transform(image=np.array(image))["image"]  # type: ignore
        image = transform(image)
        pred = lit_model.models.predict(image.unsqueeze(0).float())
        pred = pred.cpu().numpy()# type: ignore
        decoded = decode(vocab, pred[0].tolist())  # type: ignore
        # print(pred.tolist())
        decoded_str = " ".join(decoded)
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {"pred": decoded_str},
        }
        return response
