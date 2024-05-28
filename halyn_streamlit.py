import streamlit as st
from PIL import Image
import torch
import time
import numpy as np
import torchxrayvision as xrv
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from torchvision.transforms import transforms

# 제목 설정
st.title("🫁 Deep Chest Doctor🩺🧑🏻‍⚕️👩🏻‍⚕️")

# 사이드바에 파일 업로드 위젯 추가
uploaded_file = st.sidebar.file_uploader("Choose a chest X-ray image...", type=["png", "jpg", "jpeg"])

### 여기에 모델 로드하기
vision_model = xrv.models.DenseNet(weights="densenet121-res224-all")
# model_path = "alpaca-lora/24040529"  # 얘가 최근 모델인데 잘 안되는듯하기도하고.. 모르겠슴
model_path = "alpaca-lora/lora-alpaca/checkpoint-2000"
model = AutoModelForCausalLM.from_pretrained(model_path)

### 여기에 이미지 임베딩 함수 처리를 했는데 이상해여
def embed_visual_data(image,vision_model):
    image = np.array(image)
    image_tensor = torch.from_numpy(image).float()  

    if image_tensor.ndim == 3 and image_tensor.shape[0] != 3: 
        image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = (image_tensor + 1024) / 2048


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor()  
    ])

    # Apply the transform
    image_tensor = transform(image_tensor)

    # Unsqueeze to add the batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor[:,0,:,:]
    image_tensor = image_tensor.unsqueeze(0)
    feats = vision_model.features(image)
    feats = F.relu(feats, inplace=True)
    feats = F.adaptive_avg_pool2d(feats, (1, 1))
    feats = feats.reshape(-1)
    return feats

if uploaded_file is not None:
    # 이미지를 열고 화면에 표시
    image = Image.open(uploaded_file)

    # 레이아웃 설정
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Chest X-ray.', use_column_width=True)

    with col2:
        st.write("Classifying...")
        start_time = time.time()

        # 이미지를 모델 입력 형식에 맞게 전처리
        inputs = embed_visual_data(image, vision_model)
        print(inputs)

        with torch.no_grad():
            outputs = model(**inputs)

        
"""
결과 표시하는 부분도 짜야할것같습니당
이정이가 짜놓은게 CLIP/MedCLIP특화된 느낌으로 prompt가 나오는거라서,
우리 모델의 결과로 어떻게 답변이 나올지 다시 짜야할것같은데 머리가 안돌아가여

"""


        