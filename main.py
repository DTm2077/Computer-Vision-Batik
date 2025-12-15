from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
import io

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = torch.load("batik_model_full.pth", map_location=torch.device('cpu'), weights_only=False)
model.eval()

class_names = ['Aceh_Pintu_Aceh (1)', 'Bali_Barong', 'Bali_Merak', 'DKI_Ondel_Ondel', 'JawaBarat_Megamendung', 'JawaTimur_Pring', 'Jawa_Barat_Megamendung', 'Jawa_Timur_Pring', 'Kalimantan_Dayak', 'Lampung_Gajah', 'Madura_Mataketeran', 'Maluku_Pala', 'NTB_Lumbung', 'Papua_Asmat', 'Papua_Cendrawasih', 'Papua_Tifa', 'Solo_Parang', 'SulawesiSelatan_Lontara', 'Sulawesi_Selatan_Lontara', 'SumateraBarat_Rumah_Minang', 'SumateraUtara_Boraspati', 'Sumatera_Barat_Rumah_Minang', 'Sumatera_Utara_Boraspati', 'Yogyakarta_Kawung', 'Yogyakarta_Parang']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {"status": "Model Batik Ready"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probs, 0)

        return {
            "class": class_names[predicted.item()],
            "confidence": f"{confidence.item():.2%}"
        }
    except Exception as e:
        return {"error": str(e)}