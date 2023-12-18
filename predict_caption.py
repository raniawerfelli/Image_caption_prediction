from transformers import pipeline, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
# Chargement des modèles 
model = VisionEncoderDecoderModel.from_pretrained("vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("vit-gpt2-image-captioning")
# Chargement des modèles de traduction
translator_fr = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-fr")
translator_ar = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-ar")
#disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#Paramètres du caption
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

#Fonction de prédiction pour générer la descrip et effectuer des traductions
def predict_step(image):
    
    pixel_values = feature_extractor(
        images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    # Translations
    french_translation = translator_fr(preds[0], max_length=max_length)[0]['translation_text']
    arabic_translation =  translator_ar(preds[0], max_length=max_length)[0]['translation_text']
    return preds[0], french_translation, arabic_translation
#chargement d'image(teste)    
image_path = "images.jpg"
image = Image.open(image_path)
#appel de la fonction
preds, french_translation, arabic_translation = predict_step(image)
#affichage de resultat
print("Final Caption is: ", preds)
print("French Translation: ", french_translation)
print("Arabic Translation: ", arabic_translation)