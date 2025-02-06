from django.shortcuts import render
from .forms import ImageUploadForm
from ultralytics import YOLO
import cv2
import os
from django.conf import settings


# Step 1: Define Disease-Cure Mappings
DISEASE_CURE = {
    "brownspot": "Apply fungicides like Mancozeb, Copper oxychloride, or Propiconazole.",
    "Bacterial_Blight": "Use copper-based bactericides like Copper hydroxide or Streptomycin sulfate.",
    "riceblast": "Apply fungicides such as Tricyclazole, Isoprothiolane, or Azoxystrobin.",
    "Healthy-Plant": "Maintain good soil health with organic matter.",
    "unknown": "No cure available for this disease. Consult an expert.",
}


def upload_and_predict(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            uploaded_image = form.cleaned_data['image']
            image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)

            # Step 2: Load YOLO model
            model = YOLO(os.path.join('yoloapp', 'best.pt'))

            # Step 3: Perform inference
            results = model.predict(source=image_path, conf=0.2)

            # Get the predicted class (disease) names
            predicted_classes = results[0].boxes.cls  # Array of predicted class IDs
            predicted_diseases = [model.names[int(cls)] for cls in predicted_classes]

            # **Remove duplicate diseases**
            unique_diseases = list(set(predicted_diseases))

            # Step 4: Normalize disease names and determine the cure
            disease_cure_pairs = []
            for disease in unique_diseases:
                normalized_disease = disease.lower().replace("_", "").replace(" ", "")
                cure = DISEASE_CURE.get(normalized_disease, DISEASE_CURE["unknown"])
                disease_cure_pairs.append((disease, cure))

            # Save the prediction image
            prediction_image = results[0].plot()
            prediction_image_path = os.path.join(settings.MEDIA_ROOT, 'prediction.jpg')
            cv2.imwrite(prediction_image_path, prediction_image)

            # Step 5: Send prediction image and cures back to the template
            return render(request, 'yoloapp/result.html', {
                'prediction_image_url': os.path.join(settings.MEDIA_URL, 'prediction.jpg'),
                'disease_cure_pairs': disease_cure_pairs,
            })

    else:
        form = ImageUploadForm()

    return render(request, 'yoloapp/upload.html', {'form': form})
