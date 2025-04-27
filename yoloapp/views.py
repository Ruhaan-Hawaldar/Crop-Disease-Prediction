from django.shortcuts import render
from .forms import ImageUploadForm
from ultralytics import YOLO
import cv2
import os
from django.conf import settings
from django.contrib.auth.decorators import login_required


# Step 1: Define Disease-Cure Mappings
DISEASE_CURE = {
    "brownspot": "Apply fungicides like Mancozeb, Copper oxychloride, or Propiconazole.",
    "Bacterial_Blight": "Use copper-based bactericides like Copper hydroxide or Streptomycin sulfate.",
    "riceblast": "Apply fungicides such as Tricyclazole, Isoprothiolane, or Azoxystrobin.",
    "unknown": "Maintain good soil health with organic matter.",
}

@login_required(login_url='/yoloapp/login/')  # Redirects to login page if not logged in
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
            # model = YOLO(os.path.join('yoloapp', 'best.pt'))
            model_path = os.path.join(settings.BASE_DIR, 'yoloapp', 'static', 'model', 'best.pt')
            model = YOLO(model_path)

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


# login view 

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def user_login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('upload_and_predict')  # Redirect to the main page after login
        else:
            return render(request, 'yoloapp/login.html', {'error': "Invalid username or password"})

    return render(request, 'yoloapp/login.html')

@login_required
def user_logout(request):
    logout(request)
    return redirect('login')  # Redirect to login page after logout



from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
from django.contrib import messages

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 != password2:
            messages.error(request, "Passwords do not match!")
        elif User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken!")
        elif User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
        else:
            user = User.objects.create_user(username=username, email=email, password=password1)
            user.save()
            login(request, user)  # Automatically log in the user after registration
            return redirect('upload_and_predict')  # Redirect to the main page
        
        messages.success(request, "Account created successfully! You are now logged in.")


    return render(request, 'yoloapp/register.html')
