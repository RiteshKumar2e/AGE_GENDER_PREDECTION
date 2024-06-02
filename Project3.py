import cv2

# Load pre-trained models
age_net = cv2.dnn.readNetFromCaffe('age_deploy (2).prototxt', 'age_net (1).caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy (2).prototxt', 'gender_net (1).caffemodel')

def predict_age_and_gender(image, face_rect, age_threshold, gender_threshold):
    # Extract face from image
    x, y, w, h = face_rect
    face = image[y:y+h, x:x+w]

    # Preprocess the face
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender_confidence = gender_preds[0].max()
    gender_label = gender_preds[0].argmax() 

    # Gender prediction logic
    if gender_label == 0:  # Male
        gender_color = (0,255,0) #Green 
        gender_text = 'Male'
    else:  # Female
        gender_color = (0, 0, 255)  # Red
        gender_text = 'Female'

    # Check gender confidence
    if gender_confidence < gender_threshold:
        gender_text = 'Uncertain'
        gender_color = (255,0,0)  # Blue

    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_confidence = max(age_preds[0])
    age_index = age_preds[0].argmax()

    # Define age ranges
    age_ranges = [(i * 5, i * 5 + 4) for i in range(0, 20)]

    # Determine age range
    age_min, age_max = age_ranges[age_index]

    # Adjust age if confidence is low
    if age_confidence <= age_threshold:
        age_min, age_max = 0, 100

    age = f'{age_min}-{age_max}'

    return gender_text, gender_color, age

# Load images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']

# Process each image and display predictions for males and females
for image_path in image_paths:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Adjust the scale factor and min neighbors
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Store predictions for each face
    predictions = []

    # Predict age and gender for each face
    for (x, y, w, h) in faces:
        gender_text, gender_color, age = predict_age_and_gender(image, (x, y, w, h), age_threshold=0.3, gender_threshold=0.5)

        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), gender_color, 2)
        # Draw text with gender and age
        cv2.putText(image, f'Gender: {gender_text}', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gender_color, 2)
        cv2.putText(image, f'Age: {age}', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gender_color, 2)

        predictions.append((gender_text, age))

    # If no faces detected, draw a box and display text
    if len(faces) == 0:
        # Define coordinates for the rectangle box
        start_point = (int(image.shape[1] * 0.25), int(image.shape[0] * 0.25))  # Start point of the rectangle
        end_point = (int(image.shape[1] * 0.75), int(image.shape[0] * 0.75))    # End point of the rectangle
        cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)              # Red rectangle
        cv2.putText(image, "No face detected", (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the image with predictions or empty rectangle
    cv2.imshow('Image', image)
    cv2.waitKey(0)

    # Print predictions for each face or a message if no faces detected
    print("Predictions for", image_path)
    if len(faces) == 0:
        print("No faces detected")
    else:
        for i, (gender, age) in enumerate(predictions):
            print(f"Face {i+1}: Gender - {gender}, Age - {age}")
    print("\n")

cv2.destroyAllWindows()

