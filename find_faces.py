from PIL import Image
import face_recognition
import time
import os

start = time.perf_counter()

KNOWN_FACES_DIR = r'H:\photo\face_finder\pictures\known'
UNKNOWN_FACES_DIR = r'H:\photo\face_finder\pictures\unknown'
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
TOLERANCE = 0.45



print('Loading known faces...')
known_faces = []
known_names = []

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        # Load an image
        print('Loading ' + filename)
        #
        try:
            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
            encoding = face_recognition.face_encodings(image)[0]

            # Append encodings and name
            known_faces.append(encoding)
            known_names.append(name)
        except IndexError:
            print("No face found in ", filename)

print('Processing unknown faces...')

for filename in os.listdir(UNKNOWN_FACES_DIR):

    # Load image
    print(f'Filename: {filename}...',)

    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')

    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)

    # Now since we know locations, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)
    print(len(encodings),'faces found in', filename)
    for face in encodings:
        compared_faces = face_recognition.compare_faces(known_faces,face,TOLERANCE)
        if(True in compared_faces):
            print(known_names[compared_faces.index(True)])




# image = face_recognition.load_image_file("H:\\photo\\face_finder\\pictures\\unknown\\Devin_reunion.jpg")
#
#
# face_locations = face_recognition.face_locations(image)
# # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
#
# print("I found {} face(s) in this photograph.".format(len(face_locations)))
# for face_location in face_locations:
#
#     # Print the location of each face in this image
#     top, right, bottom, left = face_location
#     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
#
#     # You can access the actual face itself like this:
#     face_image = image[top:bottom, left:right]
#     pil_image = Image.fromarray(face_image)
#     pil_image.show()
#
#
# print(time.perf_counter() - start)