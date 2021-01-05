from PIL import Image
import face_recognition
import time
import os

start = time.perf_counter()

KNOWN_FACES_DIR = r'/Volumes/Elements/BrentWood/known_faces'
#UNKNOWN_FACES_DIR = r'/Volumes/Elements/BrentWood/test_pictures/July 7 - General Coverage Part C/Jennifer Glagola-Poomsae - Round 2'
UNKNOWN_FACES_DIR = r'/Volumes/Elements/BrentWood/test_pictures/July 7 - General Coverage Part C'
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
TOLERANCE = 0.60

known_faces = []
known_names = []




import concurrent.futures
import math
import numpy as np


URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://some-made-up-domain.com/']






def loadKnownFace(faceFileName):
    # Load an image
    print('  - Loading ' + faceFileName)
    #
    try:
        image = face_recognition.load_image_file(f'{faceFileName}')
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(faceFileName)
    except IndexError:
        print("  ** No face found in ", faceFileName)    


def findKnownFaces(directory):
    # We oranize known faces as subfolders of KNOWN_FACES_DIR
    # Each subfolder's name becomes our label (name)
    for name in os.listdir(directory):

        if (os.path.isdir(f'{directory}/{name}')):
           findKnownFaces(f'{directory}/{name}')
        elif (name.endswith(".png") or name.endswith(".jpg")):   
            loadKnownFace(f'{directory}/{name}')




def loadUnknownFace(faceFileName):
  # Load image
    print(f'  - Filename: {faceFileName}...',)

    image = face_recognition.load_image_file(f'{faceFileName}')

    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)

    # Now since we know locations, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)
    print('  - ', len(encodings),' faces found in', faceFileName)
    for face in encodings:
        compared_faces = face_recognition.compare_faces(known_faces,face,TOLERANCE)
        if(True in compared_faces):
            print(' *** ', known_names[compared_faces.index(True)], ' found in ',f'{faceFileName}' )


# def findUnknownFaces(directory):
#     print(f'Searching directory: {directory}...',)
#     # Each subfolder's name becomes our label (name)
#     for name in os.listdir(directory):

#         if (os.path.isdir(f'{directory}/{name}')):
#            findUnknownFaces(f'{directory}/{name}')
#         elif (name.endswith(".png") or name.endswith(".jpg")):   
#             loadUnknownFace(f'{directory}/{name}')


def processUnknowns( files ):

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        # Start the load operations and mark each future with its URL
        future_to_file = {executor.submit(loadUnknownFace, file): file for file in files}
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (file, exc))
            # else:
            #     print('%r page is %d bytes' % (file, len(data)))



def findUnknownFaces(directory):
    print(f'Searching directory: {directory}...',)
    files = []
    # Each subfolder's name becomes our label (name)
    for name in os.listdir(directory):
        if (os.path.isdir(f'{directory}/{name}')):
            findUnknownFaces(f'{directory}/{name}')
        elif (name.endswith(".png") or name.endswith(".jpg")):
            files.append(f'{directory}/{name}')
    
    num_splits =  math.ceil(len(files)/50)
    if num_splits > 0 :
        print('nuber of splits: ', num_splits)
        split_array = np.array_split(files, num_splits)

        for current_array in split_array:
            print('split size: ', len(current_array))
            processUnknowns(current_array)




print('Loading known faces...')
findKnownFaces(KNOWN_FACES_DIR)

print('Processing unknown faces...')
findUnknownFaces(UNKNOWN_FACES_DIR)





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