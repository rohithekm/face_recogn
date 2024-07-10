from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# convert static images to RGB
try:
    messi_image = cv2.imread("messi/leo.png")
    messi_image = cv2.cvtColor(messi_image, cv2.COLOR_BGR2RGB)
    messi_face_encoding = face_recognition.face_encodings(messi_image)[0]
except Exception as e:
    print(f"Error loading Messi image: {str(e)}")
    messi_face_encoding = None

try:
    rohith_image = cv2.imread("Rohith\\rohith1.png")
    rohith_image = cv2.cvtColor(rohith_image, cv2.COLOR_BGR2RGB)
    rohith_face_encoding = face_recognition.face_encodings(rohith_image)[0]
except Exception as e:
    print(f"Error loading Rohith image: {str(e)}")
    rohith_face_encoding = None

known_face_encodings = []
known_face_names = []

if messi_face_encoding is not None:
    known_face_encodings.append(messi_face_encoding)
    known_face_names.append("MESSI")

if rohith_face_encoding is not None:
    known_face_encodings.append(rohith_face_encoding)
    known_face_names.append("ROHITH")

# Initialize variables for face recognition
face_locations = []
face_encodings = []
face_names = []

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color to RGB color 
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
