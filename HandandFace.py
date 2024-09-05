import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness = 1,
    # Circle radius. Default to 2 pixels
    circle_radius = 1)

video = cv2.VideoCapture(0)

with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
        mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while True:
        #read the video
        ret,image = video.read()
        #convert image to rgb
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results = hands.process(image)
        face_mesh_results = face_mesh.process(image)

        image.flags.writeable=True
        #convert to main background
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image=image,
                                       landmark_list=hand_landmark,
                                       connections=mp_hand.HAND_CONNECTIONS)

        # Draw face mesh landmarks
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                mp_draw.draw_landmarks(image=image,
                                       landmark_list=face_landmarks,
                                       connections=mp_face_mesh.FACEMESH_CONTOURS,
                                       landmark_drawing_spec=drawing_spec,
                                       connection_drawing_spec=drawing_spec)

        cv2.imshow("Frame",image)
        k=cv2.waitKey(1)
        if k==ord('q'):
            break;
video.release()
cv2.destroyAllWindows()