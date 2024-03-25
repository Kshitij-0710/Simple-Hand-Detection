import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)  # Replace 0 with the camera index if you are using an external camera
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

try:
    while True:
        success, img = cap.read()

        # Convert the image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # You can access landmarks and other hand information here
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

        cv2.imshow("Image", img)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    # Release resources on keyboard interrupt
    cap.release()
    cv2.destroyAllWindows()
