import cv2
import dlib
from imutils import face_utils
"""
* Zła platforma do wyświetlania wideo - detekcja oczu wraz z zatrzymywaniem wideo jeśli użytkownik zamknie oczy *

Autorzy:
- Bartosz Krystowski s19545
- Robert Brzoskowski s21162

Przygotowanie środowiska:
Instalacja bibliotek cv2, dlib, face_utils
"""
"""Wczytaj obraz z kamerki lub wczytaj film z pliku"""
cap = cv2.VideoCapture('video1.mp4')

"""Wczytaj wideo / 'reklamę' z pliku"""
ad_video = cv2.VideoCapture('video.mp4')

"""Argument dla wczytywania obrazka (zastępuje cap = VideoCapture)"""
# cap = cv2.imread('image5.jpg')

"""Tworzymy detektor twarzy na bazie "zbioru twarzy" wraz z 'landmarkami'"""
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

"""Domyślny wczytany obrazek"""
image_nr = 1

"""Domyślna wartość zapauzowanej reklamy"""
is_paused = True

"""Czcionka do wyświetlania tekstu w oknie"""
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    """Odczytaj klatkę wideo"""
    ret, frame = cap.read()

    """Odczytaj obrazek (zastępuje ret, frame = cap.read())"""
    # frame = cap

    """Jeśli koniec filmu, odtwórz film ponownie (działa tylko w przypadku odczytywania obrazu z filmu)"""
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    """Konwertuj rozmiar okna"""
    frame = cv2.resize(frame, (640, 480))

    """Konwertuj obraz do skali szarości"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    """Przypisanie wykrytych twarzy"""
    faces = detector(gray)

    """Czy wykryto twarze"""
    if faces:


        """Dla każdej twarzy"""
        for face in faces:

            """Przypisz pozycje wykrytych twarzy"""
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            """Narysuj kwadrat określający położenie wykrytych twarzy"""
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            """Wykrywamy punkty twarzy"""
            shape = predictor(gray, face)

            """Konwersja punktów twarzy do tablicy numpy"""
            shape = face_utils.shape_to_np(shape)

            """Narysuj okrąg na każdym wykrytym punkcie twarzy"""
            for (sX, sY) in shape:
                cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)

            """Obliczamy dystans między górną a dolną powieką"""
            left_eye_height = abs((shape[41][1]-shape[37][1])+(shape[40][1]-shape[38][1]))
            right_eye_height = abs((shape[47][1]-shape[43][1])+(shape[46][1]-shape[44][1]))

            """Skalowanie odległości wykrytej twarzy"""
            scaling = ((x2+y2)-(x1+y1))/50

            """Jeśli dystans między powiekami jest mniejszy niż 15px dystansu między oczami, oznacza to zamknięcie oka"""
            if (left_eye_height + right_eye_height)/2 < (2.7*scaling):
                cv2.putText(frame, "Eyes: closed", (0, 50), font, 1, (255,255,255), 2)
                is_paused = True
            elif (left_eye_height + right_eye_height)/2 >= (2.7*scaling):
                cv2.putText(frame, "Eyes: opened", (0, 50), font, 1, (255,255,255), 2)
                is_paused = False

            print(left_eye_height, right_eye_height)
            print(2.7*scaling)
            print((x2+y2)-(x1+y1))

            """Jeśli reklama nie jest zatrzymana wyświetl jej obraz"""
            if is_paused is not True:
                ret_ad, frame_ad = ad_video.read()
                """Jeśli reklama się skończyła, przerwij"""
                if not ret_ad:
                    break
                frame_ad = cv2.resize(frame_ad, (640, 480))
                cv2.imshow("ad", frame_ad)

    # Jeśli nie wykryto twarzy wyświetl komunikat
    else:
        cv2.putText(frame, "No face detected", (0, 50), font, 1, (0, 0, 255), 2)

    """Wyświetl obraz"""
    cv2.imshow("images", frame)

    #cv2.imshow("image1", frame)
    #cv2.imshow("image2", output)

    """Jeśli naciśnięto klawisz 'q', zakończ pętlę"""
    if cv2.waitKey(1) == ord('q'):
        break

    """
    Jeśli naciśnięto klawisz 'z', zmień obrazek (działa tylko w przypadku wczytania obrazka
    #if cv2.waitKey(1) == ord('z'):
    #    if image_nr == 1:
    #        cap = cv2.imread('image5.jpg')
    #        image_nr = 2
    #    else:
    #        cap = cv2.imread('image4.jpg')
    #        image_nr = 1
    """


"""Zwolnij wszystkie zasoby"""
cap.release()
cv2.destroyAllWindows()
