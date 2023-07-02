import streamlit as st
import face_recognition
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import cv2
from glob import glob
from PIL import Image
import shutil

shutil.rmtree("frames", ignore_errors=True)
os.mkdir('frames')


# Функция blur_frame принимает на вход image - изображение и face_encodings - закодированные изображения лиц тех людей,
# которых не нужно замазывать. Функция замазывает тех людей, которых нет в face_encodings
def blur_frame(image, face_encodings):
    # Получаем координаты лиц на изображении
    face_locations = face_recognition.face_locations(image)

    # Выделяем каждое лицо и добавляем их в список faces_allocate
    faces_allocate = []
    for loc in face_locations:
        face = image[loc[0] - 30:loc[2] + 30, loc[3] - 30:loc[1] + 30]
        faces_allocate.append((face, loc))

    for face, coord in faces_allocate:
        face_enc = None
        try:
            # Кодируем изображение лица
            face_enc = face_recognition.face_encodings(face.astype(np.uint8))[0]
        except:
            face_enc = None

        if face_enc is not None:
            # Если лица похожего лица нету в results, то делаем blur этого лица
            results = face_recognition.compare_faces(face_encodings, face_enc)
            if True not in results:
                image[coord[0]:coord[2], coord[3]:coord[1]] = cv2.blur(
                    cv2.blur(image[coord[0]:coord[2], coord[3]:coord[1]], (31, 31)), (31, 31))

    return image


# Функция blur_video принимает на вход frame_paths - список путей до кадров видео
# Эта функция применяет blur, ко всем кадрам видео
def blur_video(frame_paths, face_encodings):
    for path in tqdm(frame_paths):
        image = face_recognition.load_image_file(f"./{path}")
        blur_image = blur_frame(image, face_encodings)
        cv2.imwrite(f"./{path}", cv2.cvtColor(blur_image, cv2.COLOR_RGB2BGR))
    return 1


# Функция split_video разделяет видео на кадры
def split_video(video_path):
    video_name = video_path.split("\\")[-1]
    os.mkdir(f"frames/{video_name}")

    vidcap = cv2.VideoCapture(video_path)
    # Получаем значение FPS
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    # Читаем по кадрову
    success, image = vidcap.read()
    cv2.imwrite(f"frames/{video_name}/image_00000.jpg", image)

    count = 1
    while success:
        cv2.imwrite(f"frames/{video_name}/image_{count:05}.jpg", image)
        success, image = vidcap.read()
        count += 1

    frame_paths = glob(f"frames/{video_name}/*.jpg")
    frame_paths.sort()
    return frame_paths, fps


# Функция images_to_video разделяет видео на кадры
def images_to_video(frame_paths, save_path):
    height, width, _ = cv2.imread(frame_paths[0]).shape
    # video_restored.avi
    video = cv2.VideoWriter(
        save_path,  # имя файла
        cv2.VideoWriter_fourcc(*'DIVX'),  # указываем кодек
        30,  # указываем кол-во fps
        (width, height)  # указываем ширину и высоту картинки
    )

    for frame_path in tqdm(frame_paths):
        img = cv2.imread(frame_path)

        # записываем изображение в конец видео
        video.write(img)

    # указвыаем, что мы завершили запись видео
    video.release()
    return 1


st.title("Blur face")
st.write("App usage rules")
st.write(
    'If you click on <b>Upload no blur faces</b>, then you will be given the opportunity to upload a photo of those people who do not need to be blurred',
    unsafe_allow_html=True)
st.write(
    "If you click on <b>Blur video</b>, then you must specify two paths, the first is the path to the video to which you want to apply blur, the second is the path to the place where this video will be saved",
    unsafe_allow_html=True)
st.write(
    "If you click on <b>Reset</b>, then all the images of people that you have added so that they are not blurry will be removed",
    unsafe_allow_html=True)
page_name = ["Reset", 'Upload no blur faces', 'Blur video']
page = st.radio('Navigation', page_name)

if page == "Reset":
    st.session_state["face_encodings"] = []
    shutil.rmtree("frames", ignore_errors=True)
    os.mkdir('frames')

if page == "Upload no blur faces":
    file = st.file_uploader("Please upload an image with not blur face", type=["jpg", "png"])

if page == "Upload no blur faces":
    # st.write("Upload no blur faces")
    if file is None:
        pass
    else:
        faces_without_blur = []
        image = Image.open(file).convert('RGB')
        image = np.array(image)
        face_locations = face_recognition.face_locations(image)

        faces_allocate = []
        for loc in face_locations:
            faces_allocate.append(image[loc[0] - 25:loc[2] + 25, loc[3] - 25:loc[1] + 25])

        faces_without_blur.extend(faces_allocate)

        for face in faces_without_blur:
            try:
                st.session_state["face_encodings"].append(face_recognition.face_encodings(face.astype(np.uint8))[0])
                st.write("Face was added")
            except:
                st.write("Face doesn't find")
        st.write(len(st.session_state["face_encodings"]))

if page == "Blur video":
    file_blur = st.text_input('Path to video', "")
    save_path = st.text_input('Path to save', "")

if page == "Blur video":
    if file_blur == "" or save_path == "":
        pass
    else:
        # C:\Education\AMOML\FinalProject2023\TestData\v2.mp4 Пример пути до видео
        # C:\Education\AMOML\FinalProject2023\TestData\video_restored1.avi Пример пути, куда сохраняем
        with st.spinner('Split video..'):
            frame_paths1, fps1 = split_video(str(file_blur))
        with st.spinner('Blur video..'):
            blur_video(frame_paths1, st.session_state["face_encodings"])
        with st.spinner('Save video..'):
            images_to_video(frame_paths1, str(save_path))
            st.write("Blur video was saved")
