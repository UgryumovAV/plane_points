import json
import os.path
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


def preprocess_floor_plan(
        image: np.ndarray,
        kernel_size: int = 3,
) -> tuple[list, np.ndarray, np.ndarray, list]:
    """
    Классические алгоритмы для нахождения контуров стен
    :param image: Изображению
    :param kernel_size: Размер ядра для морфологических операций
    :return: Результаты обработки и изображения
    """
    # 0. Бинаризация и инвертирование изображения для выделения контуров стен
    _, binary = cv2.threshold(image.copy(), 220, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # 1. Морфологические изменения для удаления тонких линий
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Находим контуры
    contours, _ = cv2.findContours(
        processed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    filtered_contours = list()
    for contour in contours:
        filtered_contours.append(contour)

    # Дополнительно. Метод хороших углов Ши-Томаси
    corners = cv2.goodFeaturesToTrack(
        processed,
        maxCorners=100,
        qualityLevel=0.1,
        minDistance=1
    )

    return filtered_contours, binary, processed, corners


def hsv_to_bgr(
        h: int,
        s: int,
        v: int,
        h_range: int = 360,
        s_range: int = 100,
        v_range: int = 100
) -> tuple[int, int, int]:
    """
    Универсальная конвертация HSV в BGR

    :param h: значение Hue - цветовой тон
    :param s: значения Saturation - насыщенность
    :param v: значения Value - Яркость
    :param h_range: максимальное значение H (обычно 360)
    :param s_range: максимальное значение S (обычно 100)
    :param v_range: максимальное значение V (обычно 100)
    :return: кортеж (b, g, r)
    """
    # Нормализуем для OpenCV
    h_cv2 = int((h / h_range) * 180)
    s_cv2 = int((s / s_range) * 255)
    v_cv2 = int((v / v_range) * 255)

    # Конвертируем
    hsv_pixel = np.uint8([[[h_cv2, s_cv2, v_cv2]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)

    b, g, r = bgr_pixel[0][0]
    return int(b), int(g), int(r)


def extract_walls_from_image(
        image_path: str,
        save_dir_path: str,
        ocr_flag: bool
) -> None:
    """
    Основная функция: загружает изображение, обрабатывает его и сохраняет изображения и JSON со стенами
    :param image_path: Путь к изображению плана
    :param save_dir_path: Директория для сохранения результатов
    :param ocr_flag: Флаг использования ocr
    :return:
    """
    image_name = Path(image_path).name.split(Path(image_path).suffix)[0]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return None

    # 1. Находим углы и контуры
    contours, binary, processed, corners = preprocess_floor_plan(image=image)

    walls = list()
    wall_counter = 1

    # 2. Обрабатываем каждый найденный контур
    for contour in contours:
        # Аппроксимируем контур прямыми линиями (упрощаем)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Преобразуем точки контура в список вида [[x1,y1], [x2,y2], ...]
        points = approx.reshape(-1, 2).tolist()

        # Если после аппроксимации осталась хотя бы одна линия (2 точки)
        if len(points) >= 2:
            wall = {
                "id": f"w{wall_counter}",
                "points": points
            }
            walls.append(wall)
            wall_counter += 1

    # 3. Преобразуем изображение обратно в BGR для отрисовки найденных контуров и углов
    if len(image.shape) == 2 or image.shape[2] == 0:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    processed_corners = processed.copy()

    colors = {i: hsv_to_bgr(int((i + 1) * 359 / len(walls)), 80, 80) for i in range(len(walls))}
    for i, wall in enumerate(walls):
        points = wall["points"]
        color = colors[i]
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, color, -1)
            cv2.circle(processed, (int(point[0]), int(point[1])), 5, color, -1)

    if corners is not None:
        corners = np.int32(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(processed_corners, (x, y), 5, (0, 255, 0), -1)


    # 4. OCR
    res_json = dict()
    if ocr_flag:
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang="ru"
        )
        result_ocr = ocr.predict(
            input=image_path,

        )
        for res in result_ocr:
            res_json = res.json
            res.save_to_img(
                os.path.join(save_dir_path, f"{image_name}_ocr.jpg")
            )

    # 5. Формируем итоговой json, изображения и сохраняем
    result_image = np.hstack((image, processed, processed_corners))
    cv2.imwrite(
        os.path.join(save_dir_path, f"{image_name}_result.jpg"),
        result_image
    )
    result = {
        "meta": {"source": image_path},
        "walls": walls,
        "OCR": res_json
    }
    with open(os.path.join(save_dir_path, f"{image_name}_result.json"), "w") as f:
        json.dump(result, f)
        return None
