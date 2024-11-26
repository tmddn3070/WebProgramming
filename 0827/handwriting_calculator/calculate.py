import math
import re
import numpy as np
import tensorflow as tf
import pickle
import cv2
import sympy
from typing import List, Tuple, Dict, Union

# 모델 정의 및 불러오기
model_paths: Dict[str, str] = {
    'main': 'models/major_model_new.h5',
    'trig': 'models/major_model_trig.h5'
}

encoder_paths: Dict[str, str] = {
    'main': 'models/label_encoder.pkl',
    'trig': 'models/label_encoder_trig.pkl',
    'scaler': 'models/scaler.pkl'
}

models: Dict[str, tf.keras.models.Model] = {key: tf.keras.models.load_model(path) for key, path in model_paths.items()}
encoders: Dict[str, pickle.Unpickler] = {key: pickle.load(open(path, 'rb')) for key, path in encoder_paths.items()}
scaler: pickle.Unpickler = pickle.load(open(encoder_paths['scaler'], 'rb'))


def is_power(c1: Tuple[Tuple[float, float], str], c2: Tuple[Tuple[float, float], str]) -> bool:
    """주어진 두 점 사이의 각도가 20도에서 70도 사이인지 확인하는 로 직

    Args:
        c1 (Tuple[Tuple[float, float], str])
        c2 (Tuple[Tuple[float, float], str])

    Returns:
        bool
    """
    delta_x: float = c2[0][0] - c1[0][0]
    delta_y: float = c2[0][1] - c1[0][1]
    angle_deg: float = math.degrees(math.atan2(delta_y, delta_x))
    return 20 <= angle_deg <= 70

def combine_elements(contour_predict: List[Tuple[Tuple[float, float], str]], element_type: str = 'number') -> List[Union[Tuple[Tuple[float, float], str], Tuple[Tuple[float, float], str]]]:
    """인식된 컨투어에서 숫자와 지수를 결합
    Args:
        contour_predict (List[Tuple[Tuple[float, float], str]])
        element_type (str, optional) default 'number'.

    Returns:
        List[Union[Tuple[Tuple[float, float], str], Tuple[Tuple[float, float], str]]]
    """
    numbers: set = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero'}
    combined_elements: List[Union[Tuple[Tuple[float, float], str], Tuple[Tuple[float, float], str]]] = []
    i: int = 0
    n: int = len(contour_predict)

    while i < n:
        current_element: Tuple[Tuple[float, float], str] = contour_predict[i]
        if (element_type == 'number' and current_element[1] in numbers) or (element_type == 'power' and i + 1 < n and is_power(current_element, contour_predict[i + 1])):
            if element_type == 'number':
                j: int = i + 1
                while j < n and contour_predict[j][1] in numbers and not is_power(current_element, contour_predict[j]):
                    current_element = (
                        ((current_element[0][0] + contour_predict[j][0][0]) / 2,
                         (current_element[0][1] + contour_predict[j][0][1]) / 2),
                        current_element[1] + ' ' + contour_predict[j][1]
                    )
                    j += 1
                combined_elements.append(current_element)
                i = j
            elif element_type == 'power':
                base_coords, base_string = current_element[0], current_element[1]
                power_string = contour_predict[i + 1][1]
                combined_elements.append((base_coords, f"{base_string} power {power_string}"))
                i += 2
        else:
            combined_elements.append(current_element)
            i += 1

    return combined_elements

def replace_words_and_operators(input_str: str, replacements: Dict[str, str]) -> str:
    """문자열에서 특정 단어나 연산자를 다른 단어나 연산자로 대체하는 로직

    Args:
        input_str (str)
        replacements (Dict[str, str])

    Returns:
        str
    """
    for word, replacement in replacements.items():
        input_str = input_str.replace(word, replacement)
    return input_str.replace(" ", "")

def replace_root(input_str: str) -> str:
    """문자열에서 제곱근 표기법을 처리하는 함ㅂ수

    Args:
        input_str (str)

    Returns:
        str
    """
    return re.sub(r'root(\d+)', lambda m: str(int(math.sqrt(int(m.group(1))))), input_str)

def calculate_expression(expression: str) -> Union[int, float, None]:
    """수학표현식을 계산합니다.

    Args:
        expression (str)

    Returns:
        Union[int, float, None]
    """
    expression = expression.replace('^', '**')
    try:
        result = eval(expression, {"math": math, **sympy.__dict__})
        return result if isinstance(result, (int, float)) else None
    except Exception:
        return None

def format_eq(exp: str) -> str:
    """수학식을 HTML 형식으로 변환

    Args:
        exp (str)

    Returns:
        str
    """
    exp = re.sub(r'root(\d+)', '√', exp)
    exp = re.sub(r'\^(\d)', r'<sup>\1</sup>', exp)
    return exp

def parse_equation(equation: str) -> Tuple[List[int], int]:
    """수학식을 파싱

    Args:
        equation (str)

    Returns:
        Tuple[List[int], int]
    """
    lhs, rhs = equation.replace(' ', '').split('=')
    lhs_coeffs: List[int] = [0, 0, 0]
    terms: List[str] = re.split(r'[+-]', lhs)

    for term in terms:
        if 'x^3' in term:
            coeff = term.split('x^3')[0]
            lhs_coeffs[0] = int(coeff or '1')
        elif 'x^2' in term:
            coeff = term.split('x^2')[0]
            lhs_coeffs[1] = int(coeff or '1')
        elif 'x' in term:
            coeff = term.split('x')[0]
            lhs_coeffs[2] = int(coeff or '1')

    return lhs_coeffs, int(rhs)

def calculate_roots(equation: str) -> str:
    """제곱근을 계산

    Args:
        equation (str)

    Returns:
        str
    """
    coeffs, rhs_value = parse_equation(equation)
    roots = np.roots(coeffs + [-rhs_value])
    return ', '.join(map(str, roots))

def predict_image(image: np.ndarray, model_type: str) -> str:
    """이미지를 추론하는 로직

    Args:
        image (np.ndarray)
        model_type (str)_

    Returns:
        str
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    resized_image = cv2.resize(bw_image, (128, 128))
    img_data = scaler.transform(resized_image.reshape(1, -1)).reshape(1, 128, 128, 1)

    model = models['trig'] if model_type == 'trigonometry' else models['main']
    encoder = encoders['trig'] if model_type == 'trigonometry' else encoders['main']

    predictions = model.predict(img_data)
    return encoder.inverse_transform([np.argmax(predictions)])[0]

def process_image(image: np.ndarray, model_type: str) -> Tuple[Union[str, int, float, None], str]:
    """이미지를 처리하는 로지 ㄱ

    Args:
        image (np.ndarray)
        model_type (str)

    Returns:
        Tuple[Union[str, int, float, None], str]
    """
    deep_copy = image.copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    contour_predict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 25 or h >= 25:
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            contour_region = cv2.bitwise_or(image, cv2.bitwise_not(mask))
            scale_factor = max(w, h) / 250
            new_w, new_h = int(w / scale_factor), int(h / scale_factor)
            contour_region = cv2.resize(contour_region[y:y + h, x:x + w], (new_w, new_h))

            canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255
            x_offset, y_offset = (300 - new_w) // 2, (300 - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = cv2.resize(contour_region, (new_w, new_h))

            predicted_label = predict_image(canvas, model_type)
            cv2.putText(deep_copy, str(predicted_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            contour_predict.append((((x + x + x + x + w + w) / 4, (-y - y - y - y - h - h) / 4), predicted_label))

    contour_predict = combine_elements(contour_predict, 'number')
    contour_predict = combine_elements(contour_predict, 'power')
    input_str = replace_words_and_operators(''.join(i[1] for i in contour_predict), {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'plus': '+', 'minus': '-', 'mul': '*', 'power': '^', 'ob': '(',
        'cb': ')'
    })
    formatted_exp = format_eq(replace_root(input_str))
    result = calculate_expression(input_str)

    if model_type == 'polynomial':
        modified_expr = ''.join([c if c not in ['-', '+', '='] or c != '=' else 'x' for c in input_str])
        modified_expr = re.sub(r'--', '=', modified_expr)
        formatted_exp = format_eq(modified_expr)
        roots = calculate_roots(modified_expr)
        return roots, formatted_exp

    elif model_type == 'trigonometry':
        expr = input_str
        expr = expr.replace('n', 'a') if expr.startswith('t') else expr
        expr = expr.replace('t', 's', 1) if expr.startswith('c') else expr
        expr = expr.replace('e', 'c', 1) if 'e' in expr else expr
        try:
            result = eval(expr, {"math": math, "sin": math.sin, "cos": math.cos, "tan": math.tan,
                                  "cot": lambda x: 1 / math.tan(x), "sec": lambda x: 1 / math.cos(x),
                                  "cosec": lambda x: 1 / math.sin(x)})
            return str(result), format_eq(expr)
        except Exception:
            return expr, expr

    return result, formatted_exp
