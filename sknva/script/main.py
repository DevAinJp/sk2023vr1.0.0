import os
import cv2
import function
import const
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def main():
    is_vertical = False
    params = define_params(is_vertical)
    output(params)

#条件演算子を使用してimgを設定する部分を簡略化
def define_params(is_vertical):
    # 画像ファイルの存在を確認
    if not os.path.isfile(const.FILE_PATH):
        raise FileNotFoundError("Image file not found!")

    # 画像の読み込みと座標の初期化
    gray_img = cv2.imread(const.FILE_PATH, 0)
    coordinates = []

    # 画像の輝度の自動調整
    gray_brightness_adjustment_img = function.brightness_adjustment(gray_img)

    # 画像に線を引く
    if is_vertical:
        img = function.input_image_vertical(gray_brightness_adjustment_img, coordinates)
    else:
        img = function.input_image(gray_brightness_adjustment_img, coordinates)

    params = {
        "img": img,
        "coordinates": coordinates,
        "show_result_coordinates": [],
        "diameter_results": [],
        "is_vertical": is_vertical,
    }
    return params


#2つの条件を1つのリスト内包表記で処理し、新しいリスト filtered_coordinates にフィルターされた座標を格納しています。
def filter_coordinates_by_position(x, y, coordinates):
    filtered_coordinates = [coord for coord in coordinates if (int(x) - 5 <= coord[0] <= int(x) + 5) and (int(y) - 5 <= coord[1] <= int(y) + 5)]
    return filtered_coordinates


def show_image(show_img, x, y, params):
    global count
    # クリックした位置で座標が存在するか確認する
    filter_coordinates = filter_coordinates_by_position(x, y, params["coordinates"])

    # 条件にヒットする座標が複数ある時は処理を終了
    if len(filter_coordinates) != 1:
        return

    clicked_coord = filter_coordinates[0]
    params["show_result_coordinates"].append(clicked_coord)

    # クリックした座標を表示する
    cv2.circle(
        show_img,
        center=(filter_coordinates[0][0], filter_coordinates[0][1]),
        radius=5,
        color=255,
        thickness=-1,
    )
    pos_str = (
        ""
        + str(count)
        + "(x,y)=("
        + str(filter_coordinates[0][0])
        + ","
        + str(filter_coordinates[0][1])
        + ")"
    )

    #ノート作成
    text_img = np.ones((100, 400), dtype=np.uint8) * 255

    pil_image = Image.fromarray(text_img)

    font_path = "C:/Windows/Fonts/meiryo.ttc"
    font = ImageFont.truetype(font_path, 35)
    draw = ImageDraw.Draw(pil_image)
    draw.text((10, 10), "result = 倍率の結果", font=font, fill=0)
    draw.text((10, 50), "diff = 誤差率の結果", font=font, fill=0)

    text_img = np.array(pil_image)
    text_img = cv2.cvtColor(text_img, cv2.COLOR_RGB2BGR)

    cv2.imshow('text_image', text_img)

    cv2.putText(
        show_img,
        pos_str,
        (filter_coordinates[0][0] + 10, filter_coordinates[0][1] + 10),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        255,
        2,
        cv2.LINE_AA,
    )

    # 計算結果の表示（答え）
    length = len(params["show_result_coordinates"])
    if length > 0 and length % 2 == 0:
        x_shaft = 1600
        y_shaft = 200 + length * 40
        var_index = 0 if params["is_vertical"] else 1

        prev_coord = params["show_result_coordinates"][length - 2][var_index]
        curr_coord = params["show_result_coordinates"][length - 1][var_index]

        l = function.minus_int_to_plus(prev_coord - curr_coord)
        result = function.diameter_calculate(l)
        diff = function.calculation_diff(const.DESIRED_VALUE, result)
        params["diameter_results"].append(result)

        # f-stringsを使用して文字列を結合
        pos_result_str = f"(result {count})=({result})"
        pos_diff_str = f"(diff {count})=({diff})"
        count += 1

        cv2.putText(
            show_img,
            pos_result_str,
            (x_shaft, y_shaft),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            255,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            show_img,
            pos_diff_str,
            (x_shaft, y_shaft + 35),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            255,
            2,
            cv2.LINE_AA,
        )
    # ウィンドウに画像を表示
    cv2.imshow("window", show_img)

# クリックしたら座標の位置を表示する処理
def click_pos(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        show_image(params["img"], x, y, params)

def output(params):
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.imshow("window", params["img"])
    cv2.setMouseCallback("window", click_pos, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 処理が終了したら平均値と標準偏差を表示する
    function.show_average(params["img"], params["diameter_results"])
    function.show_standard_deviation(params["img"], params["diameter_results"])
    result_name = "result/clickanh_vertical_" if params["is_vertical"] else "result/click_"
    cv2.imwrite(result_name + const.FILE_NAME, params["img"])

if __name__ == "__main__":
    # 何回目かを示すための値
    count = 1
    main()
