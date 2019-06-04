# 幻想濾鏡桌面應用 Fantastic Filter Desktop
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

幻想濾鏡的圖形化桌面工具

## 開始使用

### 環境要求

您需要使用Python 3.6，並且安裝以下套件，確保程序正確運行：
- tensorflow==1.13.1 or tensorflow-gpu==1.13.1
- cv2 (opencv)
- numpy
- pillow (PIL)
- tkinter

或是使用`pip3 install -r requirements.txt` 一鍵安裝

## 運行

本軟件主程序為`app.py`，直接運行即可。請搭配 **幻想濾鏡（項目準備中）** 所輸出的模型使用。

## 使用自己的模型

本軟件支援TensorFlow Frozen模型格式如下：

位置    |形狀(shape)|型態| 範圍
-------|----------|------------|-------|
輸入   |(n, m, 3)|`tf.float32`|-1 ~ +1
輸出   |(n, m, 3)| `tf.uint8` |0~255



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
