# Cow action detection 側拍牛隻影像辨識行為處理
## 介紹

這個專案基於 YOWOv2 架構修改，模型為 YoloX + Yolov7 版本組合並結合 3D CNN 模型架構。並加入時空卷積網路進行影片序列型資料辨識，最後再以卡爾曼濾波器以及匈牙利演算法增加追蹤效果，用於追蹤影片當中牛隻可能被遮擋而不易被辨識的牛隻。從牛舍中側拍的影像中提取牛隻影像並做目標追蹤牛隻，藉由牛隻隨時間變化的特徵透過模型處理進一步去分析各別牛隻行為統計，用於輔助農民觀測牛舍場域牛隻，牛隻正常行為判別、發情前徵兆行為判斷，用以幫助農場管理。


## 功能  

- 使用 YOWOv2 模型進行即時牛隻目標追蹤位置和行為檢測，每秒對畫面中各別牛隻作行為辨識區分是哪種行為。

  

## 呈現效果  

| 原始圖片                                    | 推論結果                          |
|:------------------------------------------:|:---------------------------------:|
|<img src="https://github.com/user-attachments/assets/b49a511c-1736-4179-80fa-7f9a0e74baa0" width="320" height="240">| <img src="https://github.com/user-attachments/assets/05f0765e-6357-4254-bce1-69d4a5d43b74" width="320" height="240">|
| <img src="https://github.com/user-attachments/assets/a245bebd-821b-4793-b63f-fee0c40b36af" width="320" height="240">| <img src="https://github.com/user-attachments/assets/f136b35b-88a6-467c-b064-d30d71b387b2" width="320" height="240">|


## 模型可視化  

使用 Grad CAM 技術進行可視化，以產生模型對牛隻活動區域的注意力熱圖，並進行解釋。  

| 原始圖片                                    | Grad CAM 推論結果                          |
|:------------------------------------------:|:------------------------------------------:|
|<img src="https://github.com/user-attachments/assets/c94b969d-dd20-4c6e-af22-31f61e9fee23" width="320" height="240">|<img src="https://github.com/user-attachments/assets/9c3147e4-0488-4313-b143-18d6c1876424" width="320" height="240">|
|<img src="https://github.com/user-attachments/assets/b49a511c-1736-4179-80fa-7f9a0e74baa0" width="320" height="240"> |<img src="https://github.com/user-attachments/assets/1e72c1cd-0e38-417d-953f-5a15e555c132" width="320" height="240">|
|<img src="https://github.com/user-attachments/assets/2029ee9b-814e-4599-be67-8e78e4089b84" width="320" height="240"> |<img src="https://github.com/user-attachments/assets/2ca6714a-c98e-4ef3-b523-ab9d367bfe29" width="320" height="240">|


## 資料集  

這個專案使用了一個自定資料集名為 Cow_AVA 的資料集，為參考 AVA Dataset 格式製作    

包含了圖片名稱、圖片順序、目標標記框 x1, y1, x2, y2 座標、動作類別 (one-hot encode)  
每一個影片會裁剪為 16 秒短影片做為訓練，每秒會裁剪出 30 張圖片，所以每一部訓練影片會有 480 張圖片  


## 自訂資料集
若想使用自訂資料集來使用模型，請參考下列網址  
從步驟四以後開始製作  [Custom-ava-dataset for Cow action detection](https://github.com/JasonHuang0119/Custom-ava-dataset_Custom-Spatio-Temporally-Action-Video-Dataset-Windows)  


## 安裝  
使用前請先安裝好 [Anaconda](https://www.anaconda.com/download)  

首先，我们建議使用 Anaconda 來創建一個 conda 的虛擬環境  

建立新的 Python 虛擬環境，Python 版本 >= 3.8

```Shell
# ENV_NAME 為自訂的環境名稱
conda create -n ENV_NAME python=3.8 -y 
```

然後，啟動虛擬環境
```Shell
# ENV_NAME 為自訂的環境名稱
conda activate ENV_NAME
```

安裝相關套件：
```Shell
# 安裝 pip
conda install pip

# 切換目錄到專案資料夾
cd Cow-action-detection

# 安裝使用套件
pip install -r requirements.txt 
```

項目作者使用的環境配置：

- boxmot =  10.0.72
- PyTorch = 2.2.0
- Torchvision = 0.17.2
- Tensorboard = 2.14.0
- opencv-python >= 4.7.0.68
- numpy = 1.24.4
- imageio = 2.33.1
- gram-cam = 1.5.0


(安裝 boxmot 套件會一同安裝 pytorch > 2.0 版本以及相關套件，請使用者注意，並確認有安裝到 cuda 版本)  

為了能夠正常運行此項目的代碼，請確保您的 torch 版本為 2.x 系列。

## 測試  

- 請將資料夾命名為 AVA_Dataset 放在 dataset 資料夾下，裡面內容存放 annotations、frame_lists、frames 這三個資料夾  
- annotations(標記文件) 、 frame_lists (val.csv 讀取圖片位置) 、 frames(val 測試圖片集)
- dataset 資料夾底下 有簡單的測試資料範例


## Demo
使用者可以参考下面的命令来測試本地的影片文件：

```Shell
# run demo
python demo.py --cuda -d ava_v2.2 -v yowo_v2_slowfast -size 224 --weight path/to/weight --video path/to/video --show
```
- 使用範例：
- python demo.py --cuda -d ava_v2.2 -v yowo_v2_slowfast -size 224 --weight ./weights/yowo_v2_slowfast_epoch_58.pth --video video/7.mp4 --show

或者使用下方腳本
``` Shell
# 在 ubuntu 環境下使用腳本
sh demo.sh
``` 

注意，使用者使用  ```path/to/weight``` 找 weights 資料夾下訓練權重    (範例： weights/ava_v2.2/yowov2_slowfast/{權重檔案}.pth)  
使用 ```path/to/video``` 修改為要測試的影片的文件路徑。   (範例： 找 video 資料夾下影片 video/10.mp4   )



# 参考文獻
內容有使用 YOWOv2 代碼，引用作者的論文：

```
@article{yang2023yowov2,
  title={YOWOv2: A Stronger yet Efficient Multi-level Detection Framework for Real-time Spatio-temporal Action Detection},
  author={Yang, Jianhua and Kun, Dai},
  journal={arXiv preprint arXiv:2302.06848},
  year={2023}
}
```
# 聯絡方式

聯絡人 : Jason Huang

Email : jason1539663@gmail.com

