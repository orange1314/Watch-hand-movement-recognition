# 動作識別算法使用說明

---

## 簡介

本說明文檔介紹了一種基於多模型和變換點檢測的動作識別算法，主要包括模型的訓練和預測兩個部分。算法利用傳感器數據的統計特性，結合高斯混合模型（GMM）、指數模型、主成分分析（PCA）和變換點檢測等技術，實現對多種動作的識別。

---

## 目錄

1. [模型訓練函數 `aw_train`](#模型訓練函數-aw_train)
   - [函數說明](#函數說明)
   - [參數解釋](#參數解釋)
   - [使用示例](#使用示例)
2. [動作預測函數 `detect_action_segments_with_changepoint_voting`](#動作預測函數-detect_action_segments_with_changepoint_voting)
   - [函數說明](#函數說明-1)
   - [參數解釋](#參數解釋-1)
   - [使用示例](#使用示例-1)
3. [整體流程說明](#整體流程說明)
   - [數據準備](#數據準備)
   - [模型訓練](#模型訓練)
   - [動作預測](#動作預測)
   - [結果分析](#結果分析)
4. [注意事項](#注意事項)

---

## 模型訓練函數 `aw_train`

### 函數說明

`aw_train` 函數用於訓練特定動作的模型，包括高斯混合模型（GMM）和指數模型。通過對傳感器數據進行統計分析，構建每個動作的概率分布模型，並保存訓練結果以供後續預測使用。

### 參數解釋

- **`base_dir`**：字符串，表示訓練數據所在的目錄路徑。該目錄下應包含特定動作的原始數據文件。

- **`saved_models_dir`**：字符串，表示訓練好的模型保存的目錄路徑。模型將以特定的文件格式保存在該目錄下。

- **`plot_dir`**：字符串，表示用於保存概率密度圖和其他可視化圖像的目錄路徑。

- **`segment_size`**：整數，表示數據分段的大小，即每個段包含的數據點數量。由於數據采集頻率為每秒 100 個數據點，`segment_size=100` 表示每個段為 1 秒的數據。

- **`window_size`**：整數，表示滑動窗口的步長。滑動窗口用於計算統計特征，如均值和方差。`window_size=10` 表示每次滑動 10 個數據點。

- **`confidence_level`**：浮點數，表示置信水平，用於確定概率密度分布的累積面積。例如，`confidence_level=0.997` 對應於 99.7% 的置信水平。

- **`gmmsmooth`**：浮點數，表示在計算高斯混合模型的置信區間時的平滑參數。較大的值會放寬 GMM（均值）的接受條件。

- **`expsmooth`**：浮點數，表示在計算指數模型的置信區間時的平滑參數。較大的值會放寬指數模型（方差）的接受條件。

- **`plot`**：布爾值，表示是否生成並保存概率密度圖和其他可視化圖像。

- **`n_components`**：整數，表示高斯混合模型中高斯成分的數量。選擇適當的成分數量可以更好地擬合數據分布。

### 使用示例

```python
from aw_write_model import aw_train

model = 'write'

aw_train(
    base_dir='data/' + model,
    saved_models_dir='saved_models/' + model,
    plot_dir='pic/ProbabilityDensity',
    segment_size=100,
    window_size=10,
    confidence_level=0.997,
    gmmsmooth=1,
    expsmooth=0.0001,
    plot=True,
    n_components=30
)
```

**說明**：

- **訓練數據路徑**：`data/write`，存放了 "write" 動作的訓練數據。
- **模型保存路徑**：`saved_models/write`，訓練好的模型將保存在此目錄下。
- **分段和窗口**：每 100 個數據點（1 秒）作為一個段，每次滑動 10 個數據點計算統計特征。
- **置信水平**：`0.997`，對應於標準正態分布的 3 個標準差範圍。
- **平滑參數**：`gmmsmooth=1`，`expsmooth=0.0001`，用於控制置信區間的範圍。
- **高斯成分數量**：`n_components=30`，根據數據覆雜度選擇合適的值。

---

## 動作預測函數 `detect_action_segments_with_changepoint_voting`

### 函數說明

`detect_action_segments_with_changepoint_voting` 函數用於對新的傳感器數據進行動作預測。通過加載訓練好的模型，對新數據進行分段預測，並結合變換點檢測和標簽平滑，輸出最終的動作識別結果。

### 參數解釋

- **`file_path`**：字符串，表示待預測的 CSV 數據文件路徑。

- **`model_dirs`**：列表，包含模型目錄的路徑，每個目錄對應一個動作的模型。例如，`['saved_models/write', 'saved_models/erase']`。

- **`variables`**：列表，指定用於分析的傳感器變量。默認值為 `['GravityX', 'GravityY', 'GravityZ', 'UserAccelerationX', 'UserAccelerationY', 'UserAccelerationZ']`。

- **`segment_size`**：整數，表示數據分段的大小，與訓練時保持一致。

- **`min_action_length`**：整數，表示在標簽平滑時的最小動作長度。小於該長度的段將被合並或忽略。

- **`smooth_labels`**：布爾值，表示是否進行標簽平滑處理。如果為 `True`，則會進行變換點檢測和標簽平滑；否則，直接返回初始預測結果。

- **`show_image`**：布爾值，表示是否顯示可視化結果圖像。

### 使用示例

```python
from your_module import detect_action_segments_with_changepoint_voting

model_dirs = ['saved_models/write', 'saved_models/erase']  # 添加您的動作模型文件夾
file_path = 'data/noise/chiahung_noise.csv'

# 調用預測函數
predictions = detect_action_segments_with_changepoint_voting(
    file_path=file_path,
    model_dirs=model_dirs,
    segment_size=100,
    min_action_length=150,
    smooth_labels=True,
    show_image=True
)
```

**說明**：

- **待預測數據**：`data/noise/chiahung_noise.csv`，包含需要預測的傳感器數據。
- **模型目錄**：`['saved_models/write', 'saved_models/erase']`，加載 "write" 和 "erase" 動作的模型。
- **參數設置**：
  - `segment_size=100`，與訓練時保持一致。
  - `min_action_length=150`，在標簽平滑時，小於 150 個數據點的段將被處理。
  - `smooth_labels=True`，啟用標簽平滑和變換點檢測。
  - `show_image=True`，顯示可視化結果圖像。

---

## 整體流程說明

### 數據準備

1. **數據采集**：通過傳感器采集原始數據，確保數據包含所需的變量，例如重力加速度和用戶加速度的各個分量。

2. **數據組織**：將數據按照動作類別分類，存儲在對應的目錄下。例如，"write" 動作的數據存放在 `data/write` 目錄中。

### 模型訓練

1. **調用 `aw_train` 函數**：針對每個動作，使用對應的數據目錄和參數，訓練模型並保存。

2. **參數調整**：根據數據特點和實驗效果，調整 `segment_size`、`window_size`、`confidence_level`、`n_components` 等參數，以獲得最佳的模型性能。

3. **模型保存**：訓練好的模型將保存在指定的 `saved_models_dir` 目錄下，供預測時使用。

### 動作預測

1. **調用 `detect_action_segments_with_changepoint_voting` 函數**：提供待預測的數據文件路徑和模型目錄列表。

2. **加載模型**：函數會自動加載指定目錄下的模型，包括 GMM、指數模型和閾值信息。

3. **數據分段和初步預測**：將數據按照 `segment_size` 進行分段，對每個段計算均值和方差，使用模型進行初步預測。

4. **變換點檢測和標簽平滑**（可選）：

   - 如果 `smooth_labels=True`，則會使用 PCA 和變換點檢測算法，識別數據中的變化點，劃分動作區間。
   - 對初步預測的結果進行標簽平滑處理，修正短暫的誤判段落，減少噪聲。

5. **結果輸出**：返回預測的動作標簽序列，並根據需要生成並顯示可視化結果。

### 結果分析

1. **預測結果**：`predictions` 變量包含了每個數據點對應的動作標簽。

2. **可視化圖像**：如果 `show_image=True`，函數會生成包含動作標注的時間序列圖，直觀展示傳感器數據與動作識別結果的關系。

3. **動作占比統計**：函數會打印各動作的占比信息，幫助分析預測結果。

---

## 注意事項

- **數據一致性**：訓練和預測時的 `segment_size`、`variables` 等參數應保持一致，確保模型和數據處理方式匹配。

- **模型路徑**：確保提供的模型目錄路徑正確，且目錄下包含訓練好的模型文件。

- **參數調整**：根據實際數據和需求，適當調整 `confidence_level`、`n_components`、`min_action_length` 等參數，以獲得最佳的識別效果。

- **模型擴展**：如果需要識別更多的動作，只需按照相同的流程，對新的動作數據進行模型訓練，並在預測時添加相應的模型目錄。

- **環境配置**：確保運行代碼的環境中安裝了必要的包，如 `numpy`、`pandas`、`scikit-learn`、`matplotlib`、`ruptures` 等。

---



```python

```
