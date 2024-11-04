# 引入所需的函式庫
import os
import warnings
import joblib

# 數值處理與資料處理
import numpy as np
import pandas as pd
import torch

# 統計與視覺化
from scipy.stats import norm, expon
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 機器學習
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score

# 自定義 EM 模組
from EM import CustomEMModel


#################################模型訓練#################################
# 忽略特定警告
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak on Windows with MKL')

def aw_train(base_dir='data/write',
             saved_models_dir='saved_models',
             plot_dir='pic/ProbabilityDensity',
             segment_size=100,
             window_size=100,
             variables=['GravityX', 'GravityY', 'GravityZ', 'UserAccelerationX', 'UserAccelerationY', 'UserAccelerationZ'],
             confidence_level=0.997,
             gmmsmooth=1,
             expsmooth=0.1,
             plot=True,
             n_components=15):
    """
    整合數據處理、模型訓練、信賴區間計算及結果保存的訓練函數。

    參數:
        base_dir (str): 資料所在的目錄。
        saved_models_dir (str): 用於保存訓練後模型的目錄。
        plot_dir (str): 用於保存繪圖的目錄。
        segment_size (int): 每段時間序列的長度。
        window_size (int): 滑動窗口的步長。
        confidence_level (float): 信賴區間的置信度。
        gmmsmooth (float): GMM 模型的平滑係數。
        expsmooth (float): 指數模型的平滑係數。
        plot (bool): 是否生成並保存繪圖。
        n_components (int): GMM 模型中高斯分量的數量。
    """
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    training_data, means, variances = [], [], []

    # 遍歷資料目錄，讀取 CSV 檔案並分割數據
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("._"):
                continue  # 忽略以 '._' 開頭的文件
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path, encoding='ISO-8859-1')
                
                # 列名檢查
                missing_columns = [col for col in variables if col not in data.columns]
                if missing_columns:
                    raise KeyError(f"文件 {file} 中缺少以下列：{missing_columns}")
                
                # 取出指定列
                scaled_df = data[variables]

                # 分割時間序列段
                for i in range(0, len(scaled_df) - segment_size + 1, window_size):
                    segment = scaled_df.iloc[i:i+segment_size]
                    if len(segment) == segment_size:
                        training_data.append(segment.values)
                        means.append(segment.mean().values)
                        variances.append(segment.var().values)

    training_data_array = np.array(training_data)
    means_array, variances_array = np.array(means), np.array(variances)
    training_data_tensor = torch.tensor(training_data_array, dtype=torch.float32)

    print(f"總共有 {len(training_data)} 段時間序列數據，每段長度為 {segment_size}")



    def fit_and_plot_model(data, model, title, xlabel, ylabel, variable, filename, bins=300, fit_curve=True, plot_flag=True):
        data = data.reshape(-1)
        x_values = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
        log_prob = model.score_samples(x_values)
        pdf = np.exp(log_prob)
    
        if plot_flag:
            plt.figure(figsize=(10, 4))
            plt.hist(data, bins=bins, density=True, alpha=0.6, color='b', label=f'{variable} Density')
            if fit_curve:
                plt.plot(x_values.flatten(), pdf, 'r', linewidth=2, label='Fitted Model')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            
            plot_save_path = os.path.join(plot_dir, f'{filename}.png')
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
            plt.savefig(plot_save_path)
            plt.close()

    # variables = ['GravityX', 'GravityY', 'GravityZ','UserAccelerationX','UserAccelerationY','UserAccelerationZ']
    for i, variable in enumerate(variables):
        exp_model = CustomEMModel(n_components=1, distribution='exponential')
        exp_model.fit(variances_array[:, i])
        fit_and_plot_model(
            variances_array[:, i],
            exp_model,
            f'{variable} Variance Exponential Fit',
            'Variance',
            'Probability Density',
            variable,
            f'{variable}_variance_exp'
        )

        exp_model_path = os.path.join(saved_models_dir, f'{variable}_variance_exp.pkl')
        joblib.dump(exp_model, exp_model_path)
        # for j in range(exp_model.n_components):
        #     print(f'exp Component {j+1} for {variable} Variance:= {exp_model.variance_[j]:.8f} Mean = {exp_model.means_[j]:.4f}')

        gmm_model = CustomEMModel(n_components=n_components, distribution='gaussian')
        gmm_model.fit(means_array[:, i])
        fit_and_plot_model(
            means_array[:, i],
            gmm_model,
            f'{variable} Mean GMM Fit',
            'Mean',
            'Probability Density',
            variable,
            f'{variable}_gmm'
        )

        gmm_model_path = os.path.join(saved_models_dir, f'{variable}_gmm.pkl')
        joblib.dump(gmm_model, gmm_model_path)
        # for j in range(gmm_model.n_components):
            # print(f'GMM Component {j+1} for {variable} Mean: Mean = {gmm_model.means_[j]:.4f}, Variance = {gmm_model.covariances_[j]:.4f}')

    from confidence import compute_boundary_density_gmm, compute_boundary_density_exp

    model_paths = {}
    for variable in variables:
        model_paths[variable] = {
            "gmm": os.path.join(saved_models_dir, f'{variable}_gmm.pkl'),
            "exp": os.path.join(saved_models_dir, f'{variable}_variance_exp.pkl')
        }

    results = {}

    for variable in variables:
        gmm_model = joblib.load(model_paths[variable]['gmm'])
        exp_model = joblib.load(model_paths[variable]['exp'])
        
        boundary_density_gmm = compute_boundary_density_gmm(
            model_paths[variable]['gmm'],
            confidence_level=confidence_level,
            smooth=gmmsmooth,
            plot=False
        )
        boundary_density_exp = compute_boundary_density_exp(
            model_paths[variable]['exp'],
            confidence_level=confidence_level,
            smooth=expsmooth,
            plot=False
        )
        
        results[f"boundary_density_gmm_{variable.lower()}"] = boundary_density_gmm
        results[f"boundary_density_exp_{variable.lower()}"] = boundary_density_exp

    results_path = os.path.join(saved_models_dir, 'boundary_densities.pkl')
    joblib.dump(results, results_path)
    print(f"信賴區間結果已保存至 {results_path}")


###############多模型################### V2


### **導入模塊**

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import ruptures as rpt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches  # 用於創建自定義圖例

# **解釋：**

# - **`os`**：提供了一系列與操作系統交互的函數，如文件路徑操作、目錄操作等。
# - **`numpy`**：用於高效的數值計算和數組操作，簡稱為 `np`。
# - **`pandas`**：用於數據處理和分析的庫，提供了強大的數據結構，如 DataFrame，簡稱為 `pd`。
# - **`joblib`**：用於對象的序列化和反序列化，常用於保存和加載機器學習模型。
# - **`matplotlib.pyplot`**：用於數據可視化，簡稱為 `plt`。
# - **`ruptures`**：用於時間序列的變換點檢測。
# - **`sklearn.decomposition.PCA`**：用於主成分分析（PCA）的降維算法。
# - **`matplotlib.patches`**：用於繪制形狀（補丁），這里用於創建自定義圖例。

# ---

### **函數 1：`load_data`**

def load_data(file_path):
    """
    讀取數據並進行預處理。
    """
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data['DateTime'] = pd.to_datetime(data['Date']) + pd.to_timedelta(data['TimeStamp'], unit='s')
        return data
    else:
        print("文件不存在，請檢查文件路徑。")
        return None

# **功能：**

# - **目的**：讀取指定路徑的 CSV 數據文件，並對數據進行初步預處理。
# - **流程**：
#   1. 檢查文件路徑 `file_path` 是否存在。
#   2. 如果存在：
#      - 使用 `pd.read_csv` 讀取 CSV 文件，存入 DataFrame `data`。
#      - 將 `Date` 列轉換為日期時間格式，`TimeStamp` 列轉換為時間增量（以秒為單位），然後相加，得到新的 `DateTime` 列。
#      - 返回處理後的 `data`。
#   3. 如果不存在：
#      - 輸出錯誤信息。
#      - 返回 `None`。

# **變量：**

# - **`file_path`**：CSV 文件的路徑。
# - **`data`**：讀取的 DataFrame 數據。
# - **`data['DateTime']`**：新的日期時間列，結合了日期和時間戳信息。


### **函數 2：`load_models`**

def load_models(model_dirs, variables):
    """
    載入所有動作的模型，並支持擴展到更多變量。
    """
    action_models = {}
    for model_dir in model_dirs:
        action_name = os.path.basename(model_dir)
        models = {}
        for var in variables:
            models[f'gmm_{var.lower()}'] = joblib.load(os.path.join(model_dir, f'{var}_gmm.pkl'))
            models[f'exp_{var.lower()}'] = joblib.load(os.path.join(model_dir, f'{var}_variance_exp.pkl'))
        models['thresholds'] = joblib.load(os.path.join(model_dir, 'boundary_densities.pkl'))
        action_models[action_name] = models
    return action_models

# **功能：**

# - **目的**：加載各個動作的預訓練模型，包括高斯混合模型（GMM）和指數模型，以及閾值信息。
# - **流程**：
#   1. 初始化空字典 `action_models`，用於存儲各動作的模型。
#   2. 遍歷每個模型目錄 `model_dir`：
#      - 獲取動作名稱 `action_name`。
#      - 初始化字典 `models`，用於存儲當前動作的模型。
#      - 遍歷每個變量 `var`：
#        - 加載對應變量的 GMM 模型，存入 `models`，鍵名為 `gmm_{變量名}`。
#        - 加載對應變量的指數模型，存入 `models`，鍵名為 `exp_{變量名}`。
#      - 加載閾值信息 `boundary_densities.pkl`，存入 `models`。
#      - 將當前動作的模型字典 `models` 存入 `action_models`，鍵名為動作名稱。
#   3. 返回 `action_models`。

# **變量：**

# - **`model_dirs`**：模型目錄列表，每個目錄對應一個動作。
# - **`variables`**：變量名列表，如 `['GravityX', 'GravityY', 'GravityZ']`。
# - **`action_models`**：存儲所有動作模型的字典。
# - **`action_name`**：當前動作的名稱。
# - **`models`**：當前動作的模型字典。
# - **`var`**：當前變量名。


### **函數 3：`predict_segments`**

def predict_segments(data, action_models, variables, segment_size):
    """
    對數據進行分段並預測每個分段的動作。
    """
    total_length = len(data)
    merged_segments = [data.iloc[i:i + segment_size] for i in range(0, total_length, segment_size)]

    segment_labels = []
    for segment in merged_segments:
        if len(segment) < 1:
            continue

        # 提取每段數據的均值和方差
        mean = segment[variables].mean().values
        var = segment[variables].var().values

        action_votes = {}
        action_scores = {}

        for action_name, models in action_models.items():
            vote = 0
            total_log_likelihood = 0

            # 對每個變量進行 GMM 和指數模型的 log-likelihood 計算
            for i, variable in enumerate(variables):
                # 確保 mean 和 var 是數值數據
                if isinstance(mean[i], (int, float)) and isinstance(var[i], (int, float)):
                    # GMM 模型 log-likelihood 計算
                    log_likelihood_mean = models[f'gmm_{variable.lower()}'].score_samples(np.array(mean[i]).reshape(-1, 1))[0]
                    if log_likelihood_mean > np.log(models['thresholds'][f'boundary_density_gmm_{variable.lower()}']):
                        vote += 1

                    # 指數模型 log-likelihood 計算
                    log_likelihood_var = models[f'exp_{variable.lower()}'].score_samples(np.array(var[i]).reshape(-1, 1))[0]
                    if log_likelihood_var > np.log(models['thresholds'][f'boundary_density_exp_{variable.lower()}']):
                        vote += 1

                    total_log_likelihood += log_likelihood_mean + log_likelihood_var

            action_votes[action_name] = vote
            action_scores[action_name] = total_log_likelihood

        max_vote = max(action_votes.values())
        top_actions = [action for action, vote in action_votes.items() if vote == max_vote]

        if max_vote > (len(variables) * 2 - 1):  # 所有變量都必須通過
            if len(top_actions) == 1:
                best_action = top_actions[0]
            else:
                best_action = max(top_actions, key=lambda action: action_scores[action])
        else:
            best_action = 'unknown'

        segment_labels.append(best_action)

    initial_predictions = np.repeat(segment_labels, segment_size)[:total_length]
    data['InitialPrediction'] = initial_predictions
    return data

# **功能：**

# - **目的**：將數據分段，對每個段落進行動作預測。
# - **流程**：
#   1. 計算數據總長度 `total_length`。
#   2. 將數據按照 `segment_size` 進行分段，存入列表 `merged_segments`。
#   3. 初始化列表 `segment_labels`，用於存儲每個段落的預測動作。
#   4. 遍歷每個段落 `segment`：
#      - 如果段落為空，跳過。
#      - 計算該段落各變量的均值 `mean` 和方差 `var`。
#      - 初始化字典 `action_votes` 和 `action_scores`，用於存儲每個動作的投票數和總對數似然值。
#      - 遍歷每個動作 `action_name` 和對應的模型 `models`：
#        - 初始化 `vote` 和 `total_log_likelihood`。
#        - 遍歷每個變量 `variable`：
#          - 檢查均值和方差是否為數值。
#          - 計算均值的 GMM 對數似然值 `log_likelihood_mean`。
#            - 如果對數似然值超過閾值，`vote` 加 1。
#          - 計算方差的指數模型對數似然值 `log_likelihood_var`。
#            - 如果對數似然值超過閾值，`vote` 加 1。
#          - 累加對數似然值到 `total_log_likelihood`。
#        - 將 `vote` 和 `total_log_likelihood` 存入對應的字典。
#      - 找到最大投票數 `max_vote` 和對應的動作列表 `top_actions`。
#      - 如果 `max_vote` 超過閾值（即所有變量都通過）：
#        - 如果只有一個動作，選擇該動作為 `best_action`。
#        - 如果有多個動作，選擇 `total_log_likelihood` 最大的動作為 `best_action`。
#      - 否則，將 `best_action` 設為 `'unknown'`。
#      - 將 `best_action` 添加到 `segment_labels`。
#   5. 使用 `segment_labels` 生成與原數據長度匹配的預測結果 `initial_predictions`，並添加到數據中。
#   6. 返回更新後的數據 `data`。

# **變量：**

# - **`data`**：原始數據 DataFrame。
# - **`action_models`**：加載的動作模型字典。
# - **`variables`**：用於預測的變量列表。
# - **`segment_size`**：每個段落的大小（數據點數量）。
# - **`merged_segments`**：分段後的數據列表。
# - **`segment_labels`**：每個段落的預測動作列表。
# - **`mean`**：段落中各變量的均值數組。
# - **`var`**：段落中各變量的方差數組。
# - **`action_votes`**：每個動作的投票數字典。
# - **`action_scores`**：每個動作的總對數似然值字典。
# - **`max_vote`**：最高的投票數。
# - **`top_actions`**：投票數最高的動作列表。
# - **`best_action`**：最終選定的最佳動作。
# - **`initial_predictions`**：初始預測結果數組。


### **函數 4：`apply_pca_changepoint_detection`**

def apply_pca_changepoint_detection(data, variables, segment_size, sampling_interval=2):
    """
    使用 PCA 和變換點檢測劃分區間，支持間隔采樣以提高性能。
    """
    # 對數據進行間隔采樣
    sampled_data = data.iloc[::sampling_interval].reset_index(drop=True)
    gravity_data = sampled_data[variables]

    # 進行 PCA
    pca = PCA(n_components=1, svd_solver='randomized')
    pc1 = pca.fit_transform(gravity_data)
    sampled_data['PC1'] = pc1.flatten()

    # 進行變換點檢測
    algo = rpt.Pelt(model="l2", min_size=segment_size).fit(sampled_data['PC1'].values)
    result = algo.predict(pen=10)

    # 將變換點映射回原始數據索引
    segments = []
    start = 0
    for breakpoint in result:
        # 計算原始數據中的索引
        original_start = start * sampling_interval
        original_end = breakpoint * sampling_interval
        segments.append((original_start, min(original_end, len(data))))
        start = breakpoint

    return data, segments

# **功能：**

# - **目的**：使用 PCA 降維和變換點檢測來劃分數據區間，以識別數據中統計特性的變化點。
# - **流程**：
#   1. **間隔采樣**：
#      - 以 `sampling_interval` 為間隔對數據進行采樣，減少計算量。
#      - 重置索引，得到 `sampled_data`。
#   2. **PCA 降維**：
#      - 對采樣數據的指定變量進行 PCA，降到一維。
#      - 將第一主成分 `PC1` 添加到 `sampled_data`。
#   3. **變換點檢測**：
#      - 使用 `ruptures` 庫的 PELT 算法對 `PC1` 進行變換點檢測。
#      - 設置模型為 `"l2"`，最小段長為 `segment_size`。
#      - 使用 `pen=10` 預測變換點，結果存入 `result`。
#   4. **映射回原始數據索引**：
#      - 初始化空列表 `segments`。
#      - 遍歷每個變換點 `breakpoint`，計算對應的原始數據索引範圍，存入 `segments`。
#   5. 返回原始數據 `data` 和劃分的區間 `segments`。

# **變量：**

# - **`sampling_interval`**：采樣間隔，默認為 5。
# - **`sampled_data`**：采樣後的數據 DataFrame。
# - **`gravity_data`**：采樣數據中指定的變量列。
# - **`pca`**：PCA 模型對象。
# - **`pc1`**：第一主成分的值。
# - **`algo`**：PELT 變換點檢測算法對象。
# - **`result`**：檢測到的變換點列表。
# - **`segments`**：劃分的區間列表，每個元素是 `(start, end)`。


### **函數 5：`smooth_labels_func`**

def smooth_labels_func(data, segments, min_action_length):
    """
    平滑處理，修正短暫的誤判段落。
    """
    labels = np.array(data['FinalPrediction'])
    segments_info = []
    prev_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != prev_label:
            segments_info.append({'label': prev_label, 'start': start_idx, 'end': i})
            start_idx = i
            prev_label = labels[i]
    segments_info.append({'label': prev_label, 'start': start_idx, 'end': len(labels)})

    for idx in range(1, len(segments_info) - 1):
        segment = segments_info[idx]
        prev_segment = segments_info[idx - 1]
        next_segment = segments_info[idx + 1]

        segment_length = segment['end'] - segment['start']
        if segment_length < min_action_length and prev_segment['label'] == next_segment['label']:
            labels[segment['start']:segment['end']] = prev_segment['label']
            segment['label'] = prev_segment['label']

    # 返回平滑後的標簽數組
    return labels

# **功能：**

# - **目的**：對預測的動作標簽進行平滑處理，修正短暫的誤分類段落，減少噪聲。
# - **流程**：
#   1. 將 `FinalPrediction` 列轉換為數組 `labels`。
#   2. 初始化列表 `segments_info`，用於存儲每個連續相同標簽的段落信息。
#   3. 遍歷標簽數組，識別標簽變化的位置，記錄每個段落的開始和結束索引及其標簽。
#   4. 遍歷 `segments_info`，對中間的段落進行檢查：
#      - 如果某個段落的長度小於最小動作長度 `min_action_length`，且其前後段落的標簽相同，則認為該段落是誤判。
#      - 將該段落的標簽修改為前一個段落的標簽。
#   5. 返回平滑處理後的標簽數組。

# **變量：**

# - **`data`**：包含預測標簽的 DataFrame。
# - **`segments`**：區間列表（未在此函數中使用）。
# - **`min_action_length`**：最小動作長度，短於此長度的段落將被平滑處理。
# - **`labels`**：預測標簽的數組。
# - **`segments_info`**：存儲段落信息的列表，每個元素包含 `label`、`start`、`end`。
# - **`segment_length`**：當前段落的長度。


### **函數 6：`visualize_results`**



def visualize_results(data, labels, file_path, show_image=False):
    """
    可視化結果，輸出兩張圖，一張使用重力變量（Gravity），一張使用用戶加速度變量（UserAcceleration）。
    """
    # 第一張圖：使用 GravityX, GravityY, GravityZ
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['GravityX'], label='GravityX', color='b')
    plt.plot(data.index, data['GravityY'], label='GravityY', color='r')
    plt.plot(data.index, data['GravityZ'], label='GravityZ', color='g')

    # 獲取動作標簽
    unique_actions = set(labels)
    colors = plt.get_cmap('tab20', lut=len(unique_actions))
    action_color_map = {action: colors(i) for i, action in enumerate(unique_actions)}
    legend_patches = {}

    current_label = labels[0]
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            end_idx = i - 1
            color = action_color_map[current_label]
            plt.axvspan(start_idx, end_idx, color=color, alpha=0.3)
            if current_label not in legend_patches:
                legend_patches[current_label] = mpatches.Patch(color=color, alpha=0.3, label=current_label)
            current_label = labels[i]
            start_idx = i
    end_idx = len(labels) - 1
    color = action_color_map[current_label]
    plt.axvspan(start_idx, end_idx, color=color, alpha=0.3)
    if current_label not in legend_patches:
        legend_patches[current_label] = mpatches.Patch(color=color, alpha=0.3, label=current_label)

    plt.xlabel('Sample Index')
    plt.ylabel('Gravity Values')
    plt.title('GravityX, GravityY, GravityZ with Action Annotations')
    plt.legend(handles=list(legend_patches.values()))

    # 保存第一張圖
    output_image_path = os.path.join('pic', 'predict', f"{os.path.splitext(os.path.basename(file_path))[0]}_gravity_action_detection.png")
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path)
    if show_image:
        plt.show()
    plt.close()

    # 第二張圖：使用 UserAccelerationX, UserAccelerationY, UserAccelerationZ
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['UserAccelerationX'], label='UserAccelerationX', color='b')
    plt.plot(data.index, data['UserAccelerationY'], label='UserAccelerationY', color='r')
    plt.plot(data.index, data['UserAccelerationZ'], label='UserAccelerationZ', color='g')

    # 繼續使用相同的動作標簽顯示
    current_label = labels[0]
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            end_idx = i - 1
            color = action_color_map[current_label]
            plt.axvspan(start_idx, end_idx, color=color, alpha=0.3)
            current_label = labels[i]
            start_idx = i
    end_idx = len(labels) - 1
    color = action_color_map[current_label]
    plt.axvspan(start_idx, end_idx, color=color, alpha=0.3)

    plt.xlabel('Sample Index')
    plt.ylabel('User Acceleration Values')
    plt.title('UserAccelerationX, UserAccelerationY, UserAccelerationZ with Action Annotations')
    plt.legend(handles=list(legend_patches.values()))

    # 保存第二張圖
    output_image_path = os.path.join('pic', 'predict', f"{os.path.splitext(os.path.basename(file_path))[0]}_useracceleration_action_detection.png")
    plt.savefig(output_image_path)
    if show_image:
        plt.show()
    plt.close()


# **功能：**

# - **目的**：可視化動作檢測的結果，生成兩張圖，分別顯示重力和用戶加速度數據，並在圖中標注動作區域。
# - **流程**：
#   1. **第一張圖（重力數據）**：
#      - 創建圖形，設置尺寸。
#      - 繪制 `GravityX`、`GravityY`、`GravityZ` 隨時間變化的曲線。
#      - 獲取唯一的動作標簽，生成顏色映射 `action_color_map`。
#      - 初始化 `legend_patches`，用於圖例。
#      - 遍歷標簽數組，在動作變化的位置繪制彩色背景區域 `axvspan`，並添加圖例。
#      - 設置軸標簽、標題和圖例。
#      - 構建輸出圖像路徑，創建目錄並保存圖像。
#      - 根據參數決定是否顯示圖像，最後關閉圖形以釋放內存。
#   2. **第二張圖（用戶加速度數據）**：
#      - 重覆上述步驟，但繪制的是 `UserAccelerationX`、`UserAccelerationY`、`UserAccelerationZ` 數據。
#      - 使用相同的顏色映射和圖例，以保持一致性。

# **變量：**

# - **`data`**：包含數據和預測標簽的 DataFrame。
# - **`labels`**：預測的動作標簽數組。
# - **`file_path`**：數據文件的路徑，用於命名輸出圖像文件。
# - **`show_image`**：是否顯示圖像的布爾值。
# - **`unique_actions`**：唯一的動作標簽集合。
# - **`colors`**：顏色映射對象。
# - **`action_color_map`**：動作標簽到顏色的映射字典。
# - **`legend_patches`**：用於創建圖例的補丁字典。
# - **`current_label`**：當前動作標簽。
# - **`start_time`**、**`end_time`**：當前動作段落的起始和結束時間。
# - **`output_image_path`**：輸出圖像的文件路徑。


### **函數 7：`detect_action_segments_with_changepoint_voting`**

def detect_action_segments_with_changepoint_voting(file_path,
                                                   model_dirs,
                                                   variables=['GravityX', 'GravityY', 'GravityZ', 'UserAccelerationX', 'UserAccelerationY', 'UserAccelerationZ'],
                                                   segment_size=100, 
                                                   min_action_length=150,
                                                   smooth_labels=True,
                                                   show_image=False):
    """
    主函數：檢測動作分段，並根據變換點檢測進行投票。
    """
    data = load_data(file_path)
    if data is None:
        return None

    action_models = load_models(model_dirs, variables)
    data = predict_segments(data, action_models, variables, segment_size)
    
    if smooth_labels:
        data, segments = apply_pca_changepoint_detection(data, variables, segment_size)

        final_labels = []
        for (start, end) in segments:
            segment_labels_in_segment = data['InitialPrediction'].iloc[start:end]
            if len(segment_labels_in_segment) == 0:
                final_label = 'unknown'
            else:
                final_label = segment_labels_in_segment.value_counts().idxmax()
            final_labels.extend([final_label] * (end - start))

        data['FinalPrediction'] = final_labels
        labels = smooth_labels_func(data, segments, min_action_length)
    else:
        labels = data['InitialPrediction'].values

    # 統計並打印各動作的占比
    action_counts = pd.Series(labels).value_counts()
    total_points = len(labels)
    for action in action_counts.index:
        count = action_counts[action]
        ratio = count / total_points * 100
        print(f"{action} 占比: {ratio:.2f}%")

    visualize_results(data, labels, file_path, show_image)
    return labels  # 返回預測結果

# **功能：**

# - **目的**：這是主函數，綜合調用之前定義的函數，完成動作檢測、變換點檢測、標簽平滑和結果可視化。
# - **流程**：
#   1. **數據加載**：
#      - 調用 `load_data` 讀取數據。
#      - 如果數據為空，返回 `None`。
#   2. **模型加載**：
#      - 調用 `load_models` 加載預訓練的動作模型。
#   3. **初始預測**：
#      - 調用 `predict_segments` 對數據進行初步的動作預測，結果存入 `data['InitialPrediction']`。
#   4. **標簽平滑和變換點檢測**：
#      - 如果 `smooth_labels` 為 `True`：
#        - 調用 `apply_pca_changepoint_detection` 獲取數據的變換點 `segments`。
#        - 初始化 `final_labels` 列表。
#        - 遍歷每個段落 `(start, end)`：
#          - 獲取該段落的初始預測標簽。
#          - 通過投票（多數原則）確定該段落的最終標簽 `final_label`。
#          - 將 `final_label` 擴展到整個段落，添加到 `final_labels`。
#        - 將 `final_labels` 添加到數據中 `data['FinalPrediction']`。
#        - 調用 `smooth_labels_func` 對標簽進行平滑處理，得到最終的標簽 `labels`。
#      - 如果 `smooth_labels` 為 `False`：
#        - 直接使用初始預測結果作為最終標簽。
#   5. **統計和可視化**：
#      - 統計各動作的數量和占比，打印結果。
#      - 調用 `visualize_results` 生成並保存可視化圖像。
#   6. **返回結果**：
#      - 返回最終的預測標簽 `labels`。

# **變量：**

# - **`file_path`**：數據文件的路徑。
# - **`model_dirs`**：模型目錄列表。
# - **`variables`**：用於分析的變量列表。
# - **`segment_size`**：分段大小。
# - **`min_action_length`**：最小動作長度。
# - **`smooth_labels`**：是否進行標簽平滑的布爾值。
# - **`show_image`**：是否顯示圖像的布爾值。
# - **`data`**：包含數據和預測結果的 DataFrame。
# - **`action_models`**：加載的動作模型字典。
# - **`segments`**：變換點劃分的區間列表。
# - **`final_labels`**：變換點檢測後的最終標簽列表。
# - **`labels`**：最終的預測標簽數組。
# - **`action_counts`**：各動作的數量統計。
# - **`total_points`**：總的數據點數量。
# - **`count`**：某個動作的數量。
# - **`ratio`**：某個動作的占比。



