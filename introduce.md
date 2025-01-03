# 基於似然度與分布模型的手寫字辨識方法

## 摘要

本研究旨在透過腕表資料偵測ADHD兒童是否正在進行手寫，以此評估其專注於課業的狀況。本研究分析了腕表重力感測器數據，以區分手寫與非手寫行為，結合了似然度分析、高斯混合模型（GMM）、期望最大化（EM）算法及信賴區間理論，建立了一個有效辨識手寫行為的模型。我們對六位參與者的數據進行實驗，驗證模型在不同情境下的性能，並對結果進行深入分析。

---

## 目錄

1. [引言](#引言)
2. [理論基礎](#理論基礎)
   - 2.1. [似然度（Likelihood）的概念](#21-似然度likelihood的概念)
   - 2.2. [機率、機率密度與似然度的區別](#22-機率機率密度與似然度的區別)
   - 2.3. [閾值的選擇與信賴區間](#23-閾值的選擇與信賴區間)
     - 2.3.1. [指數分佈的信賴區間估計](#231-指數分佈的信賴區間估計)
     - 2.3.2. [高斯分佈的信賴區間估計](#232-高斯分佈的信賴區間估計)
3. [高斯混合模型（GMM）](#高斯混合模型gmm)
   - 3.1. [模型的定義與背景](#31-模型的定義與背景)
   - 3.2. [數學描述](#32-數學描述)
4. [最大似然估計與 GMM 中的似然函數](#最大似然估計與-gmm-中的似然函數)
   - 4.1. [最大似然估計（MLE）](#41-最大似然估計mle)
   - 4.2. [GMM 中的似然函數](#42-gmm-中的似然函數)
5. [期望最大化（EM）算法](#期望最大化em算法)
   - 5.1. [EM 的基本流程](#51-em-的基本流程)
   - 5.2. [EM 在 GMM 中的應用](#52-em-在-gmm-中的應用)
6. [手寫字辨識演算法介紹](#手寫字辨識演算法介紹)
   - 6.1. [資料收集](#61-資料收集)
   - 6.2. [資料預處理](#62-資料預處理)
   - 6.3. [模型構建](#63-模型構建)
     - 6.3.1. [高斯混合模型擬合](#631-平均值的高斯混合模型擬合)
     - 6.3.2. [指數分佈擬合](#632-變異數的指數分佈擬合)
   - 6.4. [信賴區間的設置](#64-信賴區間的設置)
     - 6.4.1. [高斯混合模型的信賴區間](#641-高斯混合模型的信賴區間)
     - 6.4.2. [指數分佈的信賴區間](#642-指數分佈的信賴區間)
7. [實驗與結果](#實驗與結果)
   - 7.1. [實驗一：未收集受測者資料的情況（留一交叉驗證）](#71-實驗一未收集受測者資料的情況留一交叉驗證)
   - 7.2. [實驗二：已收集受測者手寫資料的情況](#72-實驗二已收集受測者手寫資料的情況)
8. [結論](#結論)
9. [參考文獻](#參考文獻)

---

## 引言

智慧型裝置的普及使得透過感測器數據來辨識使用者行為成為熱門的研究領域。手寫行為的辨識在許多方面具有重要應用價值，例如專注力評估、行為分析及學習狀態的監測等。本研究針對ADHD兒童的手寫行為辨識進行深入探討，利用腕表重力感測器數據，結合模型分布以及信賴區間理論，構建一個能有效區分手寫與非手寫行為的模型。我們對數據進行實驗以驗證模型的性能，並分析其在實際應用中的表現。

---

## 理論基礎

### 2.1. 似然度（Likelihood）的概念

**似然度（Likelihood）** 是在給定觀測數據的情況下，模型參數的可能性度量。與機率不同，似然度將數據視為已知，評估不同參數下數據出現的可能性。數學上，對於參數 $\theta$ 和觀測數據 $X$，似然函數定義為：

$$
L(\theta | X) = P(X | \theta)
$$

我們的目標是找到使似然函數最大的參數 $\theta$，即最大似然估計。

### 2.2. 機率、機率密度與似然度的區別

- **機率（Probability）**：對於離散隨機變量，機率是事件發生的可能性，滿足 
$$
0 \leq P(X = x) \leq 1
$$  
and  
$$\sum_x P(X = x) = 1
$$
- **機率密度函數（Probability Density Function, PDF）**：對於連續隨機變量，機率密度函數 $f(x)$ 滿足 $f(x) \geq 0$，且 $\int_{-\infty}^{\infty} f(x) dx = 1$。機率為某區間上的積分：

  $$
  P(a \leq X \leq b) = \int_{a}^{b} f(x) dx
  $$

- **似然度（Likelihood）**：給定觀測數據 $X$，似然函數 $L(\theta | X)$ 描述了不同參數 $\theta$ 下數據出現的可能性。

### 2.3. 閾值的選擇與信賴區間

在統計學中，**信賴區間**是對未知參數的一種區間估計，表示參數落在某個區間的概率。我們使用信賴區間來設定閾值，判斷新數據是否屬於某個分佈。

#### 2.3.1. 指數分佈的信賴區間估計

**指數分佈**適用於描述隨機事件發生的時間間隔，其概率密度函數為：

$$
f(x; \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

其中，$\lambda > 0$ 是速率參數。累積分佈函數為：

$$
F(x; \lambda) = 1 - e^{-\lambda x}
$$

##### （1）參數估計

給定樣本 $\{ x_1, x_2, \dots, x_n \}$，指數分佈的參數 $\lambda$ 的最大似然估計為：

$$
\hat{\lambda} = \frac{n}{\sum_{i=1}^{n} x_i}
$$

##### （2）信賴區間的構建

我們利用似然比檢驗或直接根據樣本分佈，使用卡方分佈構建 $\lambda$ 的信賴區間。對於樣本均值 $\bar{x}$，有：

$$
2n\lambda \bar{x} \sim \chi^2_{2n}
$$

因此，$\lambda$ 的 $(1 - \alpha)$ 信賴區間為：

$$
\left[ \frac{2n}{\chi^2_{2n, 1 - \alpha/2} \cdot \bar{x}}, \frac{2n}{\chi^2_{2n, \alpha/2} \cdot \bar{x}} \right]
$$

#### 2.3.2. 高斯分佈的信賴區間估計

對於**高斯分佈** $\mathcal{N}(\mu, \sigma^2)$，其概率密度函數為：

$$
f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

##### （1）參數估計

給定樣本 $\{ x_1, x_2, \dots, x_n \}$，均值 $\mu$ 和方差 $\sigma^2$ 的估計值為：

$$
\hat{\mu} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

$$
\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{\mu})^2
$$

##### （2）信賴區間的構建

- **對於均值 $\mu$**，當 $\sigma^2$ 已知時，樣本均值 $\hat{\mu}$ 的分佈為：

$$
\hat{\mu} \sim \mathcal{N}\left( \mu, \frac{\sigma^2}{n} \right)
$$

  因此，$\mu$ 的 $(1 - \alpha)$ 信賴區間為：

$$
\left[ \hat{\mu} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}, \hat{\mu} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \right]
$$

  其中，$z_{\alpha/2}$ 是標準正態分佈的臨界值。

- **對於方差 $\sigma^2$**，使用 $\chi^2$ 分佈構建信賴區間：

$$
\frac{(n - 1)\hat{\sigma}^2}{\sigma^2} \sim \chi^2_{n - 1}
$$

  因此，$\sigma^2$ 的 $(1 - \alpha)$ 信賴區間為：

$$
\left[ \frac{(n - 1)\hat{\sigma}^2}{\chi^2_{n - 1, 1 - \alpha/2}}, \frac{(n - 1)\hat{\sigma}^2}{\chi^2_{n - 1, \alpha/2}} \right]
$$

---

## 高斯混合模型（GMM）

### 3.1. 模型的定義與背景

**高斯混合模型（Gaussian Mixture Model, GMM）** 是多個高斯分佈的線性組合，用於描述具有多模態的數據分佈。GMM 常用於聚類和密度估計，特別適合於數據呈現多個峰值的情況。

### 3.2. 數學描述

GMM 的概率密度函數為：

$$
p(x) = \sum_{i=1}^{K} w_i \cdot \mathcal{N}(x; \mu_i, \sigma_i^2)
$$

其中：

- $K$：高斯成分的數量。
- $w_i$：第 $i$ 個成分的權重，滿足 $\sum_{i=1}^{K} w_i = 1$。
- $\mathcal{N}(x; \mu_i, \sigma_i^2)$：第 $i$ 個高斯成分的概率密度函數，定義為：

$$
\mathcal{N}(x; \mu_i, \sigma_i^2) = \frac{1}{\sqrt{2\pi \sigma_i^2}} \exp\left( -\frac{(x - \mu_i)^2}{2\sigma_i^2} \right)
$$

---

## 最大似然估計與 GMM 中的似然函數

### 4.1. 最大似然估計（MLE）

**最大似然估計（Maximum Likelihood Estimation, MLE）**旨在找到參數 $\theta$，使得在該參數下觀測數據的似然度最大。對於獨立同分佈的樣本 $\{ x_1, x_2, \dots, x_n \}$，似然函數為：

$$
L(\theta) = \prod_{i=1}^{n} f(x_i; \theta)
$$

為了計算方便，通常取對數似然函數：

$$
\ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln f(x_i; \theta)
$$

### 4.2. GMM 中的似然函數

在 GMM 中，似然函數為：

$$
L(\{ w_i, \mu_i, \sigma_i^2 \}_ {i=1}^{K}) = \prod_{j=1}^{n} \sum_{i=1}^{K} w_i \cdot \mathcal{N}(x_j; \mu_i, \sigma_i^2)
$$

直接最大化該似然函數較為複雜，因此採用 **期望最大化（EM）算法** 進行參數估計。

---

## 期望最大化（EM）算法

### 5.1. EM 的基本流程

EM 算法是一種迭代方法，用於含有隱變量的模型參數估計。基本步驟：

1. **初始化**：設定初始參數 $\theta^{(0)}$。
2. **E 步（Expectation）**：計算給定當前參數下的隱變量的期望值。
3. **M 步（Maximization）**：最大化對數似然函數，更新參數估計值。
4. **迭代**：重複 E 步和 M 步，直到參數收斂。

### 5.2. EM 在 GMM 中的應用

在 GMM 中，隱變量表示樣本 $x_j$ 來自於第 $i$ 個成分的概率。EM 步驟如下：

**E 步**：計算後驗概率（責任度） $\gamma_{ij}$：

$$
\gamma_{ij} = \frac{w_i \cdot \mathcal{N}(x_j; \mu_i, \sigma_i^2)}{\sum_{k=1}^{K} w_k \cdot \mathcal{N}(x_j; \mu_k, \sigma_k^2)}
$$

**M 步**：更新參數 $w_i$、$\mu_i$、$\sigma_i^2$：

- 更新權重 $w_i$：

$$
w_i^{(t+1)} = \frac{1}{n} \sum_{j=1}^{n} \gamma_{ij}
$$

- 更新均值 $\mu_i$：

$$
\mu_i^{(t+1)} = \frac{\sum_{j=1}^{n} \gamma_{ij} x_j}{\sum_{j=1}^{n} \gamma_{ij}}
$$

- 更新方差 $\sigma_i^2$：

$$
(\sigma_i^2)^{(t+1)} = \frac{\sum_{j=1}^{n} \gamma_{ij} (x_j - \mu_i^{(t+1)})^2}{\sum_{j=1}^{n} \gamma_{ij}}
$$

---

## 手寫字辨識演算法介紹

### 6.1. 資料收集

我們從六位參與者處收集了兩類行為數據：

1. **手寫資料**：參與者進行純手寫活動，總計 **337,246** 筆記錄。
2. **手寫後進行其他活動的資料**：參與者在完成手寫後進行其他活動，如畫畫、轉筆等，初始包含 **54,810** 筆記錄。

以下為一筆測試資料，反映了手寫過程中的轉筆動作。

**圖 1：手寫過程中轉筆動作的時序圖**

![時序圖](https://hackmd.io/_uploads/ryLsZfb1ke.png)

*說明：圖中展示了重力感測器數據隨時間的變化，可以觀察到在轉筆動作時，重力值出現明顯的波動。*

### 6.2. 資料預處理

為了降低數據的雜訊影響，我們對數據進行了以下預處理：

- **統計量計算**：每秒（100 筆數據）計算一次平均值和變異數，使用移動窗格方式，步長為 50 筆數據。
- **特徵提取**：主要關注 'GravityX'、'GravityY'、'GravityZ' 三個變數的平均值和變異數。
- **結果**：最終獲得 **7,863** 組特徵資料。

以下是各類動作的統計圖表。

**圖 2：各類動作的統計圖表**

![各類動作統計圖表](https://hackmd.io/_uploads/S1hxXfZyye.jpg)

*說明：圖中展示了不同動作的平均值和變異數分佈，可以明顯看出手寫與其他活動之間的差異。*

我們對 'GravityX'、'GravityY'、'GravityZ' 的概率密度進行了分析。

**圖 3：GravityX 的概率密度分佈**

![GravityX_densityplot_subplot](https://hackmd.io/_uploads/ByFaDDZy1g.png)

**圖 4：GravityY 的概率密度分佈**

![GravityY_densityplot_subplot](https://hackmd.io/_uploads/Hkqk_v-1yl.png)

**圖 5：GravityZ 的概率密度分佈**

![GravityZ_densityplot_subplot](https://hackmd.io/_uploads/r10JOP-1ke.png)

*說明：從上述圖表可見，手寫與其他活動的數據分佈存在顯著差異，這為我們後續的模型構建提供了依據。*

### 6.3. 模型構建

#### 6.3.1. 平均值的高斯混合模型擬合

我們對手寫資料的平均值進行了高斯混合模型擬合。以下是擬合結果：

**圖 6：GravityX 平均值的高斯混合模型擬合**

![GravityX_gmm.pkl](https://hackmd.io/_uploads/B1az2vZ1ye.png)

**圖 7：GravityY 平均值的高斯混合模型擬合**

![GravityY_gmm.pkl](https://hackmd.io/_uploads/r1crTmGkJx.png)

**圖 8：GravityZ 平均值的高斯混合模型擬合**

![GravityZ_gmm.pkl](https://hackmd.io/_uploads/BkWV2vW1Jl.png)

*說明：圖中展示了實際數據的直方圖以及擬合的高斯混合模型曲線，可以看到模型較好地描述了數據的分佈。*

#### 6.3.2. 變異數的指數分佈擬合

對於變異數，由於其非負且集中在零附近，我們選擇使用指數分佈進行擬合。

**圖 9：GravityX 變異數的指數分佈擬合**

![GravityX_variance_emm.pkl](https://hackmd.io/_uploads/SkYhnw-y1g.png)

**圖 10：GravityY 變異數的指數分佈擬合**

![GravityY_variance_emm.pkl](https://hackmd.io/_uploads/H1hNaXf1yx.png)

**圖 11：GravityZ 變異數的指數分佈擬合**

![GravityZ_variance_emm.pkl](https://hackmd.io/_uploads/BJMk6Pbykg.png)

*說明：圖中展示了變異數的實際分佈和擬合的指數分佈曲線，結果表明指數分佈能夠較好地描述變異數的分佈特性。*

### 6.4. 信賴區間的設置

#### 6.4.1. 高斯混合模型的信賴區間

由於高斯混合模型是多個高斯分佈的組合，傳統的信賴區間無法直接應用。我們採用了一種基於累積概率的方法，來構建 GMM 的信賴區間。

**（1）問題描述**

高斯混合模型的累積分佈函數（CDF）沒有解析解，因此我們需要使用數值方法計算信賴區間，即找到一個區間，使該區間內的累積概率等於目標信心水準（如 95%）。

**（2）方法步驟**

1. **定義概率密度函數**

   高斯混合模型的概率密度函數為：

$$
p(x) = \sum_{i=1}^{K} w_i \cdot \mathcal{N}(x; \mu_i, \sigma_i^2)
$$

2. **生成密度函數的網格**

   在數據範圍內建立一個細密的網格點 $x_1, x_2, \dots, x_N$，計算每個點的概率密度 $p(x_i)$。

3. **排序網格點**

   將網格點按照其對應的密度值從高到低排序，得到排序後的密度序列 $p(x_{(1)}), p(x_{(2)}), \dots, p(x_{(N)})$。

4. **計算累積概率**

   對排序後的密度值進行累積，計算累積概率：

$$
A_{\text{cumulative}} = \sum_{i=1}^{M} p(x_{(i)}) \Delta x
$$

   其中，$\Delta x$ 是網格間距，$M$ 是滿足 $A_{\text{cumulative}} \geq \alpha$ 的最小整數，$\alpha$ 是信心水準（如 0.95）。

5. **確定信賴區間**

   將累積概率達到 $\alpha$ 的網格點範圍作為信賴區間：

$$
[x_{\text{min}}, x_{\text{max}}] = [\min\{ x_{(1)}, \dots, x_{(M)} \}, \max\{ x_{(1)}, \dots, x_{(M)} \}]
$$

**（3）數學解釋**

這種方法實際上是在概率密度函數上找到一個最高密度區域，使該區域內的累積概率達到所需的信心水準。這種區間稱為 **最高密度區間（Highest Density Interval, HDI）**。

**（4）示例**

以標準正態分佈為例，我們可以通過上述方法計算其 95% 信賴區間。

**圖 12：標準正態分佈的信賴區間**

![標準正態分佈的信賴區間](https://hackmd.io/_uploads/B1jpLzG1Jx.jpg)

*說明：圖中陰影部分表示累積概率達到 95% 的區域，對應的信賴區間約為 \([-1.96, 1.96]\)。*

將此方法應用於高斯混合模型，我們得到以下結果：

**圖 13：高斯混合模型的信賴區間**

![高斯混合模型的信賴區間](https://hackmd.io/_uploads/HyGTUGzykg.jpg)

*說明：圖中陰影部分表示累積概率達到 95% 的區域，該區域由密度最高的部分組成，可能是不連續的區間。*

**（5）注意事項**

- **非連續性**：由於 GMM 的特性，信賴區間可能是不連續的多個區間之和。
- **計算效率**：需要在密度函數上進行高精度的數值積分，計算量較大。
- **應用性**：該方法適用於任何概率密度函數未知解析解但可數值計算的情況。

#### 6.4.2. 指數分佈的信賴區間

對於指數分佈，信賴區間可以通過累積分佈函數直接計算：

$$
F(x; \lambda) = 1 - e^{-\lambda x}
$$

求解 $x$ 使得：

$$
F(x; \lambda) = \alpha
$$

則：

$$
x = -\frac{\ln(1 - \alpha)}{\lambda}
$$

對於 95% 信心水準，$\alpha = 0.95$，則：

$$
x_{0.95} = -\frac{\ln(0.05)}{\lambda}
$$

---

## 實驗與結果

### 7.1. 實驗一：未收集受測者資料的情況（留一交叉驗證）

**實驗設計**：

- 使用五位參與者的手寫資料建立模型。
- 第六位參與者的手寫和非手寫資料作為測試集。
- 重複上述過程，對每位參與者進行測試。

**判斷標準**：

- 若新數據的所有特徵（平均值和變異數）皆落在信賴區間內，則判定為手寫；否則為非手寫。

**結果**：

- **混淆矩陣**：

  |                   | 預測為非手寫（負類別） | 預測為手寫（正類別） |
  |-------------------|--------------------|----------------|
  | **實際為非手寫** | TN: 1,194          | FP: 141        |
  | **實際為手寫**   | FN: 16             | TP: 114        |

- **性能指標**：

  - 準確率（Accuracy）：

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{114 + 1,194}{114 + 1,194 + 141 + 16} \approx 89.28\%
$$

  - 精確率（Precision）：

$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{114}{114 + 141} \approx 44.71\%
$$

  - 召回率（Recall）：

$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{114}{114 + 16} \approx 87.69\%
$$

  - F1 分數（F1 Score）：

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \approx 59.52\%
$$

**討論**：

- **精確率偏低**：模型在預測手寫時誤報率較高，可能是因為個體差異導致手寫特徵的變異。
- **召回率較高**：模型能夠檢測到大部分的手寫樣本，對手寫行為較為敏感。

### 7.2. 實驗二：已收集受測者手寫資料的情況

**實驗設計**：

- 使用所有參與者的部分手寫資料建立模型。
- 測試集包含參與者的額外手寫資料和非手寫資料。

以下為多個測試結果：

**圖 14：手寫辨識結果示例 1**

![handwriting_detection](https://hackmd.io/_uploads/rko5u_WJke.png)

*說明：圖中紅色區域表示被識別為手寫的時間段，可以看到模型能夠準確地辨識手寫活動。*

**圖 15：手寫辨識結果示例 2**

![handwriting_detection](https://hackmd.io/_uploads/HJTCu_W1Jx.png)

*說明：同樣地，模型對手寫活動的辨識效果良好，非手寫活動未被錯誤識別。*

**圖 16：手寫辨識結果示例 3**

![handwriting_detection](https://hackmd.io/_uploads/Hy5gFOWyyg.png)

**圖 17：手寫辨識結果示例 4**

![handwriting_detection](https://hackmd.io/_uploads/rJZGY_Zkkl.png)

**圖 18：手寫辨識結果示例 5**

![handwriting_detection](https://hackmd.io/_uploads/r1zNY_-J1l.png)

**圖 19：手寫辨識結果示例 6**

![handwriting_detection](https://hackmd.io/_uploads/SyjHt_Z1yg.png)

**圖 20：手寫辨識結果示例 7**

![handwriting_detection](https://hackmd.io/_uploads/SkfDKuZ1Jx.png)

**圖 21：模仿手寫方式的繪畫活動辨識結果**

![handwriting_detection](https://hackmd.io/_uploads/Bk-7DMG1Jg.png)

*說明：在圖 21 中，參與者刻意以模仿手寫的方式進行繪畫，模型在 15:37:15 之前識別為手寫，之後的活動未被識別為手寫，說明模型具有一定的鑑別能力。*

**結果**：

- **混淆矩陣**：

  |                   | 預測為非手寫（負類別） | 預測為手寫（正類別） |
  |-------------------|--------------------|----------------|
  | **實際為非手寫** | TN: 26,500         | FP: 300        |
  | **實際為手寫**   | FN: 3,229          | TP: 9,449      |

- **性能指標**：

  - 準確率（Accuracy）：

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{9,449 + 26,500}{9,449 + 26,500 + 300 + 3,229} \approx 91.06\%
$$

  - 精確率（Precision）：

$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{9,449}{9,449 + 300} \approx 96.92\%
$$

  - 召回率（Recall）：

$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{9,449}{9,449 + 3,229} \approx 74.53\%
$$

  - F1 分數（F1 Score）：

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \approx 84.26\%
$$

**討論**：

- **精確率顯著提高**：模型熟悉受測者的手寫特徵，誤報率降低。
- **召回率略有下降**：部分手寫樣本未被識別，可能是因為手寫與其他活動特徵相似。

---

## 結論

本研究成功地將統計學理論應用於手寫辨識，透過重力感測器數據，結合高斯混合模型和指數分佈，構建了一個有效的辨識模型。

- **未收集受測者資料的情況**：

  - 模型準確率約為 **89.28%**。
  - 精確率較低（**44.71%**），召回率較高（**87.69%**）。
  - 模型對新受測者的泛化能力有限。

- **已收集受測者手寫資料的情況**：

  - 模型準確率提高至 **91.06%**。
  - 精確率顯著提升至 **96.92%**，召回率為 **74.53%**。
  - 模型對已知受測者的辨識能力增強。

**未來工作方向**：

- **提升模型泛化能力**：增加參與者數量，採用更複雜的模型。
- **改進特徵工程**：引入更多感測器數據或提取更豐富的特徵。
- **平衡精確率與召回率**：根據應用需求，調整模型以達到最佳性能。

---

## 參考文獻

1. 先前吳老師的研究指出，不同動作的統計量可能明顯區分。
