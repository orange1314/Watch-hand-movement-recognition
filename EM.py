import numpy as np
from scipy.stats import expon
from sklearn.mixture import GaussianMixture

# 自定義的 EM 模型 (支持高斯混合模型和指數分布)
class CustomEMModel:
    def __init__(self, n_components=1, distribution='gaussian', random_state=None):
        # 對於高斯分布，允許設置 n_components，但指數分布固定 n_components 為 1
        if distribution == 'exponential':
            self.n_components = 1  # 不管輸入多少，指數分佈只會有一個成分
        else:
            self.n_components = n_components  # 高斯分佈則使用設置的 n_components

        self.distribution = distribution
        self.random_state = np.random.RandomState(random_state)

        if self.distribution == 'gaussian':
            self.model = GaussianMixture(n_components=self.n_components, random_state=random_state)
            self.means_ = None  # 初始化為 None，稍後設置
            self.covariances_ = None  # 初始化為 None，稍後設置
            self.weights_ = None  # 初始化為 None，稍後設置
        elif self.distribution == 'exponential':
            self.lambdas_ = np.zeros(1)  # 用於保存 λ 參數
            self.means_ = None  # 指數分布的均值
            self.variance_ = None  # 指數分布的方差
            self.std_dev_ = None  # 指數分布的標準差

    def fit(self, data):
        if self.distribution == 'gaussian':
            # 使用 sklearn 的 GaussianMixture 進行高斯混合模型的擬合
            self.model.fit(data.reshape(-1, 1))
            self.weights_ = self.model.weights_
            self.means_ = self.model.means_.flatten()
            self.covariances_ = self.model.covariances_.flatten()
        elif self.distribution == 'exponential':
            # 使用 scipy 的 expon.fit 估計 λ 參數
            loc, scale = expon.fit(data, floc=0)  # loc 是位置參數，scale 是 1/λ
            # scale = scale*4
            self.lambdas_ = np.array([1 / scale])  # 保存 λ

            # 計算並保存均值、方差、標準差
            self.means_ = np.array([1 / self.lambdas_[0]])  # 均值是 1/λ
            self.variance_ = np.array([expon.var(scale=1 / self.lambdas_[0])])  # 方差
            self.std_dev_ = np.array([expon.std(scale=1 / self.lambdas_[0])])  # 標準差

    def score_samples(self, data):
        if self.distribution == 'gaussian':
            # 使用 GaussianMixture 的 score_samples 來計算 log-likelihood
            return self.model.score_samples(data.reshape(-1, 1))
        elif self.distribution == 'exponential':
            # 對於指數分布，計算 log-likelihood
            log_likelihood = expon.logpdf(data, scale=1 / self.lambdas_[0]).flatten()
            return log_likelihood

    def get_params(self):
        if self.distribution == 'exponential':
            # 返回和高斯混合模型相似的參數格式
            return {"Means": self.means_, "Variance": self.variance_, "Standard Deviation": self.std_dev_}
        elif self.distribution == 'gaussian':
            # 返回高斯混合模型的權重、均值和方差
            return {"Means": self.means_, "Covariances": self.covariances_, "Weights": self.weights_}
