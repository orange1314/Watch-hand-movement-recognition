import os
import joblib
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, multivariate_normal

warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak on Windows with MKL')


# GMM 相關函數
def compute_mean_boundary_density_gmm(model_path, confidence_level=0.95, smooth=1, plot=True):
    """ 計算高斯混合模型 (GMM) 的邊界密度 """
    
    # Step 1: 加載 GMM 模型
    gmm_model = joblib.load(model_path)
    
    # 提取 GMM 參數
    weights_gmm = gmm_model.weights_
    means_gmm = gmm_model.means_
    covariances_gmm = gmm_model.covariances_

    # 處理協方差矩陣
    if covariances_gmm.ndim == 1:
        std_devs_gmm = np.sqrt(covariances_gmm)  # 對角協方差
    else:
        std_devs_gmm = np.sqrt(np.array([np.diag(cov) for cov in covariances_gmm]))  # 完整協方差矩陣
    
    means_gmm = means_gmm.flatten()
    std_devs_gmm = std_devs_gmm.flatten()

    # GMM 概率密度函數
    def gmm_pdf(x, weights, means, covariances):
        pdf = np.zeros_like(x)
        for w, m, cov in zip(weights, means, covariances):
            pdf += w * multivariate_normal.pdf(x, mean=m, cov=cov)
        return pdf

    # Step 2: 生成 x 值並計算 GMM PDF
    x_min = means_gmm.min() - 5 * std_devs_gmm.max()
    x_max = means_gmm.max() + 5 * std_devs_gmm.max()
    x_values_gmm = np.linspace(x_min, x_max, 100000)
    pdf_values_gmm = gmm_pdf(x_values_gmm, weights_gmm, means_gmm, covariances_gmm)
    
    # 繪圖
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(x_values_gmm, pdf_values_gmm, label='GMM PDF', color='black', lw=2)
        plt.title('Gaussian Mixture Model (GMM) Distribution')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Step 3: 自適應積分方法找到置信區間
    def adaptive_integration(pdf_function, x_values, weights, means, covariances, target_area=confidence_level):
        pdf_values = pdf_function(x_values, weights, means, covariances)
        sorted_indices = np.argsort(-pdf_values)
        sorted_x_values = x_values[sorted_indices]
        sorted_pdf_values = pdf_values[sorted_indices]
        cumulative_area = 0
        delta_x = x_values[1] - x_values[0]
        marked_x_ranges = []
        for i in range(len(sorted_pdf_values)):
            area_contribution = sorted_pdf_values[i] * delta_x
            cumulative_area += area_contribution
            marked_x_ranges.append(sorted_x_values[i])
            if cumulative_area >= target_area:
                break
        boundary_density = sorted_pdf_values[i]
        return marked_x_ranges, boundary_density
    
    # 計算邊界密度
    marked_x_ranges_gmm, boundary_density_gmm = adaptive_integration(gmm_pdf, x_values_gmm, weights_gmm, means_gmm, covariances_gmm)

    # 繪制自適應積分區域
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(x_values_gmm, pdf_values_gmm, label='GMM PDF', color='black', lw=2)
        plt.scatter(marked_x_ranges_gmm, gmm_pdf(np.array(marked_x_ranges_gmm), weights_gmm, means_gmm, covariances_gmm),
                    color='green', label=f'Top {confidence_level * 100}% Integrated Area', s=10)
        plt.title('Adaptive Integration - GMM 95% Area')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.show()

    return boundary_density_gmm * smooth


# 指數分布相關函數
def compute_mean_boundary_density_exp(model_path, confidence_level=0.95, smooth=1, plot=True):
    """ 計算指數分布的邊界密度 """
    
    # 加載模型
    model = joblib.load(model_path)
    
    # 獲取 λ 參數
    lambda_value = model.lambdas_[0]

    # 計算均值 (1/λ)
    scale = 1 / lambda_value
    confidence_interval = expon.interval(confidence_level, scale=scale)

    # 計算邊界密度
    lower_density = expon.pdf(confidence_interval[0], scale=scale)
    upper_density = expon.pdf(confidence_interval[1], scale=scale)
    boundary_density = (lower_density + upper_density) / 2  # 平均邊界密度

    if plot:
        # 繪制指數分布 PDF
        x_values = np.linspace(0, confidence_interval[1] * smooth, 1000)
        pdf_values = expon.pdf(x_values, scale=scale)

        plt.figure(figsize=(8, 6))
        plt.plot(x_values, pdf_values, label=f'Exponential PDF (λ = {lambda_value:.4f})')
        plt.fill_between(x_values, 0, pdf_values, where=(x_values >= confidence_interval[0]) & (x_values <= confidence_interval[1]), 
                         color='gray', alpha=0.5, label=f'{confidence_level*100:.0f}% Confidence Interval')
        plt.axvline(confidence_interval[0], color='red', linestyle='--', label=f'Lower Bound: {confidence_interval[0]:.4f}')
        plt.axvline(confidence_interval[1], color='green', linestyle='--', label=f'Upper Bound: {confidence_interval[1]:.4f}')
        plt.title(f'Exponential Distribution with Confidence Interval (λ = {lambda_value:.4f})')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        plt.show()

    return boundary_density * smooth


# 處理模型文件的函數
def compute_threshold_for_models(models_dir, confidence_level=0.95, smooth=1, plot=True):
    """ 處理 GMM 模型並計算邊界密度 """
    variables = ['GravityX', 'GravityY', 'GravityZ']
    thresholds = {}
    for variable in variables:
        model_path = os.path.join(models_dir, f'{variable}_gmm.pkl')
        mean_boundary_density = compute_mean_boundary_density_gmm(model_path, confidence_level, smooth, plot)
        thresholds[variable] = mean_boundary_density
    return thresholds
