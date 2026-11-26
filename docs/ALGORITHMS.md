# Multivariate Fault Detection Algorithms

1. **`robust_mad`**: Uses per-metric Median and Median Absolute Deviation ($\text{scale} = 1.4826 \cdot \text{MAD}$) so extreme fault spikes do not corrupt baseline statistics.
2. **`ewma_residual`**: Tracks short-term expected level via Exponentially Weighted Moving Average ($\alpha = 0.25$) and flags sudden step changes or spikes.
3. **`mahalanobis`**: Computes $d(z) = \sqrt{z^T (\Sigma + \lambda I)^{-1} z}$ on robustly standardized multivariate vectors, catching **correlated degradation** where individual metrics appear borderline normal.
4. **`pca_reconstruction`**: Projects standardized metrics onto top-$k$ principal components (85% variance via SVD) and measures Squared Prediction Error (SPE) $\|z - \hat{z}\|_2$.
5. **`ensemble`**: Combines peak multivariate/univariate evidence with consensus weighting:
   $$s = 0.55 \max(s_{\text{mad}}, s_{\text{maha}}, s_{\text{pca}}) + 0.25 s_{\text{mad}} + 0.15 s_{\text{maha}} + 0.05 s_{\text{ewma}}$$
