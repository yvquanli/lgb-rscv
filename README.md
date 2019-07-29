# RSCV for LightGBM
RSCV (Randomized SearchCV), customized for LightGBM phase save search results, with CV, with early stop, not restricted by the sklearn framework of lightgbm custom RandomSearchCV, welcome to download and use.
为LightGBM定制的RSCV(Randomized Search CV)，阶段性保存搜索结果，带CV，带早停的，不受sklearn框架制约的 lightgbm 定制RandomSearchCV，欢迎大家下载使用

# 回归
    from lgb_rscv import RandomizedSearchCV
    rscv = RandomizedSearchCV(n_iter=20)
    for i in range(20):
    	rscv.fit(x_train, y_train)
    	rscv.results_store(path='../output/rscv_results{}.xls'.format(i))
# 分类
    from lgb_rscv import RandomizedSearchCV
    rscv = RandomizedSearchCV(objective='multiclass'， n_iter=20)
    for i in range(20):
    	rscv.fit(x_train, y_train)
    	rscv.results_store(path='../output/rscv_results{}.xls'.format(i))

## 参数的范围？  
因为本人对LightGBM模型理解并不深，代码中设置的参数范围可能有所不妥，如果您有任何想法，请联系我  

## 联系我
如果有任何疑问，请联系   
Email: yvquan.li@gmail.com  
QQ: 1581071905  

## 一些疑问
Q1: 为什么不让RSCV并行化？  
A1: LightGBM可以处理器全线并开
