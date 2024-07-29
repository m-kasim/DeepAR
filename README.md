![DeepAR results](https://raw.githubusercontent.com/m-kasim/DeepAR/main/assets/01_deepar_results.png)
# Time series forecasting with DeepAR
This is an improved (fixed) implementation of the timeseries forecasting algorithm DeepAR by Amazon. I am providing a clear implementation in a Jupyter Notebook and clean Cython 3, without requiring SageMaker.

- Original paper: https://arxiv.org/pdf/1704.04110
- Documentation by Amazon: https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_how-it-works.html

### What is DeepAR?
DeepAR is an algorithm developed by Amazon Research producing accurate probabilistic timeseries forecasts, based on training an auto regressive recurrent network model on a large number of related time series.

### Know issues with selected python module versions
- Be careful with your `pandas` version as `freq` parameter has been recently deprecated for `df.Timestamp()`
- If you are using Google Collab to run the code with a `TPU` it might fail with error `TypeError: cannot pickle 'generator' object`, due to PyTorch's generator handling. Therefore, you might need to run in via a CPU instance itself
