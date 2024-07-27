# Time series forecasting with DeepAR
This is a n improved (fixed) implementation of the timeseries forecasting algorithm DeepAR by Amazon.
- Original paper: https://arxiv.org/abs/1704.04110
- Documentation by Amazon: https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_how-it-works.html

NOTES: 
- I am providing ra clear implementation in a Jupyter Notebook and clean Cython 3, without requiring SageMaker.
- Be careful with your `pandas` version as `freq` parameter has been recently deprecated for `df.Timestamp()`
- If you are using Google Collab to run the code with a `TPU` it might fail with error `TypeError: cannot pickle 'generator' object`, due to PyTorch's generator handling. Therefore, you might need to run in via a CPU instance itself
