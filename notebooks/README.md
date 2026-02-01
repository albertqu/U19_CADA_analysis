## RR notebook/analysis description

### Notebook structure
This notebook folder contains main analysis for RR project. Key ipynb are as following:
* `RR_paper.ipynb`: this contains all scripts to generate all the paper plots, including stats
    * importantly: currently the time series statistics are done via the sup T test, which is less powerful than bootstrap test. Refer to the paper/method/boostrap time series testing section for algorithmic details
* `supT_test.ipynb`: This notebook describes test functions using bootstrap test (algo 1 in method section) and supT test (algo 2 in method section)
    * this is important as a supplementary figure to showcase the correctness/sensitivity of the test
* `RR_pose*.ipynb` contains several pose analysis experiments that I have tried. But the most updated ones included in the paper are in `RR_paper.ipynb`
* `RR_behavior_*.ipynb`: has various behavior model experiments (again final results are in RR_paper.ipynb). This is where I did the logistic regression analysis for RR
* `RR_GAM.ipynb / RR_neuroGLM.ipynb`: these are the GAM/GLM analysis experiments, this didnt make it to the final paper due to data noisiness. 

### Time Warp