import numpy as np

from CML_tool.DeLong import auc_ci, delong_roc_test 


# Perfect case
probs = np.array(45*[1,1,1,1,0,1,0,0,1,1,1,1,1,1])
gt = np.array(45*[1,1,1,1,0,1,0,0,1,1,1,1,1,1])
perf_results = auc_ci(alpha=0.05, ground_truth=gt,predictions=probs)
print('Perfect case AUC results: ', perf_results)

print('\n')

#Comparison
probs1 = np.array(45*[0.5,0.6,0.9,0.1,0.001,0.67,0.87,0.35,0.75,0.5,0.5,0.4,0.6,0.7])
probs2 = np.array(45*[0.5,0.6,0.99,0.001,0.25,0.8,0.4,0.9,0.7,0.5,0.5,0.4,0.6,0.7])
print('AUC 1: ', auc_ci(alpha = 0.05,ground_truth=gt,predictions=probs1))
print('AUC2: ', auc_ci(alpha = 0.05,ground_truth=gt,predictions=probs2))
print('Test results: ', delong_roc_test(gt, probs1, probs2, ci_type='wald', alpha = 0.05))
