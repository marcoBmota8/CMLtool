import numpy as np

from CML_tool.auprc_ci import auc_ci, auprc_comparison_test

# Perfect case
probs = np.array([1,1,1,1,0,1,0,0,1,1,1,1,1,1])
gt = np.array([1,1,1,1,0,1,0,0,1,1,1,1,1,1])
print('\t\tPerfect case\n', 
      '-Wald CI:',auc_ci(alpha=0.05, ground_truth=gt,predictions=probs, ci_type='wald'),
      '\n-Logistic CI:',auc_ci(alpha=0.05, ground_truth=gt,predictions=probs, ci_type='logistic'))


#Comparison
probs1 = np.array([0.5,0.6,0.9,0.1,0.001,0.67,0.87,0.35,0.75,0.5,0.5,0.4,0.6,0.7])
probs2 = np.array([0.45,0.2,0.99,0.001,0.25,0.8,0.4,0.9,0.7,0.5,0.5,0.4,0.6,0.7])

print('AUC1: ', auc_ci(alpha = 0.05,ground_truth=gt,predictions=probs1, ci_type='wald'))
print('AUC2: ', auc_ci(alpha = 0.05,ground_truth=gt,predictions=probs2, ci_type='wald'))
print(auprc_comparison_test(gt,probs1,probs2,alpha = 0.05))

