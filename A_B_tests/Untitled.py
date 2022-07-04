#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.stats import norm, chi2_contingency
import statsmodels.api as sm
import numpy as np


# In[3]:


s1 = 135
n1 = 1781
s2 = 47
n2 = 1443
p1 = s1/n1
p2 = s2/n2
p = (s1 + s2)/(n1 + n2)
z = (p2 - p1)/((p*(1-p)*((1/n1)+(1/n2)))**0.5) # z-метка

p_value = norm.cdf(z)

print(['{:.12f}'.format(a) for a in (abs(z), p_value*2)])


# In[8]:


z1, p_value1 = sm.stats.proportions_ztest([s1, s2], [n1, n2])

print(['{:.12f}'.format(b) for b in (abs(z1), p_value1)])


# In[9]:


arr = np.array([[s1, n1 - s1], [s2, n2-s2]])
chi2, p_value3, dof, exp = chi2_contingency(arr, correction = False)

print(['{:.12f}'.format(d) for d in (chi2**0.5, p_value3)])


# In[11]:


count = 5
nobs = 83
value = .05
stat, pval = sm.stats.proportions_ztest(count, nobs, value)
print('{0:0.3f}'.format(pval))


# In[12]:


import numpy as np
from statsmodels.stats.proportion import proportions_ztest
count = np.array([5, 12])
nobs = np.array([83, 99])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))


# In[13]:


from scipy.stats import chi2_contingency
obs = np.array([[10, 10, 20], [20, 20, 20]])
chi2_contingency(obs)


# In[14]:


g, p, dof, expctd = chi2_contingency(obs, lambda_="log-likelihood")
g, p


# In[15]:


obs = np.array(
    [[[[12, 17],
       [11, 16]],
      [[11, 12],
       [15, 16]]],
     [[[23, 15],
       [30, 22]],
      [[14, 17],
       [15, 16]]]])
chi2_contingency(obs)


# In[ ]:




