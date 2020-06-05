# Global Robustness Verification Networks

Here we provide the code of SDN in "net_model.py",the back-propagation algorithm in "multpro_back_pro.py", and the verification algorithm in "GeometricDescriptionSample.py".


Take the case study in Section 5.3 as an example, we provide some intermediate results as well.
Such as the net model "model / net.pkl" and the rule-based back-propagation result "rule_TREE".


All the  commands  for the result in Section 5.3 are in "GeometricDescriptionSample.py" where "g_info" is the architecture of SDN, "cons" is the input classification rule, and "configs" is the bound of inputs.


What's more， "GeometricDescriptionSample.py" returns all the connected components and the classifications which are sorted by their volume and "r" respectively.


If you want to test this algorithm, please run "GeometricDescriptionSample.py".
