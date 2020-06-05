# Global Robustness Verification Networks

Here we provide the code of all case studys in Section5. The architecture of SDN is in "net_model.py", "train_SDN.py" shows how to train SDN, the back-propagation algorithm is in "multpro_back_pro.py", and the verification algorithm is in "GeometricDescriptionSample.py".

Take the case study in Section 5.3 as an example, we provide some intermediate results as well.
Such as the net model "model / net.pkl" and the rule-based back-propagation result "rule_TREE".


The example commands  of the result in Section 5.3 are in "GeometricDescriptionSample.py" where "g_info" is the architecture of SDN, "cons" is the input classification rule, and "configs" is the bound of inputs.


Besides the verification result， "GeometricDescriptionSample.py" returns all the connected components and the classifications which are sorted by their volume and "r" respectively.


If you want to test the verification algorithm, please run "GeometricDescriptionSample.py".
