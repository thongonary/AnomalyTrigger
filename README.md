# AnomalyTrigger
Anomaly Trigger with AutoEncoder

### Code Structure

Currently I just dump everything into one notebook. To move forward, we likely
need to restructure the code into divisions:
IO     : File reading interface for Delphes, Level-1 ntuples
Models : different AE models
Performance: Besides the general ROC and AUC, we likely need code for the trigger rate, efficiency etc.
