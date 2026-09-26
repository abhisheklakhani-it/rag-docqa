# Federated Learning

Federated learning trains a shared model across many devices or data silos without moving raw data to a central server. Each client trains locally and sends only model updates, which a coordinator aggregates, most commonly with Federated Averaging (FedAvg).

## Privacy

Because raw data never leaves the client, federated learning reduces exposure of personal information. It is often combined with secure aggregation and differential privacy for stronger guarantees.

## Solid PODs

Solid stores user data in personal online data stores called PODs. Users control which applications can read their POD, which makes Solid a natural fit for privacy-focused, decentralized machine learning experiments.
