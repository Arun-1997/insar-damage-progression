This project applies deep spatiotemporal learning to Sentinel-1 InSAR data from the 2023 Kahramanmaraş earthquake (Türkiye, Mw 7.8) to classify per-pixel damage trajectory types — not binary damaged/undamaged, but how damage evolved over a three-month post-seismic window.

Novelty of this approach
•	Temporal framing: almost all prior work uses a single pre/post image pair. This project uses 8–12 post-event scenes over 3 months.
•	Trajectory labels: generated via DTW clustering on coherence time series within Copernicus EMS damage zones — no clean ground truth exists for this formulation.
•	Architecture progression: Conv-LSTM U-Net baseline, followed by ablations with Video Swin Transformer and SegMamba.

