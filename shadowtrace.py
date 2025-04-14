import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from scipy.special import expit as sigmoid

data_traffic = []
suspicious_count = 0
normal_count = 0

for t in range(100):
    ip = f"192.168.{random.randint(0,255)}.{random.randint(0,255)}"
    timestamp = t
    
    if random.random() < 0.3:  # suspicious
        ukuran = random.randint(1000, 2000)
        suspicious_count += 1
        print(f"[!] Suspicious Traffic Detected at t={timestamp}")
        print(f"    IP: {ip}")
        print(f"    Packet Size: {ukuran} bytes\n")
    else:  # normal
        ukuran = random.randint(50, 500)
        normal_count += 1
        print(f"[+] Normal Traffic at t={timestamp}")
        print(f"    IP: {ip}")
        print(f"    Packet Size: {ukuran} bytes\n")
        
    data_traffic.append((ip, timestamp, ukuran))

print("\nTraffic Summary:")
print(f"Total Data Points: {len(data_traffic)}")
print(f"Normal Traffic: {normal_count}")
print(f"Suspicious Traffic: {suspicious_count}")
print("----------------------------------------\n")

# Build HMM
print("[2/4] Building HMM model...")
print("----------------------------------------")
print("Extracting features...")
ukuran_paket = [data[2] for data in data_traffic]
X = np.array(ukuran_paket).reshape(-1, 1)

print("\nInitializing HMM parameters:")
model = GaussianHMM(
    n_components=2,
    covariance_type="diag",
    init_params="",
    random_state=42
)

print("- States: 2 (Normal, Suspicious)")
print("- Initial probabilities: [0.7, 0.3]")
print("- Transition matrix:")
print("  [0.9, 0.1] (Normal -> Normal, Normal -> Suspicious)")
print("  [0.3, 0.7] (Suspicious -> Normal, Suspicious -> Suspicious)")

model.startprob_ = np.array([0.7, 0.3])
model.transmat_ = np.array([[0.9, 0.1],
                           [0.3, 0.7]])
model.means_ = np.array([[200.],
                        [1500.]])
model.covars_ = np.array([[50.**2],
                         [200.**2]])

print("\nTraining HMM...")
model.fit(X)
print("Model training completed!")
print("----------------------------------------\n")
    
# Calculate risk scores
print("[3/4] Calculating risk scores...")
print("----------------------------------------")
print("Computing posterior probabilities...")
skor_risiko = model.predict_proba(X)[:, 1]

print("\nRisk Score Statistics:")
print(f"Average Risk: {np.mean(skor_risiko):.3f}")
print(f"Max Risk: {np.max(skor_risiko):.3f}")
print(f"Min Risk: {np.min(skor_risiko):.3f}")
print("----------------------------------------\n")

# Create visualization
print("[4/4] Generating visualization...")
print("----------------------------------------")
print("Creating heatmap...")
plt.figure(figsize=(15, 3))
risk_matrix = skor_risiko.reshape(1, -1)

sns.heatmap(
    risk_matrix,
    cmap="RdBu_r",
    cbar=True,
    cbar_kws={'label': 'Tingkat Risiko'},
    xticklabels=20,
    vmin=0,
    vmax=1
)

plt.xlabel('timestamp (detik)')
plt.ylabel('traffic dark web')
plt.title('peta risiko aktivitas dark web')
plt.tight_layout()

print("Saving visualization...")
plt.savefig('risk_heatmap.png')
plt.close()
print("Heatmap saved as 'risk_heatmap.png'")
print("----------------------------------------\n")

print("=== Analysis Completed Successfully! ===")
print("All steps executed:")
print("✓ Data generation")
print("✓ HMM model building")
print("✓ Risk calculation")
print("✓ Visualization\n")
