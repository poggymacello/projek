import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os

# Buat direktori static jika belum ada
os.makedirs('static', exist_ok=True)

def simulasi_jaringan():
    """
    fungsi ini mensimulasikan jaringan iot dengan 10 node smart meter.
    mengacu pada ieee std 1547 untuk fluktuasi tegangan dan arus normal.
    """
    try:
        # inisialisasi array untuk menyimpan data [tegangan, arus]
        data = np.zeros((20, 10, 2))
        
        # node yang akan diberi anomali (contoh: node 2, 5, 8)
        node_anomali = [2, 5, 8]
        
        # simulasi untuk 20 timestep
        for t in range(20):
            for n in range(10):
                # cek apakah node ini termasuk yang bisa anomali
                if n in node_anomali and random.random() > 0.5:
                    # kondisi anomali: tegangan rendah, arus tinggi
                    data[t, n, 0] = random.uniform(180, 200)  # tegangan anomali
                    data[t, n, 1] = random.uniform(15, 20)    # arus anomali
                else:
                    # kondisi normal sesuai standar distribusi tegangan rendah
                    data[t, n, 0] = random.uniform(220, 240)  # tegangan normal
                    data[t, n, 1] = random.uniform(5, 10)     # arus normal
        
        return data
    except Exception as e:
        print(f"error pada simulasi jaringan: {str(e)}")
        return None

def pembelajaran_q(data):
    """
    implementasi q-learning untuk deteksi anomali.
    kompleksitas o(n*s*a) - optimasi pembelajaran berdasarkan state dan action.
    """
    try:
        # parameter q-learning
        alpha = 0.1  # learning rate
        gamma = 0.9  # discount factor
        
        # inisialisasi q-table untuk 10 node, 2 state, 2 action
        q_table = np.zeros((10, 2, 2))
        
        # proses pembelajaran
        for t in range(19):  # sampai timestep 19 agar ada next state
            for n in range(10):
                # tentukan state sekarang
                voltage_now = data[t, n, 0]
                current_now = data[t, n, 1]
                state = 1 if voltage_now < 200 or current_now > 15 else 0
                
                # tentukan state berikutnya
                voltage_next = data[t+1, n, 0]
                current_next = data[t+1, n, 1]
                next_state = 1 if voltage_next < 200 or current_next > 15 else 0
                
                # pilih action (0: aman, 1: risiko)
                action = np.argmax(q_table[n, state, :])
                if random.random() < 0.3:  # eksplorasi 30%
                    action = random.randint(0, 1)
                
                # hitung reward
                reward = 1 if action == state else -1
                
                # update q-table
                q_table[n, state, action] += alpha * (
                    reward + gamma * np.max(q_table[n, next_state, :]) - 
                    q_table[n, state, action]
                )
        
        return q_table
    except Exception as e:
        print(f"error pada pembelajaran q: {str(e)}")
        return None

def hitung_risiko(data, q_table):
    """
    menghitung skor risiko (0-1) untuk setiap node berdasarkan prediksi q-learning.
    semakin tinggi skor, semakin tinggi risiko anomali pada node tersebut.
    """
    try:
        skor_risiko = np.zeros(10)
        
        for n in range(10):
            count_risiko = 0
            for t in range(20):
                # tentukan state
                voltage = data[t, n, 0]
                current = data[t, n, 1]
                state = 1 if voltage < 200 or current > 15 else 0
                
                # prediksi action dari q-table
                action = np.argmax(q_table[n, state, :])
                
                # hitung jumlah prediksi risiko
                if action == 1:  # action risiko
                    count_risiko += 1
            
            # normalisasi skor
            skor_risiko[n] = count_risiko / 20
        
        return skor_risiko
    except Exception as e:
        print(f"error pada perhitungan risiko: {str(e)}")
        return None

def analisis_dan_visualisasi(data, skor_risiko):
    """
    membuat visualisasi dan menyimpan data analisis
    """
    try:
        # Simpan data analisis
        hasil = []
        nodes_status = {"BAHAYA": 0, "WASPADA": 0, "AMAN": 0}
        
        for n in range(10):
            tegangan = data[-1, n, 0]
            arus = data[-1, n, 1]
            risiko = skor_risiko[n]
            
            status = "AMAN" if risiko < 0.3 else "WASPADA" if risiko < 0.7 else "BAHAYA"
            nodes_status[status] += 1
            
            hasil.append({
                "node": n,
                "tegangan": f"{tegangan:.1f}",
                "arus": f"{arus:.1f}",
                "risiko": f"{risiko:.2f}",
                "status": status
            })
            
            # Output di terminal
            print(f"Node {n}: Tegangan={tegangan:.1f}V, Arus={arus:.1f}A, Risiko={risiko:.2f}, Status={status}")

        # Buat visualisasi 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        tegangan_akhir = data[-1, :, 0]
        scatter = ax.scatter(range(10), tegangan_akhir, skor_risiko,
                           c=skor_risiko, cmap='RdYlGn_r', s=100)
        
        ax.set_xlabel('indeks node')
        ax.set_ylabel('tegangan (v)')
        ax.set_zlabel('skor risiko')
        ax.set_title('electroshield: peta risiko iot')
        plt.colorbar(scatter, label='tingkat risiko')
        ax.grid(True)
        
        # Simpan plot
        plt.savefig('static/plot.png')
        plt.close()
        
        # Simpan data untuk dashboard
        with open('static/data.json', 'w') as f:
            json.dump({
                "nodes": hasil,
                "summary": nodes_status
            }, f)
            
    except Exception as e:
        print(f"error pada analisis dan visualisasi: {str(e)}")

def main():
    """
    fungsi utama untuk menjalankan simulasi dan analisis
    """
    # jalankan simulasi
    print("memulai simulasi jaringan iot...")
    data = simulasi_jaringan()
    if data is None:
        return
    
    # pembelajaran q-learning
    print("melakukan pembelajaran q-learning...")
    q_table = pembelajaran_q(data)
    if q_table is None:
        return
    
    # hitung skor risiko
    print("menghitung skor risiko...")
    skor_risiko = hitung_risiko(data, q_table)
    if skor_risiko is None:
        return
    
    # visualisasi
    print("membuat analisis dan visualisasi...")
    analisis_dan_visualisasi(data, skor_risiko)

if __name__ == "__main__":
    main()
