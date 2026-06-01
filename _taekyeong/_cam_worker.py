import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def run(q):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_title("Body Cam"); ax1.axis("off")
    ax2.set_title("Arm Cam");  ax2.axis("off")
    blank = np.zeros((360, 480, 3), dtype=np.uint8)
    h1 = ax1.imshow(blank)
    h2 = ax2.imshow(blank)
    plt.tight_layout()

    while True:
        try:
            item = q.get(timeout=0.5)
            if item is None:
                break
            h1.set_data(item[0])
            h2.set_data(item[1])
            fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            plt.pause(0.001)
    plt.close()
