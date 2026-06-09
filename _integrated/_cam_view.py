"""_cam_view.py — 로봇 카메라 별창 표시 (별도 프로세스).
body_cam / arm_cam / 신호등 비전 3개 패널을 실시간 갱신.
main_task.py 가 (body_img, arm_img, light_img) 튜플을 큐로 보낸다.
"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def run(q):
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
    fig.canvas.manager.set_window_title("Spot Robot Cameras")
    for ax, title in ((ax1, "Body Cam"), (ax2, "Arm Cam"), (ax3, "Traffic-Light Vision")):
        ax.set_title(title)
        ax.axis("off")
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    h1 = ax1.imshow(blank)
    h2 = ax2.imshow(blank)
    h3 = ax3.imshow(blank)
    plt.tight_layout()

    while True:
        try:
            item = q.get(timeout=0.5)
            if item is None:
                break
            h1.set_data(item[0])
            h2.set_data(item[1])
            h3.set_data(item[2])
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception:
            try:
                fig.canvas.flush_events()
            except Exception:
                break
    plt.close()
