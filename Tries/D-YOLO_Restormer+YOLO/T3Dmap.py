import matplotlib
matplotlib.use('Agg')  # Ekran gerektirmez, buffer'a çizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

class DroneTrajectory3D:
    def __init__(self, canvas_w=500, canvas_h=767, history=1000):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.history = history
        self.positions = []
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.last_canvas = None

    def update(self, dx, dy):
        self.x += dx * 0.5
        self.y += dy * 0.5

        speed = (dx**2 + dy**2) ** 0.5
        self.z += (speed - 2.0) * 0.05

        self.positions.append((self.x, self.y, self.z))
        if len(self.positions) > self.history:
            self.positions.pop(0)

    def draw_canvas(self):
        fig = plt.figure(figsize=(self.canvas_w / 100, self.canvas_h / 100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#E8E8E8')
        fig.patch.set_facecolor('#E8E8E8')

        if len(self.positions) >= 2:
            positions = np.array(self.positions)
            x_vals = positions[:, 0]
            y_vals = positions[:, 1]
            z_vals = positions[:, 2]

            colors = plt.cm.cool(np.linspace(0, 1, len(x_vals)))

            for i in range(1, len(x_vals)):
                ax.plot(x_vals[i-1:i+1], y_vals[i-1:i+1], z_vals[i-1:i+1],
                        color=colors[i], linewidth=1.5)

            ax.scatter(x_vals[:-1:5], y_vals[:-1:5], z_vals[:-1:5],
                      c='green', s=10, zorder=5)

            ax.scatter([x_vals[-1]], [y_vals[-1]], [z_vals[-1]],
                      c='red', s=50, zorder=6)
            
            ax.plot(x_vals, y_vals, min(z_vals),
                   color='gray', alpha=0.2, linewidth=1)

        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_title('Drone Trajectory', fontsize=10)
        ax.tick_params(labelsize=6)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # 4 channel (RGBA)
        buf = buf[:, :, :3]  # Throw the alpha channel, remain the RGB channels
        plt.close(fig)

        canvas = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        canvas = cv2.resize(canvas, (self.canvas_w, self.canvas_h))
        self.last_canvas = canvas
        return canvas