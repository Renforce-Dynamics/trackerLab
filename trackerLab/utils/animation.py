import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib import gridspec

def animate_skeleton(
    x: torch.Tensor, 
    edge_index: torch.Tensor, 
    vel: torch.Tensor, 
    ang_vel: torch.Tensor = None,
    interval=100, desc= "Skeleton Animation with Velocity",
    draw_skeleton=True,
    save_path=None):
    if x.dim() == 4:
        x = x[0]
    assert x.dim() == 3 and x.shape[2] == 3, "[T, J, 3]"
    T, J, _ = x.shape
    x = x.cpu().numpy()
    vel = vel.cpu().numpy()
    edge_index = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else np.array(edge_index)

    total_speed = np.linalg.norm(vel, axis=1)  # [T]
    heights = x[:, 0, -1].reshape((-1))

    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(desc, fontsize=16, fontweight='bold')

    if ang_vel is None:
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    else:
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
    ax_vel = fig.add_subplot(gs[0, 0])
    ax_height = fig.add_subplot(gs[1, 0])
    ax3d = fig.add_subplot(gs[:, 1], projection='3d')
    fig.subplots_adjust(wspace=0.0, hspace=0.4, bottom=0.2)

    joint_colors = plt.cm.rainbow(np.linspace(0, 1, J))
    scat = ax3d.scatter([], [], [], s=25)
    if draw_skeleton:
        lines = [ax3d.plot([], [], [], 'b-')[0] for _ in range(edge_index.shape[0])]
    else:
        lines = []
    arrows = []

    # Velocity plot
    vel_x_line, = ax_vel.plot([], [], label='Vx', color='blue')
    vel_y_line, = ax_vel.plot([], [], label='Vy', color='green')
    vel_z_line, = ax_vel.plot([], [], label='Vz', color='orange')
    total_speed_line, = ax_vel.plot([], [], label='Total Speed', color='red')
    ax_vel.set_xlim(0, T)
    all_vel_values = np.concatenate([vel, total_speed[:, None]], axis=1).flatten()
    ax_vel.set_ylim(np.min(all_vel_values), np.max(all_vel_values))
    ax_vel.set_title("Velocity Over Time", fontsize=12)
    ax_vel.set_xlabel("Frame")
    ax_vel.set_ylabel("Velocity")
    
    # Height plot
    height_line, = ax_height.plot([], [], label='Height', color='black')
    ax_height.set_xlim(0, T)
    ax_height.set_ylim(np.min(heights), np.max(heights))
    ax_height.set_title("Height Over Time", fontsize=12)
    ax_height.set_xlabel("Frame")
    ax_height.set_ylabel("Height")

    # ax_vel.legend(loc='lower center', bbox_to_anchor=(0.5, -1.2), ncol=4, fontsize=9, frameon=False)
    # ax_height.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=9, frameon=False)

    all_handles = [
        Line2D([0], [0], label='Vx', color='blue'),
        Line2D([0], [0], label='Vy', color='green'),
        Line2D([0], [0], label='Vz', color='orange'),
        Line2D([0], [0], label='Total Speed', color='red'),
        Line2D([0], [0], label='Height', color='black'),
    ]
    
    # Ang Plot
    if ang_vel is not None:
        ax_ang = fig.add_subplot(gs[2, 0])
        ax_ang.set_xlim(0, T)
        vel_x_line, = ax_vel.plot([], [], label='Vax', color='blue')
        vel_y_line, = ax_vel.plot([], [], label='Vay', color='green')
        vel_z_line, = ax_vel.plot([], [], label='Vaz', color='orange')
        ax_ang.set_ylim(np.min(ang_vel), np.max(ang_vel))
        ax_ang.set_title("Ang Velocity Over Time", fontsize=12)
        ax_ang.set_xlabel("Frame")
        ax_ang.set_ylabel("Velocity")
        
        all_handles += [
            Line2D([0], [0], label='Vax', color='blue'),
            Line2D([0], [0], label='Vay', color='green'),
            Line2D([0], [0], label='Vaz', color='orange'),
        ]
        
    fig.legend(
        handles=all_handles,
        loc='lower center',
        ncol=5,
        bbox_to_anchor=(0.5, 0.05),
        fontsize=9,
        frameon=False
    )

    pbar = tqdm.tqdm(total=T, desc="Animating")

    def init():
        ax3d.set_xlim(np.min(x[:, :, 0]), np.max(x[:, :, 0]))
        ax3d.set_ylim(np.min(x[:, :, 1]), np.max(x[:, :, 1]))
        ax3d.set_zlim(np.min(x[:, :, 2]), np.max(x[:, :, 2]))
        ax3d.set_title("Skeleton Animation", fontsize=13)
        ax3d.set_box_aspect([1,1,1])
        # ax3d.set_box_aspect([
        #     np.ptp(x[:, :, 0]),
        #     np.ptp(x[:, :, 1]),
        #     np.ptp(x[:, :, 2])
        # ])

        proxy_points = [
            Line2D([0], [0], marker='o', color='w', label=f'Jo_{i}',
                   markerfacecolor=joint_colors[i], markersize=6)
            for i in range(J)
        ]
        ax3d.legend(
            handles=proxy_points,
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            ncol=1,
            fontsize=8,
            borderaxespad=0.0,
            frameon=False
        )
        return [scat] + lines + [vel_x_line, vel_y_line, vel_z_line, total_speed_line]

    def update(frame):
        joints = x[frame]  # [J, 3]
        root = joints[0]
        velocity = vel[frame] / 10  # [3]
        
        ax3d.set_xlim(root[0] - 1, root[0] + 1)
        ax3d.set_ylim(root[1] - 1, root[1] + 1)
        ax3d.set_zlim(root[2] - 1, root[2] + 1)
        
        scat._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
        if frame == 0:
            scat.set_color(joint_colors)

        if draw_skeleton:
            for idx, (i, j) in enumerate(edge_index):
                pt1, pt2 = joints[i], joints[j]
                lines[idx].set_data([pt1[0], pt2[0]], [pt1[1], pt2[1]])
                lines[idx].set_3d_properties([pt1[2], pt2[2]])

        while arrows:
            arrow = arrows.pop()
            arrow.remove()

        arrow = ax3d.quiver(
            root[0], root[1], root[2],
            velocity[0], velocity[1], velocity[2],
            color='r', length=np.linalg.norm(velocity), normalize=True
        )
        arrows.append(arrow)

        frames = np.arange(frame + 1)
        vel_x_line.set_data(frames, vel[:frame + 1, 0])
        vel_y_line.set_data(frames, vel[:frame + 1, 1])
        vel_z_line.set_data(frames, vel[:frame + 1, 2])
        total_speed_line.set_data(frames, total_speed[:frame + 1])
        height_line.set_data(frames, heights[:frame + 1])
        
        if ang_vel is not None:
            vel_x_line.set_data(frames, ang_vel[:frame + 1, 0])
            vel_y_line.set_data(frames, ang_vel[:frame + 1, 1])
            vel_z_line.set_data(frames, ang_vel[:frame + 1, 2])

        pbar.update(1)
        return [scat] + lines + arrows + [vel_x_line, vel_y_line, vel_z_line, total_speed_line, height_line]

    ani = FuncAnimation(fig, update, frames=T, init_func=init, interval=interval, blit=False)

    if save_path is not None:
        from matplotlib.animation import FFMpegWriter
        ani.save(save_path, writer=FFMpegWriter(fps=1000 // interval))
        print(f"Saved to {save_path}")


def recover_from_deltas(x0: torch.Tensor, deltas: torch.Tensor):
    x0_expanded = x0.unsqueeze(0).expand(deltas.shape[0], -1, -1)
    recovered_traj = deltas + x0_expanded
    return recovered_traj