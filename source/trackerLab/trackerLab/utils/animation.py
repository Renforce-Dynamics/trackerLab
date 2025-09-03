import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib import gridspec


class SkeletonAnimator:
    def __init__(self, x, edge_index, vel, ang_vel=None, dof_pos=None,
                 interval=100, desc="Skeleton Animation with Velocity",
                 draw_skeleton=True):
        """
        Skeleton animation with optional velocity, angular velocity, and DoF position curves.

        Args:
            x: [T, J, 3] joint positions
            edge_index: [E, 2] skeleton connectivity
            vel: [T, 3] root velocity
            ang_vel: [T, 3], optional angular velocity
            dof_pos: [T, D], optional dof positions
        """
        self.x = x
        self.edge_index = edge_index
        self.vel = vel
        self.ang_vel = ang_vel
        self.dof_pos = dof_pos
        self.interval = interval
        self.desc = desc
        self.draw_skeleton = draw_skeleton

        self.T, self.J, _ = self.x.shape
        self.total_speed = np.linalg.norm(self.vel, axis=1)
        self.heights = self.x[:, 0, -1].reshape((-1))
        self.joint_colors = plt.cm.rainbow(np.linspace(0, 1, self.J))

        self._setup_figure()
        self._setup_lines()

    def _setup_figure(self):
        """Set up matplotlib figure and axes grid."""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle(self.desc, fontsize=16, fontweight='bold')

        # Layout adapts if ang_vel and dof_pos are provided
        rows = 2 + (1 if self.ang_vel is not None else 0) + (1 if self.dof_pos is not None else 0)
        self.gs = gridspec.GridSpec(rows, 2, width_ratios=[1, 1])

        self.ax_vel = self.fig.add_subplot(self.gs[0, 0])
        self.ax_height = self.fig.add_subplot(self.gs[1, 0])
        self.ax3d = self.fig.add_subplot(self.gs[:, 1], projection='3d')

        row_idx = 2
        self.ax_ang = None
        if self.ang_vel is not None:
            self.ax_ang = self.fig.add_subplot(self.gs[row_idx, 0])
            row_idx += 1

        self.ax_dof = None
        if self.dof_pos is not None:
            self.ax_dof = self.fig.add_subplot(self.gs[row_idx, 0])

        self.fig.subplots_adjust(wspace=0.0, hspace=0.4, bottom=0.2)

    def _setup_lines(self):
        """Initialize all plots (velocity, height, angular velocity, dof_pos)."""
        # 3D Scatter & skeleton
        self.scat = self.ax3d.scatter([], [], [], s=25)
        if self.draw_skeleton:
            self.lines = [self.ax3d.plot([], [], [], 'b-')[0] for _ in range(self.edge_index.shape[0])]
        else:
            self.lines = []
        self.arrows = []

        # Velocity
        self.vel_x_line, = self.ax_vel.plot([], [], label='Vx', color='blue')
        self.vel_y_line, = self.ax_vel.plot([], [], label='Vy', color='green')
        self.vel_z_line, = self.ax_vel.plot([], [], label='Vz', color='orange')
        self.total_speed_line, = self.ax_vel.plot([], [], label='Total Speed', color='red')
        self.ax_vel.set_xlim(0, self.T)
        all_vel_values = np.concatenate([self.vel, self.total_speed[:, None]], axis=1).flatten()
        self.ax_vel.set_ylim(np.min(all_vel_values), np.max(all_vel_values))
        self.ax_vel.set_title("Velocity Over Time", fontsize=12)

        # Height
        self.height_line, = self.ax_height.plot([], [], label='Height', color='black')
        self.ax_height.set_xlim(0, self.T)
        self.ax_height.set_ylim(np.min(self.heights), np.max(self.heights))
        self.ax_height.set_title("Height Over Time", fontsize=12)

        # Angular velocity
        if self.ang_vel is not None:
            self.ang_x_line, = self.ax_ang.plot([], [], label='Vax', color='blue')
            self.ang_y_line, = self.ax_ang.plot([], [], label='Vay', color='green')
            self.ang_z_line, = self.ax_ang.plot([], [], label='Vaz', color='orange')
            self.ax_ang.set_xlim(0, self.T)
            self.ax_ang.set_ylim(np.min(self.ang_vel), np.max(self.ang_vel))
            self.ax_ang.set_title("Ang Velocity Over Time", fontsize=12)

        # DoF pos
        if self.dof_pos is not None:
            self.dof_lines = []
            D = self.dof_pos.shape[1]
            for i in range(D):
                line, = self.ax_dof.plot([], [], label=f'DoF {i}')
                self.dof_lines.append(line)
            self.ax_dof.set_xlim(0, self.T)
            self.ax_dof.set_ylim(np.min(self.dof_pos), np.max(self.dof_pos))
            self.ax_dof.set_title("Joint DoF Positions", fontsize=12)

        # Legend
        all_handles = [
            Line2D([0], [0], label='Vx', color='blue'),
            Line2D([0], [0], label='Vy', color='green'),
            Line2D([0], [0], label='Vz', color='orange'),
            Line2D([0], [0], label='Total Speed', color='red'),
            Line2D([0], [0], label='Height', color='black'),
        ]
        if self.ang_vel is not None:
            all_handles += [
                Line2D([0], [0], label='Vax', color='blue'),
                Line2D([0], [0], label='Vay', color='green'),
                Line2D([0], [0], label='Vaz', color='orange'),
            ]
        if self.dof_pos is not None:
            all_handles += [Line2D([0], [0], label=f'DoF {i}') for i in range(self.dof_pos.shape[1])]
        self.fig.legend(handles=all_handles, loc='lower center', ncol=6,
                        bbox_to_anchor=(0.5, 0.05), fontsize=9, frameon=False)

    def init(self):
        """Initialization for FuncAnimation."""
        self.ax3d.set_xlim(np.min(self.x[:, :, 0]), np.max(self.x[:, :, 0]))
        self.ax3d.set_ylim(np.min(self.x[:, :, 1]), np.max(self.x[:, :, 1]))
        self.ax3d.set_zlim(np.min(self.x[:, :, 2]), np.max(self.x[:, :, 2]))
        self.ax3d.set_title("Skeleton Animation", fontsize=13)
        self.ax3d.set_box_aspect([1, 1, 1])

        proxy_points = [
            Line2D([0], [0], marker='o', color='w', label=f'Jo_{i}',
                   markerfacecolor=self.joint_colors[i], markersize=6)
            for i in range(self.J)
        ]
        self.ax3d.legend(handles=proxy_points, loc='center left',
                         bbox_to_anchor=(1.05, 0.5), fontsize=8, frameon=False)
        return [self.scat] + self.lines

    def update(self, frame):
        """Update function for FuncAnimation."""
        joints = self.x[frame]
        root = joints[0]
        velocity = self.vel[frame] / 10

        # Update 3D
        self.ax3d.set_xlim(root[0] - 1, root[0] + 1)
        self.ax3d.set_ylim(root[1] - 1, root[1] + 1)
        self.ax3d.set_zlim(root[2] - 1, root[2] + 1)
        self.scat._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
        if frame == 0:
            self.scat.set_color(self.joint_colors)
        if self.draw_skeleton:
            for idx, (i, j) in enumerate(self.edge_index):
                pt1, pt2 = joints[i], joints[j]
                self.lines[idx].set_data([pt1[0], pt2[0]], [pt1[1], pt2[1]])
                self.lines[idx].set_3d_properties([pt1[2], pt2[2]])
        while self.arrows:
            arrow = self.arrows.pop()
            arrow.remove()
        arrow = self.ax3d.quiver(root[0], root[1], root[2],
                                 velocity[0], velocity[1], velocity[2],
                                 color='r', length=np.linalg.norm(velocity), normalize=True)
        self.arrows.append(arrow)

        # Update curves
        frames = np.arange(frame + 1)
        self.vel_x_line.set_data(frames, self.vel[:frame + 1, 0])
        self.vel_y_line.set_data(frames, self.vel[:frame + 1, 1])
        self.vel_z_line.set_data(frames, self.vel[:frame + 1, 2])
        self.total_speed_line.set_data(frames, self.total_speed[:frame + 1])
        self.height_line.set_data(frames, self.heights[:frame + 1])
        if self.ang_vel is not None:
            self.ang_x_line.set_data(frames, self.ang_vel[:frame + 1, 0])
            self.ang_y_line.set_data(frames, self.ang_vel[:frame + 1, 1])
            self.ang_z_line.set_data(frames, self.ang_vel[:frame + 1, 2])
        if self.dof_pos is not None:
            for i, line in enumerate(self.dof_lines):
                line.set_data(frames, self.dof_pos[:frame + 1, i])
        return [self.scat] + self.lines + self.arrows


def animate_skeleton(x, edge_index, vel, ang_vel=None, dof_pos=None,
                     interval=100, desc="Skeleton Animation with Velocity",
                     draw_skeleton=True, save_path=None):
    if x.dim() == 4:
        x = x[0]
    assert x.dim() == 3 and x.shape[2] == 3, "[T, J, 3]"
    x = x.cpu().numpy()
    vel = vel.cpu().numpy()
    edge_index = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else np.array(edge_index)
    ang_vel = ang_vel.cpu().numpy() if ang_vel is not None else None
    dof_pos = dof_pos.cpu().numpy() if dof_pos is not None else None

    animator = SkeletonAnimator(x, edge_index, vel, ang_vel, dof_pos,
                                interval=interval, desc=desc, draw_skeleton=draw_skeleton)
    pbar = tqdm.tqdm(total=x.shape[0], desc="Animating")

    def wrapped_update(frame):
        out = animator.update(frame)
        pbar.update(1)
        return out

    ani = FuncAnimation(animator.fig, wrapped_update, frames=x.shape[0],
                        init_func=animator.init, interval=interval, blit=False)
    if save_path is not None:
        from matplotlib.animation import FFMpegWriter
        ani.save(save_path, writer=FFMpegWriter(fps=1000 // interval))
        print(f"Saved to {save_path}")
    return ani


def recover_from_deltas(x0: torch.Tensor, deltas: torch.Tensor):
    """Recover trajectory from initial x0 and deltas."""
    x0_expanded = x0.unsqueeze(0).expand(deltas.shape[0], -1, -1)
    recovered_traj = deltas + x0_expanded
    return recovered_traj
