import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from equations import *

if __name__ == '__main__':
    particles = [
        Particle("alpha cen A", 1.1, np.array([-0.5, 0, 0], dtype="float64"),
                 np.array([0.01, 0.01, 0], dtype="float64")),
        Particle("alpha cen B", 0.907, np.array([0.5, 0, 0], dtype="float64"),
                 np.array([-0.05, 0, -0.1], dtype="float64")),
        Particle("alpha cen C", 1.0, np.array([0, 1, 0], dtype="float64"), np.array([0, -0.01, 0], dtype="float64"))
    ]
    solutions = run_simulation(particles, 5000)
    animation_step = 5
    positions = solutions[:, :3, :]
    r1_sol = positions[:, 0, :][::animation_step]
    r2_sol = positions[:, 1, :][::animation_step]
    r3_sol = positions[:, 2, :][::animation_step]

    r_sols = [r1_sol, r2_sol, r3_sol]
    for r in r_sols:
        print(r[-1])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(np.min([np.min(r[:, 0]) for r in r_sols]), np.max([np.max(r[:, 0]) for r in r_sols]))
    ax.set_ylim(np.min([np.min(r[:, 1]) for r in r_sols]), np.max([np.max(r[:, 1]) for r in r_sols]))
    ax.set_zlim(np.min([np.min(r[:, 2]) for r in r_sols]), np.max([np.max(r[:, 2]) for r in r_sols]))

    line1, = ax.plot([], [], [], label='Body 1', color='darkblue')
    line2, = ax.plot([], [], [], label='Body 2', color='red')
    line3, = ax.plot([], [], [], label='Body 3', color='orange')

    marker1, = ax.plot([], [], [], color='darkblue', marker='o')
    marker2, = ax.plot([], [], [], color='red', marker='o')
    marker3, = ax.plot([], [], [], color='orange', marker='o')


    # Initialization function (called once)
    def init():
        line1.set_data([], [])
        line1.set_3d_properties([])

        line2.set_data([], [])
        line2.set_3d_properties([])

        line3.set_data([], [])
        line3.set_3d_properties([])

        marker1.set_data([], [])
        marker1.set_3d_properties([])

        marker2.set_data([], [])
        marker2.set_3d_properties([])

        marker3.set_data([], [])
        marker3.set_3d_properties([])

        return line1, line2, line3, marker1, marker2, marker3


    # Animation function (called for each frame)
    def animate(i):
        if i > 0:  # Check if i is greater than 0 to avoid accessing -1 index
            x1 = r1_sol[:i, 0]
            y1 = r1_sol[:i, 1]
            z1 = r1_sol[:i, 2]

            x2 = r2_sol[:i, 0]
            y2 = r2_sol[:i, 1]
            z2 = r2_sol[:i, 2]

            x3 = r3_sol[:i, 0]
            y3 = r3_sol[:i, 1]
            z3 = r3_sol[:i, 2]

            line1.set_data(x1, y1)
            line1.set_3d_properties(z1)

            line2.set_data(x2, y2)
            line2.set_3d_properties(z2)

            line3.set_data(x3, y3)
            line3.set_3d_properties(z3)

            if len(x1) > 0:  # Check if the arrays are not empty
                marker1.set_data(x1[-1], y1[-1])
                marker1.set_3d_properties(z1[-1])

            if len(x2) > 0:
                marker2.set_data(x2[-1], y2[-1])
                marker2.set_3d_properties(z2[-1])

            if len(x3) > 0:
                marker3.set_data(x3[-1], y3[-1])
                marker3.set_3d_properties(z3[-1])

        return line1, line2, line3, marker1, marker2, marker3


    # Create the animation
    ani = FuncAnimation(fig, animate, init_func=init, frames=len(r1_sol), interval=50, blit=True)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectories of Three Bodies')

    # Add a legend
    ax.legend()

    # To display the animation in a Jupyter Notebook
    # from IPython.display import HTML
    #
    # HTML(ani.to_jshtml())

    # To save the animation as a file (e.g., GIF)
    writervideo = matplotlib.animation.FFMpegWriter(fps=60)
    ani.save('three_body_animation.mp4', writer=writervideo)

    # To display the animation in a standalone Python script
    # plt.show()