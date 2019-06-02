import numpy as np
import src.Utils.plot as plot
import src.Models.velocityMotionModel as vm

def plotRobotMotion():
    N = 100
    control = np.array([1, 0.2])
    pose = np.array([8, 15, 0])
    deltaT = 5

    plot.limit(0, 30, 0, 30)
    plot.plotRobotPose(pose)

    model = vm.VelocityMotionModel(0.001, 0.001, 0.05, 0.05, 0.001, 0.001)
    poses = []
    for i in range(N):
        newPose = model.sampleNewPose(control, pose, deltaT)
        poses.append(newPose)
        plot.plotRobotPose(newPose)
    plot.show()


if __name__ == '__main__':
    plotRobotMotion()
