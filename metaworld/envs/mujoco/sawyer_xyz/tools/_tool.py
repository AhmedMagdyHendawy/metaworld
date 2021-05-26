import abc
import itertools


def get_position_of(tool, mjsim):
    if type(tool) is not str:
        tool = tool.name
    return mjsim.data.get_body_xpos(tool)


def set_position_of(tool, mjsim, mjmodel):
    assert(len(tool.specified_pos) == 3)
    mjsim.model.body_pos[mjmodel.body_name2id(tool.name)] = tool.specified_pos


def get_joint_pos_of(tool, mjsim):
    if type(tool) is not str:
        tool = tool.name
    return mjsim.data.get_joint_qpos(tool + 'Joint')


def set_joint_pos_of(tool, mjsim, pos):
    mjsim.data.set_joint_qpos(tool.name + 'Joint', pos)


def get_joint_vel_of(tool, mjsim):
    if type(tool) is not str:
        tool = tool.name
    return mjsim.data.get_joint_qvel(tool + 'Joint')


def set_joint_vel_of(tool, mjsim, vel):
    mjsim.data.set_joint_qvel(tool.name + 'Joint', vel)


def get_quat_of(tool, mjsim):
    if type(tool) is not str:
        tool = tool.name
    return mjsim.data.get_body_xquat(tool)


def set_quat_of(tool, mjsim, mjmodel):
    assert(len(tool.specified_quat) == 4)
    mjsim.model.body_quat[mjmodel.body_name2id(tool.name)] = tool.specified_quat
    return


def get_vel_of(tool, mjsim, mjmodel):
    if type(tool) is not str:
        tool = tool.name
    return mjsim.data.cvel[mjmodel.body_name2id(tool)]


def set_vel_of(tool, mjsim, mjmodel):
    '''
    mjmodel does not have access to set cvel values since they are calculated:
    mjsim.model.cvel[mjmodel.body_name2id(tool.name)] = new_vel
    '''
    raise NotImplementedError


class Tool(abc.ABC):
    """
    Any object involved in accomplishing a task, excepting the robot itself and
    the table. If a `Tool` is meant to be frozen in place over the course of a
    trajectory, you should use the `Artifact` subclass instead.
    """
    def __init__(self, enabled=True):
        self.specified_pos = None
        self.specified_quat = None
        self.enabled = enabled

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def pos_is_static(self):
        return False

    @property
    @abc.abstractmethod
    def bbox(self):
        """
        6 element np.ndarray representing an oversized bounding box for the
        object. First 3 are bottom-left corner, last 3 are upper-right corner
        """
        pass

    def get_bbox_corners(self):
        bbox = self.bbox.tolist()
        assert len(bbox) == 6
        return itertools.product(*zip(bbox[:3], bbox[3:]))

    @property
    def resting_pos_z(self) -> float:
        """The Z position perceived by Mujoco when object lies flat on table"""
        return 0.0


class Artifact(Tool, abc.ABC):
    @property
    def pos_is_static(self):
        return True
