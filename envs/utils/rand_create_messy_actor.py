import sapien.core as sapien
import numpy as np
import transforms3d as t3d
import sapien.physx as sapienp
from .create_actor import *

def check_overlap(radius, x, y, area):
    if x <= area[0]:
        dx = area[0] - x
    elif area[0] < x and x < area[2]:
        dx = 0
    elif x >= area[2]:
        dx = x - area[2]
    if y <= area[1]:
        dy = area[1] - y
    elif area[1] < y and y < area[3]:
        dy = 0
    elif y >= area[3]:
        dy = y - area[3]
    
    return dx * dx + dy * dy <= radius * radius


def rand_pose_messy(
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop = False,
    rotate_rand = False,
    rotate_lim = [0,0,0],
    qpos = [1,0,0,0],
    size_dict = None,
    obj_radius = 0.1,
    z_offset = 0.001,
    z_max = 0,
    prohibited_area = None
) -> sapien.Pose:  
    if (len(xlim)<2 or xlim[1]<xlim[0]):
        xlim=np.array([xlim[0],xlim[0]])
    if (len(ylim)<2 or ylim[1]<ylim[0]):
        ylim=np.array([ylim[0],ylim[0]])
    if (len(zlim)<2 or zlim[1]<zlim[0]):
        zlim=np.array([zlim[0],zlim[0]])
    
    times = 0
    while True:
        times += 1
        if times > 100000:
            return False, None
        x = np.random.uniform(xlim[0], xlim[1])
        y = np.random.uniform(ylim[0], ylim[1])
        is_overlap = False
        for area in prohibited_area:
            # if x + obj_radius >= area[0] and x - obj_radius <= area[1] and y - obj_radius >= area[2] and y + obj_radius <= area[3]:
            #     continue
            # if x > area[0] and x < area[1] and y > area[2] and y < area[3]:
            #     continue
            if check_overlap(obj_radius, x, y, area):
                is_overlap = True
                break
        if is_overlap:
            continue
        distances = np.sqrt((np.array([sub_list[0] for sub_list in size_dict]) - x) ** 2 + (np.array([sub_list[1] for sub_list in size_dict]) - y) ** 2)
        max_distances = np.array([sub_list[3] + obj_radius for sub_list in size_dict])

        if x > -0.24 and x < 0.25 and y > -0.34 and y < 0:
            if z_max > 0.07:
                continue
        if x - obj_radius < -0.6 or x + obj_radius > 0.6 or y - obj_radius < -0.34 or y + obj_radius > 0.34:
            continue
        if np.all(distances > max_distances) and y + obj_radius < ylim[1]:
            break
        
    z = np.random.uniform(zlim[0],zlim[1])
    z = z - z_offset

    rotate = qpos
    if (rotate_rand):
        angles = [0,0,0]
        for i in range(3):
            angles[i] = np.random.uniform(-rotate_lim[i],rotate_lim[i])
        rotate_quat = t3d.euler.euler2quat(angles[0], angles[1], angles[2])
        rotate = t3d.quaternions.qmult(rotate, rotate_quat)

    return True,sapien.Pose([x, y, z],rotate)

# def rand_create_obj(
#     scene: sapien.Scene,
#     modelname: str,
#     xlim: np.ndarray,
#     ylim: np.ndarray,
#     zlim: np.ndarray,
#     ylim_prop = False,
#     rotate_rand = False,
#     rotate_lim = [0,0,0],
#     qpos = [1,0,0,0],
#     scale = (1,1,1),
#     convex = False,
#     is_static = False,
#     model_id = None,
#     z_val_protect = False
# ) -> sapien.Entity:
    
#     obj_pose = rand_pose_messy(
#         xlim=xlim,
#         ylim=ylim,
#         zlim=zlim,
#         ylim_prop=ylim_prop,
#         rotate_rand=rotate_rand,
#         rotate_lim=rotate_lim,
#         qpos=qpos
#     )

#     return create_obj(
#         scene=scene,
#         pose=obj_pose,
#         modelname=modelname,
#         scale=scale,
#         convex=convex,
#         is_static=is_static,
#         model_id = model_id,
#         z_val_protect = z_val_protect
#     )

def rand_create_messy_obj(
    scene: sapien.Scene,
    modelname: str,
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop = False,
    rotate_rand = False,
    rotate_lim = [0,0,0],
    qpos = [1,0,0,0],
    scale = (1,1,1),
    convex = False,
    is_static = False,
    model_id = None,
    z_val_protect = False,
    size_dict = None,
    obj_radius = 0.1,
    z_offset = 0.001,
    prohibited_area = None
) -> sapien.Entity:
    
    success, obj_pose = rand_pose_messy(
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        ylim_prop=ylim_prop,
        rotate_rand=rotate_rand,
        rotate_lim=rotate_lim,
        qpos=qpos,
        size_dict=size_dict,
        obj_radius = obj_radius,
        z_offset=z_offset,
        prohibited_area=prohibited_area
    )
    if not success:
        return False, None
    return True, create_obj(
        scene=scene,
        pose=obj_pose,
        modelname=modelname,
        scale=scale,
        convex=convex,
        is_static=is_static,
        model_id = model_id,
        z_val_protect = z_val_protect
    )

def create_urdf_obj(
    scene: sapien.Scene,
    pose: sapien.Pose,
    modelname: str,
    scale = 1.0,
    fix_root_link = True
)->sapienp.PhysxArticulation: 
    modeldir = "./assets/"+modelname+"/"
    file_name = modeldir + "base.glb"
    json_file_path = modeldir + 'model_data.json'
    
    try:
        with open(json_file_path, 'r') as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    modeldir = "./assets/"+modelname+"/"
    object: sapien.Articulation = loader.load_multiple(modeldir+"model.urdf")[1][0]
    object.set_pose(pose)
    return object, model_data


def rand_create_messy_urdf_obj(
    scene: sapien.Scene,
    modelname: str,
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop = False,
    rotate_rand = False,
    rotate_lim = [0,0,0],
    qpos = [1,0,0,0],
    scale = 1.0,
    fix_root_link = True,
    size_dict = None,
    obj_radius = 0.1,
    z_offset = 0.001,
    z_max = 0,
    prohibited_area = None
)->sapienp.PhysxArticulation: 
    
    success, obj_pose = rand_pose_messy(
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        ylim_prop=ylim_prop,
        rotate_rand=rotate_rand,
        rotate_lim=rotate_lim,
        qpos=qpos,
        size_dict=size_dict,
        obj_radius = obj_radius,
        z_offset = z_offset,
        z_max = z_max,
        prohibited_area = prohibited_area
    )
    if not success:
        return False, None

    return True, create_urdf_obj(
        scene,
        pose= obj_pose,
        modelname=modelname,
        scale=scale,
        fix_root_link = fix_root_link
    )