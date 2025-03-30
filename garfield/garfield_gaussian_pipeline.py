import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Mapping, Any, Optional, List, Dict
from torchtyping import TensorType
from pathlib import Path
import trimesh
import viser
import viser.transforms as vtf
import open3d as o3d
import cv2
import time
import numpy as np
import os
import torch
import copy
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch.cuda.amp.grad_scaler import GradScaler
from nerfstudio.viewer.viewer_elements import *
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.models.splatfacto import SplatfactoModel

from cuml.cluster.hdbscan import HDBSCAN
from nerfstudio.models.splatfacto import RGB2SH,SH2RGB

import tqdm

from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors

from scipy.spatial.transform import Rotation as Rot

from garfield.garfield_datamanager import GarfieldDataManagerConfig, GarfieldDataManager
from garfield.garfield_model import GarfieldModel, GarfieldModelConfig
from garfield.garfield_pipeline import GarfieldPipelineConfig, GarfieldPipeline

from collections import deque, defaultdict
import torch.nn.functional as F

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def quat_to_rotmat(quat):
    assert quat.shape[-1] == 4, quat.shape
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    
    return mat.reshape(quat.shape[:-1] + (3, 3))

def generate_random_colors(N=5000) -> torch.Tensor:
    """Generate random colors for visualization"""
    hs = np.random.uniform(0, 1, size=(N, 1))
    ss = np.random.uniform(0.6, 0.61, size=(N, 1))
    vs = np.random.uniform(0.84, 0.95, size=(N, 1))
    hsv = np.concatenate([hs, ss, vs], axis=-1)
    # convert to rgb
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8)[None, ...], cv2.COLOR_HSV2RGB)
    return torch.Tensor(rgb.squeeze() / 255.0)


@dataclass
class GarfieldGaussianPipelineConfig(VanillaPipelineConfig):
    """Gaussian Splatting, but also loading GARField grouping field from ckpt."""
    _target: Type = field(default_factory=lambda: GarfieldGaussianPipeline)
    garfield_ckpt: Optional[Path] = None  # Need to specify this


class GarfieldGaussianPipeline(VanillaPipeline):
    """
    Trains a Gaussian Splatting model, but also loads a GARField grouping field from ckpt.
    This grouping field allows you to:
     - interactive click-based group selection (you can drag it around)
     - scene clustering, then group selection (also can drag it around)

    Note that the pipeline training must be stopped before you can interact with the scene!!
    """
    
    """
    WMX
    Now add some new simple rotation controller for the scene.
    """   
    
    
    model: SplatfactoModel
    garfield_pipeline: List[GarfieldPipeline]  # To avoid importing Viewer* from nerf pipeline
    state_stack: List[Dict[str, TensorType]]  # To revert to previous state
    click_location: Optional[TensorType]  # For storing click location
    click_handle: Optional[viser.GlbHandle]  # For storing click handle
    crop_group_list: List[TensorType]  # For storing gaussian crops (based on click point)
    crop_transform_handle: Optional[viser.TransformControlsHandle]  # For storing scene transform handle -- drag!
    cluster_labels: Optional[TensorType]  # For storing cluster labels

    # new joint controller
    conjunction_joint: Optional[viser.TransformControlsHandle]  # For storing joint controller

    def __init__(
        self,
        config: GarfieldGaussianPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: typing.Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

        print("Loading instance feature model...")
        assert config.garfield_ckpt is not None, "Need to specify garfield checkpoint"
        from nerfstudio.utils.eval_utils import eval_setup
        _, garfield_pipeline, _, _ = eval_setup(
            config.garfield_ckpt, test_mode="inference"
        )
        self.garfield_pipeline = [garfield_pipeline]
        self.state_stack = []

        self.colormap = generate_random_colors()

        self.viewer_control = ViewerControl()

        self.a_interaction_method = ViewerDropdown(
            "Interaction Method",
            default_value="Interactive",
            options=["Interactive", "Clustering"],
            cb_hook=self._update_interaction_method
        )

        self.click_gaussian = ViewerButton(name="Click", cb_hook=self._click_gaussian)
        self.click_location = None
        self.click_handle = None

        self.crop_to_click = ViewerButton(name="Crop to Click", cb_hook=self._crop_to_click, disabled=True)
        self.crop_to_group_level = ViewerSlider(name="Group Level", min_value=0, max_value=29, step=1, default_value=0, cb_hook=self._update_crop_vis, disabled=True)
        self.crop_group_list = []

        self.move_current_crop = ViewerButton(name="Drag Current Crop", cb_hook=self._drag_current_crop, disabled=True)
        self.crop_transform_handle = None
        
        self.cluster_scene = ViewerButton(name="Cluster Scene", cb_hook=self._cluster_scene, disabled=False, visible=False)
        
        # new joint controller 
        self.add_joint_controller = ViewerButton(name="Add Joint Controller", cb_hook=self._add_joint_controller, disabled=True, visible=False)
        self.conjunction_joint = None
        self.currnet_selected_joint = None
        self.skeleton = None    
        #visualization of the conjunction area
        self.visualize_conjunction_area = ViewerButton(name="Visualize Conjunction Area", cb_hook=self._visualize_conjunction_area, disabled=True, visible=False)

        self.cluster_scene_scale = ViewerSlider(name="Cluster Scale", min_value=0.0, max_value=2.0, step=0.01, default_value=0.0, disabled=False, visible=False)
        self.cluster_scene_shuffle_colors = ViewerButton(name="Reshuffle Cluster Colors", cb_hook=self._reshuffle_cluster_colors, disabled=False, visible=False)
        self.cluster_labels = None

       
        self.reset_state = ViewerButton(name="Reset State", cb_hook=self._reset_state, disabled=True)

        self.z_export_options = ViewerCheckbox(name="Export Options", default_value=False, cb_hook=self._update_export_options)
        self.z_export_options_visible_gaussians = ViewerButton(
            name="Export Visible Gaussians",
            visible=False,
            cb_hook=self._export_visible_gaussians
            )
        self.z_export_options_camera_path_filename = ViewerText("Camera Path Filename", "", visible=False)
        self.z_export_options_camera_path_render = ViewerButton("Render Current Pipeline", cb_hook=self.render_from_path, visible=False)
        
        
        #save/load groupings state
        self.z_a = ViewerCheckbox(name="Grouping State", default_value=False, cb_hook=self._update_save_state_options)
        self.z_b = ViewerButton(name="Save State",visible=False,cb_hook=self._save_state)
        self.z_c = ViewerButton("Load State", cb_hook=self._load_state, visible=False)
        self.z_d = ViewerText("State Path Filename", "", visible=False)

    def _update_interaction_method(self, dropdown: ViewerDropdown):
        """Update the UI based on the interaction method"""
        hide_in_interactive = (not (dropdown.value == "Interactive")) # i.e., hide if in interactive mode

        self.cluster_scene.set_hidden((not hide_in_interactive))
        self.cluster_scene_scale.set_hidden((not hide_in_interactive))
        self.cluster_scene_shuffle_colors.set_hidden((not hide_in_interactive))
        
        self.click_gaussian.set_hidden(hide_in_interactive)
        self.crop_to_click.set_hidden(hide_in_interactive)
        self.crop_to_group_level.set_hidden(hide_in_interactive)
        self.move_current_crop.set_hidden(hide_in_interactive)

    def _update_export_options(self, checkbox: ViewerCheckbox):
        """Update the UI based on the export options"""
        self.z_export_options_camera_path_filename.set_hidden(not checkbox.value)
        self.z_export_options_camera_path_render.set_hidden(not checkbox.value)
        self.z_export_options_visible_gaussians.set_hidden(not checkbox.value)

    def _update_save_state_options(self, checkbox: ViewerCheckbox):
        """Update the UI based on the export options"""
        # self.z_state_path_filename.set_hidden(not checkbox.value)
        # self.z_save_state_options_visible_gaussians.set_hidden(not checkbox.value)
        # self.z_save_load_state.set_hidden(not checkbox.value)
        self.z_b.set_hidden(not checkbox.value)
        self.z_c.set_hidden(not checkbox.value)
        self.z_d.set_hidden(not checkbox.value)
    
    
    def _reset_state(self, button: ViewerButton):
        """Revert to previous saved state"""
        assert len(self.state_stack) > 0, "No previous state to revert to"
        prev_state = self.state_stack.pop()
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name]

        self.click_location = None
        if self.click_handle is not None:
            self.click_handle.remove()
        self.click_handle = None

        self.click_gaussian.set_disabled(False)

        self.crop_to_click.set_disabled(True)
        self.crop_to_group_level.set_disabled(True)
        # self.crop_to_group_level.value = 0
        self.move_current_crop.set_disabled(True)
        self.crop_group_list = []
        if self.crop_transform_handle is not None:
            self.crop_transform_handle.remove()
            self.crop_transform_handle = None
        if len(self.state_stack) == 0:
            self.reset_state.set_disabled(True)

        self.cluster_labels = None
        self.cluster_scene.set_disabled(False)
        
        #reset controller
        # if self.conjunction_joint is not None:
        #     self.conjunction_joint.remove()
        if self.conjunction_joint is not None:
            for i in range(len(self.conjunction_joint)):
                self.conjunction_joint[i].remove()

        if self.skeleton is not None:
            for i in range(len(self.skeleton)):
                self.skeleton[i].remove()
        
        if self.currnet_selected_joint is not None:
            self.currnet_selected_joint.remove()
        
        self.add_joint_controller.set_disabled(True)
        self.add_joint_controller.set_hidden(True)
        
        
        
        #reset visualization of the conjunction area
        self.visualize_conjunction_area.set_disabled(True)
        self.visualize_conjunction_area.set_hidden(True)
        
    def _queue_state(self):
        """Save current state to stack"""
        import copy
        self.state_stack.append(copy.deepcopy({k:v.detach() for k,v in self.model.gauss_params.items()}))
        self.reset_state.set_disabled(False)

    def _click_gaussian(self, button: ViewerButton):
        """Start listening for click-based 3D point specification.
        Refer to garfield_interaction.py for more details."""
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.click_gaussian.set_disabled(False)
            self.crop_to_click.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)

        self.click_gaussian.set_disabled(True)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Refer to garfield_interaction.py for more details."""

        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]
        import viser.transforms as vtf

        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()
        # rotate the ray around into cam coordinates
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()
        # project it into coordinates with matrix
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])
        self.model.eval()
        outputs = self.model.get_outputs(cam.to(self.device))
        self.model.train()
        with torch.no_grad():
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()

        self.click_location = np.array(click.origin) + np.array(click.direction) * (depth / z_dir)

        sphere_mesh = trimesh.creation.icosphere(radius=0.2)
        sphere_mesh.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)  # type: ignore
        self.click_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/click",
            mesh=sphere_mesh,
            position=VISER_NERFSTUDIO_SCALE_RATIO * self.click_location,
        )

    def _crop_to_click(self, button: ViewerButton):
        """Crop to click location"""
        assert self.click_location is not None, "Need to specify click location"

        self._queue_state()  # Save current state
        curr_means = self.model.gauss_params['means'].detach()
        self.model.eval()

        # The only way to reset is to reset the state using the reset button.
        self.click_gaussian.set_disabled(True)  # Disable user from changing click
        self.crop_to_click.set_disabled(True)  # Disable user from changing click
        # Get the 3D location of the click
        location = self.click_location
        location = torch.tensor(location).view(1, 3).to(self.device)

        # The list of positions to query for garfield features. The first one is the click location.
        positions = torch.cat([location, curr_means])  # N x 3

        # Create a kdtree, to get the closest gaussian to the click-point.
        points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(curr_means.cpu().numpy()))
        kdtree = o3d.geometry.KDTreeFlann(points)
        _, inds, _ = kdtree.search_knn_vector_3d(location.view(3, -1).float().detach().cpu().numpy(), 10)

        # get the closest point to the sphere, using kdtree
        sphere_inds = inds
        scales = torch.ones((positions.shape[0], 1)).to(self.device)

        keep_list = []
        prev_group = None

        # Iterate over different scales, to get the a range of possible groupings.
        grouping_model = self.garfield_pipeline[0].model
        for s in tqdm.tqdm(torch.linspace(0, 1.5, 30)):
            # Calculate the grouping features, and calculate the affinity between click point and scene
            instances = grouping_model.get_grouping_at_points(positions, s)  # (1+N, 256)
            click_instance = instances[0]
            affinity = torch.norm(click_instance - instances, dim=1)[1:]

            # Filter out points that have affinity < 0.5 (i.e., not likely to be in the same group)
            keeps = torch.where(affinity < 0.5)[0].cpu()
            keep_points = points.select_by_index(keeps.tolist())  # indices of gaussians

            # Here, we desire the gaussian groups to be grouped tightly together spatially. 
            # We use DBSCAN to group the gaussians together, and choose the cluster that contains the click point.
            # Note that there may be spuriously high affinity between points that are spatially far apart,
            #  possibly due two different groups being considered together at an odd angle / far viewpoint.

            # If there are too many points, we downsample them first before DBSCAN.
            # Then, we assign the filtered points to the cluster of the nearest downsampled point.
            if len(keeps) > 5000:
                curr_point_min = keep_points.get_min_bound()
                curr_point_max = keep_points.get_max_bound()

                downsample_size = 0.01 * s
                _, _, curr_points_ds_ids = keep_points.voxel_down_sample_and_trace(
                    voxel_size=max(downsample_size, 0.0001),
                    min_bound=curr_point_min,
                    max_bound=curr_point_max,
                )
                curr_points_ds_ids = np.array([points[0] for points in curr_points_ds_ids])
                curr_points_ds = keep_points.select_by_index(curr_points_ds_ids)
                curr_points_ds_selected = np.zeros(len(keep_points.points), dtype=bool)
                curr_points_ds_selected[curr_points_ds_ids] = True

                _clusters = np.asarray(curr_points_ds.cluster_dbscan(eps=0.02, min_points=5))
                nn_model = NearestNeighbors(
                    n_neighbors=1, algorithm="auto", metric="euclidean"
                ).fit(np.asarray(curr_points_ds.points))

                _, indices = nn_model.kneighbors(np.asarray(keep_points.points)[~curr_points_ds_selected])

                clusters = np.zeros(len(keep_points.points), dtype=int)
                clusters[curr_points_ds_selected] = _clusters
                clusters[~curr_points_ds_selected] = _clusters[indices[:, 0]]

            else:
                clusters = np.asarray(keep_points.cluster_dbscan(eps=0.02, min_points=5))

            # Choose the cluster that contains the click point. If there is none, move to the next scale.
            cluster_inds = clusters[np.isin(keeps, sphere_inds)]
            cluster_inds = cluster_inds[cluster_inds != -1]
            if len(cluster_inds) == 0:
                continue
            cluster_ind = cluster_inds[0]

            keeps = keeps[np.where(clusters == cluster_ind)]

            if prev_group is None:
                prev_group = keeps
                keep_list.append(keeps)
                continue

            keeps = torch.cat([prev_group, keeps])
            keeps = torch.unique(keeps)

            # # Deduplication, based on the # of current points included in the previous group.
            # overlap = torch.isin(keeps, prev_group).sum()
            # if overlap < 0.8 * len(keeps):
            #     prev_group = keeps
            keep_list.append(keeps)

        if len(keep_list) == 0:
            print("No gaussians within crop, aborting")
            # The only way to reset is to reset the state using the reset button.
            self.click_gaussian.set_disabled(False)
            self.crop_to_click.set_disabled(False)
            return

        # Remove the click handle + visualization
        self.click_location = None
        self.click_handle.remove()
        self.click_handle = None
        
        self.crop_group_list = keep_list
        self.crop_to_group_level.set_disabled(False)
        # self.crop_to_group_level.value = 29   # MX : There uis something wrong with this line!!!!!!!!!!!!!!!
        self.move_current_crop.set_disabled(False)


    def _update_crop_vis(self, number: ViewerSlider):
        """Update which click-based crop to visualize -- this requires that _crop_to_click has been called."""
        # If there is no click-based crop or saved state to crop from, do nothing
        if len(self.crop_group_list) == 0:
            return
        if len(self.state_stack) == 0:
            return
        
        # Clamp the number to be within the range of possible crops
        if number.value > len(self.crop_group_list) - 1:
            number.value = len(self.crop_group_list) - 1
            return
        elif number.value < 0:
            number.value = 0
            return

        keep_inds = self.crop_group_list[number.value]
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name][keep_inds]

    def _drag_current_crop(self, button: ViewerButton):
        """Add a transform control to the current scene, and update the model accordingly."""
        self.crop_to_group_level.set_disabled(True)  # Disable user from changing crop
        self.move_current_crop.set_disabled(True)  # Disable user from creating another drag handle
        
        scene_centroid = self.model.gauss_params['means'].detach().mean(dim=0)
        self.crop_transform_handle = self.viewer_control.viser_server.add_transform_controls(
            name=f"/scene_transform",
            position=(VISER_NERFSTUDIO_SCALE_RATIO*scene_centroid).cpu().numpy(),
        )

        # Visualize the whole scene -- the points corresponding to the crop will be controlled by the transform handle.
        crop_inds = self.crop_group_list[self.crop_to_group_level.value]
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name].clone()

        curr_means = self.model.gauss_params['means'].clone().detach()
        curr_rotmats = quat_to_rotmat(self.model.gauss_params['quats'][crop_inds].detach())

        @self.crop_transform_handle.on_update
        def _(_):
            handle_position = torch.tensor(self.crop_transform_handle.position).to(self.device)
            handle_position = handle_position / VISER_NERFSTUDIO_SCALE_RATIO
            handle_rotmat = quat_to_rotmat(torch.tensor(self.crop_transform_handle.wxyz).to(self.device).float())

            means = self.model.gauss_params['means'].detach()
            quats = self.model.gauss_params['quats'].detach()

            means[crop_inds] = handle_position.float() + torch.matmul(
                handle_rotmat, (curr_means[crop_inds] - curr_means[crop_inds].mean(dim=0)).T
            ).T
            quats[crop_inds] = torch.Tensor(Rot.from_matrix(
                torch.matmul(handle_rotmat.float(), curr_rotmats.float()).cpu().numpy()
            ).as_quat()).to(self.device)  # this is in xyzw format
            quats[crop_inds] = quats[crop_inds][:, [3, 0, 1, 2]]  # convert to wxyz format

            self.model.gauss_params['means'] = torch.nn.Parameter(means.float())
            self.model.gauss_params['quats'] = torch.nn.Parameter(quats.float())

            self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender
    
    def joint_controller(self):
        
        """Add a joint controller to the scene, and update the model accordingly."""
        self.crop_to_group_level.set_disabled(True)  # Disable user from changing crop
        self.move_current_crop.set_disabled(True)  # Disable user from creating another drag handle
        
      
        #rgb_visualization
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name].clone()

        curr_means = self.model.gauss_params['means'].clone().detach()
        curr_labels = self.cluster_labels.clone().detach()
        print("get initial means and rotmats")

        #find joint for each groupings
        #1.for each groupings, find the the connection groupings and build the graph(how to define connection?)
        #2.set the root node as the groupings connected to the most groupings
        #3.build the hierarchy tree (how to define the node structure?) 
        
        #find the connection groupings
        def find_connections(means, labels, n, threshold=0.02):
            """
            find the connection between groupings
            """
            unique_labels = labels.unique()
            connections = []
            connected_graph = torch.eye(len(unique_labels))
            connected_points = torch.zeros(len(unique_labels), len(unique_labels),3)
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    group_i_points = means[labels == unique_labels[i]]
                    group_j_points = means[labels == unique_labels[j]]
                    # Compute pairwise distances between points in the two groups
                    dists = torch.cdist(group_i_points, group_j_points).detach().cpu()
                    # Identify points within the threshold distance
                    close_pairs = dists < threshold
                    if close_pairs.sum() >= n:
                        # Extract the indices of points that are close
                        close_i_indices, close_j_indices = torch.where(close_pairs)
                        # Calculate the mean position of these points
                        close_points = torch.cat((group_i_points[close_i_indices], group_j_points[close_j_indices]), dim=0)
                        mean_pos = close_points.mean(dim=0)
                        connections.append((int(unique_labels[i].item()), int(unique_labels[j].item()), mean_pos.tolist()))
                        # else:print(f"Group {unique_labels[i]} and {unique_labels[j]} are not connected. Found {close_pairs.sum()} close points out of required {n}.")
                        connected_graph[i, j] = 1
                        connected_graph[j, i] = 1
                        connected_points[i, j] = mean_pos
                        connected_points[j, i] = mean_pos
            return connections,connected_graph,connected_points
                        
        #set the root node as the groupings connected to the most groupings
        def find_root_node(connections):
            """
            Finds the root node based on the number of connections.
            """
            connection_counts = {}
            for c in connections:
                connection_counts[c[0]] = connection_counts.get(c[0], 0) + 1
                connection_counts[c[1]] = connection_counts.get(c[1], 0) + 1
            root_node = max(connection_counts, key=connection_counts.get)
            return root_node
        
        def build_tree_with_hierarchy(connections, root_label):
            bidirectional_map = defaultdict(list)
            for parent, child, _ in connections:
                bidirectional_map[parent].append(child)
                bidirectional_map[child].append(parent)

            unidirectional_map = {}
            level = {root_label: 0}  # Initialize root level here
            queue = deque([root_label])
            joint_position = {root_label: curr_means[curr_labels == root_label].mean(dim=0).tolist()}
            
            while queue:
                node = queue.popleft()
                
                for neighbor in bidirectional_map[node]:
                    if neighbor not in level:  # Ensure neighbor hasn't been visited
                        # Establish parent-child relationship
                        if node not in unidirectional_map:
                            unidirectional_map[node] = []
                        unidirectional_map[node].append(neighbor)
                        for c in connections:
                            if (c[0], c[1]) == (neighbor, node) or (c[0], c[1]) == (node, neighbor):
                                joint_position[neighbor] = c[2]  # x[2] is the mean position
                                break  # Exit the loop after finding the first match
                        queue.append(neighbor)
                        level[neighbor] = level[node] + 1  # Assign level safely

            return unidirectional_map, joint_position
        
        def find_all_descendants(tree, node_i):
            """
            Find all descendants of node_i in the tree, including children, grandchildren, etc.
            """
            descendants = []
            # Direct children of node_i
            children = tree.get(node_i, [])
            
            for child in children:
                # Add the child
                descendants.append(child)
                # Recursively find and add the child's descendants
                descendants.extend(find_all_descendants(tree, child))
                
            return descendants
        
        def find_children(tree, node_i):
            """
            Find all of node_i's children in the tree.
            """
            return tree.get(node_i, [])
        
        def build_conjunction_joint(tree, joint_position):
            conjunction_joint= {}
            def traverse_and_add_controls(parent , node , path_name):
                # Update the path name to include the current node
                current_path_name = f"{path_name}/group_{node}" if path_name else f"group_{node}"  
                # Calculate group centroid for the current node, if applicable
                if parent != node:
                    relative_joint_position = np.array(joint_position[node]) - np.array(joint_position[parent])
                else:
                    relative_joint_position = np.array(joint_position[node])
                    
                group_transform_handle = self.viewer_control.viser_server.add_transform_controls(
                    name = current_path_name,
                    position = (VISER_NERFSTUDIO_SCALE_RATIO * relative_joint_position),
                    scale=0.6*(0.75 ** current_path_name.count("/")),
                    # disable_axes = True,
                    disable_sliders = True,
                )
                
                conjunction_joint[node] = group_transform_handle
                
                # Recursively add controls for the children, updating the path name
                for child in find_children(tree, node):
                    traverse_and_add_controls(node , child, current_path_name)

            traverse_and_add_controls(root_label, root_label, "")
            return conjunction_joint    
  
        def build_skeleton(tree, joint_position):
            if self.skeleton is not None:
                for i in range(len(self.skeleton)):
                    self.skeleton[i].remove()
                    
            skeleton= {}
            def traverse_and_add_line(parent , node , path_name):
                # Update the path name to include the current node
                current_path_name = f"{path_name}/{node}" if path_name else f"{node}"   
                 
                line =  self.viewer_control.viser_server.add_spline_catmull_rom(
                    name = current_path_name,
                    positions =np.array([joint_position[parent], joint_position[node]]) * VISER_NERFSTUDIO_SCALE_RATIO,
                    tension = 0.5,
                    line_width = 3.0,
                    color=np.array([1.0, 0.0, 1.0]),
                    segments = 100 )
                
                skeleton[node] = line
                
                # Recursively add controls for the children, updating the path name
                for child in find_children(tree, node):
                    traverse_and_add_line(node , child, current_path_name)

            traverse_and_add_line(root_label, root_label, "")
            return skeleton   
        
        def build_parent_mapping(tree_structure):
            parent_map = {}
            for parent, children in tree_structure.items():
                for child in children:
                    parent_map[child] = parent
            return parent_map

        def pose2mat(pose:torch.Tensor):
            """
            Converts a 7-vector pose to a 4x4 transformation matrix
            """
            t = pose[:,:3]
            q = pose[:,3:]
            rot = quat_to_rotmat(q)
            mat = torch.eye(4)[None].repeat(pose.size(0),1,1)
            mat[:,:3,:3] = rot
            mat[:,:3,3] = t
            return mat
                
         
        connections,self.connected_graph ,self.connected_points = find_connections(curr_means, curr_labels, n = 2)
        root_label = find_root_node(connections)
        tree, joint_position = build_tree_with_hierarchy(connections, root_label) 
        
        self.curr_joints =  torch.tensor([joint_position[i] for i in sorted(joint_position.keys())]).to(self.device)   # world space joint positions
        self.tree = tree
        self.root_label = root_label
        self.parent_map = build_parent_mapping(tree)
         
        #visualization (controller + skeleton)
        self.conjunction_joint = build_conjunction_joint(tree, joint_position)
        self.skeleton = build_skeleton(tree, self.curr_joints.detach().cpu().numpy())
        self.relative_joint_position = []
        for i in range(len(self.curr_joints)):
            if i == root_label:
                self.relative_joint_position.append(self.curr_joints[i])
            else:
                self.relative_joint_position.append(self.curr_joints[i]-self.curr_joints[self.parent_map[i]])
        self.relative_joint_position = torch.stack(self.relative_joint_position).to(self.device)                   
        #To update the joint position
        print(tree)
        self.tree_list = [root_label] + find_all_descendants(tree, root_label)
        
        ###########################################################################################################################
        #set up for the transformation
        self.init_means = self.model.gauss_params['means'].detach().clone()
        self.init_quats = self.model.gauss_params['quats'].detach().clone()
        self.group_masks = self.cluster_labels
        self.joint_position = self.curr_joints
        self.num_joints = len(self.joint_position)
        self.updated_joint_position = self.joint_position.detach().clone()
        
        def relative_joint_positions(joints):
            """
            input: world position of the joints 
            Returns the relative joint positions given the joint positions 4*4
            """
            root_transformation = torch.eye(4,device=self.device)
            root_transformation[:3,3] = joints[self.root_label]
            
            relative_joints = {self.root_label: root_transformation}
            for p_node,c_node in self.tree.items():
                for node in c_node:
                    relative_joints[node] = torch.eye(4,device=self.device)
                    relative_joints[node][:3,3] = joints[node] - joints[p_node]
                
            return relative_joints
        ###########################################################################################################################
        #set up for the transformation
        self.relative_joint_poses = relative_joint_positions(self.joint_position)
        
        def gaussian_local_pose():
            """
            Returns the local pose of each gaussian in the camera coordinate system
            """
            # V_wg = T_wc V_cg -> V_cg = T_wc^-1 V_wg : gaussian pose in child frame
            updated_curr_means = self.init_means.clone()
            updated_curr_rotmats = quat_to_rotmat(self.init_quats.clone())
            
            local_transoform = torch.tile(torch.eye(4),(len(self.group_masks),1,1)).to(self.device)
            for node in self.relative_joint_poses.keys():
                group_inds = np.isin(self.group_masks,[node])
                group_inds = torch.tensor(group_inds).to(self.device)
                curr_means = updated_curr_means[group_inds]
                curr_rotates = updated_curr_rotmats[group_inds]
                
                w_vector = torch.tile(torch.eye(4),(len(curr_means),1,1)).to(self.device)
                w_vector[:,:3,:3] = curr_rotates
                w_vector[:,:3,3] = curr_means
                
                w_frame = torch.eye(4).to(self.device)
                w_frame[:3,3] = self.joint_position[node]
                
                local_transoform[group_inds] = torch.matmul(torch.inverse(w_frame)[None].repeat(w_vector.shape[0],1,1),w_vector)
            
            return local_transoform
        ###########################################################################################################################
        #set up for the transformation
        self.local_gaussian_pose = gaussian_local_pose()  
        def apply_to_model(pose_deltas):
            with torch.no_grad():
                self.model.gauss_params['means'] = self.init_means.clone()
                self.model.gauss_params['quats'] = self.init_quats.clone()
            updated_joint_poses = {}
            delta_mat = pose2mat(pose_deltas).to(self.device)
            for node,j_pose in self.relative_joint_poses.items():
                updated_joint_poses[node] = j_pose @ delta_mat[node] 
            
            #run forward kinematics to update the joint positions
            world_poses = {}
            queue = deque([self.root_label])
            #make sure the root is updated first, update parent first then children
            while queue:
                current_node = queue.popleft()
                if current_node in self.tree:
                    for child in self.tree[current_node]:
                        queue.append(child) 
                
                if current_node == self.root_label:
                    world_poses[current_node] = updated_joint_poses[current_node]
                else:
                    #T_wc = T_wp @ T_pc
                    world_poses[current_node] = world_poses[self.parent_map[current_node]] @ updated_joint_poses[current_node]

            #apply transformation at one time
            means = self.model.gauss_params['means'].detach()
            rotmats = self.model.gauss_params['quats'].detach()
            
            updated_curr_means = self.init_means.clone()
            updated_curr_rotmats = quat_to_rotmat(self.init_quats.clone())
            w_poses = torch.tile(torch.eye(4),(self.model.num_points,1,1)).to(self.device)
            for node,w_pose in world_poses.items():
                group_inds = np.isin(self.group_masks,[node])
                group_inds = torch.tensor(group_inds).to(self.device)
                w_poses[group_inds] = w_pose
        
            local_transoform = self.local_gaussian_pose
            updated_transform = torch.matmul(w_poses,local_transoform)
            
            updated_curr_means = updated_transform[:,:3, 3]
            updated_curr_rotmats = updated_transform[:,:3,:3]

            means = updated_curr_means
            rotmats = torch.Tensor(Rot.from_matrix(updated_curr_rotmats.detach().cpu().numpy()).as_quat()).to(self.device)
            rotmats = rotmats[:, [3, 0, 1, 2]]
            
            self.model.gauss_params['means'] = means.float()
            self.model.gauss_params['quats'] = rotmats.float()

            #update joint_position
            self.updated_joint_position = torch.stack([world_poses[i][:3,3] for i in sorted(world_poses.keys())]).to(self.device)

        for i in self.tree_list:
            def set_callback_in_closure(i: int) -> None:
                @self.conjunction_joint[i].on_update
                def _(_) -> None: 
                    pose_deltas = torch.zeros(int(self.group_masks.unique().max()+1),7,dtype=torch.float32,device='cuda')
                    pose_deltas[:,3:] = torch.tensor([1,0,0,0],dtype=torch.float32,device='cuda')
                    for j in range(len(self.tree_list)):
                        pose_deltas[j,:3] = torch.tensor(self.conjunction_joint[j].position).to(self.device)/VISER_NERFSTUDIO_SCALE_RATIO  -  self.relative_joint_position[j].clone()
                        pose_deltas[j,3:] = torch.tensor(self.conjunction_joint[j].wxyz).to(self.device).float()
                        
                    apply_to_model(pose_deltas)
                    self.skeleton = build_skeleton(self.tree, self.updated_joint_position.clone().detach().cpu().numpy())
                    self.viewer_control.viewer._trigger_rerender()  
                    
                    
            set_callback_in_closure(i)
            
    def _add_joint_controller (self, button: ViewerButton):
        self.joint_controller()
    
    def _visualize_conjunction_area(self, button: ViewerButton):
        
        """find all the conjunciton area"""
        def find_conjunction_points(points, groupings, threshold):
            # Ensure groupings is on the same device as points
            groupings = groupings.detach().cpu()
            points = points.detach().cpu()
            # Calculate pairwise distances between points
            dist_matrix = torch.cdist(points, points, p=2)

            # Initialize a tensor of False, indicating non-conjunction points
            conjunction_mask = torch.zeros(points.size(0), dtype=torch.bool, device=points.device)

            # Iterate over each point
            for i in range(points.size(0)):
                # Get distances to all other points and check if they are in a different grouping
                distances = dist_matrix[i]
                other_groupings = groupings != groupings[i]

                # Find if there's any point within the threshold distance belonging to a different group
                is_conjunction = (distances <= threshold) & other_groupings

                # If any such point exists, mark the current point as a conjunction point
                if is_conjunction.any():
                    conjunction_mask[i] = True
            
            return conjunction_mask
        
        """Add controllers to all the groupings."""
        self.crop_to_group_level.set_disabled(True)  # Disable user from changing crop
        self.move_current_crop.set_disabled(True)  # Disable user from creating another drag handle
            
        #rgb_visualization
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name].clone()
        
        
        curr_means = self.model.gauss_params['means'].clone().detach()
        curr_labels = self.cluster_labels.clone().detach()
        
        conjunction_mask = find_conjunction_points(curr_means, curr_labels, 0.01).to(self.device)
        
        #visualize the conjunction area-show them in red
        features_dc = self.model.gauss_params['features_dc'].detach()
        features_rest = self.model.gauss_params['features_rest'].detach()
        red_color = torch.tensor([1.0, 0.0, 0.0])
        features_dc[conjunction_mask] = RGB2SH(red_color.to(self.model.gauss_params['features_dc']))
        features_rest[conjunction_mask] = 0
        self.model.gauss_params['features_dc'] = torch.nn.Parameter(self.model.gauss_params['features_dc'])
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(self.model.gauss_params['features_rest'])
        
        opacities = self.model.gauss_params['opacities'].detach()
        opacities = torch.logit(torch.ones_like(opacities)* 0.1)
        opacities[conjunction_mask] = torch.logit(torch.ones_like(opacities)* 0.5)[conjunction_mask]
        self.model.gauss_params['opacities'] = torch.nn.Parameter(opacities)
    
    def _reshuffle_cluster_colors(self, button: ViewerButton):
        """Reshuffle the cluster colors, if clusters defined using `_cluster_scene`."""
        if self.cluster_labels is None:
            return
        self.cluster_scene_shuffle_colors.set_disabled(True)  # Disable user from reshuffling colors
        self.colormap = generate_random_colors()
        colormap = self.colormap

        labels = self.cluster_labels

        features_dc = self.model.gauss_params['features_dc'].detach()
        features_rest = self.model.gauss_params['features_rest'].detach()
        for c_id in range(0, labels.max().int().item() + 1):
            # set the colors of the gaussians accordingly using colormap from matplotlib
            cluster_mask = np.where(labels == c_id)
            features_dc[cluster_mask] = RGB2SH(colormap[c_id, :3].to(self.model.gauss_params['features_dc']))
            features_rest[cluster_mask] = 0

        self.model.gauss_params['features_dc'] = torch.nn.Parameter(self.model.gauss_params['features_dc'])
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(self.model.gauss_params['features_rest'])
        self.cluster_scene_shuffle_colors.set_disabled(False)

    def _cluster_scene(self, button: ViewerButton):
        """Cluster the scene, and assign gaussian colors based on the clusters.
        Also populates self.crop_group_list with the clusters group indices."""

        self._queue_state()  # Save current state
        self.cluster_scene.set_disabled(True)  # Disable user from clustering, while clustering

        scale = self.cluster_scene_scale.value
        grouping_model = self.garfield_pipeline[0].model
        
        positions = self.model.gauss_params['means'].detach()
        group_feats = grouping_model.get_grouping_at_points(positions, scale).cpu().numpy()  # (N, 256)
        positions = positions.cpu().numpy()

        start = time.time()

        # Cluster the gaussians using HDBSCAN.
        # We will first cluster the downsampled gaussians, then 
        #  assign the full gaussians to the spatially closest downsampled gaussian.

        vec_o3d = o3d.utility.Vector3dVector(positions)
        pc_o3d = o3d.geometry.PointCloud(vec_o3d)
        min_bound = np.clip(pc_o3d.get_min_bound(), -1, 1)
        max_bound = np.clip(pc_o3d.get_max_bound(), -1, 1)
        # downsample size to be a percent of the bounding box extent
        downsample_size = 0.01 * scale
        pc, _, ids = pc_o3d.voxel_down_sample_and_trace(
            max(downsample_size, 0.0001), min_bound, max_bound
        )
        if len(ids) > 1e6:
            print(f"Too many points ({len(ids)}) to cluster... aborting.")
            print( "Consider using interactive select to reduce points before clustering.")
            print( "Are you sure you want to cluster? Press y to continue, else return.")
            # wait for input to continue, if yes then continue, else return
            if input() != "y":
                self.cluster_scene.set_disabled(False)
                return

        id_vec = np.array([points[0] for points in ids])  # indices of gaussians kept after downsampling
        group_feats_downsampled = group_feats[id_vec]
        positions_downsampled = np.array(pc.points)

        print(f"Clustering {group_feats_downsampled.shape[0]} gaussians... ", end="", flush=True)

        # Run cuml-based HDBSCAN
        clusterer = HDBSCAN(
            cluster_selection_epsilon=0.1,
            min_samples=30,
            min_cluster_size=30,
            allow_single_cluster=True,
        ).fit(group_feats_downsampled)

        non_clustered = np.ones(positions.shape[0], dtype=bool)
        non_clustered[id_vec] = False
        labels = clusterer.labels_.copy()
        clusterer.labels_ = -np.ones(positions.shape[0], dtype=np.int32)
        clusterer.labels_[id_vec] = labels

        # Assign the full gaussians to the spatially closest downsampled gaussian, with scipy NearestNeighbors.
        positions_np = positions[non_clustered]
        if positions_np.shape[0] > 0:  # i.e., if there were points removed during downsampling
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(positions_downsampled)
            _, indices = nn_model.kneighbors(positions_np)
            clusterer.labels_[non_clustered] = labels[indices[:, 0]]

        labels = clusterer.labels_
        print(f"done. Took {time.time()-start} seconds. Found {labels.max() + 1} clusters.")

        # Relabel the noise points to the nearest cluster
        noise_mask = labels == -1
        if noise_mask.sum() != 0 and (labels>=0).sum() > 0:
            # if there is noise, but not all of it is noise, relabel the noise
            valid_mask = labels >=0
            valid_positions = positions[valid_mask]
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(valid_positions)
            noise_positions = positions[noise_mask]
            _, indices = nn_model.kneighbors(noise_positions)
            # for now just pick the closest cluster
            noise_relabels = labels[valid_mask][indices[:, 0]]
            labels[noise_mask] = noise_relabels
            clusterer.labels_ = labels
        
        #color the groupings
        labels = clusterer.labels_
        
        colormap = self.colormap

        opacities = self.model.gauss_params['opacities'].detach()
        opacities[labels < 0] = -100  # hide unclustered gaussians
        self.model.gauss_params['opacities'] = torch.nn.Parameter(opacities.float())

        self.cluster_labels = torch.Tensor(labels)
        features_dc = self.model.gauss_params['features_dc'].detach()
        features_rest = self.model.gauss_params['features_rest'].detach()
        for c_id in range(0, labels.max() + 1):
            # set the colors of the gaussians accordingly using colormap from matplotlib
            cluster_mask = np.where(labels == c_id)
            features_dc[cluster_mask] = RGB2SH(colormap[c_id, :3].to(self.model.gauss_params['features_dc']))
            features_rest[cluster_mask] = 0

        self.model.gauss_params['features_dc'] = torch.nn.Parameter(self.model.gauss_params['features_dc'])
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(self.model.gauss_params['features_rest'])

        self.cluster_scene.set_disabled(False)
        self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender
        
        #now add the joint controller based on the clustering results
        #in cluster mode, we can add joint controller
        self.add_joint_controller.set_disabled(False)
        self.add_joint_controller.set_visible(True)
        
        self.visualize_conjunction_area.set_disabled(False)
        self.visualize_conjunction_area.set_visible(True)

    def _export_visible_gaussians(self, button: ViewerButton):
        """Export the visible gaussians to a .ply file"""
        # location to save
        output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
        filename = Path(output_dir) / f"gaussians.ply"

        # Copied from exporter.py
        map_to_tensors = {}

        with torch.no_grad():
            positions = self.model.gauss_params['means'].cpu().numpy()
            map_to_tensors["positions"] = o3d.core.Tensor(positions, o3d.core.float32)
            map_to_tensors["normals"] = o3d.core.Tensor(np.zeros_like(positions), o3d.core.float32)

            colors = self.model.colors.data.cpu().numpy()
            map_to_tensors["colors"] = (colors * 255).astype(np.uint8)
            for i in range(colors.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = colors[:, i : i + 1]

            shs = self.model.shs_rest.data.cpu().numpy()
            if self.model.config.sh_degree > 0:
                shs = shs.reshape((colors.shape[0], -1, 1))
                for i in range(shs.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs[:, i]

            map_to_tensors["opacity"] = self.model.gauss_params['opacities'].data.cpu().numpy()

            scales = self.model.gauss_params['scales'].data.cpu().unsqueeze(-1).numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i]

            quats = self.model.gauss_params['quats'].data.cpu().unsqueeze(-1).numpy()

            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i]

        pcd = o3d.t.geometry.PointCloud(map_to_tensors)

        o3d.t.io.write_point_cloud(str(filename), pcd)

    def render_from_path(self, button: ViewerButton):
        from nerfstudio.cameras.camera_paths import get_path_from_json
        import json
        from nerfstudio.scripts.render import _render_trajectory_video

        assert self.z_export_options_camera_path_filename.value != ""
        camera_path_filename = Path(self.z_export_options_camera_path_filename.value)
        
        with open(camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        camera_path = get_path_from_json(camera_path)
        self.model.eval()
        with torch.no_grad():
            _render_trajectory_video(
                self,
                camera_path,
                output_filename=Path('render.mp4'),
                rendered_output_names=['rgb'],
                rendered_resolution_scaling_factor=1.0 ,
                seconds=seconds,
                output_format="video",
            )
        self.model.train()
    
    def _save_state(self, button: ViewerButton):
        """Save the current state of the model."""
        """the only thing I need to save is the mask?"""
        #save the current state
        params_to_save = {}
        for k, v in self.model.gauss_params.items():
            params_to_save[k] = v  # assuming v is already a torch.Tensor

        #save label
        params_to_save["cluster_labels"] = self.cluster_labels

        # Save to file
        state_path_filename = Path ("state/" + self.z_d.value+ ".pt")    
        state_path_filename = state_path_filename 
        
        torch.save(params_to_save, state_path_filename)

    def load_state_from_path(self,state_path_filename,blend_color=False):
        self._queue_state()
        loaded_state = torch.load(state_path_filename)
        for name in self.model.gauss_params.keys():
            if blend_color and name == "features_dc":
                # self.model.gauss_params[name] =  self.model.gauss_params[name]
                # self.model.gauss_params[name] = (self.model.gauss_params[name] + loaded_state[name].clone().to(self.device)) / 2
                self.model.gauss_params[name] = loaded_state[name].clone().to(self.device) *0.5+ self.model.gauss_params[name] *0.5
            else: 
                self.model.gauss_params[name] = loaded_state[name].clone().to(self.device)

        
        #not good to load the cluster labels
        if  loaded_state["cluster_labels"] is not None:
            self.cluster_labels = loaded_state["cluster_labels"].clone()
            self._queue_state()
            self.joint_controller()
        
        
    def _load_state(self, button: ViewerButton):
        """Load the previous state of the model."""
        state_path_filename = Path ("state/" + self.z_d.value+ ".pt")
        self.load_state_from_path(state_path_filename)