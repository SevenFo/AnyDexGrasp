import open3d as o3d
import numpy as np

def visualize_grasps(scene_cloud, pre_collision_gg, post_collision_gg, inspire_pre_gg, inspire_post_gg,cfgs):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis_objects = [scene_cloud, frame]
    
    for i in range(min(10, len(pre_collision_gg))):
        grasp_mesh = inspire_pre_gg[i].load_mesh(cfgs.inspire_mesh_json_path, pre_collision_gg[i])
        grasp_mesh.paint_uniform_color([0, 1, 0])
        vis_objects.append(grasp_mesh)
    
    for i in range(min(10, len(post_collision_gg))):
        grasp_mesh = inspire_post_gg[i].load_mesh(cfgs.inspire_mesh_json_path, post_collision_gg[i])
        grasp_mesh.paint_uniform_color([1, 0, 0])
        vis_objects.append(grasp_mesh)
    
    o3d.visualization.draw_geometries(vis_objects)

def visualize_grasp_proposals(scene_cloud, gg_array, inspire_gg_array, cfgs, title="Grasp Proposals", save_path=None):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis_objects = [scene_cloud, frame]
    grasp_meshs = o3d.geometry.TriangleMesh()
    combined_mesh = o3d.geometry.TriangleMesh()
    color_map = {
        1: [1, 0, 0],
        2: [0, 1, 0],
        3: [0, 0, 1],
        4: [1, 1, 0],
        5: [1, 0, 1],
        6: [0, 1, 1],
        7: [0.5, 0.5, 0],
        8: [0.5, 0, 0.5]
    }
    # 添加场景点云到合并mesh（作为顶点）
    scene_points = np.asarray(scene_cloud.points)
    scene_colors = np.asarray(scene_cloud.colors) if scene_cloud.has_colors() else np.zeros((len(scene_points), 3))
    combined_mesh.vertices = o3d.utility.Vector3dVector(scene_points)
    combined_mesh.vertex_colors = o3d.utility.Vector3dVector(scene_colors)
    # 添加坐标系到合并mesh
    combined_mesh += frame
    for i in range(min(80, len(gg_array))):
        grasp_type = int(inspire_gg_array.grasp_types[i])
        color = color_map.get(grasp_type, [0.5, 0.5, 0.5])
        grasp_mesh = inspire_gg_array[i].load_mesh(cfgs.inspire_mesh_json_path, gg_array[i])
        grasp_mesh.paint_uniform_color(color)
        vis_objects.append(grasp_mesh)
        # 新增：合并mesh用于保存
        grasp_meshs += grasp_mesh
        combined_mesh += grasp_mesh
    # 新增：保存功能
    if save_path:
        # 保存场景点云
        o3d.io.write_point_cloud(save_path.replace('.ply', '_cloud.ply'), scene_cloud)
        # 保存抓取mesh
        o3d.io.write_triangle_mesh(save_path.replace('.ply', '_grasps.ply'), grasp_meshs)
        # 优化网格数据
        combined_mesh.remove_duplicated_vertices()
        combined_mesh.remove_duplicated_triangles()
        
        # 保存完整场景（包含点云+坐标系+抓取提案）
        o3d.io.write_triangle_mesh(save_path.replace('.ply', '_scene.ply'), combined_mesh)
        print(f"场景已保存至 {save_path}")
    o3d.visualization.draw_geometries(vis_objects, window_name=title)
