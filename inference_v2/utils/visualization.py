import open3d as o3d
def visualize_grasp_proposals(scene_cloud, gg_array, inspire_gg_array, cfgs, title="Grasp Proposals"):
    """
    Visualize grasp proposals with different colors for different grasp types
    """
    # Create coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Create visualization objects
    vis_objects = [scene_cloud, frame]
    
    # Color map for different grasp types
    color_map = {
        1: [1, 0, 0],   # Red
        2: [0, 1, 0],   # Green
        3: [0, 0, 1],   # Blue
        4: [1, 1, 0],   # Yellow
        5: [1, 0, 1],   # Magenta
        6: [0, 1, 1],   # Cyan
        7: [0.5, 0.5, 0], # Olive
        8: [0.5, 0, 0.5]  # Purple
    }
    
    # Add grasps with color coding by type
    for i in range(min(80, len(gg_array))):
        grasp_type = int(inspire_gg_array.grasp_types[i])
        color = color_map.get(grasp_type, [0.5, 0.5, 0.5])  # Default to gray
        
        grasp_mesh = inspire_gg_array[i].load_mesh(cfgs['mesh_json_path'], gg_array[i])
        grasp_mesh.paint_uniform_color(color)
        vis_objects.append(grasp_mesh)
    
    # Visualize
    o3d.visualization.draw_geometries(vis_objects, window_name=title)
