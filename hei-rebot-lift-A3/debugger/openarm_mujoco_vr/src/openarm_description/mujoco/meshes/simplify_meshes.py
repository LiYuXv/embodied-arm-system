import bpy
import os

def simplify_mesh(file_path, ratio=0.5, min_faces=1500):
    # 清空场景
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    file_ext = os.path.splitext(file_path)[1].lower()

    # 1. 导入文件
    try:
        if file_ext == '.stl':
            bpy.ops.import_mesh.stl(filepath=file_path)
        elif file_ext == '.dae':
            bpy.ops.wm.collada_import(filepath=file_path)
        elif file_ext == '.obj':
            # 尝试新版 API (3.2+)，如果失败则尝试旧版 API
            try:
                bpy.ops.wm.obj_import(filepath=file_path)
            except Exception:
                try:
                    bpy.ops.import_scene.obj(filepath=file_path)
                except Exception as e:
                    print(f"  [Error] 无法导入 OBJ: {e}")
                    return
    except Exception as e:
        print(f"  [Error] 导入失败 {file_path}: {e}")
        return

    # 2. 获取导入的网格物体
    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not mesh_objs:
        print(f"  [Skip] 未找到有效网格: {file_path}")
        return

    # 3. 计算总面数
    total_faces = sum(len(obj.data.polygons) for obj in mesh_objs)
    
    if total_faces < min_faces:
        print(f"  [Skip] 面数 {total_faces} < {min_faces}")
        return
    else:
        print(f"  [Process] 面数: {total_faces} -> 减少一半")

    # 4. 应用精简
    for obj in mesh_objs:
        bpy.context.view_layer.objects.active = obj
        mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
        mod.ratio = ratio
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # 5. 导出文件
    try:
        if file_ext == '.stl':
            bpy.ops.export_mesh.stl(filepath=file_path)
        elif file_ext == '.dae':
            bpy.ops.wm.collada_export(filepath=file_path)
        elif file_ext == '.obj':
            # 同样对导出做双向兼容处理
            try:
                bpy.ops.wm.obj_export(filepath=file_path)
            except Exception:
                try:
                    bpy.ops.export_scene.obj(filepath=file_path)
                except Exception as e:
                    print(f"  [Error] 无法导出 OBJ: {e}")
                    return
        print(f"  [Success] 已保存: {file_path}")
    except Exception as e:
        print(f"  [Error] 导出失败 {file_path}: {e}")

def process_directory(root_dir):
    valid_extensions = ('.stl', '.dae', '.obj')
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                # 排除脚本自己
                if file == "simplify_meshes.py": continue
                file_path = os.path.join(subdir, file)
                print(f"正在处理: {file_path}")
                simplify_mesh(file_path)

if __name__ == "__main__":
    target_dir = os.path.dirname(os.path.abspath(__file__))
    process_directory(target_dir)