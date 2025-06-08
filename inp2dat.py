import math
import re
import numpy as np

# 3D 旋转辅助函数 (罗德里格旋转公式)
def rotate_point(p, angle_deg, axis_p1, axis_p2):
    theta = math.radians(angle_deg)
    axis_vec = [axis_p2[i] - axis_p1[i] for i in range(3)]
    norm = math.sqrt(sum(c * c for c in axis_vec))
    if norm == 0: return p  # 如果轴向量为零则无法旋转
    k = [c / norm for c in axis_vec]

    # 平移点，使 axis_p1 成为原点
    p_translated = [p[i] - axis_p1[i] for i in range(3)]

    # 罗德里格旋转公式
    k_cross_p = [k[1] * p_translated[2] - k[2] * p_translated[1],
                 k[2] * p_translated[0] - k[0] * p_translated[2],
                 k[0] * p_translated[1] - k[1] * p_translated[0]]
    k_dot_p = sum(k[i] * p_translated[i] for i in range(3))

    p_rotated = [
        p_translated[i] * math.cos(theta) +
        k_cross_p[i] * math.sin(theta) +
        k[i] * k_dot_p * (1 - math.cos(theta))
        for i in range(3)
    ]

    # 将点平移回去
    return [p_rotated[i] + axis_p1[i] for i in range(3)]


def parse_inp_to_dat(inp_filepath, dat_filepath):
    inp_data = {
        "heading": "默认标题",
        "parts": {},
        # {part_name: {"nodes": {id: [x,y,z]}, "elements": {id: {"type": str, "nodes": [], "elset": str}}, "elsets": {name:[]}, "nsets":{name:[]}, "sections": {}, "materials": {}}}
        "assembly": {"instances": [], "nsets": {}, "elsets": {}},
        "materials_global": {},  # {mat_name: {"type": "elastic", "props": [E, nu], "density": d}}
        "boundaries": [],  # [(set_name/node_id, dof, value)] (对于约束，value 通常是 min=max)
        "loads": {}  # 用于重力
    }

    current_part_name = None
    current_material_name = None
    current_keyword = None
    reading_instance_transform = False
    instance_transform_lines_read = 0

    print(f"开始解析 INP 文件: {inp_filepath}")

    with open(inp_filepath, 'r', encoding='utf-8') as f:  # 指定UTF-8编码以处理可能的中文路径或内容
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("**"):
            continue

        if line.startswith("*Heading"):
            current_keyword = "*Heading"
            # 下一个非注释行是标题
            while i < len(lines):
                heading_line = lines[i].strip()
                i += 1
                if heading_line and not heading_line.startswith("**"):
                    job_name_match = re.search(r"Job name:\s*(\S+)", heading_line, re.IGNORECASE)
                    model_name_match = re.search(r"Model name:\s*(\S+)", heading_line, re.IGNORECASE)
                    title_parts = []
                    if job_name_match: title_parts.append(job_name_match.group(1))
                    if model_name_match: title_parts.append(model_name_match.group(1))
                    if title_parts:
                        inp_data["heading"] = " ".join(title_parts)
                    else:  # 如果未找到特定格式，则回退
                        inp_data["heading"] = heading_line.replace("** ", "")
                    break
            continue

        if line.startswith("*Part"):
            current_keyword = "*Part"
            current_part_name = line.split("name=")[1].strip()
            inp_data["parts"][current_part_name] = {"nodes": {}, "elements": {}, "elsets": {}, "nsets": {},
                                                    "sections": {}}
            print(f"  正在解析部件: {current_part_name}")
            continue

        if line.startswith("*End Part"):
            current_part_name = None
            current_keyword = None
            continue

        if line.startswith("*Node"):
            current_keyword = "*Node"
            continue

        if line.startswith("*Element"):
            current_keyword = "*Element"
            params = {p.split("=")[0].strip().lower(): p.split("=")[1].strip() for p in line.split(",")[1:] if "=" in p}
            current_element_type = params.get("type")
            current_element_elset = params.get("elset")
            continue

        if line.startswith("*Elset"):
            current_keyword = "*Elset"
            params = {p.split("=")[0].strip().lower(): p.split("=")[1].strip() for p in line.split(",") if "=" in p}
            elset_name = params["elset"]
            # is_internal = "internal" in line.lower() # 内部标志在readme.md格式中似乎不直接使用
            generate = "generate" in line.lower()

            if current_part_name:
                if elset_name not in inp_data["parts"][current_part_name]["elsets"]:
                    inp_data["parts"][current_part_name]["elsets"][elset_name] = []
                target_elset_list = inp_data["parts"][current_part_name]["elsets"][elset_name]
            elif "assembly" in inp_data:  # 装配体单元集
                if elset_name not in inp_data["assembly"]["elsets"]:
                    inp_data["assembly"]["elsets"][elset_name] = []
                target_elset_list = inp_data["assembly"]["elsets"][elset_name]
            else:
                continue

            if generate:
                line_data = lines[i].strip()
                i += 1
                start_id, end_id, step = map(int, [val.strip() for val in line_data.split(",")])
                target_elset_list.extend(list(range(start_id, end_id + 1, step)))
            else:
                while i < len(lines) and not lines[i].strip().startswith("*") and lines[i].strip():
                    data_line = lines[i].strip()
                    i += 1
                    ids = [int(val.strip()) for val in data_line.split(",") if val.strip()]
                    target_elset_list.extend(ids)
            continue

        if line.startswith("*Nset"):
            current_keyword = "*Nset"
            params = {p.split("=")[0].strip().lower(): p.split("=")[1].strip() for p in line.split(",") if "=" in p}
            nset_name = params["nset"]
            instance_name = params.get("instance")
            generate = "generate" in line.lower()

            if instance_name:  # 装配体节点集
                if nset_name not in inp_data["assembly"]["nsets"]:
                    inp_data["assembly"]["nsets"][nset_name] = []
                target_nset_list_container = inp_data["assembly"]["nsets"]
            elif current_part_name:  # 部件节点集
                if nset_name not in inp_data["parts"][current_part_name]["nsets"]:
                    inp_data["parts"][current_part_name]["nsets"][nset_name] = []
                target_nset_list_container = inp_data["parts"][current_part_name]["nsets"]
            else:
                continue

            entry_data_ids = []
            if generate:
                line_data = lines[i].strip()
                i += 1
                parts_gen = [val.strip() for val in line_data.split(",")]
                start_id, end_id, step = int(parts_gen[0]), int(parts_gen[1]), int(parts_gen[2])
                entry_data_ids.extend(list(range(start_id, end_id + 1, step)))
            else:
                while i < len(lines) and not lines[i].strip().startswith("*") and lines[i].strip():
                    data_line = lines[i].strip()
                    i += 1
                    ids = [int(val.strip()) for val in data_line.split(",") if val.strip()]
                    entry_data_ids.extend(ids)

            # NSETs in assembly can refer to multiple instances or parts if not specified otherwise by instance=
            # For simplicity, we store a list of dicts, each dict specifying the instance and its local IDs for that set.
            if nset_name not in target_nset_list_container:
                target_nset_list_container[nset_name] = []

            target_nset_list_container[nset_name].append({
                'instance_ref': instance_name if instance_name else current_part_name,
                # if part nset, instance_ref is the part_name
                'ids': entry_data_ids
            })
            continue

        if line.startswith("*Solid Section"):
            current_keyword = "*Solid Section"
            params = {p.split("=")[0].strip().lower(): p.split("=")[1].strip() for p in line.split(",") if "=" in p}
            elset_name = params["elset"]
            material_name = params["material"]
            prop_line = lines[i].strip()
            i += 1
            props = [float(p.strip()) for p in prop_line.split(",") if p.strip()]
            section_data = {"type": "Solid", "elset": elset_name, "material": material_name, "props": props}
            inp_data["parts"][current_part_name]["sections"][elset_name] = section_data
            continue

        if line.startswith("*Shell Section"):
            current_keyword = "*Shell Section"
            params = {p.split("=")[0].strip().lower(): p.split("=")[1].strip() for p in line.split(",") if "=" in p}
            elset_name = params["elset"]
            material_name = params["material"]
            prop_line = lines[i].strip()
            i += 1
            props_str = [p.strip() for p in prop_line.split(",") if p.strip()]
            thickness = float(props_str[0])
            section_data = {"type": "Shell", "elset": elset_name, "material": material_name, "thickness": thickness}
            inp_data["parts"][current_part_name]["sections"][elset_name] = section_data
            continue

        if line.startswith("*Beam Section"):
            current_keyword = "*Beam Section"
            params = {p.split("=")[0].strip().lower(): p.split("=")[1].strip() for p in line.split(",") if "=" in p}
            elset_name = params["elset"]
            material_name = params["material"]
            section_type_str = params["section"].upper()

            geom_line = lines[i].strip()
            i += 1
            geom_props_raw = [float(p.strip()) for p in geom_line.split(",") if p.strip()]
            # 如果是 BOX，而用户只写了 4 个参数(b, d, tf, tw)，补 2 个壁厚 = tw
            if section_type_str == "BOX" and len(geom_props_raw) == 4:
                b, d, tf, tw = geom_props_raw
                geom_props_val = [b, d, tf, tf, tw, tw]          # 按 “顶/底/左/右” 厚度补全
            else:
                geom_props_val = geom_props_raw  # 原样保存(常见 6 参数)

            orientation_props_val = None
            if i < len(lines) and not lines[i].strip().startswith("*") and lines[i].strip():
                try:
                    orientation_candidate_val = [float(p.strip()) for p in lines[i].strip().split(",") if p.strip()]
                    if len(orientation_candidate_val) == 3:
                        orientation_props_val = orientation_candidate_val
                        i += 1
                except ValueError:
                    pass

            section_data = {"type": "Beam", "elset": elset_name, "material": material_name,
                            "section_type": section_type_str, "geom_props": geom_props_val,
                            "orientation": orientation_props_val}
            inp_data["parts"][current_part_name]["sections"][elset_name] = section_data
            continue

        if line.startswith("*Material"):
            current_keyword = "*Material"
            current_material_name = line.split("name=")[1].strip()
            inp_data["materials_global"][current_material_name] = {"props": [], "density": None}
            continue

        if line.startswith("*Density"):
            current_keyword = "*Density"
            if current_material_name:
                density_val = float(lines[i].strip().split(",")[0])
                i += 1
                inp_data["materials_global"][current_material_name]["density"] = density_val
            continue

        if line.startswith("*Elastic"):
            current_keyword = "*Elastic"
            if current_material_name:
                elastic_props_str = [p.strip() for p in lines[i].strip().split(",") if p.strip()]
                i += 1
                inp_data["materials_global"][current_material_name]["props"] = [float(p) for p in
                                                                                elastic_props_str[:2]]  # E, nu
            continue

        if line.startswith("*Assembly"):
            current_keyword = "*Assembly"
            current_part_name = None
            continue

        if line.startswith("*End Assembly"):
            current_keyword = None
            continue

        if line.startswith("*Instance"):
            current_keyword = "*Instance"
            parts_inst_line = line.split(",")
            instance_name = None
            part_ref_name = None
            for p_inst in parts_inst_line:
                if "name=" in p_inst: instance_name = p_inst.split("=")[1].strip()
                if "part=" in p_inst: part_ref_name = p_inst.split("=")[1].strip()

            instance_data = {"name": instance_name, "part": part_ref_name, "translation": [0.0, 0.0, 0.0],
                             "rotation": None}
            inp_data["assembly"]["instances"].append(instance_data)
            reading_instance_transform = True
            instance_transform_lines_read = 0  # Counter for lines belonging to instance transform data
            continue

        if line.startswith("*Boundary"):
            current_keyword = "*Boundary"
            continue

        if line.startswith("*Dload"):
            current_keyword = "*Dload"
            load_data_line = lines[i].strip()
            i += 1
            if "GRAV" in load_data_line.upper():
                grav_params_str = [p.strip() for p in load_data_line.split(",")]
                magnitude_grav = float(grav_params_str[2])
                comp_grav = [float(grav_params_str[3]), float(grav_params_str[4]), float(grav_params_str[5])]
                inp_data["loads"]["gravity"] = {"magnitude": magnitude_grav, "components": comp_grav}
            continue

        if line.startswith("*Step"):
            current_keyword = "*Step"
            continue
        if line.startswith("*End Step"):
            current_keyword = None
            continue

        # 数据行处理
        if current_keyword == "*Node" and current_part_name:
            parts_node_data = [p.strip() for p in line.split(",")]
            node_id_val = int(parts_node_data[0])
            coords_val = [float(c) for c in parts_node_data[1:4]]
            inp_data["parts"][current_part_name]["nodes"][node_id_val] = coords_val

        elif current_keyword == "*Element" and current_part_name:
            parts_elem_data = [p.strip() for p in line.split(",")]
            elem_id_val = int(parts_elem_data[0])
            elem_nodes_val = [int(n_str) for n_str in parts_elem_data[1:] if n_str]
            inp_data["parts"][current_part_name]["elements"][elem_id_val] = {
                "type": current_element_type,
                "nodes": elem_nodes_val,
                "elset": current_element_elset
            }
        elif reading_instance_transform:
            current_instance_data = inp_data["assembly"]["instances"][-1]
            coords_transform = []
            try:
                coords_transform = [float(c.strip()) for c in line.split(",")]
            except ValueError:  # 不是坐标行，变换数据结束
                reading_instance_transform = False
                instance_transform_lines_read = 0
                i -= 1  # 重新处理这一行
                continue

            if instance_transform_lines_read == 0:  # 期望平移数据
                if len(coords_transform) >= 3:
                    current_instance_data["translation"] = coords_transform[:3]
                instance_transform_lines_read += 1
                # 检查下一行是否是旋转数据或新的关键词
                if not (i < len(lines) and not lines[i].strip().startswith("*") and len(
                        lines[i].strip().split(',')) >= 6):  # 没有下一行旋转数据
                    reading_instance_transform = False  # 结束变换读取

            elif instance_transform_lines_read == 1:  # 期望旋转数据
                if len(coords_transform) == 7:  # xA,yA,zA, xB,yB,zB, angle
                    current_instance_data["rotation"] = {
                        "p1": coords_transform[0:3], "p2": coords_transform[3:6], "angle": coords_transform[6]
                    }
                elif len(coords_transform) == 4 and coords_transform == [0.0, 0.0, 0.0, 0.0]:  # 有时Abaqus会生成这样的空旋转行
                    pass

                reading_instance_transform = False  # 无论如何，读完两行（或一行平移后无旋转）就结束
                instance_transform_lines_read = 0
            else:  # 不应该到达这里
                reading_instance_transform = False
                instance_transform_lines_read = 0

        elif current_keyword == "*Boundary":
            parts_bc_data = [p.strip() for p in line.split(",")]
            target_bc = parts_bc_data[0]
            dof_start_bc = int(parts_bc_data[1])
            dof_end_bc = int(parts_bc_data[2]) if len(parts_bc_data) > 2 and parts_bc_data[2] else dof_start_bc
            for dof_val in range(dof_start_bc, dof_end_bc + 1):
                inp_data["boundaries"].append({"target_set_or_node": target_bc, "dof": dof_val})

    # --- 处理和全局编号 ---
    print("处理实例并创建全局模型...")
    global_nodes_map = {}  # (instance_name, local_node_id) -> global_node_id
    global_nodes_coords_bcs = {}  # global_node_id -> {"coords": [], "bcode": [0,0,0]}
    global_elements_data = []  # 包含单元字典的列表 (global_id, type, global_nodes, material_key)

    next_global_node_id = 1
    next_global_element_id = 1

    for inst in inp_data["assembly"]["instances"]:
        inst_name = inst["name"]
        part_name_ref = inst["part"]
        part_data_ref = inp_data["parts"].get(part_name_ref)
        if not part_data_ref:
            print(f"  警告: 未找到实例 {inst_name} 的部件 {part_name_ref}。")
            continue

        print(f"  实例化 {inst_name} (来自 {part_name_ref})")

        # 此实例的全局节点
        for local_node_id_iter, local_coords_iter in part_data_ref["nodes"].items():
            global_id_key_iter = (inst_name, local_node_id_iter)
            if global_id_key_iter not in global_nodes_map:
                global_nodes_map[global_id_key_iter] = next_global_node_id

                # 应用变换
                tx_coords_iter = [local_coords_iter[j] + inst["translation"][j] for j in range(3)]
                if inst["rotation"]:
                    rot_data = inst["rotation"]
                    final_coords_iter = rotate_point(tx_coords_iter, rot_data["angle"], rot_data["p1"], rot_data["p2"])
                else:
                    final_coords_iter = tx_coords_iter

                global_nodes_coords_bcs[next_global_node_id] = {"coords": final_coords_iter, "bcode": [0, 0, 0]}
                next_global_node_id += 1

        # 此实例的全局单元
        for local_elem_id_iter, elem_def_iter in part_data_ref["elements"].items():
            global_node_ids_for_elem_iter = []
            valid_element_iter = True
            for ln_id_iter in elem_def_iter["nodes"]:
                gn_id_key_iter = (inst_name, ln_id_iter)
                if gn_id_key_iter in global_nodes_map:
                    global_node_ids_for_elem_iter.append(global_nodes_map[gn_id_key_iter])
                else:
                    print(f"  警告: 在实例 {inst_name} 的单元 {local_elem_id_iter} 中未找到节点 {ln_id_iter}。")
                    valid_element_iter = False
                    break
            if not valid_element_iter:
                continue

            elem_section_info = None
            elem_material_name_str = None
            elem_abaqus_type_str = elem_def_iter["type"]
            found_section_for_elem_iter = False

            for section_key_iter, section_info_iter in part_data_ref["sections"].items():
                elset_of_section_iter = section_info_iter["elset"]

                if elem_def_iter["elset"] and elem_def_iter["elset"] == elset_of_section_iter:
                    elem_section_info = section_info_iter
                    elem_material_name_str = section_info_iter["material"]
                    found_section_for_elem_iter = True
                    break
                elif elset_of_section_iter in part_data_ref["elsets"] and local_elem_id_iter in part_data_ref["elsets"][
                    elset_of_section_iter]:
                    elem_section_info = section_info_iter
                    elem_material_name_str = section_info_iter["material"]
                    found_section_for_elem_iter = True
                    break

            if not found_section_for_elem_iter:
                if elem_def_iter["elset"] and elem_def_iter["elset"] in part_data_ref["sections"]:
                    elem_section_info = part_data_ref["sections"][elem_def_iter["elset"]]
                    elem_material_name_str = elem_section_info["material"]
                else:
                    print(
                        f"  警告: 无法找到实例 {inst_name} 中单元 {local_elem_id_iter} (类型 {elem_abaqus_type_str}) 的截面/材料。单元集: {elem_def_iter.get('elset')}")
                    continue

            mat_props_tuple_iter = None
            if elem_material_name_str and elem_material_name_str in inp_data["materials_global"]:
                raw_mat_iter = inp_data["materials_global"][elem_material_name_str]
                section_props_for_key_iter = []
                if elem_section_info:
                    if elem_section_info["type"] == "Solid" and elem_abaqus_type_str == "T3D2":
                        section_props_for_key_iter.append(elem_section_info["props"][0])
                    elif elem_section_info["type"] == "Shell":
                        section_props_for_key_iter.append(elem_section_info["thickness"])
                    elif elem_section_info["type"] == "Beam":
                        section_props_for_key_iter.extend(elem_section_info["geom_props"])

                mat_props_tuple_iter = (
                    elem_abaqus_type_str,
                    elem_material_name_str,
                    tuple(raw_mat_iter["props"]),
                    tuple(section_props_for_key_iter)
                )

            global_elements_data.append({
                "id": next_global_element_id,
                "type": elem_abaqus_type_str,
                "nodes": global_node_ids_for_elem_iter,
                "material_key": mat_props_tuple_iter,
                "original_instance_element_id": (inst_name, local_elem_id_iter)
            })
            next_global_element_id += 1

    # 应用边界条件
    print("应用边界条件...")
    for bc_item in inp_data["boundaries"]:
        target_name_bc = bc_item["target_set_or_node"]
        dof_bc = bc_item["dof"]
        nodes_to_constrain_list = []

        if target_name_bc in inp_data["assembly"]["nsets"]:
            for nset_entry_item in inp_data["assembly"]["nsets"][target_name_bc]:
                inst_n_ref = nset_entry_item['instance_ref']  # Instance name
                for local_n_id_ref in nset_entry_item['ids']:
                    gn_id_key_ref = (inst_n_ref, local_n_id_ref)
                    if gn_id_key_ref in global_nodes_map:
                        nodes_to_constrain_list.append(global_nodes_map[gn_id_key_ref])
                    else:
                        print(
                            f"  警告: 在 NSET {target_name_bc} (实例 {inst_n_ref}) 中的节点 {local_n_id_ref} 未找到，无法应用BC。")
        else:
            # 尝试将 target_name_bc 视为直接的全局节点 ID (可能性较小)
            try:
                node_id_direct = int(target_name_bc)
                if node_id_direct in global_nodes_coords_bcs:
                    nodes_to_constrain_list.append(node_id_direct)
                else:
                    print(f"  警告: BC 目标 {target_name_bc} 不是已知的 Set，也不是有效的全局节点 ID。")
            except ValueError:
                print(f"  警告: 无法解析 BC 目标 {target_name_bc}。")

        for gn_id_bc in nodes_to_constrain_list:
            if gn_id_bc in global_nodes_coords_bcs:
                if 1 <= dof_bc <= 3:
                    global_nodes_coords_bcs[gn_id_bc]["bcode"][dof_bc - 1] = 1
            else:
                print(f"  警告: 用于 BC 的全局节点 {gn_id_bc} 在最终映射中未找到。")

    # --- 准备 DAT 输出 ---
    print("准备 DAT 输出...")
    dat_title_str = inp_data["heading"]

    dat_element_groups_dict = {}
    for elem_iter in global_elements_data:
        if elem_iter["material_key"] not in dat_element_groups_dict:
            dat_element_groups_dict[elem_iter["material_key"]] = []
        dat_element_groups_dict[elem_iter["material_key"]].append(elem_iter)

    numeg_val = len(dat_element_groups_dict)
    nlcase_val = 1 if "gravity" in inp_data["loads"] else 0
    if nlcase_val == 0 and any(step_key for step_key in inp_data if step_key.startswith("*Step")):
        nlcase_val = 1

    modex_val = 1

    abaqus_to_dat_type_map_dict = {'T3D2': 1, 'S4R': 6, 'C3D8R': 4, 'B31': 5}

    # find duplicated nodes
    old_to_new_id = {}
    new_global_nodes_coords_bcs = {}
    coords_seen = {}
    new_id = 1
    intersections = {}

    sorted_global_node_ids_list = sorted(global_nodes_coords_bcs.keys())
    for gn_id_out in sorted_global_node_ids_list:
        node_info_out = global_nodes_coords_bcs[gn_id_out]
        coords_out = tuple(node_info_out["coords"])
        # let coordinates with extremely small numbers be considered as zero
        coords_out = tuple(round(coord, 7) for coord in coords_out)  # 保留7位小数
        if coords_out not in coords_seen:
            coords_seen[coords_out] = new_id
            new_global_nodes_coords_bcs[new_id] = node_info_out
            old_to_new_id[gn_id_out] = new_id
            new_id += 1
        else:
            old_bcode = new_global_nodes_coords_bcs[coords_seen[coords_out]]["bcode"]
            new_bcode = node_info_out["bcode"]
            node_info_out["bcode"] = [max(old_bcode[i], new_bcode[i]) for i in range(3)]
            print(f"  更新: 节点 {gn_id_out} 重复，更新为 {coords_seen[coords_out]}")
            old_to_new_id[gn_id_out] = coords_seen[coords_out]
    numnp_val = len(new_global_nodes_coords_bcs)

    # --- 指定Node类型，更新element的Node编号 ---
    node_type = [set() for _ in range(numnp_val)]
    for mat_key_out, group_elements_list in dat_element_groups_dict.items():
        if not group_elements_list: continue
        abaqus_elem_type_str_out = mat_key_out[0]
        elem_type = abaqus_to_dat_type_map_dict[abaqus_elem_type_str_out]
        for elem_data_out in group_elements_list:
            elem_data_out["nodes"] = [old_to_new_id[node] for node in elem_data_out["nodes"]]
            for node in elem_data_out["nodes"]:
                node_type[node-1].add(elem_type)

    print(f"写入 DAT 文件: {dat_filepath}")
    with open(dat_filepath, 'w', encoding='utf-8') as f_out:
        f_out.write(f"{dat_title_str}\n")
        f_out.write(f"{numnp_val} {numeg_val} {nlcase_val} {modex_val}\n")

        sorted_global_node_ids_list = sorted(new_global_nodes_coords_bcs.keys())
        for gn_id_out in sorted_global_node_ids_list:
            node_info_out = new_global_nodes_coords_bcs[gn_id_out]
            coords_out = node_info_out["coords"]
            bcodes_out = node_info_out["bcode"]
            if 5 in node_type[gn_id_out-1] or 6 in node_type[gn_id_out-1]:
                last = 0 if 5 in node_type[gn_id_out-1] else 1
                if 1 in node_type[gn_id_out-1] or 4 in node_type[gn_id_out-1]:
                    f_out.write(
                    f"{gn_id_out} {bcodes_out[0]} {bcodes_out[1]} {bcodes_out[2]} 0 0 {last} {coords_out[0]:.7e} {coords_out[1]:.7e} {coords_out[2]:.7e}\n")
                else:
                    first_two = "0 0" if 5 in node_type[gn_id_out-1] else "1 1"
                    f_out.write(
                    f"{gn_id_out} {first_two} 0 0 0 {last} {coords_out[0]:.7e} {coords_out[1]:.7e} {coords_out[2]:.7e}\n")
            else:
                f_out.write(
                f"{gn_id_out} {bcodes_out[0]} {bcodes_out[1]} {bcodes_out[2]} {coords_out[0]:.7e} {coords_out[1]:.7e} {coords_out[2]:.7e}\n")

        if nlcase_val > 0:
            f_out.write(f"1 0\n")

        for mat_key_out, group_elements_list in dat_element_groups_dict.items():
            if not group_elements_list: continue

            abaqus_elem_type_str_out = mat_key_out[0]
            dat_elem_type_id_out = abaqus_to_dat_type_map_dict.get(abaqus_elem_type_str_out)
            if dat_elem_type_id_out is None:
                print(f"  警告: 未知的 Abaqus 单元类型 {abaqus_elem_type_str_out} 无法转换为 DAT。跳过此组。")
                continue

            num_elems_in_group_out = len(group_elements_list)
            num_mat_for_group_out = 1

            f_out.write(f"{dat_elem_type_id_out} {num_elems_in_group_out} {num_mat_for_group_out}\n")

            e_nu_tuple_out = mat_key_out[2]
            section_props_out = mat_key_out[3]

            # if mat_key_out not in processed_material_keys_to_dat_id_map:
            #     processed_material_keys_to_dat_id_map[mat_key_out] = next_dat_material_id_val
            #     current_dat_mat_id_out = next_dat_material_id_val
            #     next_dat_material_id_val += 1
            # else:
            #     current_dat_mat_id_out = processed_material_keys_to_dat_id_map[mat_key_out]

            mat_line_parts_list = ["1", f"{e_nu_tuple_out[0]:.7e}", f"{e_nu_tuple_out[1]:.3f}"]

            if dat_elem_type_id_out == 1:
                area_val = section_props_out[0] if section_props_out else 0.0
                mat_line_parts_list.append(f"{area_val:.7e}")
            elif dat_elem_type_id_out == 3:
                thickness_val = section_props_out[0] if section_props_out else 1.0
                plane_stress_flag_val = 1
                mat_line_parts_list.extend([f"{thickness_val:.7e}", str(plane_stress_flag_val)])
            elif dat_elem_type_id_out == 5:
                if section_props_out:
                    mat_line_parts_list.extend([f"{p_val:.7e}" for p_val in section_props_out])
                else:
                    mat_line_parts_list.extend(["0.0"] * 6)  # BOX截面有6个参数
            elif dat_elem_type_id_out == 6:
                thickness_val = section_props_out[0] if section_props_out else 1.0
                mat_line_parts_list.append(f"{thickness_val:.7e}")   # 只要厚度即可

            f_out.write(" ".join(mat_line_parts_list) + "\n")

            for local_idx, elem_data_out in enumerate(group_elements_list, start=1):
                elem_nodes_str_out = " ".join(map(str, elem_data_out["nodes"]))
                f_out.write(f"{local_idx} {elem_nodes_str_out} 1\n")

    print(f"完成 DAT 文件生成: {dat_filepath}")

    # 返回解析后的数据和统计信息，以便在主脚本中打印
    return {
        "inp_filepath": inp_filepath,
        "dat_filepath": dat_filepath,
        "dat_title": dat_title_str,
        "numnp": numnp_val,
        "numeg": numeg_val,
        "nlcase": nlcase_val,
        "modex": modex_val,
        "abaqus_to_dat_type_map": abaqus_to_dat_type_map_dict
    }


# --- 主执行块 ---
inp_file_path = 'inp2dat/Bridge-1.inp'  # 你的 INP 文件路径
dat_file_path = 'inp2dat/Bridge-1.dat'  # 输出的 DAT 文件路径

# 调用解析函数
conversion_summary_data = parse_inp_to_dat(inp_file_path, dat_file_path)

# 打印转换摘要
print(f"\n转换摘要:")
print(f"输入 Abaqus INP: {conversion_summary_data['inp_filepath']}")
print(f"输出 DAT: {conversion_summary_data['dat_filepath']}")
print(f"标题: {conversion_summary_data['dat_title']}")
print(f"总全局节点数 (NUMNP): {conversion_summary_data['numnp']}")
print(f"总单元组数 (NUMEG): {conversion_summary_data['numeg']}")
print(f"荷载工况数 (NLCASE): {conversion_summary_data['nlcase']}")
print(f"MODEX: {conversion_summary_data['modex']}")
print("\n使用的单元类型映射 (Abaqus -> DAT 类型 ID):")
for abq_type, dat_id_type in conversion_summary_data['abaqus_to_dat_type_map'].items():
    print(f"  {abq_type} -> {dat_id_type}")
print("\n关于荷载的说明: INP 文件中的重力荷载未转换为 DAT 文件中的集中荷载。")
print("INP *Boundary 卡片定义的边界条件已应用于节点的 bcode。")
print("材料属性已按 (Abaqus 单元类型, 材料名称, E, nu, 特定截面属性) 分组。")