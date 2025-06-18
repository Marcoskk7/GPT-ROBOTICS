import sys
import os
from typing import Dict, Any
import datetime

# 添加项目根目录到Python路径
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(parent_directory)
sys.path.append(project_root)

from envs.demo_stack import CupPlacementPlanning as StackPlanning
from envs.llm_interface import DeepSeekLLM, LocalLLM
from config.llm_config import DEFAULT_LLM_CONFIG, ROBOT_CONFIG, INTERACTION_CONFIG

class LLMEnhancedStackEnvironment(StackPlanning):
    """集成大语言模型的方块叠加环境"""
    
    def __init__(self, config_path=None, llm_config=None):
        super().__init__(config_path)
        self.llm = None
        self.task_history = []
        self.block_names = ['block1', 'block2', 'block3']
        self.block_colors = {'block1': '红色', 'block2': '绿色', 'block3': '蓝色'}
        self.setup_llm(llm_config)
    
    def setup_llm(self, llm_config=None):
        """设置LLM模块"""
        if llm_config is None:
            llm_config = DEFAULT_LLM_CONFIG
        
        try:
            if llm_config["type"] == "deepseek":
                api_key = llm_config.get("api_key")
                if not api_key or api_key == "your-deepseek-api-key-here":
                    print("⚠️ 未设置DeepSeek API密钥，使用本地模式")
                    self.llm = LocalLLM()
                else:
                    model = llm_config.get("model", "deepseek-chat")
                    base_url = llm_config.get("base_url", "https://api.deepseek.com")
                    self.llm = DeepSeekLLM(api_key, model, base_url)
            else:
                self.llm = LocalLLM()
                
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}，使用本地模式")
            self.llm = LocalLLM()

    # 直接继承demo_stack.py中的核心方法（完全不修改）
    def get_block_pose(self, block_name):
        """获取指定方块的当前位置"""
        block = getattr(self, block_name, None)
        if block is None:
            return None
        
        block_pose = block.get_pose()
        
        class SimplePose:
            def __init__(self, position, quaternion):
                self.p = position
                self.q = quaternion
        
        return SimplePose(block_pose.p.tolist(), block_pose.q.tolist())

    def get_block1_pose(self):
        """获取方块1的位置"""
        return self.get_block_pose('block1')

    def get_block2_pose(self):
        """获取方块2的位置"""
        return self.get_block_pose('block2')

    def get_block3_pose(self):
        """获取方块3的位置"""
        return self.get_block_pose('block3')

    def plan_block_stacking(self):
        """
        规划将三个方块进行叠加的完整动作序列
        策略：将block2和block3依次叠加到block1上
        （直接从demo_stack.py完整移植）
        """
        print("开始规划方块叠加任务...")
        
        # 获取所有方块的位置
        block1_pose = self.get_block1_pose()
        block2_pose = self.get_block2_pose()
        block3_pose = self.get_block3_pose()
        
        if not all([block1_pose, block2_pose, block3_pose]):
            print("无法获取所有方块的位置")
            return False
        
        print(f"方块1位置: {block1_pose.p}")
        print(f"方块2位置: {block2_pose.p}")
        print(f"方块3位置: {block3_pose.p}")
        
        # 创建简单的目标位置对象
        class SimplePose:
            def __init__(self, position, quaternion=[0, 1, 0, 0]):
                self.p = position
                self.q = quaternion
        
        try:
            # 方块尺寸信息
            block_half_size = 0.025  # 方块半尺寸
            block_height = block_half_size * 2  # 方块完整高度
            
            # 记录初始手腕角度 - 修复
            initial_qpos = []
            for joint in self.active_joints[:7]:
                drive_target = joint.get_drive_target()
                if hasattr(drive_target, 'item'):
                    initial_qpos.append(drive_target.item())
                else:
                    initial_qpos.append(float(drive_target))
            
            wrist_rotation_angle = initial_qpos[6] if len(initial_qpos) > 6 else 0.79
            
            print(f"锁定手腕旋转角度为: {wrist_rotation_angle:.3f}")
            # === 第一阶段：将block2叠加到block1上 ===
            print("\n=== 阶段1：将方块2叠加到方块1上 ===")

            # 1. 移动到block2上方
            print("步骤1: 移动到方块2上方...")
            above_block2_pose = SimplePose([
                block2_pose.p[0],
                block2_pose.p[1], 
                block2_pose.p[2] + 0.20
            ])
            if not self.move_to_pose(above_block2_pose, wrist_rotation_angle):
                print("移动到方块2上方失败")
                return False        
                
            # 2. 打开夹爪
            print("步骤2: 打开夹爪...")
            self.open_gripper(0.06)
            
            # 3. 下降抓取block2
            print("步骤3: 下降抓取方块2...")
            grasp_block2_pose = SimplePose([
                block2_pose.p[0],
                block2_pose.p[1], 
                block2_pose.p[2] + 0.10  # 在方块中心稍上方
            ])
            if not self.move_to_pose(grasp_block2_pose, wrist_rotation_angle):
                print("下降到方块2失败")
                return False
            
            # 4. 夹取block2
            print("步骤4: 夹取方块2...")
            self.close_gripper()
            
            # 5. 提升block2
            print("步骤5: 提升方块2...")
            lift_block2_pose = SimplePose([
                block2_pose.p[0],
                block2_pose.p[1], 
                block2_pose.p[2] + 0.25
            ])
            if not self.move_to_pose(lift_block2_pose, wrist_rotation_angle):
                print("提升方块2失败")
                return False
            
            # 6. 移动到block1上方
            print("步骤6: 移动到方块1上方...")
            above_block1_pose = SimplePose([
                block1_pose.p[0],
                block1_pose.p[1], 
                block1_pose.p[2] + 0.2
            ])
            if not self.move_to_pose(above_block1_pose, wrist_rotation_angle):
                print("移动到方块1上方失败")
                return False
            
            # 7. 将block2放置到block1上
            print("步骤7: 将方块2放置到方块1上...")
            place_block2_pose = SimplePose([
                block1_pose.p[0],
                block1_pose.p[1], 
                block1_pose.p[2] + 0.15 # block1顶部 + block2半高度
            ])
            if not self.move_to_pose(place_block2_pose, wrist_rotation_angle):
                print("放置方块2失败")
                return False
            
            # 8. 释放block2
            print("步骤8: 释放方块2...")
            self.open_gripper(0.06)
            
            # 9. 撤回
            print("步骤9: 撤回...")
            retreat_pose = SimplePose([
                block1_pose.p[0],
                block1_pose.p[1], 
                block1_pose.p[2] + 0.25
            ])
            if not self.move_to_pose(retreat_pose, wrist_rotation_angle):
                print("撤回失败")
                return False
            
            # === 第二阶段：将block3叠加到block2上 ===
            print("\n=== 阶段2：将方块3叠加到方块2上 ===")
            
            # 10. 移动到block3上方
            print("步骤10: 移动到方块3上方...")
            above_block3_pose = SimplePose([
                block3_pose.p[0],
                block3_pose.p[1], 
                block3_pose.p[2] + 0.15
            ])
            if not self.move_to_pose(above_block3_pose, wrist_rotation_angle):
                print("移动到方块3上方失败")
                return False
            
            # 11. 下降抓取block3
            print("步骤11: 下降抓取方块3...")
            grasp_block3_pose = SimplePose([
                block3_pose.p[0],
                block3_pose.p[1], 
                block3_pose.p[2] + 0.1
            ])
            if not self.move_to_pose(grasp_block3_pose, wrist_rotation_angle):
                print("下降到方块3失败")
                return False
            
            # 12. 夹取block3
            print("步骤12: 夹取方块3...")
            self.close_gripper()
            
            # 13. 提升block3
            print("步骤13: 提升方块3...")
            lift_block3_pose = SimplePose([
                block3_pose.p[0],
                block3_pose.p[1], 
                block3_pose.p[2] + 0.3
            ])
            if not self.move_to_pose(lift_block3_pose, wrist_rotation_angle):
                print("提升方块3失败")
                return False
            
            # 14. 移动到block2(现在在block1上)上方
            print("步骤14: 移动到叠加后的方块2上方...")
            above_stacked_pose = SimplePose([
                block1_pose.p[0],
                block1_pose.p[1], 
                block1_pose.p[2] + 0.30  # 两个方块的高度 + 安全距离
            ])
            if not self.move_to_pose(above_stacked_pose, wrist_rotation_angle):
                print("移动到叠加方块上方失败")
                return False
            
            # 15. 将block3放置到block2上
            print("步骤15: 将方块3放置到方块2上...")
            place_block3_pose = SimplePose([
                block1_pose.p[0],
                block1_pose.p[1], 
                block1_pose.p[2] + 0.25  # 三层方块的高度
            ])
            if not self.move_to_pose(place_block3_pose, wrist_rotation_angle):
                print("放置方块3失败")
                return False
            
            # 16. 释放block3
            print("步骤16: 释放方块3...")
            self.open_gripper(0.06)
            
            print("方块叠加任务完成!")
            return True
            
        except Exception as e:
            print(f"执行方块叠加时出错: {e}")
            return False

    # 继承demo_stack.py的move_to_pose和follow_path方法
    def move_to_pose(self, target_pose, use_point_cloud=False):
        """
        移动到目标位置
        （直接从demo_stack.py移植，不修改）
        """
        if self.planner is None:
            print("规划器未可用，使用简单插值移动")
            return self._simple_move_to_pose(target_pose)
            
        try:
            from mplib import Pose
            
            # 将SimplePose转换为mplib.Pose
            if hasattr(target_pose, 'p') and hasattr(target_pose, 'q'):
                mplib_pose = Pose(target_pose.p, target_pose.q)
            else:
                mplib_pose = target_pose
            
            # 获取当前关节角度
            current_qpos = self.robot.get_qpos()
            
            # 使用正确的API进行路径规划
            result = self.planner.plan_pose(
                mplib_pose, 
                current_qpos, 
                time_step=1.0/250.0
            )
            
            if result['status'] != "Success":
                print(f"路径规划失败: {result['status']}")
                return self._simple_move_to_pose(target_pose)
                
            # 执行路径 - 使用示例代码中的follow_path方法
            self.follow_path(result)
            return True
            
        except Exception as e:
            print(f"路径规划执行失败: {e}")
            return self._simple_move_to_pose(target_pose)
        
    def follow_path(self, result):
        """Helper function to follow a path generated by the planner"""
        n_step = result["position"].shape[0]
        
        for i in range(n_step):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            
            # 设置关节位置和速度
            for j in range(len(self.planner.move_group_joint_indices)):
                self.active_joints[j].set_drive_target(result["position"][i][j])
                self.active_joints[j].set_drive_velocity_target(result["velocity"][i][j])
            
            # 仿真步骤
            self.scene.step()
            
            # 每4个仿真步骤渲染一次
            if i % 4 == 0:
                self.scene.update_render()
                if hasattr(self, 'viewer') and self.viewer is not None:
                    self.viewer.render()

    def process_natural_language_command(self, user_input: str, debug=True) -> bool:
        """处理自然语言命令"""
        if self.llm is None:
            print("❌ LLM模块未可用")
            return False
        
        print(f"📝 用户输入: {user_input}")
        
        # 解析命令
        command = self.parse_stack_command(user_input, debug=debug)
        
        print(f"🔍 解析结果: {command['description']} (信心度: {command['confidence']:.2f})")
        
        if command["confidence"] < 0.3:
            print("⚠️ 命令解析信心度较低，请重新输入更清晰的指令")
            self.show_help_examples()
            return False
        
        # 执行命令
        print("🤖 正在执行任务...")
        success = self.execute_parsed_command(command)
        
        # 记录历史
        self.task_history.append({
            "user_input": user_input,
            "parsed_command": command,
            "success": success,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        return success
    
    def parse_stack_command(self, user_input: str, debug=True) -> Dict[str, Any]:
        """解析方块叠加相关的自然语言命令（只保留stack_all_blocks功能）"""
        
        # 构建针对方块叠加任务的专用提示
        system_prompt = f"""请将用户指令转换为JSON格式。只返回JSON，不要任何其他文字。

当前环境包含三个方块：
- block1: 红色方块
- block2: 绿色方块  
- block3: 蓝色方块

任务类型：
- stack_all_blocks: 完整叠加所有方块 (默认顺序：block2和block3叠到block1上)
- open_gripper: 打开夹爪
- close_gripper: 关闭夹爪
- show_status: 显示环境状态
- go_home: 回到初始位置
- reset_scene: 重置场景

用户输入：{user_input}

返回格式（只返回这个JSON）：
{{"task_type": "任务类型", "parameters": {{}}, "confidence": 0.9, "description": "描述"}}"""
        
        try:
            if hasattr(self.llm, 'generate_response'):
                response = self.llm.generate_response(system_prompt, temperature=0.1, max_tokens=150)
                
                if debug:
                    print(f"🔍 API原始响应: {response}")
                
                # 提取JSON
                json_result = self._extract_json_from_response(response, debug)
                
                if json_result:
                    required_fields = ["task_type", "parameters", "confidence", "description"]
                    if all(key in json_result for key in required_fields):
                        if debug:
                            print(f"✅ JSON解析成功: {json_result}")
                        return json_result
            
            if debug:
                print("❌ JSON解析失败，使用关键词匹配")
            return self._fallback_parse_stack(user_input)
            
        except Exception as e:
            if debug:
                print(f"❌ 命令解析失败: {e}")
            return self._fallback_parse_stack(user_input)
    
    def _extract_json_from_response(self, response: str, debug=False) -> Dict[str, Any]:
        """从响应中提取JSON"""
        import json
        import re
        
        # 方法1: 直接解析
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # 方法2: 正则匹配
        try:
            json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    result = json.loads(match)
                    if debug:
                        print(f"🔍 找到JSON: {match}")
                    return result
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            if debug:
                print(f"正则匹配失败: {e}")
        
        return None
    
    def _fallback_parse_stack(self, user_input: str) -> Dict[str, Any]:
        """方块叠加专用的关键词匹配解析（只保留stack_all_blocks）"""
        user_input_lower = user_input.lower().strip()
        
        # 0. 重置场景
        if any(word in user_input_lower for word in ["再来一次", "重置", "reset", "重新开始", "restart", "重来", "again"]):
            return {
                "task_type": "reset_scene",
                "parameters": {},
                "confidence": 0.9,
                "description": "重置场景，再来一次"
            }
        
        # 1. 完整叠加任务（只保留这一个功能）
        if any(word in user_input_lower for word in ["叠加", "堆叠", "stack", "叠起来", "堆起来", "叠放"]):
            return {
                "task_type": "stack_all_blocks",
                "parameters": {},
                "confidence": 0.9,
                "description": "按默认顺序叠加所有方块"
            }
        
        # 2. 夹爪操作
        elif any(word in user_input_lower for word in ["打开", "open"]) and any(word in user_input_lower for word in ["夹爪", "gripper"]):
            return {
                "task_type": "open_gripper",
                "parameters": {"opening": 0.06},
                "confidence": 0.8,
                "description": "打开夹爪"
            }
        
        elif any(word in user_input_lower for word in ["关闭", "close"]) and any(word in user_input_lower for word in ["夹爪", "gripper"]):
            return {
                "task_type": "close_gripper",
                "parameters": {},
                "confidence": 0.8,
                "description": "关闭夹爪"
            }
        
        # 3. 状态查询
        elif any(word in user_input_lower for word in ["状态", "status", "情况", "现在", "当前"]):
            return {
                "task_type": "show_status",
                "parameters": {},
                "confidence": 0.9,
                "description": "显示当前环境状态"
            }
        
        # 4. 回家
        elif any(word in user_input_lower for word in ["回家", "home", "初始", "复位"]):
            return {
                "task_type": "go_home",
                "parameters": {},
                "confidence": 0.8,
                "description": "机器人回到初始位置"
            }
        
        else:
            return {
                "task_type": "unknown",
                "parameters": {},
                "confidence": 0.0,
                "description": f"无法识别的命令: {user_input}"
            }
    
    def execute_parsed_command(self, command: Dict[str, Any]) -> bool:
        """执行解析后的命令（只保留stack_all_blocks功能）"""
        task_type = command["task_type"]
        parameters = command["parameters"]
        
        try:
            if task_type == "stack_all_blocks":
                return self.plan_block_stacking()
            
            elif task_type == "open_gripper":
                opening = parameters.get("opening", 0.06)
                self.open_gripper(opening)
                print(f"夹爪已打开到 {opening:.3f}")
                return True
            
            elif task_type == "close_gripper":
                self.close_gripper()
                print("夹爪已关闭")
                return True
            
            elif task_type == "show_status":
                print(self.get_environment_description())
                return True
            
            elif task_type == "go_home":
                self.move_robot_to_home()
                print("机器人已回到初始位置")
                return True
            
            elif task_type == "reset_scene":
                success = self.reset_and_reinitialize()
                if success:
                    print("🎉 场景重置成功，可以重新开始了!")
                    print(self.get_environment_description())
                return success
            
            else:
                print(f"❌ 未知的任务类型: {task_type}")
                return False
                
        except Exception as e:
            print(f"❌ 执行命令时出错: {e}")
            return False
    
    def reset_and_reinitialize(self):
        """重置并重新初始化整个环境"""
        print("🔄 完全重置方块叠加环境...")
        
        try:
            # 清理现有方块
            for block_name in self.block_names:
                block = getattr(self, block_name, None)
                if block:
                    self.scene.remove_actor(block)
                    setattr(self, block_name, None)
            
            # 重置机器人
            self.move_robot_to_home()
            
            # 重新加载方块
            self.load_objects()
            
            # 清空任务历史
            self.task_history.clear()
            
            # 等待物理仿真稳定
            for _ in range(100):
                self.scene.step()
            
            print("✅ 方块叠加环境重置完成!")
            return True
            
        except Exception as e:
            print(f"❌ 环境重置失败: {e}")
            return False
    
    def get_environment_description(self) -> str:
        """获取环境描述"""
        description = "=== 🤖 方块叠加环境状态 ===\n"
        
        # 基本信息
        description += f"🧱 场景中有3个方块: {', '.join([f'{color}({name})' for name, color in self.block_colors.items()])}\n"
        
        # 夹爪状态
        try:
            if hasattr(self, 'active_joints') and len(self.active_joints) >= 9:
                gripper_pos = self.active_joints[-1].get_drive_target()
                gripper_state = "打开" if gripper_pos > 0.02 else "关闭"
                description += f"🤏 夹爪状态: {gripper_state} (位置: {gripper_pos:.3f})\n"
        except:
            description += "🤏 夹爪状态: 未知\n"
        
        # 方块位置
        try:
            for block_name in self.block_names:
                block_pose = self.get_block_pose(block_name)
                if block_pose:
                    color = self.block_colors[block_name]
                    description += f"🟥 {color}方块({block_name}): ({block_pose.p[0]:.3f}, {block_pose.p[1]:.3f}, {block_pose.p[2]:.3f})\n"
        except Exception as e:
            description += f"⚠️ 方块位置获取失败: {e}\n"
        
        # 任务历史统计
        if self.task_history:
            total_tasks = len(self.task_history)
            successful_tasks = sum(1 for task in self.task_history if task["success"])
            description += f"📊 任务统计: 总计{total_tasks}个任务，成功{successful_tasks}个\n"
        
        return description
    
    def interactive_mode(self):
        """交互模式"""
        welcome_msg = """
🤖 欢迎使用LLM增强的方块叠加系统！

📦 当前环境包含：
  • 红色方块 (block1) 🟥
  • 绿色方块 (block2) 🟩  
  • 蓝色方块 (block3) 🟦

💬 您可以用自然语言控制机器人进行方块叠加操作
        """
        print(welcome_msg)
        print("输入 'help' 查看帮助，输入 'quit' 退出\n")
        print(self.get_environment_description())
        
        while True:
            try:
                user_input = input(f"\n🎯 请输入指令> ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 退出交互模式")
                    break
                
                elif user_input.lower() in ['help', '帮助', 'h']:
                    self.show_help()
                
                elif user_input.lower() in ['status', '状态', 's']:
                    print(self.get_environment_description())
                
                elif user_input.lower() in ['history', '历史']:
                    self.show_task_history()
                
                elif user_input.lower() in ['clear', '清屏']:
                    os.system('clear' if os.name == 'posix' else 'cls')
                
                elif user_input.strip() == '':
                    continue
                
                else:
                    success = self.process_natural_language_command(user_input)
                    if success:
                        print("✅ 任务执行成功!")
                    else:
                        print("❌ 任务执行失败!")
                
            except KeyboardInterrupt:
                print("\n👋 用户中断，退出交互模式")
                break
            except Exception as e:
                print(f"❌ 处理输入时出错: {e}")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
=== 📚 方块叠加系统使用帮助 ===

🎯 自然语言命令示例:
  叠加操作:
  • "叠加所有方块" / "堆叠方块" / "将方块叠起来"
  
  基本操作:
  • "打开夹爪" / "张开夹爪"
  • "关闭夹爪" / "闭合夹爪"
  • "显示状态" / "当前状态"
  • "回到初始位置" / "回家"
  • "再来一次" / "重置" / "重新开始" 🔄

🎨 方块颜色对应：
  • 🟥 红色方块 = block1
  • 🟩 绿色方块 = block2
  • 🟦 蓝色方块 = block3

🔧 系统命令:
  • help/h/帮助: 显示此帮助
  • status/s/状态: 显示环境状态
  • history/历史: 显示任务历史
  • clear/清屏: 清除屏幕
  • quit/q/exit/退出: 退出程序

💡 提示: 
  - 支持中文和英文指令
  - 系统会按block2→block1、block3→block2的顺序自动叠加
"""
        print(help_text)
    
    def show_help_examples(self):
        """显示简短示例"""
        print("\n💡 命令示例:")
        print("  • 叠加所有方块")
        print("  • 显示状态")
    
    def show_task_history(self):
        """显示任务执行历史"""
        if not self.task_history:
            print("📝 暂无任务执行历史")
            return
        
        print("=== 📋 任务执行历史 ===")
        max_history = 10
        recent_tasks = self.task_history[-max_history:]
        
        for i, task in enumerate(recent_tasks, 1):
            status = "✅" if task["success"] else "❌"
            time_str = task["timestamp"].split("T")[1][:8]
            confidence = task["parsed_command"].get("confidence", 0)
            print(f"{i:2d}. {status} [{time_str}] {task['user_input']} -> {task['parsed_command']['description']} (信心度: {confidence:.2f})")