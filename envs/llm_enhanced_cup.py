import sys
import os
from typing import Dict, Any
import datetime

# 添加项目根目录到Python路径
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(parent_directory)
sys.path.append(project_root)

from envs.demo_cup import CupPlacementPlanning
from envs.llm_interface import DeepSeekLLM, LocalLLM
from config.llm_config import DEFAULT_LLM_CONFIG, ROBOT_CONFIG, INTERACTION_CONFIG

class LLMEnhancedCupPlacement(CupPlacementPlanning):
    """集成大语言模型的杯子放置环境"""
    
    def __init__(self, config_path=None, llm_config=None):
        super().__init__(config_path)
        self.llm = None
        self.task_history = []
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
    
    def process_natural_language_command(self, user_input: str, debug=True) -> bool:
        """处理自然语言命令"""
        if self.llm is None:
            print("❌ LLM模块未可用")
            return False
        
        print(f"📝 用户输入: {user_input}")
        
        # 解析命令，传入debug参数
        if hasattr(self.llm, 'parse_robot_command'):
            if isinstance(self.llm, DeepSeekLLM):
                command = self.llm.parse_robot_command(user_input, debug=debug)
            else:
                command = self.llm.parse_robot_command(user_input)
        else:
            command = {"task_type": "unknown", "parameters": {}, "confidence": 0.0, "description": "解析失败"}
        
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
    
    def execute_parsed_command(self, command: Dict[str, Any]) -> bool:
        """执行解析后的命令"""
        task_type = command["task_type"]
        parameters = command["parameters"]
        
        try:
            if task_type == "move_cup_to_coaster":
                return self.plan_cup_to_coaster()
            
            elif task_type == "grasp_cup":
                return self.grasp_cup_only()
            
            elif task_type == "release_cup":
                self.open_gripper(0.06)
                print("夹爪已打开，杯子已释放")
                return True
            
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
                # 清空任务历史
                self.task_history.clear()
                print("🗑️ 已清空任务历史")
                
                # 重置场景
                success = self.reset_and_reinitialize()
                if success:
                    print("🎉 场景重置成功，可以重新开始了!")
                    # 显示重置后的环境状态
                    print(self.get_environment_description())
                return success            
            else:
                print(f"❌ 未知的任务类型: {task_type}")
                return False
                
        except Exception as e:
            print(f"❌ 执行命令时出错: {e}")
            return False
    
    def grasp_cup_only(self) -> bool:
        """单独的抓取杯子功能"""
        cup_pose = self.get_cup_pose()
        if cup_pose is None:
            print("❌ 未找到杯子")
            return False
        
        class SimplePose:
            def __init__(self, pos, quat=[0, 1, 0, 0]):
                self.p = pos
                self.q = quat
        
        try:
            print("📍 移动到杯子上方...")
            above_cup = SimplePose([cup_pose.p[0], cup_pose.p[1] + 0.05, cup_pose.p[2] + 0.15])
            if not self.move_to_pose(above_cup):
                return False
            
            print("🖐️ 打开夹爪...")
            self.open_gripper(0.06)
            
            print("⬇️ 下降抓取杯子...")
            grasp_pose = SimplePose([cup_pose.p[0], cup_pose.p[1] + 0.05, cup_pose.p[2] + 0.12])
            if not self.move_to_pose(grasp_pose):
                return False
            
            print("🤏 抓取杯子...")
            self.close_gripper()
            
            print("⬆️ 提升杯子...")
            lift_pose = SimplePose([cup_pose.p[0], cup_pose.p[1] + 0.05, cup_pose.p[2] + 0.20])
            self.move_to_pose(lift_pose)
            
            return True
            
        except Exception as e:
            print(f"❌ 抓取杯子失败: {e}")
            return False
    
    def get_environment_description(self) -> str:
        """获取环境描述"""
        description = "=== 🤖 当前环境状态 ===\n"
        
        # 场景基本信息
        scene_info = self.get_scene_info()
        description += f"🏠 场景状态: 杯子{'存在' if scene_info['cup_exists'] else '不存在'}, 杯垫{'存在' if scene_info['coaster_exists'] else '不存在'}\n"
        
        # 夹爪状态
        try:
            if hasattr(self, 'active_joints') and len(self.active_joints) >= 9:
                gripper_pos = self.active_joints[-1].get_drive_target()
                gripper_state = "打开" if gripper_pos > 0.02 else "关闭"
                description += f"🤏 夹爪状态: {gripper_state} (位置: {gripper_pos:.3f})\n"
        except:
            description += "🤏 夹爪状态: 未知\n"
        
        # 物体位置
        try:
            if self.cup:
                cup_pose = self.get_cup_pose()
                description += f"☕ 杯子位置: ({cup_pose.p[0]:.3f}, {cup_pose.p[1]:.3f}, {cup_pose.p[2]:.3f})\n"
            
            if self.coaster:
                coaster_pose = self.get_coaster_pose()
                description += f"🎯 杯垫位置: ({coaster_pose.p[0]:.3f}, {coaster_pose.p[1]:.3f}, {coaster_pose.p[2]:.3f})\n"
        except:
            description += "⚠️ 物体位置获取失败\n"
        
        # 任务历史统计
        if self.task_history:
            total_tasks = len(self.task_history)
            successful_tasks = sum(1 for task in self.task_history if task["success"])
            description += f"📊 任务统计: 总计{total_tasks}个任务，成功{successful_tasks}个\n"
        
        return description
    
    def interactive_mode(self):
        """交互模式"""
        print(INTERACTION_CONFIG["welcome_message"])
        print("输入 'help' 查看帮助，输入 'quit' 退出\n")
        print(self.get_environment_description())
        
        while True:
            try:
                user_input = input(f"\n{INTERACTION_CONFIG['prompt_symbol']}").strip()
                
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
                        print(f"{INTERACTION_CONFIG['success_symbol']} 任务执行成功!")
                    else:
                        print(f"{INTERACTION_CONFIG['error_symbol']} 任务执行失败!")
                
            except KeyboardInterrupt:
                print("\n👋 用户中断，退出交互模式")
                break
            except Exception as e:
                print(f"❌ 处理输入时出错: {e}")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
=== 📚 使用帮助 ===

🎯 自然语言命令示例:
  • "将杯子移动到杯垫上" / "把杯子放到杯垫上"
  • "抓取杯子" / "拿起杯子" / "夹住杯子" / "将杯子提起来"
  • "释放杯子" / "放下杯子" / "松开杯子"
  • "打开夹爪" / "张开夹爪"
  • "关闭夹爪" / "闭合夹爪"
  • "显示状态" / "当前状态" / "环境情况"
  • "回到初始位置" / "回家" / "复位"
  • "再来一次" / "重置" / "重新开始" / "重建场景"  ⭐ 新功能

🔧 系统命令:
  • help/h/帮助: 显示此帮助
  • status/s/状态: 显示当前环境状态
  • history/历史: 显示任务执行历史
  • clear/清屏: 清除屏幕
  • quit/q/exit/退出: 退出程序

💡 提示: 
  - 支持中文和英文指令
  - 可以用自然语言描述任务
  - 系统会自动解析您的意图
  - 使用"再来一次"可以重置整个场景 🔄
"""
        print(help_text)
    
    def show_help_examples(self):
        """显示简短的示例"""
        print("\n💡 命令示例:")
        print("  • 将杯子放到杯垫上")
        print("  • 抓取杯子")
        print("  • 打开夹爪")
        print("  • 显示状态")
    
    def show_task_history(self):
        """显示任务执行历史"""
        if not self.task_history:
            print("📝 暂无任务执行历史")
            return
        
        print("=== 📋 任务执行历史 ===")
        max_history = INTERACTION_CONFIG.get("max_history", 10)
        recent_tasks = self.task_history[-max_history:]
        
        for i, task in enumerate(recent_tasks, 1):
            status = INTERACTION_CONFIG["success_symbol"] if task["success"] else INTERACTION_CONFIG["error_symbol"]
            time_str = task["timestamp"].split("T")[1][:8]
            confidence = task["parsed_command"].get("confidence", 0)
            print(f"{i:2d}. {status} [{time_str}] {task['user_input']} -> {task['parsed_command']['description']} (信心度: {confidence:.2f})")