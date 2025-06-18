"""
机器人语言交互系统主启动脚本
"""
import sys
import os

# 添加项目根目录到Python路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
sys.path.append(project_root)

from envs.llm_enhanced_cup import LLMEnhancedCupPlacement
from config.llm_config import ROBOT_CONFIG

def main():
    """主函数"""
    print("🚀 启动机器人语言交互系统...")
    
    # 创建LLM增强环境
    env = LLMEnhancedCupPlacement()
    
    try:
        # 创建环境
        print("🔧 正在创建机器人环境...")
        env.create_environment_with_planning(ROBOT_CONFIG)
        print("✅ 环境创建完成!")
        
        # 进入交互模式
        env.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n👋 用户中断程序")
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass
        print("🔚 程序结束")

if __name__ == "__main__":
    main()