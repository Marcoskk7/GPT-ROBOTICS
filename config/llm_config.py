"""
LLM配置文件
"""

# DeepSeek API配置
DEEPSEEK_CONFIG = {
    "type": "deepseek",
    "api_key": "sk-76641a19d7174b26ad821c8dbd74b822",  # 您的API密钥
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com",
    "max_tokens": 1000,
    "temperature": 0.7
}

# 本地模型配置（备用）
LOCAL_CONFIG = {
    "type": "local",
    "model": "simple"
}

# 默认使用DeepSeek
DEFAULT_LLM_CONFIG = DEEPSEEK_CONFIG

# 机器人配置
ROBOT_CONFIG = {
    'urdf_path': './assets/embodiments/panda/panda.urdf',
    'joint_stiffness': 1000,
    'joint_damping': 200,
    'wrist_joint_stiffness': 1500,
    'wrist_joint_damping': 500,
}

# 交互模式配置
INTERACTION_CONFIG = {
    "welcome_message": "🚀 欢迎使用机器人语言交互系统！",
    "prompt_symbol": "🤖 请输入指令: ",
    "success_symbol": "✅",
    "error_symbol": "❌",
    "max_history": 20,
}