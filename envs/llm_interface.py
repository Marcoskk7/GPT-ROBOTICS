import json
import re
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import time

class LLMInterface(ABC):
    """大语言模型接口基类"""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应"""
        pass
    
    @abstractmethod
    def parse_robot_command(self, user_input: str) -> Dict[str, Any]:
        """解析用户输入为机器人命令"""
        pass

class DeepSeekLLM(LLMInterface):
    """DeepSeek API实现（兼容ChatGPT格式）"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 测试API连接
        self._test_connection()
    
    def _test_connection(self):
        """测试API连接"""
        try:
            test_response = self.generate_response("你好", max_tokens=10)
            if test_response and "抱歉" not in test_response:
                print("✅ DeepSeek API连接成功")
            else:
                print("⚠️ DeepSeek API连接可能有问题")
        except Exception as e:
            print(f"⚠️ DeepSeek API测试失败: {e}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """调用DeepSeek API生成响应"""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的机器人控制助手。请严格按照要求返回JSON格式，不要包含任何其他文字或解释。"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": kwargs.get("max_tokens", 200),
                "temperature": kwargs.get("temperature", 0.1),  # 降低温度提高一致性
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"API调用失败: {response.status_code} - {response.text}")
                return "抱歉，我无法处理这个请求。"
                
        except requests.exceptions.Timeout:
            print("API调用超时")
            return "请求超时，请稍后重试。"
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            return "抱歉，我无法处理这个请求。"
    
    def parse_robot_command(self, user_input: str, debug=True) -> Dict[str, Any]:
        """解析用户自然语言为机器人命令"""
        
        # 简化的系统提示，更明确的JSON要求
        system_prompt = f"""请将用户指令转换为JSON格式。只返回JSON，不要任何其他文字。

任务类型：
- move_cup_to_coaster: 将杯子移到杯垫
- grasp_cup: 抓取杯子
- release_cup: 释放杯子  
- open_gripper: 打开夹爪
- close_gripper: 关闭夹爪
- show_status: 显示状态
- go_home: 回到初始位置
- reset_scene: 重置场景，再来一次

用户输入：{user_input}

返回格式（只返回这个JSON）：
{{"task_type": "任务类型", "parameters": {{}}, "confidence": 0.9, "description": "描述"}}"""
        
        try:
            response = self.generate_response(system_prompt, temperature=0.1, max_tokens=150)
            
            if debug:
                print(f"🔍 API原始响应: {response}")
            
            # 多种JSON提取方法
            json_result = self._extract_json_from_response(response, debug)
            
            if json_result:
                # 验证必要字段
                required_fields = ["task_type", "parameters", "confidence", "description"]
                if all(key in json_result for key in required_fields):
                    if debug:
                        print(f"✅ JSON解析成功: {json_result}")
                    return json_result
            
            if debug:
                print("❌ JSON解析失败，使用关键词匹配")
            return self._fallback_parse(user_input)
            
        except Exception as e:
            if debug:
                print(f"❌ 命令解析失败: {e}")
            return self._fallback_parse(user_input)
    
    def _extract_json_from_response(self, response: str, debug=False) -> Dict[str, Any]:
        """从响应中提取JSON，使用多种方法"""
        
        # 方法1: 直接解析整个响应
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # 方法2: 查找第一个完整的JSON对象
        try:
            # 改进的正则表达式，支持嵌套
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
        
        # 方法3: 逐行查找JSON
        try:
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        
        # 方法4: 手动构建JSON（最后的备用方案）
        if debug:
            print("尝试手动解析响应内容...")
        
        return None
    
    def _fallback_parse(self, user_input: str) -> Dict[str, Any]:
        """改进的备用关键词匹配解析"""
        user_input_lower = user_input.lower().strip()
        
        # 更精确的关键词匹配，优先级从高到低
   
        # 0. 重置场景（新增）
        if any(word in user_input_lower for word in ["再来一次", "重置", "reset", "重新开始", "restart", "重来", "again", "重建", "初始化"]):
            return {
                "task_type": "reset_scene",
                "parameters": {},
                "confidence": 0.9,
                "description": "重置场景，再来一次"
            }        
        
        # 1. 杯子相关操作
        if any(word in user_input_lower for word in ["杯子", "cup"]) and any(word in user_input_lower for word in ["放到", "移动", "杯垫", "coaster", "放置"]):
            return {
                "task_type": "move_cup_to_coaster",
                "parameters": {},
                "confidence": 0.8,
                "description": "将杯子移动到杯垫上"
            }
        
        # 2. 抓取操作（优先级高于一般移动）
        elif any(word in user_input_lower for word in ["抓取", "grasp", "拿起", "抓住", "夹取", "取"]) and not any(word in user_input_lower for word in ["放", "松"]):
            return {
                "task_type": "grasp_cup",
                "parameters": {},
                "confidence": 0.8,
                "description": "抓取杯子"
            }
        
        # 3. 释放操作
        elif any(word in user_input_lower for word in ["释放", "放下", "release", "松开", "放手", "松"]):
            return {
                "task_type": "release_cup",
                "parameters": {},
                "confidence": 0.8,
                "description": "释放杯子"
            }
        
        # 4. 夹爪操作 - 关闭优先判断
        elif any(word in user_input_lower for word in ["关闭", "close", "闭合", "夹紧", "合上", "闭", "夹"]) and any(word in user_input_lower for word in ["夹爪", "gripper", "手"]):
            return {
                "task_type": "close_gripper",
                "parameters": {},
                "confidence": 0.8,
                "description": "关闭夹爪"
            }
        
        # 5. 打开夹爪
        elif any(word in user_input_lower for word in ["打开", "open", "张开", "张", "开"]) and any(word in user_input_lower for word in ["夹爪", "gripper", "手"]):
            return {
                "task_type": "open_gripper",
                "parameters": {"opening": 0.06},
                "confidence": 0.8,
                "description": "打开夹爪"
            }
        
        # 6. 状态查询
        elif any(word in user_input_lower for word in ["状态", "status", "情况", "现在", "当前", "怎样", "如何"]):
            return {
                "task_type": "show_status",
                "parameters": {},
                "confidence": 0.9,
                "description": "显示当前环境状态"
            }
        
        # 7. 回家操作
        elif any(word in user_input_lower for word in ["回家", "home", "初始", "复位", "归位", "回"]):
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

class LocalLLM(LLMInterface):
    """本地LLM实现（简单关键词匹配）"""
    
    def __init__(self, model_name: str = "simple"):
        self.model_name = model_name
        print("📱 使用本地关键词匹配模式")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        return "我是本地助手，可以通过关键词帮您控制机器人。"
    
    def parse_robot_command(self, user_input: str) -> Dict[str, Any]:
        """使用DeepSeekLLM的备用解析方法"""
        # 直接使用改进的关键词匹配
        dummy_llm = DeepSeekLLM("dummy", "dummy", "dummy")
        return dummy_llm._fallback_parse(user_input)