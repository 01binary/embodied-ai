#!/usr/bin/env python3

import ast
import json
import os
import re
import sys
import urllib.error
import urllib.request

import rospy
import rospkg
from embodied_ai.msg import ActuatorCommand
from embodied_ai.msg import ActuatorState


class LlmChatNode:
    def __init__(self):
        self.api_base = rospy.get_param("~api_base", "http://127.0.0.1:1234/v1")
        self.model = rospy.get_param("~model", "local-model")
        self.temperature = float(rospy.get_param("~temperature", 0.2))
        self.max_tokens = int(rospy.get_param("~max_tokens", 512))
        self.timeout_sec = float(rospy.get_param("~timeout_sec", 300.0))
        self.api_key = rospy.get_param("~api_key", os.environ.get("OPENAI_API_KEY", ""))
        self.system_prompt_file = rospy.get_param("~system_prompt_file", "system-prompt.md")
        self.few_shot_file = rospy.get_param("~few_shot_file", "few-shot-prompts.json")
        self.cmd_topic = rospy.get_param("~cmd_topic", "actuator/cmd")
        self.state_topic = rospy.get_param("~state_topic", "actuator/state")

        package_root = rospkg.RosPack().get_path("embodied_ai")
        self.system_prompt_path = self._resolve_path(package_root, self.system_prompt_file)
        self.few_shot_path = self._resolve_path(package_root, self.few_shot_file)

        self.cmd_pub = rospy.Publisher(self.cmd_topic, ActuatorCommand, queue_size=10)
        self.state_sub = rospy.Subscriber(self.state_topic, ActuatorState, self._on_state)
        self.latest_state = None

        self.current_cmd = ActuatorCommand()
        self.current_cmd.led = False
        self.current_cmd.solenoid = False
        self.current_cmd.servo_enable = False
        self.current_cmd.servo_cmd = 0.0

        self.seed_messages = self._build_seed_messages()
        self.messages = list(self.seed_messages)

    @staticmethod
    def _resolve_path(package_root, maybe_relative):
        if os.path.isabs(maybe_relative):
            return maybe_relative
        return os.path.join(package_root, maybe_relative)

    def _build_seed_messages(self):
        if not os.path.isfile(self.system_prompt_path):
            raise RuntimeError("Missing system prompt file: {}".format(self.system_prompt_path))
        with open(self.system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        if not system_prompt:
            raise RuntimeError("System prompt file is empty: {}".format(self.system_prompt_path))

        seed = [{"role": "system", "content": system_prompt}]

        if not os.path.isfile(self.few_shot_path):
            raise RuntimeError("Missing few-shot file: {}".format(self.few_shot_path))
        with open(self.few_shot_path, "r", encoding="utf-8") as f:
            few_shot_data = json.load(f)

        if isinstance(few_shot_data, dict):
            few_shot_messages = few_shot_data.get("messages", [])
        else:
            few_shot_messages = few_shot_data

        if not isinstance(few_shot_messages, list):
            raise RuntimeError("Few-shot data must be a list or {\"messages\": [...]} object")

        for i, msg in enumerate(few_shot_messages):
            if not isinstance(msg, dict):
                raise RuntimeError("Few-shot message #{} must be an object".format(i))
            role = msg.get("role")
            content = msg.get("content")
            if role not in ("user", "assistant", "system"):
                raise RuntimeError("Few-shot message #{} has invalid role: {}".format(i, role))
            if content is None:
                raise RuntimeError("Few-shot message #{} is missing content".format(i))
            seed.append({"role": role, "content": content})

        return seed

    def reset_conversation(self):
        self.messages = list(self.seed_messages)
        rospy.loginfo("Conversation reset to system prompt + few-shot seed.")

    def _on_state(self, msg):
        self.latest_state = msg
        self.current_cmd.led = msg.led
        self.current_cmd.solenoid = msg.solenoid

    def _publish_current_command(self):
        self.cmd_pub.publish(self.current_cmd)
        rospy.loginfo(
            "Published actuator command: led=%s solenoid=%s servo_enable=%s servo_cmd=%.4f",
            self.current_cmd.led,
            self.current_cmd.solenoid,
            self.current_cmd.servo_enable,
            self.current_cmd.servo_cmd,
        )

    def send_chat_completion(self):
        url = "{}/chat/completions".format(self.api_base.rstrip("/"))
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer {}".format(self.api_key)

        req = urllib.request.Request(url=url, data=payload_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            response_text = response.read().decode("utf-8")
            return json.loads(response_text)

    @staticmethod
    def _extract_json_payloads(text):
        payloads = []
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch not in ("{", "["):
                continue
            try:
                payload, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, (dict, list)):
                payloads.append(payload)
        return payloads

    @staticmethod
    def _split_top_level(text, sep=","):
        parts = []
        buf = []
        depth = 0
        in_single = False
        in_double = False
        escape = False

        for ch in text:
            if escape:
                buf.append(ch)
                escape = False
                continue

            if ch == "\\" and (in_single or in_double):
                buf.append(ch)
                escape = True
                continue

            if ch == "'" and not in_double:
                in_single = not in_single
                buf.append(ch)
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                buf.append(ch)
                continue

            if not in_single and not in_double:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                elif ch == sep and depth == 0:
                    parts.append("".join(buf).strip())
                    buf = []
                    continue

            buf.append(ch)

        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    @staticmethod
    def _parse_value(raw):
        v = raw.strip()
        if not v:
            return None

        low = v.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low == "null":
            return None

        try:
            return ast.literal_eval(v)
        except Exception:
            pass

        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        return v

    def _extract_function_calls(self, text):
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return []

        body = text[start + 1:end].strip()
        if not body:
            return []

        calls = []
        for call_text in self._split_top_level(body):
            m = re.match(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", call_text, re.DOTALL)
            if not m:
                continue
            name = m.group(1)
            args_blob = m.group(2).strip()
            arguments = {}
            if args_blob:
                for arg_piece in self._split_top_level(args_blob):
                    if "=" not in arg_piece:
                        continue
                    key, raw_val = arg_piece.split("=", 1)
                    arguments[key.strip()] = self._parse_value(raw_val)
            calls.append({"name": name, "arguments": arguments})

        return calls

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("1", "true", "on", "yes"):
                return True
            if low in ("0", "false", "off", "no"):
                return False
        raise ValueError("Cannot convert to bool: {}".format(value))

    @staticmethod
    def _to_float(value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip().replace("%", ""))
        raise ValueError("Cannot convert to number: {}".format(value))

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, value))

    def _state_as_dict(self):
        if self.latest_state is None:
            return {"available": False, "error": "No actuator/state message received yet."}

        servo_norm = float(self.latest_state.servo_pos)
        servo_norm = self._clamp(servo_norm, 0.0, 1.0)
        servo_percent = servo_norm * 100.0
        servo_angle_deg = servo_norm * 180.0

        return {
            "available": True,
            "led": bool(self.latest_state.led),
            "solenoid": bool(self.latest_state.solenoid),
            "servo_pos_norm": round(servo_norm, 4),
            "servo_percent": round(servo_percent, 2),
            "servo_angle_deg": round(servo_angle_deg, 2),
            "adc_raw": int(self.latest_state.adc_raw),
            "encoder_min": int(self.latest_state.encoder_min),
            "encoder_max": int(self.latest_state.encoder_max),
        }

    def _dispatch_tool(self, name, arguments):
        args = arguments or {}

        if name == "set_led":
            on = self._to_bool(args.get("on", args.get("state")))
            self.current_cmd.led = on
            self._publish_current_command()
            return {"ok": True, "tool": name, "led": on}

        if name == "set_solenoid":
            on = self._to_bool(args.get("on", args.get("state")))
            self.current_cmd.solenoid = on
            self._publish_current_command()
            return {"ok": True, "tool": name, "solenoid": on}

        if name == "set_servo_percent":
            percent = self._to_float(args.get("percent"))
            percent = self._clamp(percent, 0.0, 100.0)
            self.current_cmd.servo_enable = True
            self.current_cmd.servo_cmd = percent / 100.0
            self._publish_current_command()
            return {
                "ok": True,
                "tool": name,
                "servo_percent": round(percent, 2),
                "servo_cmd": round(self.current_cmd.servo_cmd, 4),
            }

        if name == "set_servo_angle":
            angle_deg = self._to_float(args.get("angle_deg", args.get("angle")))
            angle_deg = self._clamp(angle_deg, 0.0, 180.0)
            self.current_cmd.servo_enable = True
            self.current_cmd.servo_cmd = angle_deg / 180.0
            self._publish_current_command()
            return {
                "ok": True,
                "tool": name,
                "servo_angle_deg": round(angle_deg, 2),
                "servo_cmd": round(self.current_cmd.servo_cmd, 4),
            }

        if name == "get_actuator_state":
            return {"ok": True, "tool": name, "state": self._state_as_dict()}

        raise ValueError("Unknown tool: {}".format(name))

    def tool_callback(self, payload):
        rospy.loginfo("toolCallback payload: %s", json.dumps(payload))
        calls = []

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    if "name" in item:
                        calls.append(item)
                    elif "tool" in item:
                        calls.append({"name": item["tool"], "arguments": item})
        elif isinstance(payload, dict):
            if "name" in payload:
                calls.append(payload)
            elif "tool" in payload:
                calls.append({"name": payload["tool"], "arguments": payload})

        if not calls:
            print("[toolCallback] Ignored unsupported payload: {}".format(json.dumps(payload)))
            return

        for call in calls:
            name = call.get("name")
            arguments = call.get("arguments", {})
            try:
                result = self._dispatch_tool(name, arguments)
                print("[toolCallback] result {}".format(json.dumps(result)))
            except Exception as e:
                print("[toolCallback] error {}".format(str(e)))

    def process_assistant_reply(self, assistant_text):
        print("assistant> {}".format(assistant_text))
        function_calls = self._extract_function_calls(assistant_text)
        for call in function_calls:
            self.tool_callback(call)

        json_payloads = self._extract_json_payloads(assistant_text)
        for payload in json_payloads:
            self.tool_callback(payload)

    def run(self):
        print("LLM chat started.")
        print("Endpoint: {}/chat/completions".format(self.api_base.rstrip("/")))
        print("Model: {}".format(self.model))
        print("Command topic: {}".format(self.cmd_topic))
        print("State topic: {}".format(self.state_topic))
        print("System prompt: {}".format(self.system_prompt_path))
        print("Few-shot file: {}".format(self.few_shot_path))
        print("Commands: /reset, /exit")

        while not rospy.is_shutdown():
            try:
                user_text = input("you> ").strip()
            except EOFError:
                print("")
                break
            except KeyboardInterrupt:
                print("")
                break

            if not user_text:
                continue
            if user_text in ("/exit", "/quit"):
                break
            if user_text == "/reset":
                self.reset_conversation()
                continue

            self.messages.append({"role": "user", "content": user_text})
            try:
                response_json = self.send_chat_completion()
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                rospy.logerr("HTTP error %s from LLM server: %s", e.code, body)
                continue
            except urllib.error.URLError as e:
                rospy.logerr("Could not reach LLM server: %s", str(e))
                continue
            except Exception as e:
                rospy.logerr("Unexpected LLM request error: %s", str(e))
                continue

            try:
                assistant_text = response_json["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                rospy.logerr("Unexpected response schema: %s", json.dumps(response_json))
                continue

            self.messages.append({"role": "assistant", "content": assistant_text})
            self.process_assistant_reply(assistant_text)


def main():
    rospy.init_node("llm_chat_node", anonymous=False)
    try:
        node = LlmChatNode()
    except Exception as e:
        rospy.logerr("Failed to initialize llm_chat_node: %s", str(e))
        sys.exit(1)
    node.run()


if __name__ == "__main__":
    main()
