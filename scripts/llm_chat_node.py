#!/usr/bin/env python3

import json
import os
import sys
import urllib.error
import urllib.request

import rospy
import rospkg


class LlmChatNode:
    def __init__(self):
        self.api_base = rospy.get_param("~api_base", "http://127.0.0.1:1234/v1")
        self.model = rospy.get_param("~model", "local-model")
        self.temperature = float(rospy.get_param("~temperature", 0.2))
        self.max_tokens = int(rospy.get_param("~max_tokens", 512))
        self.timeout_sec = float(rospy.get_param("~timeout_sec", 120.0))
        self.api_key = rospy.get_param("~api_key", os.environ.get("OPENAI_API_KEY", ""))
        self.system_prompt_file = rospy.get_param("~system_prompt_file", "system-prompt.md")
        self.few_shot_file = rospy.get_param("~few_shot_file", "few-shot-prompts.json")

        package_root = rospkg.RosPack().get_path("embodied_ai")
        self.system_prompt_path = self._resolve_path(package_root, self.system_prompt_file)
        self.few_shot_path = self._resolve_path(package_root, self.few_shot_file)

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
    def tool_callback(payload):
        rospy.loginfo("toolCallback payload: %s", json.dumps(payload))
        print("[toolCallback] {}".format(json.dumps(payload, indent=2)))

    def process_assistant_reply(self, assistant_text):
        print("assistant> {}".format(assistant_text))
        json_payloads = self._extract_json_payloads(assistant_text)
        for payload in json_payloads:
            self.tool_callback(payload)

    def run(self):
        print("LLM chat started.")
        print("Endpoint: {}/chat/completions".format(self.api_base.rstrip("/")))
        print("Model: {}".format(self.model))
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
