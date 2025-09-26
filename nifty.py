#!/usr/bin/env python3
"""
Nifty Matrix bot.

Major features:

1. Robust environment handling (supports MATRIX_* and legacy names, warns if aliases are used).
2. Clean login lifecycle with proper session closure on failure.
3. Background request queue to keep nio callbacks responsive.
4. Structured code via NiftyBot class for easier maintenance.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import yaml
from aiohttp import ClientSession
from dotenv import load_dotenv
from nio import (
    Api,
    AsyncClient,
    AsyncClientConfig,
    InviteMemberEvent,
    LoginResponse,
    MatrixRoom,
    Response,
    RoomMessageText,
)

# ------------------------------------------------------------------------------
# Configuration & constants
# ------------------------------------------------------------------------------

# Load environment variables (override OS-provided ones such as USERNAME)
load_dotenv(override=True)


def require_env(primary: str, *aliases: str) -> str:
    for name in (primary, *aliases):
        value = os.getenv(name)
        if value:
            if name != primary:
                print(f"[CONFIG] Using environment variable '{name}' for '{primary}'.")
            return value
    tried = ", ".join((primary, *aliases)) if aliases else primary
    raise RuntimeError(f"Missing required environment variable: {primary} (tried {tried}).")


HOMESERVER = require_env("MATRIX_HOMESERVER", "HOMESERVER")
USERNAME = require_env("MATRIX_USERNAME", "USERNAME")
PASSWORD = require_env("MATRIX_PASSWORD", "PASSWORD")

OPENROUTER_API_KEY = require_env("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

JINA_API_KEY = os.getenv("JINA_API_KEY")

SETTINGS_FILE = "settings.json"
PROMPT_FILE = "prompt.yaml"

DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"

REACTION_TRIGGERS: Dict[str, List[str]] = {
    "based": ["💊", "😎", "👍"],
    "cringe": ["😬", "🤔"],
    "awesome": ["🔥", "⚡", "🚀"],
    "thanks": ["👍", "🙏", "✨"],
    "nifty": ["😊", "👋"],
    "linux": ["🐧", "💻", "⚡"],
    "windows": ["🪟", "🤷"],
    "monero": ["💸", "💰", "🤑"],
    "python": ["🐍", "💻"],
    "rust": ["🦀", "⚡"],
    "uncle cmos": ["🐐", "👑", "🙏"],
    "security": ["🔒", "🛡️", "🔐"],
    "privacy": ["🕵️", "🔒", "🛡️"],
    "good morning": ["☀️", "👋", "🌅"],
    "good night": ["🌙", "😴", "💤"],
    "lmao": ["🤣", "💀"],
    "wtf": ["🤯", "😵", "🤔"],
    "nice": ["👌", "✨", "💯"],
}

FILTERED_WORDS = ["kyoko"]
HISTORY_LIMIT = 100
QUEUE_MAXSIZE = 5

IMPORTANT_INDICATORS = ["?", "help", "error", "problem", "issue", "important", "announcement", "urgent"]

CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".java": "java",
    ".sh": "bash",
    ".go": "go",
    ".rb": "ruby",
}

TECH_TOPICS = {
    "python": ["python", "pip", "django", "flask", "pandas", "numpy"],
    "javascript": ["javascript", "js", "node", "react", "vue", "angular"],
    "linux": ["linux", "ubuntu", "debian", "arch", "kernel", "bash"],
    "security": ["security", "encryption", "vpn", "tor", "privacy", "hack"],
    "crypto": ["bitcoin", "monero", "ethereum", "blockchain", "defi"],
    "ai": ["ai", "machine learning", "neural", "gpt", "llm", "deepseek"],
    "networking": ["network", "tcp", "udp", "http", "dns", "firewall"],
    "database": ["database", "sql", "mongodb", "redis", "postgresql"],
}

CURRENT_INFO_INDICATORS = [
    "latest",
    "current",
    "today",
    "yesterday",
    "this week",
    "recent",
    "news",
    "headlines",
    "update",
    "breaking",
    "announced",
    "price",
    "stock",
    "weather",
    "score",
    "results",
    "released",
    "launched",
    "published",
    "version",
    "status",
    "outage",
    "down",
    "working",
]

NO_SEARCH_PATTERNS = [
    r"what is (?:a|an|the)? (?:function|variable|loop|class)",
    r"how (?:do|does|to) .{0,20} work",
    r"why (?:is|are|do|does)",
    r"explain",
    r"define",
    r"who (?:is|are) you",
    r"what (?:is|are) you",
    r"help me with",
    r"debug",
    r"fix",
]

ENTITY_QUERY_PATTERNS = [
    r"what.{0,10}happening with",
    r"how is .{0,20} doing",
    r"status of",
    r"news about",
    r"updates? on",
    r"latest.{0,10}from",
]

KNOWN_BOTS = {"@kyoko:xmr.mx"}

SUMMARY_KEYWORDS = ["summary", "summarize", "recap", "what happened", "what was discussed", "catch me up", "tldr"]

# ------------------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------------------


def load_settings() -> Dict[str, Any]:
    if not os.path.exists(SETTINGS_FILE):
        settings = {"llm_model": DEFAULT_MODEL}
        save_settings(settings)
        return settings
    with open(SETTINGS_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_settings(settings: Dict[str, Any]) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as handle:
        json.dump(settings, handle)


def load_personality() -> str:
    default_personality = (
        "You are Nifty, a helpful and knowledgeable Matrix chatbot. "
        "You're friendly, witty, and technically proficient. "
        "You specialize in programming, Linux systems, security, and general tech support. "
        "Keep responses concise but informative, and maintain a casual yet professional tone."
    )
    
    if not os.path.exists(PROMPT_FILE):
        print(f"[WARNING] {PROMPT_FILE} not found, using default personality.")
        return default_personality
        
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data and "personality" in data:
            return data["personality"]
        else:
            print(f"[WARNING] 'personality' key not found in {PROMPT_FILE}, using default personality.")
            return default_personality
    except Exception as exc:
        print(f"[WARNING] Error loading {PROMPT_FILE}: {exc}, using default personality.")
        return default_personality


SETTINGS = load_settings()
BOT_PERSONALITY = load_personality()

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------


def get_display_name(user_id: str) -> str:
    if ":" in user_id and user_id.startswith("@"):
        return user_id.split(":", 1)[0][1:]
    return user_id


def filter_bot_triggers(text: str) -> str:
    filtered = text
    for word in FILTERED_WORDS:
        filtered = re.sub(re.escape(word), "[another bot]", filtered, flags=re.IGNORECASE)
    return filtered


def detect_language_from_url(url: str) -> str:
    for ext, lang in CODE_EXTENSIONS.items():
        if url.lower().endswith(ext):
            return lang
    return "text"


def extract_urls_from_message(message: str) -> List[str]:
    pattern = r"https?://[^\s<>'\"{}|\\^`\[\]]+(?:[.,!?;](?=\s|$))?"
    urls = re.findall(pattern, message)
    return [url.rstrip(".,!?;") for url in urls]


def detect_code_in_message(message: str) -> bool:
    indicators = [
        "```",
        "`",
        "def ",
        "class ",
        "function ",
        "const ",
        "let ",
        "var ",
        "import ",
        "from ",
        "export ",
        "()",
        "=>",
        "{}",
        "[]",
        "error:",
        "exception:",
        "traceback:",
    ]
    lower = message.lower()
    return any(indicator in lower for indicator in indicators)


def summarize_search_results(results: List[Dict[str, str]], query: str) -> str:
    if not results:
        return ""
    lines = [f"🔍 Search results for '{query}':", ""]
    for index, result in enumerate(results, start=1):
        lines.append(f"**{index}. {result['title']}**")
        lines.append(f"   {result['snippet']}")
        if result.get("url"):
            lines.append(f"   🔗 {result['url']}")
        lines.append("")
    return "\n".join(lines)


def extract_code_from_response(response: str) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    pattern = r"```(\w*)\n(.*?)```"
    last = 0
    for match in re.finditer(pattern, response, re.DOTALL):
        if match.start() > last:
            text = response[last : match.start()].strip()
            if text:
                parts.append({"type": "text", "content": text})
        language = match.group(1) or "text"
        code = match.group(2).strip()
        parts.append({"type": "code", "language": language, "content": code})
        last = match.end()
    if last < len(response):
        text = response[last:].strip()
        if text:
            parts.append({"type": "text", "content": text})
    return parts if parts else [{"type": "text", "content": response}]


# ------------------------------------------------------------------------------
# Conversation context tracking
# ------------------------------------------------------------------------------


class ConversationContext:
    def __init__(self) -> None:
        self.topics: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.user_interests: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.conversation_threads: Dict[str, List[Any]] = defaultdict(list)
        self.important_messages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def update(self, room_id: str, message: Dict[str, Any]) -> None:
        sender = message["sender"]
        body_lower = message["body"].lower()

        for topic, keywords in TECH_TOPICS.items():
            if any(keyword in body_lower for keyword in keywords):
                self.topics[room_id][topic] += 1.0
                for other_topic in list(self.topics[room_id].keys()):
                    if other_topic != topic:
                        self.topics[room_id][other_topic] *= 0.95
                if topic not in self.user_interests[room_id][sender]:
                    self.user_interests[room_id][sender].append(topic)

        if any(indicator in body_lower for indicator in IMPORTANT_INDICATORS):
            entry = {
                "sender": sender,
                "body": message["body"],
                "timestamp": message["timestamp"],
                "type": (
                    "question"
                    if "?" in body_lower
                    else "issue"
                    if any(word in body_lower for word in ["error", "problem"])
                    else "announcement"
                ),
            }
            self.important_messages[room_id].append(entry)
            self.important_messages[room_id] = self.important_messages[room_id][-20:]

    def analyze_flow(self, messages: List[Dict[str, Any]]) -> str:
        if len(messages) < 3:
            return "just_started"
        timestamps = [msg["timestamp"] for msg in messages]
        diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        avg = sum(diffs) / len(diffs) if diffs else 0
        if avg < 30:
            flow = "very_active"
        elif avg < 120:
            flow = "active"
        elif avg < 600:
            flow = "moderate"
        else:
            flow = "slow"
        recent_senders = [msg["sender"] for msg in messages[-10:]]
        unique_recent = set(recent_senders)
        if len(unique_recent) == 2 and len(recent_senders) >= 6:
            flow += "_dialogue"
        elif len(unique_recent) >= 4:
            flow += "_group_discussion"
        return flow

    def get_room_context(self, room_id: str, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not history:
            return None
        top_topics = sorted(self.topics[room_id].items(), key=lambda item: item[1], reverse=True)[:5]
        user_expertise = {
            get_display_name(user): interests[:3]
            for user, interests in self.user_interests[room_id].items()
            if interests
        }
        important = self.important_messages[room_id][-5:]
        flow = self.analyze_flow(history)
        return {
            "top_topics": top_topics,
            "user_expertise": user_expertise,
            "recent_important": important,
            "conversation_flow": flow,
            "message_count": len(history),
            "unique_participants": len({msg["sender"] for msg in history}),
        }

    async def cleanup(self) -> None:
        cutoff = datetime.now().timestamp() - 24 * 3600
        for room_id, topics in list(self.topics.items()):
            for topic in list(topics.keys()):
                topics[topic] *= 0.5
                if topics[topic] <= 0.1:
                    del topics[topic]
            if not topics:
                del self.topics[room_id]
        for room_id, important in self.important_messages.items():
            self.important_messages[room_id] = [msg for msg in important if msg["timestamp"] > cutoff]


# ------------------------------------------------------------------------------
# Request queue
# ------------------------------------------------------------------------------


@dataclass
class MessageTask:
    room: MatrixRoom
    event: RoomMessageText
    received_at: datetime


# ------------------------------------------------------------------------------
# Bot core
# ------------------------------------------------------------------------------


class NiftyBot:
    def __init__(self) -> None:
        config = AsyncClientConfig(store_sync_tokens=True)
        self.client = AsyncClient(HOMESERVER, USERNAME, config=config)
        self.room_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))
        self.joined_rooms: set[str] = set()
        self.context = ConversationContext()
        self.request_queue: asyncio.Queue[MessageTask] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self.queue_worker: Optional[asyncio.Task[Any]] = None
        self.cleanup_task: Optional[asyncio.Task[Any]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        login_response = await self._login()
        if login_response is None:
            await self.client.close()
            return

        await self._post_login_setup()
        await self._initial_sync()
        self.queue_worker = asyncio.create_task(self._request_worker(), name="request-worker")
        self.cleanup_task = asyncio.create_task(self._context_cleanup_loop(), name="context-cleanup")

        self._print_banner()
        try:
            await self.client.sync_forever(timeout=30000, full_state=False)
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt - shutting down...")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Sync error: {exc}")
        finally:
            await self._shutdown()

    async def _login(self) -> Optional[LoginResponse]:
        try:
            response = await self.client.login(PASSWORD)
            if isinstance(response, LoginResponse):
                print(f"Logged in as {response.user_id}")
                return response
            print(f"Failed to login: {response}")
            return None
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Login error: {exc}")
            return None

    async def _post_login_setup(self) -> None:
        print("Fetching joined rooms...")
        try:
            joined = await self.client.joined_rooms()
            if getattr(joined, "rooms", None):
                for room_id in joined.rooms:
                    self.joined_rooms.add(room_id)
                    print(f"Already in room: {room_id}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error fetching joined rooms: {exc}")

        self.client.add_event_callback(self._on_message, RoomMessageText)
        self.client.add_event_callback(self._on_invite, InviteMemberEvent)

    async def _initial_sync(self) -> None:
        print("Performing initial sync...")
        try:
            sync = await self.client.sync(timeout=30000, full_state=True)
            print(f"Initial sync complete. Next batch: {sync.next_batch}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Initial sync failed: {exc}")

    async def _shutdown(self) -> None:
        print("Closing down tasks...")
        if self.queue_worker:
            self.queue_worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.queue_worker
        if self.cleanup_task:
            self.cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.cleanup_task
        await self.client.close()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _on_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        if event.state_key != self.client.user_id:
            return
        print(f"[INVITE] Received invite to {room.room_id} from {event.sender}")
        try:
            result = await self.client.join(room.room_id)
            if hasattr(result, "room_id"):
                self.joined_rooms.add(room.room_id)
                print(f"[INVITE] Joined {room.room_id}")
                await self._send_plain_text(
                    room.room_id,
                    (
                        "Hey! I'm Nifty! 👋 Thanks for inviting me! Just say 'nifty' followed by your message "
                        "to chat, or reply to any of my messages! 🚀\n\n"
                        "I specialize in:\n"
                        "• 💻 Programming & debugging\n"
                        "• 🐧 Linux/Unix systems\n"
                        "• 🌐 Web dev & networking\n"
                        "• 🔒 Security & cryptography\n"
                        "• 🤖 General tech support\n"
                        "• 📱 Mobile dev tips\n"
                        "• 🎮 Gaming & internet culture\n\n"
                        "Commands:\n"
                        "• `nifty !reset` - Clear my context\n"
                        "• `nifty summary` - Get a detailed chat analysis\n"
                        "• Share URLs and I'll read and analyze them!\n"
                        "• `?set <model>` - Change LLM model (admin only)\n"
                        "• `?set list` - Show current LLM model\n\n"
                        "I also react to messages with emojis when appropriate! 😊 Let's build something cool! 💪"
                    ),
                )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[INVITE] Failed to join {room.room_id}: {exc}")

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if event.sender == self.client.user_id:
            return
        if any(event.sender.startswith(bot) for bot in KNOWN_BOTS):
            return

        print(f"[DEBUG] Incoming message in {room.room_id} from {event.sender}: {event.body}")

        if not await self._handle_model_commands(room, event):
            self._store_message(room.room_id, event)

            await self._maybe_react(room.room_id, event.event_id, event.body)

            is_reply, replied_to_bot, previous_message = await self._analyze_reply(room, event)
            should_respond = "nifty" in event.body.lower() or replied_to_bot

            if should_respond:
                task = MessageTask(room=room, event=event, received_at=datetime.now())
                try:
                    self.request_queue.put_nowait(task)
                except asyncio.QueueFull:
                    await self._send_plain_text(
                        room.room_id,
                        "Yo I'm getting slammed with requests rn, gimme a sec! 😅",
                    )
                else:
                    print(f"[QUEUE] Task queued (size={self.request_queue.qsize()})")

    async def _handle_model_commands(self, room: MatrixRoom, event: RoomMessageText) -> bool:
        if not event.body.startswith("?set "):
            return False
        command = event.body[5:].strip()
        if command == "list":
            await self._send_plain_text(
                room.room_id,
                f"Current LLM model: {SETTINGS['llm_model']}",
            )
        else:
            SETTINGS["llm_model"] = command
            save_settings(SETTINGS)
            await self._send_plain_text(
                room.room_id,
                f"LLM model updated to: {SETTINGS['llm_model']}",
            )
        return True

    def _store_message(self, room_id: str, event: RoomMessageText) -> None:
        msg = {
            "sender": event.sender,
            "body": event.body,
            "timestamp": (event.server_timestamp or datetime.now().timestamp() * 1000) / 1000,
            "event_id": event.event_id,
        }
        self.room_history[room_id].append(msg)
        self.context.update(room_id, msg)

    async def _maybe_react(self, room_id: str, event_id: str, body: str) -> None:
        lower = body.lower()
        for trigger, reactions in REACTION_TRIGGERS.items():
            if trigger in lower:
                chance = {
                    "uncle cmos": 0.7,
                    "based": 0.4,
                    "cringe": 0.5,
                    "lol": 0.2,
                    "lmao": 0.2,
                    "wtf": 0.3,
                    "monero": 0.5,
                    "good morning": 0.6,
                    "good night": 0.6,
                }.get(trigger, 0.3)
                if random.random() < chance:
                    reaction = random.choice(reactions)
                    try:
                        await self.client.room_send(
                            room_id,
                            message_type="m.reaction",
                            content={
                                "m.relates_to": {
                                    "rel_type": "m.annotation",
                                    "event_id": event_id,
                                    "key": reaction,
                                }
                            },
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        print(f"Failed to react in {room_id}: {exc}")
                break

    async def _analyze_reply(
        self,
        room: MatrixRoom,
        event: RoomMessageText,
    ) -> tuple[bool, bool, Optional[str]]:
        source = getattr(event, "source", {})
        relates_to = source.get("content", {}).get("m.relates_to", {})
        reply_to = relates_to.get("m.in_reply_to", {}).get("event_id")
        if not reply_to:
            return False, False, None
        try:
            response = await self.client.room_get_event(room.room_id, reply_to)
            replied_event = getattr(response, "event", None)
            if replied_event and replied_event.sender == self.client.user_id:
                return True, True, replied_event.body
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to fetch replied event: {exc}")
        return True, False, None

    # ------------------------------------------------------------------
    # Queue worker
    # ------------------------------------------------------------------

    async def _request_worker(self) -> None:
        while True:
            try:
                task = await self.request_queue.get()
                await self._process_task(task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WORKER] Error: {exc}")

    async def _process_task(self, task: MessageTask) -> None:
        room = task.room
        event = task.event

        if "!reset" in event.body.lower():
            self.room_history[room.room_id].clear()
            await self._send_plain_text(room.room_id, "✨ Nifty's context cleared! Fresh start! 🧹")
            return

        urls = extract_urls_from_message(event.body)
        url_contents: List[Dict[str, Any]] = []
        if urls:
            await self.client.room_typing(room.room_id, True)
            for url in urls[:3]:
                content = await self._fetch_url_content(url)
                if content:
                    url_contents.append(content)

        previous_message = None
        _, replied_to_bot, previous_message = await self._analyze_reply(room, event)  # Re-check inside worker

        prompt = event.body
        await self.client.room_typing(room.room_id, True)
        reply = await self._draft_reply(
            prompt=prompt,
            room_id=room.room_id,
            previous_message=previous_message if replied_to_bot else None,
            url_contents=url_contents or None,
        )
        await self.client.room_typing(room.room_id, False)

        if any(word.lower() in reply.lower() for word in FILTERED_WORDS):
            reply = filter_bot_triggers(reply)

        response = await self._send_formatted_message(room.room_id, reply)
        event_id = getattr(response, "event_id", None)
        if event_id:
            bot_msg = {
                "sender": self.client.user_id,
                "body": reply,
                "timestamp": datetime.now().timestamp(),
                "event_id": event_id,
            }
            self.room_history[room.room_id].append(bot_msg)
            self.context.update(room.room_id, bot_msg)

    # ------------------------------------------------------------------
    # Content processing
    # ------------------------------------------------------------------

    async def _fetch_url_content(self, url: str) -> Optional[Dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                reader_url = f"https://r.jina.ai/{aiohttp.helpers.quote(url, safe='')}"
                headers = {
                    "Accept": "application/json",
                    "X-With-Links-Summary": "true",
                    "X-With-Images-Summary": "true",
                    "X-With-Generated-Alt": "true",
                }
                if JINA_API_KEY:
                    headers["Authorization"] = f"Bearer {JINA_API_KEY}"
                async with session.get(reader_url, headers=headers) as response:
                    if response.status != 200:
                        print(f"[URL] Failed to fetch {url}: {response.status}")
                        print(f"[URL] Body: {await response.text()}")
                        return None
                    data = await response.json()
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[URL] Error fetching {url}: {exc}")
                return None

        content = data.get("content", "")
        title = data.get("title", url)
        result: Dict[str, Any] = {"type": "article", "content": content[:5000], "title": title}

        file_lang = detect_language_from_url(url)
        code_info = data.get("code", {})
        if file_lang != "text":
            result.update({"type": "code", "language": file_lang, "content": content[:5000]})
            return result
        if code_info.get("language"):
            result.update({"type": "code", "language": code_info["language"], "content": content[:5000]})
            return result

        for key in ("description", "images", "links"):
            if key in data:
                result[key] = data[key][:5] if isinstance(data[key], list) else data[key]
        return result

    async def _needs_web_search(self, prompt: str) -> bool:
        lower = prompt.lower()
        needs_current = any(indicator in lower for indicator in CURRENT_INFO_INDICATORS)
        entity_query = any(re.search(pattern, lower) for pattern in ENTITY_QUERY_PATTERNS)
        general = any(re.search(pattern, lower) for pattern in NO_SEARCH_PATTERNS)
        if "latest version" in lower or "new features" in lower:
            return True
        return (needs_current or entity_query) and not general

    async def _search_with_jina(self, query: str, num_results: int = 5) -> Optional[List[Dict[str, str]]]:
        timeout = aiohttp.ClientTimeout(total=15)
        headers = {"Accept": "application/json"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                url = f"https://s.jina.ai/{aiohttp.helpers.quote(filter_bot_triggers(query))}"
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        print(f"[JINA] Query failed ({response.status}): {query}")
                        print(f"[JINA] Body: {await response.text()}")
                        return None
                    if "application/json" not in response.headers.get("Content-Type", ""):
                        text = await response.text()
                        return [{"title": f"Search results for: {query}", "url": url, "snippet": text[:300]}]
                    payload = await response.json()
            except asyncio.TimeoutError:
                print(f"[JINA] Timeout for query: {query}")
                return None
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[JINA] Error: {exc}")
                return None

        results: List[Dict[str, str]] = []
        for item in payload.get("data", [])[:num_results]:
            results.append(
                {
                    "title": filter_bot_triggers(item.get("title", "No title")),
                    "url": item.get("url", ""),
                    "snippet": filter_bot_triggers(
                        item.get("description") or item.get("content", "No description available")
                    )[:300],
                }
            )
        return results

    async def _search_technical_docs(self, query: str) -> Optional[List[Dict[str, str]]]:
        tech_queries = [
            f"{query} site:stackoverflow.com",
            f"{query} site:docs.python.org OR site:github.com",
            f"{query} programming documentation tutorial",
        ]
        for enhanced in tech_queries:
            results = await self._search_with_jina(enhanced)
            if results:
                return results
        return None

    async def _draft_reply(
        self,
        prompt: str,
        room_id: str,
        previous_message: Optional[str],
        url_contents: Optional[List[Dict[str, Any]]],
    ) -> str:
        filtered_prompt = filter_bot_triggers(prompt)
        room_context = self.context.get_room_context(room_id, list(self.room_history[room_id])[-30:])
        system_prompt = self._build_system_prompt(room_context, url_contents)

        wants_summary = any(keyword in filtered_prompt.lower() for keyword in SUMMARY_KEYWORDS)
        if wants_summary and room_id:
            filtered_prompt = await self._build_summary_prompt(filtered_prompt, room_id)

        about_nifty = any(keyword in filtered_prompt.lower() for keyword in ["who are you", "what are you", "nifty"])
        is_technical = detect_code_in_message(filtered_prompt) or any(
            keyword in filtered_prompt.lower()
            for keyword in [
                "code",
                "programming",
                "debug",
                "error",
                "function",
                "script",
                "compile",
                "python",
                "javascript",
                "rust",
                "linux",
                "bash",
                "git",
                "docker",
                "api",
                "database",
                "sql",
                "server",
                "network",
                "security",
            ]
        )

        if url_contents:
            filtered_prompt = self._append_url_content(filtered_prompt, url_contents)

        should_search = False
        if not wants_summary and not about_nifty and not url_contents:
            should_search = await self._needs_web_search(filtered_prompt)

        search_summary = ""
        if should_search:
            query = self._extract_search_query(filtered_prompt)
            if is_technical:
                results = await self._search_technical_docs(query)
            else:
                results = await self._search_with_jina(query)
            if results:
                search_summary = summarize_search_results(results, query)
                filtered_prompt = self._enhance_prompt_with_search(filtered_prompt, search_summary, is_technical)
            else:
                filtered_prompt += "\n\n(Note: I couldn't retrieve web search results for this query, so I'll rely on my existing knowledge.)"

        messages = [{"role": "system", "content": system_prompt}]
        if is_technical:
            messages[0]["content"] += (
                "\n\nThe user has a technical question. Provide detailed, accurate technical help "
                "with code examples when appropriate. Be thorough but concise."
            )

        history = list(self.room_history[room_id])
        if len(history) > 3:
            context_messages = []
            for msg in history[-11:-1]:
                role = "assistant" if msg["sender"] == self.client.user_id else "user"
                snippet = msg["body"][:200]
                context_messages.append({"role": role, "content": f"{get_display_name(msg['sender'])}: {snippet}"})
            messages.extend(context_messages[-5:])

        if previous_message:
            messages.append({"role": "assistant", "content": f"[Previous message I sent]: {previous_message}"})
            messages.append({"role": "user", "content": f"[User is replying to the above message]: {filtered_prompt}"})
        else:
            messages.append({"role": "user", "content": filtered_prompt})

        payload = {
            "model": SETTINGS["llm_model"],
            "messages": messages,
            "temperature": 0.7 if is_technical else 0.8,
            "max_tokens": 1000,
        }

        try:
            async with ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                async with session.post(OPENROUTER_URL, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        print(f"[LLM] Error {response.status}: {text}")
                        return f"Hey, I'm Nifty and I hit a snag (error {response.status}). Mind trying again? 🔧"
                    data = await response.json()
        except asyncio.TimeoutError:
            print("[LLM] Timeout after 30 seconds")
            return "Yo, the AI servers are being slow af rn. Try again in a sec? 🔧"
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[LLM] Request failed: {exc}")
            return "Hmm, Nifty here - something went wonky on my end! Could you try that again? 🤔"

        reply = data["choices"][0]["message"]["content"]
        return filter_bot_triggers(reply)

    def _build_system_prompt(
        self,
        room_context: Optional[Dict[str, Any]],
        url_contents: Optional[List[Dict[str, Any]]],
    ) -> str:
        prompt = BOT_PERSONALITY
        if room_context:
            prompt += "\n\n**ROOM CONTEXT**:\n"
            if room_context["top_topics"]:
                topics_str = ", ".join(
                    f"{topic} ({score:.1f})" for topic, score in room_context["top_topics"][:3]
                )
                prompt += f"Current hot topics: {topics_str}\n"
            if room_context["user_expertise"]:
                prompt += "User expertise in the room:\n"
                for user, interests in list(room_context["user_expertise"].items())[:5]:
                    prompt += f"  - {user}: {', '.join(interests)}\n"
            flow = room_context["conversation_flow"]
            if "dialogue" in flow:
                prompt += "This is a focused dialogue between two people. Be direct and helpful.\n"
            elif "group_discussion" in flow:
                prompt += "This is a group discussion. Consider multiple perspectives.\n"
            if "very_active" in flow:
                prompt += "The conversation is very active. Keep responses concise and on-point.\n"
            if room_context["recent_important"]:
                prompt += "\nRecent important points:\n"
                for item in room_context["recent_important"][-3:]:
                    body = item["body"][:100]
                    prompt += f"  - [{item['type']}] {get_display_name(item['sender'])}: {body}...\n"
        if url_contents:
            prompt += "\n\nIMPORTANT: The user has shared URLs with content. Analyze and discuss the content thoroughly."
        return prompt

    async def _build_summary_prompt(self, original_prompt: str, room_id: str) -> str:
        minutes = 30
        lower = original_prompt.lower()
        patterns = [(r"last (\d+) minutes?", 1), (r"past (\d+) minutes?", 1), (r"last (\d+) hours?", 60), (r"past (\d+) hours?", 60)]
        for pattern, multiplier in patterns:
            match = re.search(pattern, lower)
            if match:
                minutes = int(match.group(1)) * multiplier
                break
        summary = self._create_comprehensive_summary(room_id, minutes)
        return (
            f"The user asked: {original_prompt}\n\n"
            f"Here's the comprehensive analysis:\n{summary}\n\n"
            "Based on this data, provide a natural, conversational summary. Focus on:\n"
            "1. Key discussion points and decisions made\n"
            "2. Questions that were asked and whether they were answered\n"
            "3. Any action items or next steps mentioned\n"
            "4. The overall mood and flow of the conversation\n\n"
            "Keep your personality but be informative. Remember you are Nifty."
        )

    def _append_url_content(self, prompt: str, contents: List[Dict[str, Any]]) -> str:
        lines = [prompt, "", "[SHARED CONTENT]:"]
        for content in contents[:3]:
            lines.append(f"\nFrom {content['title']}:")
            if content["type"] == "code":
                lines.append(f"Programming language: {content.get('language', 'unknown')}")
                lines.append(f"Code content:\n{content['content'][:2000]}\n")
            else:
                lines.append(f"Content:\n{content['content'][:2000]}\n")
        return "\n".join(lines)

    def _extract_search_query(self, prompt: str) -> str:
        lower = prompt.lower()
        for keyword in ["search for", "look up", "find", "tell me about", "what is", "who is", "how to", "how do i"]:
            if keyword in lower:
                return prompt.lower().split(keyword)[-1].strip()
        return prompt

    def _enhance_prompt_with_search(self, prompt: str, summary: str, is_technical: bool) -> str:
        if is_technical:
            return (
                f"User asked a technical question: {prompt}\n\n"
                f"{summary}\n"
                "Based on these search results, provide a comprehensive technical answer. Include:\n"
                "1. Direct solution to their problem\n"
                "2. Code examples if applicable (properly formatted)\n"
                "3. Best practices and common pitfalls\n"
                "4. Alternative approaches if relevant\n\n"
                "Remember to maintain your personality while being technically accurate and helpful. You are Nifty."
            )
        return (
            f"User asked: {prompt}\n\n"
            f"{summary}\n"
            "Based on these search results, provide a comprehensive but concise answer. Focus on:\n"
            "1. Directly answering the user's question\n"
            "2. Highlighting the most important/relevant information\n"
            "3. Mentioning any interesting or surprising facts\n"
            "4. Being accurate while maintaining your personality\n\n"
            "Remember you are Nifty."
        )

    def _create_comprehensive_summary(self, room_id: str, minutes: int) -> str:
        history = list(self.room_history[room_id])
        if not history:
            return "No recent messages to summarize."
        cutoff = datetime.now().timestamp() - minutes * 60
        recent = [msg for msg in history if msg["timestamp"] > cutoff]
        if not recent:
            return f"No messages in the last {minutes} minutes."
        context = self.context.get_room_context(room_id, recent)
        participants = sorted({get_display_name(msg["sender"]) for msg in recent})
        lines = [
            f"📊 **Comprehensive Chat Analysis (last {minutes} minutes)**",
            "",
            f"**Active Participants** ({len(participants)}): {', '.join(participants)}",
            f"**Total Messages**: {len(recent)}",
            "",
        ]
        if context and context["top_topics"]:
            lines.append("**🔥 Hot Topics**:")
            for topic, score in context["top_topics"]:
                lines.append(f"  • {topic.capitalize()} (relevance: {score:.1f})")
            lines.append("")
        if context and context["user_expertise"]:
            lines.append("**👥 User Interests/Expertise**:")
            for user, interests in list(context["user_expertise"].items())[:5]:
                lines.append(f"  • {user}: {', '.join(interests)}")
            lines.append("")
        if context and context["recent_important"]:
            lines.append("**⚡ Important Messages**:")
            for imp in context["recent_important"][-5:]:
                body = imp["body"][:100] + "..." if len(imp["body"]) > 100 else imp["body"]
                lines.append(f"  • [{imp['type']}] {get_display_name(imp['sender'])}: {body}")
            lines.append("")
        if context:
            flow_map = {
                "very_active": "🔥 Very Active",
                "active": "💬 Active",
                "moderate": "🗨️ Moderate",
                "slow": "🐌 Slow",
                "very_active_dialogue": "🔥 Intense Dialogue",
                "active_dialogue": "💬 Active Dialogue",
                "very_active_group_discussion": "🔥 Heated Group Discussion",
                "active_group_discussion": "💬 Active Group Discussion",
            }
            lines.append(f"**Conversation Style**: {flow_map.get(context['conversation_flow'], 'normal')}")
        return "\n".join(lines)

    async def _send_plain_text(self, room_id: str, body: str) -> Response:
        return await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": body},
        )

    async def _send_formatted_message(self, room_id: str, reply: str) -> Response:
        parts = extract_code_from_response(reply)
        formatted_body_parts: List[str] = []
        plain_parts: List[str] = []
        for part in parts:
            if part["type"] == "text":
                text = part["content"]
                formatted_text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
                formatted_body_parts.append(f"<p>{formatted_text}</p>")
                plain_parts.append(text)
            else:
                language = part["language"]
                code = html.escape(part["content"])
                if language and language != "text":
                    formatted_body_parts.append(f"<p><strong>{language}:</strong></p>")
                    plain_parts.append(f"{language}:")
                formatted_body_parts.append(f'<pre><code class="language-{language}">{code}</code></pre>')
                plain_parts.append(f"```{language}\n{part['content']}\n```")
        content = {
            "msgtype": "m.text",
            "body": "\n\n".join(plain_parts).strip(),
            "format": "org.matrix.custom.html",
            "formatted_body": "".join(formatted_body_parts).strip(),
        }
        return await self.client.room_send(room_id=room_id, message_type="m.room.message", content=content)

    async def _context_cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(3600)
                await self.context.cleanup()
                print(f"[CLEANUP] Completed at {datetime.now().isoformat(timespec='seconds')}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[CLEANUP] Error: {exc}")

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        print("=" * 60)
        print("🤖 Nifty Bot is running!")
        print("=" * 60)
        print(f"✅ Identity: {self.client.user_id}")
        print("✅ Listening for messages in all joined rooms")
        print("✅ Auto-accepting room invites")
        print("📝 Trigger: Say 'nifty' anywhere in a message")
        print("💬 Or reply directly to any of my messages")
        print("❌ Random responses: DISABLED")
        print("👀 Emoji reactions: ENABLED (various triggers)")
        print("🧹 Reset: 'nifty !reset' to clear context")
        print("📊 Summary: 'nifty summary' for comprehensive chat analysis")
        print(f"🧠 Optimized Context: Tracking {HISTORY_LIMIT} messages")
        print("📈 Context Features: Topic tracking, user expertise, important messages")
        print("💻 Technical expertise: Programming, Linux, Security, etc.")
        print("🔗 URL Analysis: Share URLs and I'll read and discuss them!")
        print("📝 Code Formatting: Proper syntax highlighting for all languages")
        print(f"🚫 Filtering words: {', '.join(FILTERED_WORDS)}")
        print("🔍 Web search: Powered by Jina.ai - Smart detection for current info")
        print("🎯 Personality: Professional, helpful, witty, context-aware")
        print("⏱️ Timeouts: 30s for LLM, 15s for search, 20s for URL fetching")
        print("🔄 Retry logic: handled at worker level")
        print("🧹 Auto-cleanup: Hourly context cleanup to maintain performance")
        print("🔄 LLM Model Switching: ?set <model> / ?set list")
        print("=" * 60)


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import contextlib

    bot = NiftyBot()
    try:
        asyncio.run(bot.start())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Application error: {exc}")
