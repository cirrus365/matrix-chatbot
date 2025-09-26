# nifty.py
import asyncio
import aiohttp
from nio import AsyncClient, MatrixRoom, RoomMessageText, LoginResponse, InviteMemberEvent, Api
from datetime import datetime, timedelta
import json
import html
import re
from urllib.parse import quote, urlparse, unquote
from collections import defaultdict, deque
import random
from asyncio import Queue, QueueFull
import os
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Load settings
settings_file = "settings.json"
try:
    with open(settings_file, "r") as f:
        settings = json.load(f)
except FileNotFoundError:
    settings = {"llm_model": "deepseek/deepseek-chat-v3-0324:free"}
    with open(settings_file, "w") as f:
        json.dump(settings, f)

# Load personality prompt from YAML
with open("prompt.yaml", "r") as f:
    personality_data = yaml.safe_load(f)
    BOT_PERSONALITY = personality_data["personality"]

# Your credentials from .env
HOMESERVER = os.getenv("HOMESERVER")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# OpenRouter config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Jina.ai config for web search only
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Create a request queue to prevent overload
request_queue = Queue(maxsize=5)  # Max 5 pending requests

# Reaction triggers - more professional and appropriate
REACTION_TRIGGERS = {
    'based': ['💊', '😎', '👍'],
    'cringe': ['😬', '🤔'],
    'awesome': ['🔥', '⚡', '🚀'],
    'thanks': ['👍', '🙏', '✨'],
    'nifty': ['😊', '👋'],
    'linux': ['🐧', '💻', '⚡'],
    'windows': ['🪟', '🤷'],
    'monero': ['💸', '💰', '🤑'],
    'python': ['🐍', '💻'],
    'rust': ['🦀', '⚡'],
    'uncle cmos': ['🐐', '👑', '🙏'],
    'security': ['🔒', '🛡️', '🔐'],
    'privacy': ['🕵️', '🔒', '🛡️'],
    'good morning': ['☀️', '👋', '🌅'],
    'good night': ['🌙', '😴', '💤'],
    'lmao': ['🤣', '💀'],
    'wtf': ['🤯', '😵', '🤔'],
    'nice': ['👌', '✨', '💯'],
}

# Bot filter - prevent triggering other bots
FILTERED_WORDS = ['kyoko']  # Only filter exact bot trigger word

# Store recent messages for context - REDUCED for better performance
room_message_history = defaultdict(lambda: deque(maxlen=100))  # Reduced from 500 to 100

# Store joined rooms
joined_rooms = set()

# Enhanced conversation context tracker
class ConversationContext:
    def __init__(self):
        self.topics = defaultdict(lambda: defaultdict(float))  # room_id -> topic -> relevance_score
        self.user_interests = defaultdict(lambda: defaultdict(list))  # room_id -> user -> interests
        self.conversation_threads = defaultdict(list)  # room_id -> list of conversation threads
        self.important_messages = defaultdict(list)  # room_id -> list of important messages
        
    def update_context(self, room_id, message_data):
        """Update conversation context with new message"""
        sender = message_data['sender']
        body = message_data['body'].lower()
        
        # Extract topics from message
        tech_topics = {
            'python': ['python', 'pip', 'django', 'flask', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue', 'angular'],
            'linux': ['linux', 'ubuntu', 'debian', 'arch', 'kernel', 'bash'],
            'security': ['security', 'encryption', 'vpn', 'tor', 'privacy', 'hack'],
            'crypto': ['bitcoin', 'monero', 'ethereum', 'blockchain', 'defi'],
            'ai': ['ai', 'machine learning', 'neural', 'gpt', 'llm', 'deepseek'],
            'networking': ['network', 'tcp', 'udp', 'http', 'dns', 'firewall'],
            'database': ['database', 'sql', 'mongodb', 'redis', 'postgresql'],
        }
        
        # Update topic scores
        for topic, keywords in tech_topics.items():
            if any(keyword in body for keyword in keywords):
                self.topics[room_id][topic] += 1.0
                # Decay older topics
                for t in self.topics[room_id]:
                    if t != topic:
                        self.topics[room_id][t] *= 0.95
        
        # Track user interests
        for topic, keywords in tech_topics.items():
            if any(keyword in body for keyword in keywords):
                if topic not in self.user_interests[room_id][sender]:
                    self.user_interests[room_id][sender].append(topic)
        
        # Mark important messages (questions, problems, announcements)
        importance_indicators = ['?', 'help', 'error', 'problem', 'issue', 'important', 'announcement', 'urgent']
        if any(indicator in body for indicator in importance_indicators):
            self.important_messages[room_id].append({
                'sender': sender,
                'body': message_data['body'],
                'timestamp': message_data['timestamp'],
                'type': 'question' if '?' in body else 'issue' if any(word in body for word in ['error', 'problem']) else 'announcement'
            })
            # Keep only last 20 important messages (reduced from 50)
            self.important_messages[room_id] = self.important_messages[room_id][-20:]
    
    def get_room_context(self, room_id, lookback_messages=30):  # REDUCED from 50 to 30
        """Get comprehensive room context"""
        if room_id not in room_message_history:
            return None
        
        messages = list(room_message_history[room_id])[-lookback_messages:]
        if not messages:
            return None
        
        # Get top topics
        top_topics = sorted(self.topics[room_id].items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get active users and their expertise
        user_expertise = {}
        for user, interests in self.user_interests[room_id].items():
            if interests:
                user_expertise[get_display_name(user)] = interests[:3]  # Top 3 interests per user
        
        # Get recent important messages
        recent_important = self.important_messages[room_id][-5:]  # Reduced from 10
        
        # Analyze conversation flow
        conversation_flow = self.analyze_conversation_flow(messages)
        
        return {
            'top_topics': top_topics,
            'user_expertise': user_expertise,
            'recent_important': recent_important,
            'conversation_flow': conversation_flow,
            'message_count': len(messages),
            'unique_participants': len(set(msg['sender'] for msg in messages))
        }
    
    def analyze_conversation_flow(self, messages):
        """Analyze how conversation is flowing"""
        if len(messages) < 3:
            return 'just_started'
        
        # Check message frequency
        if len(messages) >= 2:
            time_diffs = []
            for i in range(1, len(messages)):
                diff = messages[i]['timestamp'] - messages[i-1]['timestamp']
                time_diffs.append(diff)
            
            avg_time = sum(time_diffs) / len(time_diffs)
            
            if avg_time < 30:  # Less than 30 seconds between messages
                flow = 'very_active'
            elif avg_time < 120:  # Less than 2 minutes
                flow = 'active'
            elif avg_time < 600:  # Less than 10 minutes
                flow = 'moderate'
            else:
                flow = 'slow'
            
            # Check if it's a back-and-forth between two people
            recent_senders = [msg['sender'] for msg in messages[-10:]]
            unique_recent = set(recent_senders)
            if len(unique_recent) == 2 and len(recent_senders) >= 6:
                flow += '_dialogue'
            elif len(unique_recent) >= 4:
                flow += '_group_discussion'
            
            return flow
        
        return 'normal'

conversation_context = ConversationContext()

def filter_bot_triggers(text):
    """Remove or replace words that might trigger other bots"""
    filtered_text = text
    
    for word in FILTERED_WORDS:
        # Replace any variation (case-insensitive) with a safe alternative
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        filtered_text = pattern.sub("[another bot]", filtered_text)
    
    return filtered_text

def get_display_name(user_id):
    """Extract display name from user ID"""
    # Remove the @ and domain parts
    if '@' in user_id:
        name = user_id.split(':')[0][1:]  # Remove @ and everything after :
        return name
    return user_id

def detect_language_from_url(url):
    """Detect programming language from file extension"""
    extensions = {
        '.py': 'python',
        '.js': 'javascript',
        '.rs': 'rust',
        '.c': 'c',
        '.cpp': 'cpp',
        '.java': 'java',
        '.sh': 'bash',
        '.go': 'go',
        '.rb': 'ruby'
    }
    
    for ext, lang in extensions.items():
        if url.endswith(ext):
            return lang
    return 'text'

def extract_urls_from_message(message):
    """Extract all URLs from a message"""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+(?:[.,!?;](?=\s|$))?'
    urls = re.findall(url_pattern, message)
    return [url.rstrip('.,!?;') for url in urls]

async def search_with_jina(query, num_results=5):
    """Search using Jina.ai's search API with timeout"""
    filtered_query = filter_bot_triggers(query)
    
    timeout = aiohttp.ClientTimeout(total=15)  # 15 second timeout for searches
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # Jina search API
            search_url = f"https://s.jina.ai/{quote(filtered_query)}"
            
            headers = {
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Accept": "application/json"
            }
            
            async with session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    # Try to parse as JSON first
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        data = await response.json()
                        results = []
                        if 'data' in data:
                            for item in data['data'][:num_results]:
                                result = {
                                    'title': filter_bot_triggers(item.get('title', 'No title')),
                                    'url': item.get('url', ''),
                                    'snippet': filter_bot_triggers(item.get('description', item.get('content', 'No description available'))[:300])
                                }
                                if 'publishedDate' in item:
                                    result['date'] = item['publishedDate']
                                results.append(result)
                        return results
                    else:
                        # If not JSON, parse the text response
                        text = await response.text()
                        # Extract results from text format if needed
                        return [{
                            'title': f"Search results for: {query}",
                            'url': search_url,
                            'snippet': text[:300]
                        }]
                else:
                    print(f"Jina search returned status {response.status}")
                    print(f"Response: {await response.text()}")
                    return None
        
        except asyncio.TimeoutError:
            print(f"[ERROR] Jina search timed out after 15 seconds")
            return None
        except Exception as e:
            print(f"Error searching with Jina: {e}")
            return None

async def fetch_url_with_jina(url):
    """Fetch and parse content from URL using Jina.ai reader with timeout"""
    timeout = aiohttp.ClientTimeout(total=20)  # 20 second timeout for URL fetching
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # Use Jina reader API
            reader_url = f"https://r.jina.ai/{quote(url, safe='')}"
            
            headers = {
                'Accept': 'application/json',
                'X-With-Links-Summary': 'true',  # Get link summaries
                'X-With-Images-Summary': 'true',  # Get image descriptions
                'X-With-Generated-Alt': 'true'   # Get AI-generated alt text
            }
            
            # Add API key if available
            if JINA_API_KEY:
                headers['Authorization'] = f'Bearer {JINA_API_KEY}'
            
            async with session.get(reader_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract content based on response structure
                    content = data.get('content', '')
                    title = data.get('title', urlparse(url).netloc)
                    
                    # Detect content type
                    content_type = 'article'
                    if any(ext in url for ext in ['.py', '.js', '.rs', '.c', '.cpp', '.java']):
                        content_type = 'code'
                        language = detect_language_from_url(url)
                        return {
                            'type': content_type,
                            'content': content[:5000],  # Limit content size
                            'title': title,
                            'language': language
                        }
                    
                    # Check if it's code content
                    if data.get('code', {}).get('language'):
                        return {
                            'type': 'code',
                            'content': content[:5000],  # Limit content size
                            'title': title,
                            'language': data['code']['language']
                        }
                    
                    # Add extra metadata if available
                    result = {
                        'type': content_type,
                        'content': content[:5000],  # Limit content size
                        'title': title
                    }
                    
                    # Add useful metadata
                    if 'description' in data:
                        result['description'] = data['description']
                    if 'images' in data:
                        result['images'] = data['images'][:5]  # Limit images
                    if 'links' in data:
                        result['links'] = data['links'][:10]  # Limit links
                    
                    return result
                else:
                    print(f"Jina reader returned status {response.status} for {url}")
                    return None
        
        except asyncio.TimeoutError:
            print(f"[ERROR] URL fetch timed out after 20 seconds for {url}")
            return None
        except Exception as e:
            print(f"Error fetching URL with Jina: {e}")
            return None

async def fetch_url_content(url):
    """Fetch and parse content from any URL using Jina.ai"""
    return await fetch_url_with_jina(url)

async def search_technical_docs(query):
    """Search technical documentation using Jina with site-specific queries"""
    tech_queries = [
        f"{query} site:stackoverflow.com",
        f"{query} site:docs.python.org OR site:github.com",
        f"{query} programming documentation tutorial"
    ]
    
    # Try the first query that returns results
    for tech_query in tech_queries[:2]:
        results = await search_with_jina(tech_query, num_results=5)
        if results:
            return results
    
    # Fallback to general technical search
    return await search_with_jina(f"{query} programming solution", num_results=5)

def format_code_blocks(text):
    """Format code blocks properly for Matrix markdown"""
    # Pattern to match code blocks with language hints
    code_block_pattern = r'```(\w*)\n(.*?)```'
    
    def replace_code_block(match):
        language = match.group(1) or 'text'
        code = match.group(2)
        return f'<pre><code class="language-{language}">{html.escape(code)}</code></pre>'
    
    # Replace code blocks
    formatted = re.sub(code_block_pattern, replace_code_block, text, flags=re.DOTALL)
    
    # Handle inline code
    inline_pattern = r'`([^`]+)`'
    formatted = re.sub(inline_pattern, r'<code>\1</code>', formatted)
    
    return formatted

def extract_code_from_response(response):
    """Extract code blocks from LLM response and format them separately"""
    parts = []
    current_pos = 0
    
    # Find all code blocks
    code_block_pattern = r'```(\w*)\n(.*?)```'
    
    for match in re.finditer(code_block_pattern, response, re.DOTALL):
        # Add text before code block
        if match.start() > current_pos:
            text_part = response[current_pos:match.start()].strip()
            if text_part:
                parts.append({
                    'type': 'text',
                    'content': text_part
                })
        
        # Add code block
        language = match.group(1) or 'text'
        code = match.group(2).strip()
        
        parts.append({
            'type': 'code',
            'language': language,
            'content': code
        })
        
        current_pos = match.end()
    
    # Add remaining text
    if current_pos < len(response):
        text_part = response[current_pos:].strip()
        if text_part:
            parts.append({
                'type': 'text',
                'content': text_part
            })
    
    return parts if parts else [{'type': 'text', 'content': response}]

def create_comprehensive_summary(room_id, minutes=30):
    """Create a detailed summary of room activity"""
    if room_id not in room_message_history:
        return "No recent messages in this room."
    
    messages = list(room_message_history[room_id])
    if not messages:
        return "No recent messages to summarize."
    
    # Get context analysis
    context = conversation_context.get_room_context(room_id)
    
    # Filter messages from the last N minutes
    cutoff_time = datetime.now().timestamp() - (minutes * 60)
    recent_messages = [msg for msg in messages if msg['timestamp'] > cutoff_time]
    
    if not recent_messages:
        return f"No messages in the last {minutes} minutes."
    
    # Create comprehensive summary
    summary = f"📊 **Comprehensive Chat Analysis (last {minutes} minutes)**\n\n"
    
    # Participants and activity
    participants = list(set(get_display_name(msg['sender']) for msg in recent_messages))
    summary += f"**Active Participants** ({len(participants)}): {', '.join(participants)}\n"
    summary += f"**Total Messages**: {len(recent_messages)}\n\n"
    
    # Main topics if available
    if context and context['top_topics']:
        summary += "**🔥 Hot Topics**:\n"
        for topic, score in context['top_topics'][:5]:
            summary += f"  • {topic.capitalize()} (relevance: {score:.1f})\n"
        summary += "\n"
    
    # User expertise
    if context and context['user_expertise']:
        summary += "**👥 User Interests/Expertise**:\n"
        for user, interests in list(context['user_expertise'].items())[:5]:
            summary += f"  • {user}: {', '.join(interests)}\n"
        summary += "\n"
    
    # Important messages
    if context and context['recent_important']:
        summary += "**⚡ Important Messages**:\n"
        for imp_msg in context['recent_important'][-5:]:
            msg_type = imp_msg['type']
            sender = get_display_name(imp_msg['sender'])
            body_preview = imp_msg['body'][:100] + "..." if len(imp_msg['body']) > 100 else imp_msg['body']
            summary += f"  • [{msg_type}] {sender}: {body_preview}\n"
        summary += "\n"
    
    # Conversation flow
    if context:
        flow = context['conversation_flow']
        flow_description = {
            'very_active': '🔥 Very Active',
            'active': '💬 Active',
            'moderate': '🗨️ Moderate',
            'slow': '🐌 Slow',
            'very_active_dialogue': '🔥 Intense Dialogue',
            'active_dialogue': '💬 Active Dialogue',
            'very_active_group_discussion': '🔥 Heated Group Discussion',
            'active_group_discussion': '💬 Active Group Discussion'
        }
        summary += f"**Conversation Style**: {flow_description.get(flow, flow)}\n"
    
    return summary

def detect_code_in_message(message):
    """Detect if message contains code blocks or code-related content"""
    code_indicators = [
        '```',  # Code blocks
        '`',    # Inline code
        'def ', 'class ', 'function ', 'const ', 'let ', 'var ',  # Common keywords
        'import ', 'from ', 'export ',
        '()', '=>', '{}', '[]',  # Common syntax
        'error:', 'exception:', 'traceback:',  # Error messages
    ]
    return any(indicator in message.lower() for indicator in code_indicators)

async def needs_web_search(prompt, room_context=None):
    """Determine if a query actually needs web search for current information"""
    prompt_lower = prompt.lower()
    
    # Keywords that strongly indicate need for current/latest information
    current_info_indicators = [
        'latest', 'current', 'today', 'yesterday', 'this week', 'recent',
        'news', 'headlines', 'update', 'breaking', 'announced',
        'price', 'stock', 'weather', 'score', 'results',
        'released', 'launched', 'published', 'version',
        'status', 'outage', 'down', 'working'
    ]
    
    # Check if asking about current events or real-time data
    needs_current_info = any(indicator in prompt_lower for indicator in current_info_indicators)
    
    # Check for questions about specific current entities (companies, people, events)
    entity_patterns = [
        r'what.{0,10}happening with',
        r'how is .{0,20} doing',
        r'status of',
        r'news about',
        r'updates? on',
        r'latest.{0,10}from'
    ]
    has_entity_query = any(re.search(pattern, prompt_lower) for pattern in entity_patterns)
    
    # Don't search for:
    # - General knowledge questions (unless they need current info)
    # - Programming/technical questions (unless about latest versions)
    # - Philosophical or opinion questions
    # - Questions about the bot itself
    
    no_search_patterns = [
        r'what is (?:a|an|the)? (?:function|variable|loop|class)',  # Basic programming concepts
        r'how (?:do|does|to) .{0,20} work',  # How things work in general
        r'why (?:is|are|do|does)',  # Why questions rarely need current info
        r'explain',  # Explanations don't need current info
        r'define',  # Definitions are static
        r'who (?:is|are) you',  # Bot identity questions
        r'what (?:is|are) you',  # Bot identity questions
        r'help me with',  # Help requests usually don't need web search
        r'debug',  # Debugging help
        r'fix',  # Fix requests
    ]
    
    is_general_knowledge = any(re.search(pattern, prompt_lower) for pattern in no_search_patterns)
    
    # Special case: version-specific technical questions might need search
    if 'latest version' in prompt_lower or 'new features' in prompt_lower:
        return True
    
    # Only search if we need current info AND it's not a general knowledge question
    should_search = (needs_current_info or has_entity_query) and not is_general_knowledge
    
    if should_search:
        print(f"[DEBUG] Web search needed - Current info: {needs_current_info}, Entity query: {has_entity_query}")
    
    return should_search

def summarize_search_results(results, query):
    """Create a better summary of search results"""
    if not results:
        return None
    
    summary = f"🔍 Search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        summary += f"**{i}. {result['title']}**\n"
        summary += f"   {result['snippet']}\n"
        if result.get('url'):
            summary += f"   🔗 {result.get('url', '')}\n"
        summary += "\n"
    
    return summary

async def send_formatted_message(room_id, reply_text):
    """Send a message with proper code formatting"""
    # Extract code blocks and text parts
    message_parts = extract_code_from_response(reply_text)
    
    # Build formatted body
    formatted_body = ""
    plain_body = ""
    
    for part in message_parts:
        if part['type'] == 'text':
            # Format text with inline code support
            text = part['content']
            formatted_text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
            formatted_body += f"<p>{formatted_text}</p>"
            plain_body += text + "\n\n"
            
        elif part['type'] == 'code':
            language = part['language']
            code = html.escape(part['content'])
            
            # Add language label if specified
            if language and language != 'text':
                formatted_body += f"<p><strong>{language}:</strong></p>"
                plain_body += f"{language}:\n"
            
            formatted_body += f'<pre><code class="language-{language}">{code}</code></pre>'
            plain_body += f"```{language}\n{part['content']}\n```\n\n"
    
    # Send formatted message
    content = {
        "msgtype": "m.text",
        "body": plain_body.strip(),
        "format": "org.matrix.custom.html",
        "formatted_body": formatted_body.strip()
    }
    
    return await client.room_send(
        room_id=room_id,
        message_type="m.room.message",
        content=content
    )

async def maybe_react(room_id, event_id, message):
    """Send reaction to message if it contains trigger words"""
    message_lower = message.lower()
    
    for trigger, reactions in REACTION_TRIGGERS.items():
        if trigger in message_lower:
            # Different reaction chances for different triggers
            reaction_chances = {
                'uncle cmos': 0.7,  # Always react to uncle cmos
                'based': 0.4,
                'cringe': 0.5,
                'lol': 0.2,
                'lmao': 0.2,
                'wtf': 0.3,
                'monero': 0.5,
                'good morning': 0.6,
                'good night': 0.6,
            }
            
            chance = reaction_chances.get(trigger, 0.3)  # Default 30% chance
            
            if random.random() < chance:
                reaction = random.choice(reactions)
                try:
                    await client.room_send(
                        room_id=room_id,
                        message_type="m.reaction",
                        content={
                            "m.relates_to": {
                                "rel_type": "m.annotation",
                                "event_id": event_id,
                                "key": reaction
                            }
                        }
                    )
                except Exception as e:
                    print(f"Error sending reaction: {e}")
                break  # Only one reaction per message

async def get_llm_reply_with_retry(prompt, context=None, previous_message=None, room_id=None, url_contents=None):
    """Wrapper with retry logic and exponential backoff"""
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            return await get_llm_reply(prompt, context, previous_message, room_id, url_contents)
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                print(f"[ERROR] All retry attempts failed due to timeout")
                return "Damn, the AI servers are timing out hard rn. Maybe try again in a minute? 💀"
            
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            print(f"[RETRY] Attempt {attempt + 1} timed out, retrying in {delay}s...")
            await asyncio.sleep(delay)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] All retry attempts failed: {e}")
                return "Yo, the AI servers are really struggling rn. Maybe try again in a minute? 🔧"
            
            delay = base_delay * (2 ** attempt)
            print(f"[RETRY] Attempt {attempt + 1} failed, retrying in {delay}s...")
            await asyncio.sleep(delay)

async def get_llm_reply(prompt, context=None, previous_message=None, room_id=None, url_contents=None):
    # Get comprehensive room context - REDUCED lookback
    room_context = None
    if room_id:
        room_context = conversation_context.get_room_context(room_id, lookback_messages=30)  # Reduced from 100
    
    # Build context-aware system prompt
    system_prompt = BOT_PERSONALITY
    
    # Add rich context to system prompt
    if room_context:
        system_prompt += f"\n\n**ROOM CONTEXT**:\n"
        
        # Add topic context
        if room_context['top_topics']:
            topics_str = ', '.join([f"{topic} ({score:.1f})" for topic, score in room_context['top_topics'][:3]])
            system_prompt += f"Current hot topics: {topics_str}\n"
        
        # Add user expertise context
        if room_context['user_expertise']:
            system_prompt += "User expertise in the room:\n"
            for user, interests in list(room_context['user_expertise'].items())[:5]:
                system_prompt += f"  - {user}: {', '.join(interests)}\n"
        
        # Add conversation flow
        flow = room_context['conversation_flow']
        if 'dialogue' in flow:
            system_prompt += "This is a focused dialogue between two people. Be direct and helpful.\n"
        elif 'group_discussion' in flow:
            system_prompt += "This is a group discussion. Consider multiple perspectives.\n"
        
        if 'very_active' in flow:
            system_prompt += "The conversation is very active. Keep responses concise and on-point.\n"
        
        # Add recent important messages for context
        if room_context['recent_important']:
            system_prompt += "\nRecent important points:\n"
            for imp in room_context['recent_important'][-3:]:
                system_prompt += f"  - [{imp['type']}] {get_display_name(imp['sender'])}: {imp['body'][:100]}...\n"
    
    # Add URL content handling to system prompt
    if url_contents:
        system_prompt += "\n\nIMPORTANT: The user has shared URLs with content. Analyze and discuss the content thoroughly, providing insights, explanations, or help based on what you read."
    
    # Check if user is asking about the bot
    about_nifty = any(keyword in prompt.lower() for keyword in ['who are you', 'what are you', 'your name', 'who is nifty', 'what is nifty'])
    
    if about_nifty:
        prompt = f"{prompt}\n\n[Remember: You are Nifty, a Matrix bot with the handle @nifty:matrix.stargazypie.xyz. Be self-aware about your identity.]"
    
    # Check for technical questions or code
    is_technical = detect_code_in_message(prompt) or any(keyword in prompt.lower() for keyword in [
        'code', 'programming', 'debug', 'error', 'function', 'script', 'compile',
        'python', 'javascript', 'rust', 'linux', 'bash', 'git', 'docker',
        'api', 'database', 'sql', 'server', 'network', 'security'
    ])
    
    # Check for chat summary request
    summary_keywords = ['summary', 'summarize', 'recap', 'what happened', 'what was discussed', 'catch me up', 'tldr']
    wants_summary = any(keyword in prompt.lower() for keyword in summary_keywords)
    
    if wants_summary and room_id:
        # Extract time frame if specified
        minutes = 30  # default
        time_patterns = [
            (r'last (\d+) minutes?', 1),
            (r'past (\d+) minutes?', 1),
            (r'last (\d+) hours?', 60),
            (r'past (\d+) hours?', 60)
        ]
        
        for pattern, multiplier in time_patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                minutes = int(match.group(1)) * multiplier
                break
        
        chat_summary = create_comprehensive_summary(room_id, minutes)
        
        # Get LLM to enhance the summary
        enhanced_prompt = f"""The user asked: {prompt}

Here's the comprehensive analysis:
{chat_summary}

Based on this data, provide a natural, conversational summary. Focus on:
1. Key discussion points and decisions made
2. Questions that were asked and whether they were answered
3. Any action items or next steps mentioned
4. The overall mood and flow of the conversation

Keep your personality but be informative. Remember you are Nifty."""
        
        prompt = enhanced_prompt
    
    # Filter the incoming prompt
    filtered_prompt = filter_bot_triggers(prompt)
    
    # If URL contents provided, add them to the prompt (with size limits)
    if url_contents:
        url_summary = "\n\n[SHARED CONTENT]:\n"
        for content in url_contents[:3]:  # Limit to 3 URLs
            url_summary += f"\nFrom {content['title']}:\n"
            
            if content['type'] == 'code':
                url_summary += f"Programming language: {content.get('language', 'unknown')}\n"
                url_summary += f"Code content:\n{content['content'][:2000]}\n"  # Limit content size
            else:
                url_summary += f"Content:\n{content['content'][:2000]}\n"  # Limit content size
        
        filtered_prompt = filtered_prompt + url_summary
    
    # Add timeout to the session
    timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Smart web search detection
        should_search = False
        if not wants_summary and not about_nifty and not url_contents:
            should_search = await needs_web_search(filtered_prompt, room_context)
        
        # Search for technical docs if it's a technical question
        if is_technical and should_search and not url_contents:
            # Extract search query
            query = filtered_prompt
            for keyword in ['search for', 'look up', 'find', 'tell me about', 'what is', 'who is', 'how to', 'how do i']:
                if keyword in query.lower():
                    query = query.lower().split(keyword)[-1].strip()
                    break
            
            # Search technical resources
            results = await search_technical_docs(query)
            
            if results:
                search_summary = summarize_search_results(results, query)
                
                # Enhanced prompt for technical answers
                enhanced_prompt = f"""User asked a technical question: {filtered_prompt}

{search_summary}

Based on these search results, provide a comprehensive technical answer. Include:
1. Direct solution to their problem
2. Code examples if applicable (properly formatted)
3. Best practices and common pitfalls
4. Alternative approaches if relevant

Remember to maintain your personality while being technically accurate and helpful. You are Nifty, a skilled technical expert."""
                
                filtered_prompt = enhanced_prompt
                
        elif should_search:
            # Regular search for non-technical queries
            query = filtered_prompt
            for keyword in ['search for', 'look up', 'find', 'tell me about', 'what is', 'who is']:
                if keyword in query.lower():
                    query = query.lower().split(keyword)[-1].strip()
                    break
            
            # Use Jina search
            results = await search_with_jina(query)
            
            if results:
                search_summary = summarize_search_results(results, query)
                
                # Enhanced prompt for better summarization
                enhanced_prompt = f"""User asked: {filtered_prompt}

{search_summary}

Based on these search results, provide a comprehensive but concise answer. Focus on:
1. Directly answering the user's question
2. Highlighting the most important/relevant information
3. Mentioning any interesting or surprising facts
4. Being accurate while maintaining your personality

Remember you are Nifty, be aware of your identity."""
                
                filtered_prompt = enhanced_prompt
            else:
                filtered_prompt += "\n\n(Note: I couldn't retrieve web search results for this query, so I'll provide information based on my training data.)"
        
        # Build messages with context
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add technical context if needed
        if is_technical:
            messages[0]["content"] += "\n\nThe user has a technical question. Provide detailed, accurate technical help with code examples when appropriate. Be thorough but concise."
        
        # Add conversation history for better context - REDUCED
        if room_id and room_id in room_message_history:
            # Get last 10 messages for context (reduced from 20)
            recent_messages = list(room_message_history[room_id])[-10:]
            if len(recent_messages) > 3:
                context_messages = []
                for msg in recent_messages[:-1]:  # Exclude the current message
                    role = "assistant" if msg['sender'] == client.user_id else "user"
                    # Truncate long messages in context
                    msg_content = msg['body'][:200] if len(msg['body']) > 200 else msg['body']
                    context_messages.append({
                        "role": role,
                        "content": f"{get_display_name(msg['sender'])}: {msg_content}"
                    })
                
                # Add only last 5 context messages (reduced from 10)
                messages.extend(context_messages[-5:])
        
        # Add previous message context if this is a reply
        if previous_message:
            messages.append({"role": "assistant", "content": f"[Previous message I sent]: {previous_message}"})
            messages.append({"role": "user", "content": f"[User is replying to the above message]: {filtered_prompt}"})
        else:
            messages.append({"role": "user", "content": filtered_prompt})
        
        # Adjust temperature based on context
        temperature = 0.7 if is_technical else 0.8
        
        data = {
            "model": settings['llm_model'],  # Use model from settings
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000  # Reduced from 2000 to 1000
        }
        
        try:
            async with session.post(OPENROUTER_URL, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    reply = result["choices"][0]["message"]["content"]
                    
                    # Filter the response
                    filtered_reply = filter_bot_triggers(reply)
                    
                    return filtered_reply
                else:
                    error_text = await response.text()
                    print(f"OpenRouter API error: {response.status} - {error_text}")
                    return f"Hey, I'm Nifty and I hit a snag (error {response.status}). Mind trying again? 🔧"
        
        except asyncio.TimeoutError:
            print(f"[ERROR] LLM request timed out after 30 seconds")
            return "Yo, the AI servers are being slow af rn. Try again in a sec? 🔧"
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            return f"Hmm, Nifty here - something went wonky on my end! Could you try that again? 🤔"

async def get_replied_to_event(room_id, reply_to_event_id):
    """Fetch the event that was replied to"""
    try:
        response = await client.room_get_event(room_id, reply_to_event_id)
        return response
    except Exception as e:
        print(f"Error fetching replied-to event: {e}")
        return None

async def cleanup_old_context():
    """Periodic cleanup of old context data"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        
        # Clean up old conversation context
        for room_id in list(conversation_context.topics.keys()):
            # Decay all topic scores
            for topic in conversation_context.topics[room_id]:
                conversation_context.topics[room_id][topic] *= 0.5
            
            # Remove topics with very low scores
            conversation_context.topics[room_id] = {
                topic: score 
                for topic, score in conversation_context.topics[room_id].items() 
                if score > 0.1
            }
        
        # Clear very old important messages
        for room_id in conversation_context.important_messages:
            cutoff_time = datetime.now().timestamp() - (24 * 3600)  # 24 hours
            conversation_context.important_messages[room_id] = [
                msg for msg in conversation_context.important_messages[room_id]
                if msg['timestamp'] > cutoff_time
            ]
        
        print(f"[CLEANUP] Context cleanup completed at {datetime.now()}")

async def process_message_request(room: MatrixRoom, event: RoomMessageText):
    """Process a message request from the queue"""
    # This is where the actual message processing happens
    # Moved from message_callback to avoid queue blocking
    pass  # The actual processing is still in message_callback for now

async def invite_callback(room: MatrixRoom, event: InviteMemberEvent):
    """Handle room invites"""
    print(f"[INVITE] Received invite to room {room.room_id} from {event.sender}")
    
    # Only process invites for our user
    if event.state_key != client.user_id:
        return
    
    # Accept the invite
    print(f"[INVITE] Accepting invite to room {room.room_id}")
    result = await client.join(room.room_id)
    
    if hasattr(result, 'room_id'):
        print(f"[INVITE] Successfully joined room {room.room_id}")
        joined_rooms.add(room.room_id)
        
        # Send a greeting message
        await client.room_send(
            room_id=room.room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": "Hey! I'm Nifty! 👋 Thanks for inviting me! Just say 'nifty' followed by your message to chat, or reply to any of my messages! 🚀\n\nI specialize in:\n• 💻 Programming & debugging\n• 🐧 Linux/Unix systems\n• 🌐 Web dev & networking\n• 🔒 Security & cryptography\n• 🤖 General tech support\n• 📱 Mobile dev tips\n• 🎮 Gaming & internet culture\n\nCommands:\n• `nifty !reset` - Clear my context\n• `nifty summary` - Get a detailed chat analysis\n• Share URLs and I'll read and analyze them!\n• `?set <model>` - Change LLM model (admin only)\n• `?set list` - Show current LLM model\n\nI also react to messages with emojis when appropriate! 😊 Let's build something cool! 💪"
            }
        )
    else:
        print(f"[INVITE] Failed to join room: {result}")

async def message_callback(room: MatrixRoom, event: RoomMessageText):
    global client, settings  # Use global client and settings
    
    print(f"[DEBUG] Message callback triggered!")
    print(f"[DEBUG] Room ID: {room.room_id}")
    print(f"[DEBUG] Sender: {event.sender}, Bot ID: {client.user_id}")
    print(f"[DEBUG] Message: {event.body}")
    
    # Ignore our own messages
    if event.sender == client.user_id:
        print("[DEBUG] Ignoring own message")
        return
    
    # Ignore messages from other known bots
    known_bots = ['@kyoko:xmr.mx']
    if any(event.sender.startswith(bot) for bot in known_bots):
        print("[DEBUG] Ignoring message from another bot")
        return
    
    # Check for setting commands (before history tracking)
    if event.body.startswith("?set "):
        command = event.body[5:].strip()
        if command == "list":
            await client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": f"Current LLM model: {settings['llm_model']}"
                }
            )
            return
        else:
            # Set new model
            settings['llm_model'] = command
            with open(settings_file, "w") as f:
                json.dump(settings, f)
            await client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": f"LLM model updated to: {settings['llm_model']}"
                }
            )
            return
    
    # Store this message in room history
    message_data = {
        'sender': event.sender,
        'body': event.body,
        'timestamp': event.server_timestamp / 1000 if event.server_timestamp else datetime.now().timestamp(),
        'event_id': event.event_id
    }
    room_message_history[room.room_id].append(message_data)
    
    # Update conversation context
    conversation_context.update_context(room.room_id, message_data)
    
    # Maybe react to the message
    await maybe_react(room.room_id, event.event_id, event.body)
    
    # Check if this is a reply to a message
    is_reply = False
    replied_to_bot = False
    previous_message = None
    
    if hasattr(event, 'source') and event.source.get('content', {}).get('m.relates_to', {}).get('m.in_reply_to'):
        is_reply = True
        reply_to_event_id = event.source['content']['m.relates_to']['m.in_reply_to']['event_id']
        print(f"[DEBUG] This is a reply to event: {reply_to_event_id}")
        
        # Fetch the replied-to event
        replied_event = await get_replied_to_event(room.room_id, reply_to_event_id)
        if replied_event and hasattr(replied_event, 'event'):
            replied_sender = replied_event.event.sender
            if replied_sender == client.user_id:
                replied_to_bot = True
                previous_message = replied_event.event.body
                print(f"[DEBUG] User is replying to bot's message: {previous_message[:50]}...")
    
    # Check for direct mention or reply
    should_respond = "nifty" in event.body.lower() or replied_to_bot
    
    if should_respond:
        # Try to add to queue (non-blocking) to prevent overload
        try:
            request_queue.put_nowait({
                'room_id': room.room_id,
                'event': event,
                'timestamp': datetime.now()
            })
        except QueueFull:
            await client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": "Yo I'm getting slammed with requests rn, gimme a sec! 😅"
                }
            )
            return
        finally:
            # Clear the queue item after processing
            try:
                request_queue.get_nowait()
            except:
                pass
        
        # Check for reset command
        if "!reset" in event.body.lower():
            # Clear the room's message history
            room_message_history[room.room_id].clear()
            
            await client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": "✨ Nifty's context cleared! Fresh start! 🧹"
                }
            )
            return
        
        prompt = event.body
        
        # Extract URLs from message
        urls_in_message = extract_urls_from_message(event.body)
        url_contents = []
        
        if urls_in_message:
            print(f"[DEBUG] Found URLs in message: {urls_in_message}")
            
            # Send typing notification while fetching URLs
            await client.room_typing(room.room_id, True)
            
            # Process URLs with timeout
            for url in urls_in_message[:3]:  # Limit to first 3 URLs
                content = await fetch_url_content(url)
                if content:
                    url_contents.append(content)
        
        print(f"Sending prompt to LLM: {prompt}")
        if replied_to_bot:
            print(f"With context from previous bot message")
        if url_contents:
            print(f"With {len(url_contents)} URL contents")
        
        # Send typing notification
        await client.room_typing(room.room_id, True)
        
        try:
            # Use the retry wrapper instead of direct call
            reply = await get_llm_reply_with_retry(
                prompt=prompt,
                previous_message=previous_message,
                room_id=room.room_id,
                url_contents=url_contents
            )
            
            # Final safety check
            if any(word.lower() in reply.lower() for word in FILTERED_WORDS):
                print("[WARNING] Reply contained filtered word, filtering again")
                reply = filter_bot_triggers(reply)
            
            print(f"LLM Reply (filtered): {reply}")
            
            # Stop typing notification
            await client.room_typing(room.room_id, False)
            
            # Send the formatted response
            response = await send_formatted_message(room.room_id, reply)
            
            # Store bot's message in history
            if hasattr(response, 'event_id'):
                bot_message_data = {
                    'sender': client.user_id,
                    'body': reply,
                    'timestamp': datetime.now().timestamp(),
                    'event_id': response.event_id
                }
                room_message_history[room.room_id].append(bot_message_data)
                
                # Update context with bot's message
                conversation_context.update_context(room.room_id, bot_message_data)
            
            print("[DEBUG] Message sent successfully!")
        except Exception as e:
            print(f"Error sending message: {e}")
            await client.room_typing(room.room_id, False)
            
            # Send error message
            await client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": "Oops, something went wrong! Try again maybe? 🤷"
                }
            )
    else:
        if is_reply:
            print("Ignoring reply (not to bot's message)")
        else:
            print("Ignoring message (doesn't contain 'nifty')")

async def main():
    global client
    client = AsyncClient(HOMESERVER, USERNAME)
    
    # Login
    response = await client.login(PASSWORD)
    if not isinstance(response, LoginResponse):
        print(f"Failed to login: {response}")
        return
    
    print(f"Logged in as {client.user_id}")
    
    # Get list of joined rooms
    print("Getting list of joined rooms...")
    joined_rooms_response = await client.joined_rooms()
    if hasattr(joined_rooms_response, 'rooms'):
        for room_id in joined_rooms_response.rooms:
            joined_rooms.add(room_id)
            print(f"Already in room: {room_id}")
    
    # Add event callbacks
    client.add_event_callback(message_callback, RoomMessageText)
    client.add_event_callback(invite_callback, InviteMemberEvent)
    
    # Do an initial sync to get the latest state
    print("Performing initial sync...")
    sync_response = await client.sync(timeout=30000, full_state=True)
    print(f"Initial sync completed. Next batch: {sync_response.next_batch}")
    
    # Start cleanup task
    asyncio.create_task(cleanup_old_context())
    
    print("=" * 50)
    print("🤖 Nifty Bot is running!")
    print("=" * 50)
    print("✅ Identity: @nifty:matrix.stargazypie.xyz")
    print("✅ Listening for messages in all joined rooms")
    print("✅ Auto-accepting room invites")
    print("📝 Trigger: Say 'nifty' anywhere in a message")
    print("💬 Or reply directly to any of my messages")
    print("❌ Random responses: DISABLED")
    print("👀 Emoji reactions: ENABLED (various triggers)")
    print("🧹 Reset: 'nifty !reset' to clear context")
    print("📊 Summary: 'nifty summary' for comprehensive chat analysis")
    print("🧠 Optimized Context: Tracking 100 messages (reduced for performance)")
    print("📈 Context Features: Topic tracking, user expertise, important messages")
    print("💻 Technical expertise: Programming, Linux, Security, etc.")
    print("🔗 URL Analysis: Share URLs and I'll read and discuss them!")
    print("📝 Code Formatting: Proper syntax highlighting for all languages")
    print(f"🚫 Filtering words: {', '.join(FILTERED_WORDS)}")
    print("🔍 Web search: Powered by Jina.ai - Smart detection for current info")
    print("🎯 Personality: Professional, helpful, witty, context-aware")
    print("⏱️ Timeouts: 30s for LLM, 15s for search, 20s for URL fetching")
    print("🔄 Retry logic: 3 attempts with exponential backoff")
    print("🧹 Auto-cleanup: Hourly context cleanup to maintain performance")
    print("📉 Reduced context: Optimized for faster response times")
    print("🔄 LLM Model Switching: ?set <model> / ?set list")
    print("=" * 50)
    
    # Sync forever
    try:
        await client.sync_forever(timeout=30000, full_state=False)
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt - shutting down...")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
