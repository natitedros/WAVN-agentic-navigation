"""
Messages and the message bus.

Every message has the same envelope as the format we defined (message_id, task_id,
reply_to, type, sender, receiver, payload). The bus stands in for the network:
agents never call each other directly, they only send messages. Later, the bus
can be replaced by ROS 2 topics without changing the agents' logic.
"""
import itertools
from collections import deque
from dataclasses import dataclass, field

_next_id = itertools.count(1)


@dataclass
class Message:
    type: str
    sender: str
    receiver: str                  # a robot name, "path_finder", or "all"
    payload: dict
    task_id: str | None = None
    reply_to: str | None = None
    message_id: str = field(default_factory=lambda: f"msg-{next(_next_id):03d}")


class MessageBus:
    """Delivers messages one at a time, in the order they were sent."""

    def __init__(self, show=True):
        self.agents = {}
        self.queue = deque()
        self.log = []          # every message ever sent (useful for the paper's examples)
        self.show = show

    def register(self, agent):
        self.agents[agent.name] = agent
        agent.bus = self

    def send(self, msg: Message):
        self.queue.append(msg)

    def run(self, max_messages=500):
        delivered = 0
        while self.queue and delivered < max_messages:
            msg = self.queue.popleft()
            self.log.append(msg)
            if self.show:
                print(f"  {msg.message_id}  {msg.sender:>11} -> {msg.receiver:<11} {msg.type:<22} {summary(msg)}")
            if msg.receiver == "all":
                targets = [a for name, a in self.agents.items() if name not in (msg.sender, "path_finder")]
            else:
                targets = [self.agents[msg.receiver]]
            for agent in targets:
                agent.handle(msg)
            delivered += 1


class Agent:
    """Base class: an agent has a name, receives messages and sends messages."""

    def __init__(self, name):
        self.name = name
        self.bus = None

    def send(self, type, receiver, payload, task_id=None, reply_to=None):
        msg = Message(type, self.name, receiver, payload, task_id, reply_to)
        self.bus.send(msg)
        return msg

    def handle(self, msg: Message):
        # Route each message type to a method called on_<type>, if the agent has one.
        method = getattr(self, f"on_{msg.type}", None)
        if method:
            method(msg)

    def think(self, text):
        # Prints the agent's reasoning, indented under the message that caused it.
        print(f"{'':16}[{self.name}] {text}")


def summary(msg: Message) -> str:
    """One short line describing a message, for the printed log."""
    p = msg.payload
    if msg.type == "homing_request":
        return f'target="{p["target_description"]}"'
    if msg.type == "observation_report":
        return ", ".join(f'"{o["description"]}"' for o in p["objects"])
    if msg.type == "clarification_question":
        return f'{p["question"]}  options={p["about_object_ids"]}'
    if msg.type == "clarification_answer":
        return f'picked={p["selected_object_id"]} ({p["confidence"]:.2f})'
    if msg.type == "route_instruction":
        return " -> ".join(f'{h["via_robot"]}:"{h["landmark_description"]}"' for h in p["hops"])
    if msg.type == "progress_update":
        return f'hop {p["hop"]}: {p["status"]}'
    return ""
