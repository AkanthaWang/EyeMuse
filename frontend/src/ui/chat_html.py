from __future__ import annotations

from html import escape
from typing import Iterable


def _items(items: Iterable[object]) -> list[object]:
    return list(items)


def build_companion_chat_html(conversation_items: Iterable[object]) -> str:
    items = [
        item
        for item in _items(conversation_items)
        if getattr(item, "role", "") == "user"
    ]
    if not items:
        return (
            "<p style='color:#6E8DB8; text-align:center; margin:8px 0;'>"
            "输入消息后，EyeMuse 会在上方气泡回复"
            "</p>"
        )

    parts: list[str] = []
    for item in items[-6:]:
        text = escape(str(getattr(item, "text", ""))).replace("\n", "<br>")
        align = "right"
        bubble = "background: linear-gradient(135deg, rgba(210, 235, 255, 0.92), rgba(165, 211, 255, 0.88)); border: 1px solid rgba(255, 255, 255, 0.90); color: #164A83;"
        parts.append(
            "<div style='margin:7px 0; text-align:%s;'>"
            "<span style='display:inline-block; max-width:92%%; padding:8px 11px; border-radius:14px; %s "
            "box-shadow:0 10px 20px rgba(131,176,228,0.16); font-size:12px; line-height:1.5;'>%s</span>"
            "</div>" % (align, bubble, text)
        )
    return "".join(parts)


def build_main_conversation_html(conversation_items: Iterable[object]) -> str:
    items = _items(conversation_items)
    if not items:
        return "<p style='color:#6E8DB8;'>暂无消息</p>"

    parts: list[str] = []
    for item in items[-40:]:
        role = getattr(item, "role", "")
        text = escape(str(getattr(item, "text", ""))).replace("\n", "<br>")
        timestamp = escape(str(getattr(item, "timestamp", "")))
        if role == "user":
            bubble = "background: linear-gradient(135deg, rgba(137, 198, 255, 0.82), rgba(93, 153, 245, 0.86)); border: 1px solid rgba(255, 255, 255, 0.84);"
            title = "你"
        elif role == "eyeMuse":
            bubble = "background: linear-gradient(135deg, rgba(228, 242, 255, 0.78), rgba(197, 222, 255, 0.72)); border: 1px solid rgba(255, 255, 255, 0.84);"
            title = "EyeMuse"
        else:
            bubble = "background: rgba(233, 246, 255, 0.62); border: 1px solid rgba(255, 255, 255, 0.82);"
            title = "系统"
        parts.append(
            f"<div style='margin:10px 0; padding:12px 14px; border-radius:16px; {bubble} color:#355E96;'>"
            f"<div style='font-size:11px; color:#6E8DB8; margin-bottom:6px;'>{title} · {timestamp}</div>"
            f"<div style='font-size:14px; line-height:1.6;'>{text}</div>"
            "</div>"
        )
    return "".join(parts)
