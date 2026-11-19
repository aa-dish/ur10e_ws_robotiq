"""Natural language -> robot command, then command -> concrete detected object(s).

Two parsers are chained:
  1. Ollama (local LLM).  It sees the list of objects the camera currently detects and
     answers with JSON (action, which object ids, where to put them).
  2. A rule-based parser that works without any LLM (keywords, colours, left/right ...).
The LLM output is *never trusted blindly*: object ids are validated against the scene,
and anything unusable falls back to the rules.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..vision.classifier import Catalog
from ..vision.detector import Detection
from .ollama_client import OllamaClient

log = logging.getLogger(__name__)

ACTIONS = ['pick_place', 'home', 'list', 'status', 'stop', 'help', 'unknown']
PLACE_WORDS = {
    'bin': ['bin', 'trash', 'trash bin', 'garbage', 'rubbish', 'waste', 'basket', 'bucket', 'dustbin'],
    'table_left': ['left side', 'left of the table', 'to the left', 'on the left', 'left'],
    'table_right': ['right side', 'right of the table', 'to the right', 'on the right', 'right'],
    'table_front': ['front', 'near me', 'closer', 'near side'],
    'table_back': ['back', 'far side', 'far end', 'behind'],
    'table_center': ['center', 'centre', 'middle'],
}
COLOURS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'white', 'black', 'gray', 'grey']
SUPERLATIVES = {
    'tallest': ('height', -1), 'highest': ('height', -1), 'shortest': ('height', +1), 'lowest': ('height', +1),
    'biggest': ('area_m2', -1), 'largest': ('area_m2', -1), 'smallest': ('area_m2', +1), 'tiniest': ('area_m2', +1),
    'longest': ('long', -1), 'widest': ('short', -1), 'thinnest': ('short', +1),
    'leftmost': ('x', +1), 'rightmost': ('x', -1), 'closest': ('dist', +1), 'nearest': ('dist', +1),
    'farthest': ('dist', -1), 'furthest': ('dist', -1), 'frontmost': ('front', +1), 'backmost': ('front', -1),
}


@dataclass
class Command:
    action: str = 'unknown'
    object_phrase: str = ''
    object_ids: List[int] = field(default_factory=list)         # detection ids chosen by the LLM (optional)
    target: str = 'bin'                                          # 'bin' | 'table_left' | ... | 'xy:0.1,-0.5'
    select_all: bool = False
    raw: str = ''
    source: str = 'rules'                                        # 'ollama' | 'rules'
    reply: str = ''                                              # something to tell the user

    def __str__(self):
        return (f'Command({self.action}, object="{self.object_phrase}", ids={self.object_ids}, '
                f'target={self.target}, all={self.select_all}, via {self.source})')


def _scene_text(dets: Sequence[Detection], catalog: Catalog, robot_xy=(0.0, 0.0)) -> str:
    lines = []
    for d in dets:
        name = catalog.display_name(d.label) if d.label else 'unknown object'
        lines.append(f'id={d.id}: {name}, colour {d.color_name}, height {d.height * 100:.0f} cm, '
                     f'x={d.center[0]:+.2f} y={d.center[1]:+.2f}')
    return '\n'.join(lines) if lines else '(no objects visible)'


SYSTEM_PROMPT = """You translate spoken instructions for a robot arm that picks objects from a table and puts them somewhere.
You are given the list of objects the camera currently sees. Answer ONLY with JSON:
{"action": "pick_place" | "home" | "list" | "status" | "stop" | "help" | "unknown",
 "object_ids": [integers from the scene list that the user means, [] if none],
 "all": true if the user wants every object (e.g. "clear the table"),
 "target": "bin" | "table_left" | "table_right" | "table_front" | "table_back" | "table_center",
 "reply": "one short sentence for the user"}
Rules: "bin", "trash", "garbage" -> target "bin". If no destination is given use "bin".
"left"/"right" refer to what you see in the camera picture; smaller x is further left.
Pick the object ids that best match the description (name, colour, size, position). Never invent ids.
If the user only chats or asks something unrelated use action "unknown" and answer in "reply"."""


class CommandParser:
    def __init__(self, catalog: Catalog, ollama: Optional[OllamaClient] = None, use_llm: bool = True):
        self.catalog = catalog
        self.ollama = ollama
        self.use_llm = use_llm and ollama is not None
        self.llm_ready = False
        if self.use_llm:
            m = ollama.ensure_model()
            self.llm_ready = m is not None
            log.info('Ollama model: %s', m if m else 'NOT AVAILABLE -> using rule-based parser')

    # ---------------------------------------------------------------- public
    def parse(self, text: str, dets: Sequence[Detection]) -> Command:
        text = text.strip()
        low = text.lower()
        quick = self._quick_actions(low)
        if quick:
            quick.raw = text
            return quick
        if self.use_llm and self.llm_ready:
            try:
                return self._parse_llm(text, dets)
            except Exception as e:
                log.warning('Ollama parsing failed (%s) - using rules', e)
        return self.parse_rules(text)

    # ---------------------------------------------------------------- LLM
    def _parse_llm(self, text: str, dets: Sequence[Detection]) -> Command:
        user = f'Objects on the table:\n{_scene_text(dets, self.catalog)}\n\nInstruction: {text}'
        out = self.ollama.chat_json(SYSTEM_PROMPT, user)
        action = str(out.get('action', 'unknown'))
        if action not in ACTIONS:
            raise ValueError(f'bad action {action!r}')
        valid = {d.id for d in dets}
        ids = [int(i) for i in out.get('object_ids', []) if isinstance(i, (int, float)) and int(i) in valid]
        target = str(out.get('target', 'bin'))
        if target not in PLACE_WORDS:
            target = 'bin'
        cmd = Command(action=action, object_ids=ids, target=target, select_all=bool(out.get('all', False)),
                      raw=text, source='ollama', reply=str(out.get('reply', '')))
        if action == 'pick_place' and not ids and not cmd.select_all:
            rules = self.parse_rules(text)                    # LLM could not ground it -> try the rules
            if rules.action == 'pick_place':
                rules.reply = cmd.reply or rules.reply
                return rules
        return cmd

    # ---------------------------------------------------------------- rules
    @staticmethod
    def _quick_actions(low: str) -> Optional[Command]:
        if re.fullmatch(r'\s*(stop|halt|abort|cancel|emergency stop|freeze)[.! ]*', low):
            return Command(action='stop')
        if re.fullmatch(r'\s*(help|\?|what can you do\??)[.! ]*', low):
            return Command(action='help')
        if re.search(r'\b(go|move|return|back)\b.*\b(home|rest|ready)\b|\bhome\b$', low) and 'pick' not in low:
            return Command(action='home')
        if re.search(r'\b(what|which)\b.*\b(see|visible|there|on the table)\b|\blist\b.*\b(objects|items)\b|\bwhat do you see\b', low):
            return Command(action='list')
        if re.fullmatch(r'\s*(status|state)[.! ]*', low):
            return Command(action='status')
        return None

    def parse_rules(self, text: str) -> Command:
        low = ' ' + ' '.join(text.lower().replace(',', ' ').replace('.', ' ').split()) + ' '
        cmd = Command(action='unknown', raw=text, source='rules')
        verbs = r'(pick|grab|take|get|fetch|lift|move|put|place|bring|throw|toss|drop|dump|remove|clear|clean|carry|transfer|relocate|push)'
        if not re.search(r'\b' + verbs + r'\b', low):
            cmd.reply = "I didn't understand. Try e.g. 'pick the mustard bottle and put it in the bin'."
            return cmd
        cmd.action = 'pick_place'
        # split "<verb> <object phrase> <prep> <target phrase>"
        m = re.search(r'\b(?:in|into|to|onto|on|inside|at|toward|towards|over)\b\s+(?:the\s+)?(.*)$', low)
        obj_part, tgt_part = low, ''
        if m:
            tgt_part = m.group(1)
            obj_part = low[:m.start()]
        cmd.target = self._parse_target(tgt_part) if tgt_part else 'bin'
        if re.search(r'\b(everything|all|every|each)\b|clear the table|clean( up)? the table', obj_part):
            cmd.select_all = True
        cmd.object_phrase = obj_part.strip()
        return cmd

    @staticmethod
    def _parse_target(phrase: str) -> str:
        p = ' ' + phrase + ' '
        for key in ('bin', 'table_left', 'table_right', 'table_front', 'table_back', 'table_center'):
            for w in PLACE_WORDS[key]:
                if re.search(r'\b' + re.escape(w) + r'\b', p):
                    return key
        return 'bin'


# ====================================================================== resolution
def _attr(d: Detection, name: str, robot_xy=(0.0, 0.0)) -> float:
    if name == 'height':
        return d.height
    if name == 'area_m2':
        return d.area_m2
    if name == 'long':
        return d.size[0]
    if name == 'short':
        return d.size[1]
    if name == 'x':
        return d.center[0]
    if name == 'dist':
        return math.hypot(d.center[0] - robot_xy[0], d.center[1] - robot_xy[1])
    if name == 'front':                       # smaller = closer to the robot (y towards 0)
        return -d.center[1]
    raise KeyError(name)


def resolve_targets(cmd: Command, dets: Sequence[Detection], catalog: Catalog,
                    left_right: str = 'image') -> List[Detection]:
    """Turn a command into the ordered list of detections to handle (may be empty)."""
    dets = list(dets)
    if not dets:
        return []
    if cmd.object_ids:
        by_id = {d.id: d for d in dets}
        return [by_id[i] for i in cmd.object_ids if i in by_id]
    if cmd.select_all:
        return sorted(dets, key=lambda d: (-d.height, d.center[0]))    # tall first: nothing falls on lower ones

    phrase = ' ' + cmd.object_phrase.lower() + ' '
    cand = dets
    # 1. name match (catalogue names / aliases)
    cids = catalog.match_name(phrase)
    named = False
    if cids:
        best_score_cid = cids[0]
        sel = [d for d in cand if d.label == best_score_cid]
        if sel:
            cand, named = sel, True
        else:
            # LLM-less fallback: the name matched a class that is not (or no longer) on the table
            # -> maybe the recognition got it wrong; use the 2nd-best candidate classes of each blob
            sel = [d for d in cand if any(c == best_score_cid for c, _ in d.candidates[:2])]
            if sel:
                cand, named = sel, True
            else:
                return []
    # 2. colour
    for col in COLOURS:
        if re.search(r'\b' + col + r'\b', phrase):
            colname = 'gray' if col == 'grey' else col
            sel = [d for d in cand if d.color_name == colname]
            if sel:
                cand = sel
            elif not named:
                return []
    # 3. generic category words (bottle, can, box ...) when no explicit name matched
    if not named:
        for cid, e in catalog.entries.items():
            for cat in e.categories:
                if re.search(r'\b' + cat + r's?\b', phrase):
                    sel = [d for d in cand if d.label and cat in catalog.entries[d.label].categories]
                    if sel:
                        cand = sel
                        break
            else:
                continue
            break
    # 4. superlatives / spatial words
    for word, (attr, sign) in SUPERLATIVES.items():
        if re.search(r'\b' + word + r'\b', phrase):
            cand = [min(cand, key=lambda d: sign * _attr(d, attr))]
            break
    else:
        sgn = 1 if left_right == 'image' else -1
        if re.search(r'\bleft\b', phrase):
            cand = [min(cand, key=lambda d: sgn * d.center[0])]
        elif re.search(r'\bright\b', phrase):
            cand = [max(cand, key=lambda d: sgn * d.center[0])]
        elif re.search(r'\b(front|nearest|closest)\b', phrase):
            cand = [min(cand, key=lambda d: _attr(d, 'dist'))]
        elif re.search(r'\b(back|far)\b', phrase):
            cand = [max(cand, key=lambda d: _attr(d, 'dist'))]
    if re.search(r'\b(both|two|2)\b', phrase):
        return sorted(cand, key=lambda d: _attr(d, 'dist'))[:2]
    if not named and len(cand) == len(dets) and not re.search(r'\b(it|one|thing|object|item|something|anything)\b', phrase):
        return []                                          # nothing in the phrase matched anything
    return sorted(cand, key=lambda d: _attr(d, 'dist'))[:1]
