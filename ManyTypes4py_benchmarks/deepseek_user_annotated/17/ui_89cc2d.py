#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: omi
# @Date:   2014-08-24 21:51:57
# KenHuang:
# 1.增加显示颜色自定义；
# 2.调整显示格式；
"""
网易云音乐 Ui
"""
import curses
import datetime
import os
import re
from shutil import get_terminal_size
from typing import Any, Dict, List, Optional, Tuple, Union

from . import logger
from .config import Config
from .scrollstring import scrollstring
from .scrollstring import truelen
from .scrollstring import truelen_cut
from .storage import Storage
from .utils import md5

log = logger.getLogger(__name__)

try:
    import dbus

    dbus_activity = True
except ImportError:
    dbus_activity = False
    log.warn("Qt dbus module is not installed.")
    log.warn("Osdlyrics is not available.")


def break_substr(s: str, start: int, max_len: int = 80) -> str:
    if truelen(s) <= max_len:
        return s
    res: List[str] = []
    current_truelen = 0
    start_pos = 0
    end_pos = 0
    for c in s:
        current_truelen += 2 if c > chr(127) else 1
        if current_truelen > max_len:
            res.append(s[start_pos:end_pos])
            current_truelen = 0
            start_pos = end_pos + 1
            end_pos += 1
        else:
            end_pos += 1
    try:
        res.append(s[start_pos:end_pos])
    except Exception:
        pass
    return "\n{}".format(" " * start).join(res)


def break_str(s: str, start: int, max_len: int = 80) -> str:
    res: List[str] = []
    for substr in s.splitlines():
        res.append(break_substr(substr, start, max_len))
    return "\n{}".format(" " * start).join(res)


class Ui(object):
    def __init__(self) -> None:
        self.screen: Any = curses.initscr()
        curses.start_color()
        if Config().get("curses_transparency"):
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_CYAN, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
        else:
            colors = Config().get("colors")
            if (
                "TERM" in os.environ
                and os.environ["TERM"] == "xterm-256color"
                and colors
            ):
                curses.use_default_colors()
                for i in range(1, 6):
                    color = colors["pair" + str(i)]
                    curses.init_pair(i, color[0], color[1])
                self.screen.bkgd(32, curses.color_pair(5))
            else:
                curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
                curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
                curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
                curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        # term resize handling
        self.config: Config = Config()
        size: Tuple[int, int] = get_terminal_size()
        self.x: int = size[0]
        self.y: int = size[1]
        self.playerX: int = 1
        self.playerY: int = 0
        self.update_margin()
        self.update_space()
        self.lyric: str = ""
        self.now_lyric: str = ""
        self.post_lyric: str = ""
        self.now_lyric_index: int = 0
        self.now_tlyric_index: int = 0
        self.tlyric: str = ""
        self.storage: Storage = Storage()
        self.newversion: bool = False

    def addstr(self, *args: Any) -> None:
        if len(args) == 1:
            self.screen.addstr(args[0])
        else:
            try:
                self.screen.addstr(args[0], args[1], args[2].encode("utf-8"), *args[3:])
            except Exception as e:
                log.error(e)

    def update_margin(self) -> None:
        # Left margin
        self.left_margin_ratio: float = self.config.get("left_margin_ratio")
        if self.left_margin_ratio == 0:
            self.startcol: int = 0
        else:
            self.startcol = max(int(float(self.x) / self.left_margin_ratio), 0)
        self.indented_startcol: int = max(self.startcol - 3, 0)
        # Right margin
        self.right_margin_ratio: float = self.config.get("right_margin_ratio")
        if self.right_margin_ratio == 0:
            self.endcol: int = 0
        else:
            self.endcol = max(
                int(float(self.x) - float(self.x) / self.right_margin_ratio),
                self.startcol + 1,
            )

        self.indented_endcol: int = max(self.endcol - 3, 0)
        self.content_width: int = self.endcol - self.startcol - 1

    def build_playinfo(
        self, song_name: str, artist: str, album_name: str, quality: str, start: int, pause: bool = False
    ) -> None:
        curses.noecho()
        curses.curs_set(0)
        # refresh top 2 line
        self.screen.move(1, 1)
        self.screen.clrtoeol()
        self.screen.move(2, 1)
        self.screen.clrtoeol()
        if pause:
            self.addstr(
                1, self.indented_startcol, "_ _ z Z Z " + quality, curses.color_pair(3)
            )
        else:
            self.addstr(
                1, self.indented_startcol, "♫  ♪ ♫  ♪ " + quality, curses.color_pair(3)
            )

        song_info: str
        if artist or album_name:
            song_info = "{}{}{}  < {} >".format(
                song_name,
                self.space,
                artist,
                album_name,
            )
        else:
            song_info = song_name

        if truelen(song_info) <= self.endcol - self.indented_startcol - 19:
            self.addstr(
                1,
                min(self.indented_startcol + 18, self.indented_endcol - 1),
                song_info,
                curses.color_pair(4),
            )
        else:
            song_info = scrollstring(song_info + " ", start)
            self.addstr(
                1,
                min(self.indented_startcol + 18, self.indented_endcol - 1),
                truelen_cut(str(song_info), self.endcol - self.indented_startcol - 19),
                curses.color_pair(4),
            )

        self.screen.refresh()

    def update_lyrics(self, now_playing: float, lyrics: List[str], tlyrics: List[str]) -> None:

        timestap_regex: str = r"[0-5][0-9]:[0-5][0-9]\.[0-9]*"

        def get_timestap(lyric_line: str) -> str:
            match_ret = re.match(r"\[(" + timestap_regex + r")\]", lyric_line)
            if match_ret:
                return match_ret.group(1)
            else:
                return ""

        def get_lyric_time(lyric_line: str) -> datetime.timedelta:
            lyric_timestap: str = get_timestap(lyric_line)
            if lyric_timestap == "":
                return datetime.timedelta(seconds=now_playing)
            else:
                return (
                    datetime.datetime.strptime(get_timestap(lyric_line), "%M:%S.%f")
                    - datetime.datetime.strptime("00:00", "%M:%S")
                    - lyric_time_offset
                )

        def strip_timestap(lyric_line: str) -> str:
            return re.sub(r"\[" + timestap_regex + r"\]", r"", lyric_line)

        def append_translation(translated_lyric: str, origin_lyric: str) -> str:
            translated_lyric = strip_timestap(translated_lyric)
            origin_lyric = strip_timestap(origin_lyric)
            if translated_lyric == "" or origin_lyric == "":
                return translated_lyric + origin_lyric
            return translated_lyric + " || " + origin_lyric

        if (
            tlyrics and self.now_tlyric_index >= len(tlyrics) - 1
        ) or self.now_lyric_index >= len(lyrics) - 1:
            self.post_lyric = ""
            return

        lyric_time_offset: datetime.timedelta = datetime.timedelta(seconds=0.5)
        next_lyric_time: datetime.timedelta = get_lyric_time(lyrics[self.now_lyric_index + 1])
        now_time: datetime.timedelta = datetime.timedelta(seconds=now_playing)
        while now_time >= next_lyric_time and self.now_lyric_index < len(lyrics) - 2:
            self.now_lyric_index = self.now_lyric_index + 1
            next_lyric_time = get_lyric_time(lyrics[self.now_lyric_index + 1])

        if tlyrics:
            next_tlyric_time: datetime.timedelta = get_lyric_time(tlyrics[self.now_tlyric_index + 1])
            while (
                now_time >= next_tlyric_time
                and self.now_tlyric_index < len(tlyrics) - 2
            ):
                self.now_tlyric_index = self.now_tlyric_index + 1
                next_tlyric_time = get_lyric_time(tlyrics[self.now_tlyric_index + 1])

        if tlyrics:
            self.now_lyric = append_translation(
                tlyrics[self.now_tlyric_index], lyrics[self.now_lyric_index]
            )
            if (
                self.now_tlyric_index < len(tlyrics) - 1
                and self.now_lyric_index < len(lyrics) - 1
            ):
                self.post_lyric = append_translation(
                    tlyrics[self.now_tlyric_index + 1], lyrics[self.now_lyric_index + 1]
                )
            else:
                self.post_lyric = ""
        else:
            self.now_lyric = strip_timestap(lyrics[self.now_lyric_index])
            if self.now_lyric_index < len(lyrics) - 1:
                self.post_lyric = strip_timestap(lyrics[self.now_lyric_index + 1])
            else:
                self.post_lyric = ""

    def build_process_bar(
        self, song: Dict[str, Any], now_playing: float, total_length: float, playing_flag: bool, playing_mode: int
    ) -> None:

        if not song or not playing_flag:
            return
        name: str = song["song_name"]
        artist: str = song["artist"]
        lyrics: List[str] = song.get("lyric", [])
        tlyrics: List[str] = song.get("tlyric", [])

        curses.noecho()
        curses.curs_set(0)
        self.screen.move(3, 1)
        self.screen.clrtoeol()
        self.screen.move(4, 1)
        self.screen.clrtoeol()
        self.screen.move(5, 1)
        self.screen.clrtoeol()
        self.screen.move(6, 1)
        self.screen.clrtoeol()
        if total_length <= 0:
            total_length = 1
        if now_playing > total_length or now_playing <= 0:
            now_playing = 0
        if int(now_playing) == 0:
            self.now_lyric_index = 0
            if tlyrics:
                self.now_tlyric_index = 0
            self.now_lyric = ""
            self.post_lyric = ""
        process: str = "["
        process_bar_width: int = self.content_width - 24
        for i in range(0, process_bar_width):
            if i < now_playing / total_length * process_bar_width:
                if (i + 1) > now_playing / total_length * process_bar_width:
                    if playing_flag:
                        process += ">"
                        continue
                process += "="
            else:
                process += " "
        process += "] "

        now: str = str(datetime.timedelta(seconds=int(now_playing))).lstrip("0").lstrip(":")
        total: str = str(datetime.timedelta(seconds=total_length)).lstrip("0").lstrip(":")
        process += "({}/{})".format(now, total)

        if playing_mode == 0:
            process = "顺序播放 " + process
        elif playing_mode == 1:
            process = "顺序循环 " + process
        elif playing_mode == 2:
            process = "单曲循环 " + process
        elif playing_mode == 3:
            process = "随机播放 " + process
        elif playing_mode == 4:
            process = "随机循环 " + process
        else:
            pass
        self.addstr(3, self.startcol - 2, process, curses.color_pair(1))
        if not lyrics:
            self.now_lyric = "暂无歌词 ~>_<~ \n"
            self.post_lyric = ""
            if dbus_activity and self.config.get("osdlyrics"):
                self.now_playing = "{} - {}\n".format(name, artist)
        else:
            self.update_lyrics(now_playing, lyrics, tlyrics)

        if dbus_activity and self.config.get("osdlyrics"):
            try:
                bus = dbus.SessionBus().get_object("org.musicbox.Bus", "/")
                if self.now_lyric == "暂无歌词 ~>_<~ \n":
                    bus.refresh_lyrics(
                        self.now_playing, dbus_interface="local.musicbox.Lyrics"
                    )
                else:
                    bus.refresh_lyrics(
                        self.now_lyric, dbus_interface="local.musicbox.Lyrics"
                    )
            except Exception as e:
                log.error(e)
                pass
        if self.now_lyric_index % 2 == 0:
            self.addstr(
                4, max(self.startcol - 2, 0), str(self.now_lyric), curses.color_pair(3)
            )
            self.addstr(
                5, max(self.startcol + 1, 0), str(self.post_lyric), curses.A_DIM
            )
        else:
            self.addstr(
                4, max(self.startcol - 2, 0), str(self.post_lyric), curses.A_DIM
            )
            self.addstr(
                5, max(self.startcol + 1, 0), str(self.now_lyric), curses.color_pair(3)
            )
        self.screen.refresh()

    def build_loading(self) -> None:
        curses.curs_set(0)
        self.addstr(7, self.startcol, "享受高品质音乐，loading...", curses.color_pair(1))
        self.screen.refresh()

    def build_submenu(self, data: Any) -> None:
        pass

    def build_menu(
        self, datatype: str, title: str, datalist: List[Any], offset: int, index: int, step: int, start: int
    ) -> None:
        curses.noecho()
        curses.curs_set(0)
        self.screen.move(7, 1)
        self.screen.clrtobot()
        self.addstr(7, self.startcol, title, curses.color_pair(1))

        if len(datalist) == 0:
            self.addstr(8, self.startcol, "这里什么都没有 -，-")
            return self.screen.refresh()

        if datatype == "main":
            for i in range(offset, min(len(datalist), offset + step)):
                if i == index:
                    self.addstr(
                        i - offset + 9,
                        self.indented_startcol,
                        "-> " + str(i) + ". " + datalist[i]["entry_name"],
                        curses.color_pair(2),
                    )
                else:
                    self.addstr(
                        i - offset + 9,
                        self.startcol,
                        str(i) + ". " + datalist[i]["entry_name"],
                    )

        elif datatype in ("songs", "djprograms", "fmsongs"):
            iter_range = min(len(datalist), offset + step)
            for i in range(offset, iter_range):
                if isinstance(datalist[i], str):
                    raise ValueError(datalist)
                if i == index:
                    self.addstr(i - offset + 9, 0, " " * self.startcol)
                    lead = "-> " + str(i) + ". "
                    self.addstr(
                        i - offset + 9,
                        self.indented_startcol,
                        lead,
                        curses.color_pair(2),
                    )
                    name = "{}{}{}  < {} >".format(
                        datalist[i]["song_name"],
                        self.space,
                        datalist[i]["artist"],
                        datalist[i]["album_name"],
                    )

                    if truelen(name) < self.content_width:
                        self.addstr(
                            i - offset + 9,
                            self.indented_startcol + len(lead),
                            name,
                            curses.color_pair(2),
                        )
                    else:
                        name = scrollstring(name + "  ", start)
                        self.addstr(
                            i - offset +