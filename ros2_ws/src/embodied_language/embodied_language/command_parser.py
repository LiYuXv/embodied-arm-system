"""中文机械臂指令的基础规则解析器."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParsedCommand:
    """自然语言指令的结构化解析结果."""

    raw_text: str
    action: str
    target: str
    target_region: str = ""


class CommandParser:
    """将基础中文指令映射为机械臂标准任务命令."""

    _NAMED_POSE_RULES = {
        "home": (
            "回到初始位置",
            "回到初始位姿",
            "回到原位",
            "回家",
            "复位",
        ),
        "observe": (
            "移动到观察位置",
            "移动到观察位姿",
            "去观察位置",
            "到观察位置",
        ),
        "ready": (
            "移动到准备位置",
            "移动到准备位姿",
            "去准备位置",
            "进入准备状态",
        ),
        "pre_pick": (
            "准备抓取",
            "移动到预抓取位置",
            "移动到预抓取位姿",
            "去预抓取位置",
            "到预抓取位置",
        ),
    }

    _GRIPPER_RULES = {
        "open": (
            "打开夹爪",
            "张开夹爪",
            "松开夹爪",
            "夹爪打开",
        ),
        "close": (
            "关闭夹爪",
            "闭合夹爪",
            "合上夹爪",
            "夹爪关闭",
        ),
    }

    _IGNORED_CHARACTERS = "，。！？!?、"

    _COLOUR_TARGETS = {
        "红色": ("red_cube", "red_target_zone"),
        "蓝色": ("blue_cube", "blue_target_zone"),
    }
    _PICK_PLACE_PREFIXES = ("把", "将")
    _PICK_PLACE_VERBS = ("抓到", "放到", "移动到")
    _PICK_PLACE_REGION_SUFFIXES = ("位置", "区域")

    @classmethod
    def normalize(cls, text: str) -> str:
        """移除空白和常见中文标点."""
        normalized = "".join(text.strip().split())

        for character in cls._IGNORED_CHARACTERS:
            normalized = normalized.replace(character, "")

        return normalized

    def parse(self, text: str) -> Optional[ParsedCommand]:
        """解析中文指令；无法识别时返回 None."""
        normalized_text = self.normalize(text)

        if not normalized_text:
            return None

        pick_place = self._parse_pick_place(text, normalized_text)
        if pick_place is not None:
            return pick_place

        for target, phrases in self._NAMED_POSE_RULES.items():
            for phrase in phrases:
                if normalized_text == self.normalize(phrase):
                    return ParsedCommand(
                        raw_text=text,
                        action="go_named_pose",
                        target=target,
                    )

        for target, phrases in self._GRIPPER_RULES.items():
            for phrase in phrases:
                if normalized_text == self.normalize(phrase):
                    return ParsedCommand(
                        raw_text=text,
                        action="set_gripper",
                        target=target,
                    )

        return None

    def _parse_pick_place(
        self,
        raw_text: str,
        normalized_text: str,
    ) -> Optional[ParsedCommand]:
        """Parse colour-independent Chinese pick-and-place sentences."""
        for colour, (target, target_region) in self._COLOUR_TARGETS.items():
            object_phrase = f"{colour}方块"
            for prefix in self._PICK_PLACE_PREFIXES:
                for verb in self._PICK_PLACE_VERBS:
                    for suffix in self._PICK_PLACE_REGION_SUFFIXES:
                        expected = (
                            f"{prefix}{object_phrase}{verb}{colour}{suffix}"
                        )
                        if normalized_text == expected:
                            return ParsedCommand(
                                raw_text=raw_text,
                                action="pick_place",
                                target=target,
                                target_region=target_region,
                            )
        return None
