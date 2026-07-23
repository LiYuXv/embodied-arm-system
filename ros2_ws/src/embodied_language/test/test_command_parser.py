"""Tests for the rule-based Chinese command parser."""

import pytest

from embodied_language.command_parser import CommandParser


@pytest.fixture
def parser() -> CommandParser:
    return CommandParser()


@pytest.mark.parametrize(
    ("text", "expected_target"),
    [
        ("回家", "home"),
        ("回到初始位置", "home"),
        ("回到初始位置。", "home"),
        ("复位", "home"),
        ("移动到观察位置", "observe"),
        ("去观察位置", "observe"),
        ("进入准备状态", "ready"),
        ("移动到准备位姿", "ready"),
        ("准备抓取", "pre_pick"),
        ("移动到预抓取位置", "pre_pick"),
    ],
)
def test_parse_named_pose_commands(
    parser: CommandParser,
    text: str,
    expected_target: str,
) -> None:
    result = parser.parse(text)

    assert result is not None
    assert result.raw_text == text
    assert result.action == "go_named_pose"
    assert result.target == expected_target


@pytest.mark.parametrize(
    ("text", "expected_target"),
    [
        ("打开夹爪", "open"),
        ("张开夹爪", "open"),
        ("松开夹爪", "open"),
        ("夹爪打开", "open"),
        ("关闭夹爪", "close"),
        ("闭合夹爪", "close"),
        ("合上夹爪", "close"),
        ("夹爪关闭", "close"),
    ],
)
def test_parse_gripper_commands(
    parser: CommandParser,
    text: str,
    expected_target: str,
) -> None:
    result = parser.parse(text)

    assert result is not None
    assert result.raw_text == text
    assert result.action == "set_gripper"
    assert result.target == expected_target


@pytest.mark.parametrize(
    ("text", "target", "target_region"),
    [
        ("把红色方块抓到红色位置", "red_cube", "red_target_zone"),
        ("将红色方块放到红色区域", "red_cube", "red_target_zone"),
        ("把红色方块移动到红色区域", "red_cube", "red_target_zone"),
        ("把蓝色方块抓到蓝色位置", "blue_cube", "blue_target_zone"),
        ("将蓝色方块放到蓝色区域", "blue_cube", "blue_target_zone"),
        ("把 蓝色 方块 移动到 蓝色 位置。", "blue_cube", "blue_target_zone"),
    ],
)
def test_parse_colour_pick_place_commands(
    parser: CommandParser,
    text: str,
    target: str,
    target_region: str,
) -> None:
    """All supported colour, verb, and region wording maps consistently."""
    result = parser.parse(text)

    assert result is not None
    assert result.action == "pick_place"
    assert result.target == target
    assert result.target_region == target_region


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "随便动一下",
        "抓取红色方块",
    ],
)
def test_parse_unknown_commands_returns_none(
    parser: CommandParser,
    text: str,
) -> None:
    assert parser.parse(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" 回 家 ", "回家"),
        ("回到初始位置。", "回到初始位置"),
        ("准备抓取！", "准备抓取"),
        ("打 开 夹 爪！", "打开夹爪"),
    ],
)
def test_normalize(
    text: str,
    expected: str,
) -> None:
    assert CommandParser.normalize(text) == expected
