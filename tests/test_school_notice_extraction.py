"""School red-header notice fixture and its expected structured extraction.

This validates the document-header parsing needed around RaNER.  RaNER itself
only identifies fixed entity categories, so title, release date, and request
are intentionally extracted from the notice's conventional layout.
"""

import re
from dataclasses import dataclass

SCHOOL_NOTICE = """华东师范大学文件

华师学工〔2026〕18号

关于做好2026届毕业生离校工作的通知

2026年8月2日
华东师范大学学生工作部（处）

各学院、各有关单位：

为做好2026届毕业生离校服务工作，现将有关事项通知如下。请各学院于2026年8月15日前
完成毕业生档案核验、离校手续确认和就业去向信息更新，并将异常情况报送学生工作部（处）。

华东师范大学学生工作部（处）
2026年8月2日
"""


@dataclass(frozen=True)
class NoticeFields:
    title: str
    published_at: str
    issuing_organization: str
    request: str


def extract_notice_fields(text: str) -> NoticeFields:
    """Extract the four explicit header/body fields in this red-header notice fixture."""
    title_match = re.search(r"^关于.+?的通知$", text, flags=re.MULTILINE)
    date_match = re.search(r"^发布时间：(?P<value>.+)$", text, flags=re.MULTILINE)
    organization_match = re.search(r"^发文单位：(?P<value>.+)$", text, flags=re.MULTILINE)
    request_match = re.search(r"(?P<value>请各学院于.+?。)", text, flags=re.DOTALL)

    assert title_match is not None
    assert date_match is not None
    assert organization_match is not None
    assert request_match is not None
    return NoticeFields(
        title=title_match.group(),
        published_at=date_match.group("value"),
        issuing_organization=organization_match.group("value"),
        request=re.sub(r"\s+", "", request_match.group("value")),
    )


def test_extract_school_red_header_notice_fields() -> None:
    assert extract_notice_fields(SCHOOL_NOTICE) == NoticeFields(
        title="关于做好2026届毕业生离校工作的通知",
        published_at="2026年8月2日",
        issuing_organization="华东师范大学学生工作部（处）",
        request=(
            "请各学院于2026年8月15日前完成毕业生档案核验、离校手续确认和就业去向信息更新，"
            "并将异常情况报送学生工作部（处）。"
        ),
    )
