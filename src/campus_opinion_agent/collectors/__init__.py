"""Data collectors for campus opinion sources."""

from campus_opinion_agent.collectors.tikhub_external import (
    TikHubExternalConfig,
    TikhubExternalCollector,
    WeiboComment,
    WeiboPost,
    XiaohongshuComment,
    XiaohongshuNote,
    ZhihuComment,
    ZhihuContent,
)

__all__ = [
    "TikHubExternalConfig",
    "TikhubExternalCollector",
    "WeiboComment",
    "WeiboPost",
    "XiaohongshuComment",
    "XiaohongshuNote",
    "ZhihuComment",
    "ZhihuContent",
]
