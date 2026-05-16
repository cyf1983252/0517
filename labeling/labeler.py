"""
增强型用户属性打标器

对每个用户独立推理，不依赖其他用户信息，确保无 train/test 泄露。
基于多字段语义分析、逐级优先级匹配、改进决策树。
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, Any

from .keywords import STRONG_KW, WEAK_KW, SUB_KW
from .rules import infer_age_group, infer_active_period, infer_account_scale, infer_content_language


# ---- 单词边界匹配 ----
_kw_cache = {}

def word_match(kw: str, text: str) -> bool:
    """带词边界的关键词匹配，避免 'ai' 误匹配 'gmail' 等"""
    if kw not in _kw_cache:
        escaped = re.escape(kw)
        if len(kw) <= 4:
            _kw_cache[kw] = re.compile(
                r'(?<![a-z0-9])' + escaped + r'(?![a-z0-9])', re.IGNORECASE)
        else:
            _kw_cache[kw] = re.compile(escaped, re.IGNORECASE)
    return bool(_kw_cache[kw].search(text))


# ---- 职业模板（按优先级逐级匹配） ----
_OCCUPATION_CHECKS = [
    # 技术/工程
    (["engineer", "developer", "programmer", "architect", "devops",
      "software", "tech lead", "cto", "工程狮", "开发", "程序员",
      "架构师", "全栈", "前端", "后端", "运维", "技术",
      "data scientist", "ml engineer", "ai engineer",
      "frontend", "backend", "full stack", "engineering",
      "infrastructure", "platform engineer", "系统",
      "blockchain", "web3", "crypto", "solidity",
      "data engineer", "ml", "data science"], "技术/工程"),

    # 艺术/设计
    (["illustrator", "artist", "painter", "画师", "插画", "画画的",
      "絵描き", "イラストレーター", "sketch",
      "animator", "アニメーター", "animation", "visual artist",
      "concept artist", "character designer",
      "graphic designer", "ux designer", "ui designer",
      "creative director", "art director",
      "freelance illustrator", "artwork", "digital art",
      "cartoonist", "mangaka", "漫画家",
      "painting", "draw", "イラスト",
      "pixiv", "commission", "絵", "絵师",
      "drawing", "paints", "painted",
      "3d artist", "modeler", "sculptor"], "艺术/设计"),

    # 媒体/内容
    (["writer", "author", "journalist", "editor", "reporter",
      "content creator", "blogger", "copywriter", "novelist",
      "记者", "编辑", "作家", "自媒体", "content creator",
      "youtuber", "streamer", "influencer", "主播", "up主",
      "媒体", "撰稿", "专栏", "vloger", "vlogger",
      "youtube", "tiktok", "correspondent",
      "screenwriter", "脚本",
      "musician", "composer", "producer", "dj", "singer",
      "音乐人", "作曲家", "歌手", "音乐制作",
      "photographer", "摄影", "摄影师", "写真",
      "cosplayer", "coser", "cosplay", "コスプレ",
      "model", "模特", "actor", "actress", "演员",
      "director", "导演", "film maker",
      "dancer", "舞者", "编舞", "choreographer"], "媒体/内容"),

    # 商业/管理（排除纯技术/艺术工种）
    (["ceo", "founder", "co-founder", "manager", "director",
      "executive", "entrepreneur", "vp", "president",
      "创始人", "总监", "总裁", "总经理", "经理",
      "management", "head of", "lead", "coo", "cfo",
      "partner", "managing director"], "商业/管理"),

    # 学术/教育
    (["professor", "teacher", "lecturer", "researcher", "scientist",
      "phd", "ph.d", "academic", "faculty",
      "教授", "老师", "教师", "研究员", "博士", "讲师",
      "teach", "education", "teaching",
      "postdoc", "post-doc", "dean", "助教"], "学术/教育"),

    # 学生
    (["student", "undergrad", "graduate", "大学", "学生",
      "在学", "在读", "college", "university student",
      "在校", "高校", "大学生"], "学生"),

    # 市场/销售
    (["marketing", "sales", "marketer", "brand manager", "pr",
      "市场", "销售", "营销", "运营", "品牌", "广告",
      "growth", "bd", "business development",
      "seo", "social media manager"], "市场/销售"),

    # 医疗/健康
    (["doctor", "nurse", "physician", "medical", "healthcare",
      "surgeon", "therapist", "psychologist",
      "医生", "护士", "医师", "药师", "医疗", "临床",
      "hospital", "clinic", "med student"], "医疗/健康"),

    # 法律/政务
    (["lawyer", "attorney", "legal", "judge", "law firm",
      "律师", "法官", "法律", "法务", "合规",
      "government", "ambassador", "politician",
      "senator", "congress", "公務員", "公务员", "officer",
      "military", "army", "navy", "diplomat",
      "diplomatic", "政"], "法律/政务"),
]

_NAME_OCCUPATION_HINTS = [
    (["teacher", "prof", "dr.", "dr "], "学术/教育"),
    (["engineer", "dev", "coder", "hack"], "技术/工程"),
    (["art", "draw", "paint", "sketch", "design", "illus"], "艺术/设计"),
    (["ceo", "founder", "boss", "coo", "cto"], "商业/管理"),
    (["writer", "blog", "press", "news", "journal"], "媒体/内容"),
    (["doctor", "nurse", "med"], "医疗/健康"),
    (["law", "legal", "attorney"], "法律/政务"),
    (["student", "pupil"], "学生"),
]

_DOMAIN_TO_SUB = {
    "科技": "软件/开发", "娱乐": "偶像/粉丝",
    "体育": "健身/户外", "新闻": "时事/政治",
    "游戏": "电竞/游戏", "商业": "创业/商业",
    "时尚": "美妆/时尚", "教育": "教育/学术",
    "生活": "美食/旅游", "其他": "其他子类",
}


def _occupation_from_desc(desc: str) -> str:
    """从描述文字逐级匹配职业"""
    for kw_list, label in _OCCUPATION_CHECKS:
        if any(word_match(kw, desc) for kw in kw_list):
            return label
    return None


def _occupation_from_name(name: str, screen: str) -> str:
    """从昵称/用户名推断职业线索"""
    for kw_list, label in _NAME_OCCUPATION_HINTS:
        if any(word_match(kw, name) for kw in kw_list):
            return label
        if any(word_match(kw, screen) for kw in kw_list):
            return label
    return "其他职业"


def label_user(row: pd.Series) -> Dict[str, Any]:
    """
    为单个用户生成全部 7 个属性标签。
    每个用户的标签完全独立，无跨用户信息泄露。
    """
    desc = str(row.get("description_norm", ""))
    dname = str(row.get("name", "")).lower()
    sname = str(row.get("screen_name", "")).lower()

    fol = max(int(row.get("followers_count", 0) or 0), 0)
    fri = max(int(row.get("friends_count", 0) or 0), 0)
    sta = max(int(row.get("statuses_count", 0) or 0), 0)
    fav_val = row.get("favourites_count", 0)
    if pd.isna(fav_val):
        fav_val = 0
    fav = max(int(fav_val), 0)
    ff_ratio = fol / max(fri, 1)

    text = f"{desc} {desc} {dname} {sname}".lower()

    # === interest_domain ===
    scores = {}
    for cat in STRONG_KW:
        s_score = sum(3 for kw in STRONG_KW[cat] if word_match(kw, text))
        s_desc = sum(2 for kw in STRONG_KW[cat] if word_match(kw, desc))
        scores[cat] = s_score + s_desc

    for cat in WEAK_KW:
        if cat not in scores:
            scores[cat] = 0
        w_score = sum(1 for kw in WEAK_KW[cat] if word_match(kw, text))
        scores[cat] += w_score * 0.5

    best_cat = max(scores, key=scores.get)
    interest_domain = "其他" if (best_cat in ("生活", "其他") and scores[best_cat] == 0
                                 and (desc == "" or desc == "nan")) else best_cat
    if scores[best_cat] == 0:
        interest_domain = "其他" if (desc == "" or desc == "nan") else "生活"

    # === interest_sub ===
    sub_scores = {}
    for cat, keywords in SUB_KW.items():
        score = sum(1 for kw in keywords if word_match(kw, text))
        if score > 0:
            sub_scores[cat] = score
    interest_sub = max(sub_scores, key=sub_scores.get) if sub_scores \
        else _DOMAIN_TO_SUB.get(interest_domain, "其他子类")

    # === occupation_type ===
    occ = _occupation_from_desc(desc)
    if occ is None:
        occ = _occupation_from_name(dname, sname)

    # === age_group / active_period ===
    age_group = infer_age_group(fol, fri, sta, fav, ff_ratio)
    active_period = infer_active_period(fol, fri, sta, ff_ratio)

    # === account_scale / content_language ===
    account_scale = infer_account_scale(fol)
    content_language = infer_content_language(desc)

    return {
        "interest_domain": interest_domain,
        "interest_sub": interest_sub,
        "occupation_type": occ,
        "age_group": age_group,
        "active_period": active_period,
        "account_scale": account_scale,
        "content_language": content_language,
    }


def label_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    为 DataFrame 中所有用户生成标签。
    每行独立推理，适合并行处理。

    返回: 原 DataFrame 副本，追加 7 个标签列
    """
    result = df.copy()
    labels = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if verbose and i > 0 and i % 2000 == 0:
            print(f"  labeling: {i}/{total}")
        labels.append(label_user(row))

    label_df = pd.DataFrame(labels)
    for col in label_df.columns:
        result[col] = label_df[col].values

    return result
