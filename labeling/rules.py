"""
基于社交指标的标签推断规则 — age_group, active_period
"""

def infer_age_group(fol: int, fri: int, sta: int, fav: int, ff_ratio: float) -> str:
    """基于社交指标的年龄组推断"""
    if fol > 500000 and ff_ratio > 50:
        return "35-50"
    if fol > 100000 and ff_ratio > 10:
        return "25-34"
    if fol > 200000 and ff_ratio < 5:
        return "35-50"
    if sta > 20000 and fol < 5000 and fri > 500:
        return "18-24"
    if fol < 100 and fri < 100 and sta < 500:
        return "50+"
    if fol > 5000 and ff_ratio > 3:
        return "25-34"
    if sta > 5000 and fol < 1000:
        return "18-24"
    if fol > 50000:
        return "35-50"
    if fol > 1000 or sta > 1000:
        return "25-34"
    if ff_ratio < 0.5:
        return "18-24"
    return "25-34"


def infer_active_period(fol: int, fri: int, sta: int, ff_ratio: float) -> str:
    """基于社交指标的活跃时段推断"""
    density = sta / max(fol, 1)
    if density > 10 and ff_ratio < 0.5:
        return "night_owl"
    if density > 2 and ff_ratio > 3:
        return "evening"
    if ff_ratio > 50 and sta > 1000:
        return "afternoon"
    if density < 0.01 and fol > 10000:
        return "afternoon"
    if sta < 100 and fol < 100:
        return "morning"
    if density > 0.5 and ff_ratio < 1:
        return "night_owl"
    return "morning"


def infer_account_scale(followers_count: int) -> str:
    """基于粉丝数的账号规模分桶"""
    if followers_count > 100000:
        return "mega"
    elif followers_count > 10000:
        return "macro"
    elif followers_count > 1000:
        return "mid"
    elif followers_count > 100:
        return "micro"
    else:
        return "nano"


def infer_content_language(text: str) -> str:
    """基于字符统计的内容语言检测"""
    text = str(text).strip()
    if not text:
        return "en"
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    cjk += sum(1 for c in text if '぀' <= c <= 'ヿ')
    cjk += sum(1 for c in text if '가' <= c <= '힯')
    ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total = cjk + ascii_chars
    if total == 0:
        return "en"
    if cjk / total > 0.3 and ascii_chars / total > 0.3:
        return "mixed"
    elif cjk / total > 0.3:
        return "zh"
    else:
        return "en"
