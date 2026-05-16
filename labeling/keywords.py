"""
关键词库 — 用于兴趣领域和子领域的多级语义匹配
"""

# ============================================================
# 强信号关键词（权重3/2）— 出现即强烈指示
# ============================================================
STRONG_KW = {
    "科技": ["ai", "artificial intelligence", "machine learning", "deep learning",
             "software", "developer", "programmer", "编程", "代码", "开源",
             "科技", "技术", "数据", "算法", "区块链", "web3", "crypto",
             "bitcoin", "saas", "api", "云计算", "人工智能",
             "nlp", "llm", "大模型", "程序员", "前端", "后端", "全栈",
             "python", "java", "rust", "golang", "react", "vue",
             "startup tech", "ai community", "decentralized", "protocol",
             "layer", "smart contract", "infrastructure", "robot",
             "nvidia", "deepmind", "openai", "stability ai",
             "generative", "diffusion", "transformer",
             "数据挖掘", "量化", "分布式", "服务器"],

    "娱乐": ["singer", "actor", "actress", "musician", "music", "film", "movie",
             "偶像", "明星", "歌手", "演员", "娱乐", "粉丝", "追星",
             "kpop", "anime", "idol", "celebrity", "drama", "concert",
             "艺人", "comedy", "show", "talents",
             "pop", "rock", "band", "album", "vocal",
             "illustrator", "artist", "画师", "插画", "画画的",
             "絵描き", "イラストレーター", "sketch",
             "animator", "アニメーター", "animation",
             "concept artist", "character designer",
             "painting", "draw", "イラスト",
             "pixiv", "commission", "artwork", "digital art",
             "fanart", "同人", "doujin", "manga", "mangaka",
             "cosplay", "cosplayer", "coser", "コスプレ",
             "vtuber", "virtual youtuber", "hololive",
             "vocaloid", "ボカロ", "初音"],

    "体育": ["sport", "football", "soccer", "basketball", "nba", "fitness",
             "体育", "足球", "篮球", "健身", "运动", "马拉松", "跑步",
             "athlete", "coach", "premier league", "champions league",
             "tennis", "golf", "nfl", "ufc", "olympic", "世界杯"],

    "新闻": ["news", "journalist", "reporter", "media", "press", "politics",
             "新闻", "记者", "媒体", "政治", "时事", "报道",
             "government", "policy", "election", "correspondent",
             "专栏", "日报", "周刊", "opinion"],

    "游戏": ["gamer", "gaming", "game", "esports", "twitch", "steam",
             "游戏", "电竞", "steam", "playstation", "xbox", "nintendo",
             "league of legends", "valorant", "genshin", "原神", "王者荣耀",
             "minecraft", "fortnite", "dota", "pubg",
             "主播", "直播", "攻略", "开黑", "电竞选手",
             "arknights", "ff14", "最终幻想", "apex", "崩坏", "星穹铁道",
             "wuthering waves", "鸣潮", "hoyoverse", "miyoho"],

    "商业": ["business", "entrepreneur", "ceo", "founder", "startup",
             "商业", "创业", "金融", "投资", "营销", "经济",
             "investor", "trader", "stock", "venture", "fund",
             "consulting", "management", "executive",
             "股市", "基金", "理财", "vc", "天使投资",
             "finance", "market", "资本"],

    "时尚": ["fashion", "beauty", "makeup", "style", "model",
             "时尚", "美妆", "穿搭", "潮流", "护肤", "彩妆",
             "luxury", "brand", "cosmetics", "skincare",
             "fashion week", "runway", "穿搭博主"],

    "教育": ["teacher", "professor", "education", "academic", "research",
             "教育", "老师", "教授", "学术", "研究", "大学",
             "phd", "ph.d", "university", "college", "school",
             "lecture", "course", "learn", "study", "博士", "硕士",
             "teach", "teaching", "讲师", "导师", "lecturer",
             "researcher", "scientist"],

    "生活": ["food", "travel", "photography", "lifestyle", "cooking",
             "美食", "旅游", "摄影", "生活", "日常", "vlog",
             "cafe", "coffee", "pet", "cat", "dog", "宠物",
             "户外", "露营", "hiking", "camping",
             "garden", "nature", "flower", "花"],
}

# ============================================================
# 弱信号关键词（权重1）— 轻度提示
# ============================================================
WEAK_KW = {
    "科技": ["tech", "code", "digital", "computer", "science", "hack",
             "cyber", "dev", "cloud", "数据"],
    "娱乐": ["fan", "pop", "dance", "entertain", "搞笑", "番剧", "动漫",
             "entertainment"],
    "体育": ["gym", "workout", "yoga", "swim", "cycle", "active"],
    "新闻": ["world", "update", "headline", "观点", "新闻"],
    "游戏": ["console", "mobile game", "rpg", "策略", "卡牌", "手游"],
    "商业": ["money", "财富", "老板", "经理", "管理"],
    "时尚": ["outfit", "化妆品", "beauty"],
    "教育": ["study", "knowledge", "读书", "学习", "学生"],
    "生活": ["life", "daily", "分享", "记录"],
}

# ============================================================
# 兴趣子领域关键词
# ============================================================
SUB_KW = {
    "AI/数据科学": ["ai", "machine learning", "deep learning", "nlp", "data science",
                   "人工智能", "机器学习", "深度学习", "数据科学", "大模型", "llm"],
    "软件/开发": ["programming", "coding", "dev", "software", "前端", "后端", "全栈",
                 "python", "java", "rust", "react", "vue", "开源", "github",
                 "developer", "code", "程序员"],
    "区块链/Web3": ["blockchain", "crypto", "web3", "bitcoin", "ethereum", "defi",
                   "区块链", "加密货币", "比特币", "以太坊", "nft", "dao"],
    "硬件/IoT": ["iot", "robotics", "hardware", "embedded", "chip", "物联网",
                 "机器人", "硬件", "嵌入式", "芯片"],
    "影视/综艺": ["film", "movie", "tv", "drama", "hollywood", "netflix",
                 "电影", "综艺", "电视剧", "韩剧", "美剧", "日剧"],
    "音乐/演出": ["music", "singer", "concert", "pop", "rock", "band",
                 "音乐", "歌手", "演唱会", "演出", "kpop",
                 "vocaloid", "ボカロ", "初音"],
    "偶像/粉丝": ["kpop", "anime", "idol", "celebrity", "fan", "cosplay",
                 "偶像", "明星", "粉丝", "追星", "二次元", "anime",
                 "虚拟主播", "vtuber"],
    "足球": ["football", "soccer", "premier league", "champions league",
            "足球", "英超", "欧冠", "西甲", "世界杯"],
    "篮球": ["basketball", "nba", "篮球"],
    "健身/户外": ["fitness", "gym", "workout", "marathon", "yoga",
                 "健身", "跑步", "马拉松", "骑行", "登山", "户外", "露营"],
    "时事/政治": ["news", "politics", "government", "policy", "election",
                 "新闻", "政治", "时事", "政府", "外交"],
    "电竞/游戏": ["gamer", "esports", "game", "gaming", "twitch", "steam",
                 "游戏", "电竞", "直播", "原神", "王者荣耀", "apex",
                 "valorant", "lol", "ff14", "arknights", "genshin",
                 "崩坏", "星穹铁道", "hoyoverse"],
    "投资/金融": ["finance", "invest", "stock", "fund", "trader",
                 "金融", "投资", "股市", "基金", "理财", "证券"],
    "创业/商业": ["startup", "entrepreneur", "founder", "business",
                 "创业", "创始人", "商业", "管理", "vc", "venture"],
    "美妆/时尚": ["beauty", "fashion", "makeup", "skincare", "luxury",
                 "时尚", "美妆", "穿搭", "护肤", "彩妆", "潮流"],
    "教育/学术": ["education", "academic", "research", "phd", "professor",
                 "教育", "学术", "研究", "教授", "博士", "学生"],
    "美食/旅游": ["food", "travel", "cooking", "restaurant",
                 "美食", "旅游", "探店", "咖啡", "旅行"],
}
